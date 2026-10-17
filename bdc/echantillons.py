"""Reconstruction du tableau des échantillons, ligne par ligne.

1. La rangée d'en-tête est repérée par son vocabulaire (Réf., Profondeur,
   HAP 16, MS…) ; chaque colonne reçoit un type (référence, parcelle,
   profondeur, matrice, case d'analyse, analyses en texte, observations…).
   Aucune position n'est codée en dur : un gabarit inconnu est lu de la même
   façon s'il emploie ce vocabulaire.
2. Chaque rangée devient un échantillon : cases cochées, cases cochées puis
   barrées (« \\/\\ non »), valeurs raturées et corrigées (la valeur
   corrigée est retenue, la valeur barrée est tracée), lignes annulées
   (barrées sur toute la largeur ou « annulé » en observation), lignes
   ajoutées à la main.
3. Les références sont harmonisées entre lignes (préfixe commun, chiffres
   et lettres attendus) uniquement en choisissant parmi les lectures des
   moteurs OCR : on n'invente jamais une référence illisible (null + alerte).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from . import encre as E
from .ocr import Lecture
from .tableau import AIRE_MIN, Cellule, TableauLu, lire_masques
from .textes import (compact, norm, normaliser_parcelle, normaliser_profondeur,
                     normaliser_ref, nettoyer_espaces, premiere_date, profondeur_valide,
                     similarite, VERS_CHIFFRE)
from .trace import Alerte, Rature, V, Valeur, union_boites


# ======================================================================
# Typage des colonnes
# ======================================================================
CASES = {"hap": "HAP16", "ms": "MS", "cot": "COT", "etm": "ETM", "ph": "pH"}

# (type, mots-clés) dans l'ordre de priorité ; comparaison sur le texte
# normalisé (minuscules, sans accents ni ponctuation), tolérante aux erreurs OCR.
LEXIQUE = [
    ("hap", ["hap", "hap16", "16hap", "hydrocarbures"]),
    ("ms", ["ms", "matiereseche", "msec"]),
    ("cot", ["cot", "carboneorganique", "carbone"]),
    ("etm", ["etm", "metaux", "elementstraces", "elementstracesmetalliques"]),
    ("ph", ["ph"]),
    ("analyses", ["analysesdemandees", "analyses", "analysedemandee", "parametres", "determinations"]),
    ("ref", ["refechantillon", "reference", "ref", "nechantillon", "echantillon", "point", "identifiant", "code"]),
    ("parcelle", ["parcelle", "ilot", "parcelles"]),
    ("profondeur", ["profondeur", "prof", "horizon"]),
    ("matrice", ["matrice", "typedesol", "materiau", "nature", "typedesolmatrice"]),
    ("date", ["dateprelev", "dateprelevement", "date"]),
    ("localisation", ["localisation", "position", "emplacement", "situation"]),
    ("observations", ["observations", "remarques", "remarque", "commentaires", "commentaire", "obs"]),
    ("autre", ["autre", "autres", "autresanalyses"]),
    ("ignore", ["commune", "adherent", "culture", "cultureprecedente", "latitude", "longitude",
                "exploitant", "lieudit", "coordonnees"]),
    ("numero", ["n", "no", "num", "numero"]),
]
TYPES_UTILES = {"ref", "profondeur", "parcelle", "matrice", "analyses", "hap", "ms", "cot", "etm", "ph",
                "observations", "localisation"}


def _score_mot(cle: str, texte: str) -> float:
    """Similarité de ``cle`` avec le meilleur passage de ``texte`` (compacts)."""
    if not texte:
        return 0.0
    if cle == texte:
        return 1.0
    if len(cle) <= 3:
        # mots courts (ms, ph, cot, ref…) : égalité ou préfixe/suffixe exact
        if texte.startswith(cle) or texte.endswith(cle):
            return 0.9 if len(texte) <= len(cle) + 3 or cle in ("ref",) else 0.75
        return 0.0
    if cle in texte:
        return 0.95
    meilleur = 0.0
    from difflib import SequenceMatcher
    for long in (len(cle) - 1, len(cle), len(cle) + 1):
        for i in range(0, max(1, len(texte) - long + 1)):
            r = SequenceMatcher(None, cle, texte[i:i + long], autojunk=False).ratio()
            meilleur = max(meilleur, r)
    return meilleur


def typer_entete(lecture: Lecture) -> tuple[str | None, float]:
    """Type d'une colonne d'après le texte de son en-tête (toutes lectures)."""
    meilleur, score = None, 0.0
    for texte in lecture.textes():
        t = compact(texte)
        if not t:
            continue
        for typ, cles in LEXIQUE:
            for cle in cles:
                s = _score_mot(cle, t)
                # les cases d'analyse ont des en-têtes très courts
                if typ in ("ms", "ph", "cot", "etm") and len(t) > 12:
                    s *= 0.5
                if s > score + 1e-9:
                    meilleur, score = typ, s
    if score < 0.75:
        return None, score
    return meilleur, score


def trouver_entete(t: TableauLu, max_rangees: int = 4):
    """(rangée d'en-tête, {colonne: type}, score) ou None."""
    meilleur = None
    for r in range(min(max_rangees, t.grille.n_rangees - 1)):
        types, score = {}, 0.0
        for c in range(t.grille.n_colonnes):
            cel = t.cellule(r, c)
            if cel.vide:
                continue
            typ, s = typer_entete(cel.lecture)
            if typ:
                types[c] = typ
                if typ in TYPES_UTILES:
                    score += s
        utiles = {v for v in types.values() if v in TYPES_UTILES}
        if len(utiles) >= 2 and (meilleur is None or score > meilleur[2]):
            meilleur = (r, types, score)
    if meilleur is None:
        return None
    r, types, score = meilleur
    # une colonne « numéro » seule n'est pas une référence ; si aucune colonne
    # n'est typée « ref », la première colonne textuelle à gauche le devient
    if "ref" not in types.values():
        for c in range(t.grille.n_colonnes):
            if types.get(c) in (None, "numero"):
                textes = [t.cellule(rr, c).texte for rr in range(r + 1, t.grille.n_rangees)]
                if sum(bool(re.search(r"[A-Za-z].*\d|\d.*[A-Za-z]", x or "")) for x in textes) >= 1:
                    types[c] = "ref"
                    break
    return r, types, score


# ======================================================================
# Structures
# ======================================================================
@dataclass
class LigneLue:
    page: int
    rangee: int
    valeurs: dict = field(default_factory=dict)      # nom -> Valeur
    annulee: bool = False
    motif: Valeur | None = None
    ratures: list = field(default_factory=list)      # Rature
    alertes: list = field(default_factory=list)      # Alerte
    boite: tuple | None = None
    candidats_ref: list = field(default_factory=list)   # (texte normalisé, conf)


@dataclass
class TableauEchantillons:
    lignes: list = field(default_factory=list)       # LigneLue
    types_colonnes: dict = field(default_factory=dict)
    alertes: list = field(default_factory=list)
    trouve: bool = False


# ======================================================================
# Cases d'analyse
# ======================================================================
def _stats_colonne(cellules: list[Cellule]) -> dict:
    marquees = [c for c in cellules if not c.vide]
    parts = [c.part_encre for c in marquees]
    couleurs = Counter(c.couleur for c in marquees)
    return {"mediane": float(np.median(parts)) if parts else 0.0,
            "n": len(marquees),
            "couleur": couleurs.most_common(1)[0][0] if couleurs else "principale"}


def etat_case(cel: Cellule, stats: dict) -> tuple[str, float, str]:
    """('vide' | 'coche' | 'barre' | 'incertain', confiance, explication)."""
    if cel.vide:
        return "vide", 0.95, "case vide"
    texte = norm(cel.texte)
    if re.search(r"\bnon\b|annul|\bpas\b", texte):
        return "barre", 0.85, f"mention « {cel.texte} » dans la case"
    tot = sum(cel.couleurs.values()) or 1
    principale = cel.couleurs.get("principale", 0) / tot
    colore = 1 - principale
    med = stats["mediane"] or cel.part_encre
    # marque imprimée surchargée au stylo (croix puis « \/\ ») : deux encres
    # dans la case, nettement plus d'encre que les autres cases cochées
    if stats["n"] >= 2 and stats["couleur"] == "principale" and colore >= 0.3 and principale >= 0.15 \
            and cel.part_encre >= 2.0 * med:
        return "barre", 0.7, "case cochée à l'imprimé puis surchargée au stylo"
    if stats["n"] >= 3 and cel.part_encre >= 3.0 * med and cel.part_encre > 0.06:
        return "incertain", 0.4, "case beaucoup plus chargée que les autres (barrée ?)"
    if cel.part_encre < 0.008:
        return "incertain", 0.45, "trace d'encre très faible"
    return "coche", 0.9 if cel.part_encre >= 0.5 * med else 0.7, "case cochée"


_RE_ANALYSES = [
    ("HAP16", re.compile(r"hap|h\.a\.p|hydrocarb", re.I)),
    ("MS", re.compile(r"(?<![a-z])ms(?![a-z])|mati[eè]re s[eè]che|m\.s\.", re.I)),
    ("COT", re.compile(r"(?<![a-z])cot(?![a-z])|carbone", re.I)),
    ("ETM", re.compile(r"(?<![a-z])etm(?![a-z])|m[ée]taux|[ée]l[ée]ments? traces", re.I)),
    ("pH", re.compile(r"(?<![a-z])ph(?![a-z])", re.I)),
]


def analyses_du_texte(texte: str) -> list[str]:
    """'HAP 16 + MS + ETM' -> ['HAP16', 'MS', 'ETM'] (tolère '+', ',', espaces absents)."""
    if not texte:
        return []
    t = re.sub(r"[+,;/&]", " ", texte)
    t = re.sub(r"(?i)hap\s*1\s*6", "HAP16 ", t)
    t = re.sub(r"(?i)(HAP16)(MS|ETM|COT|PH)", r"\1 \2", t)
    trouve = []
    for nom, rx in _RE_ANALYSES:
        if rx.search(t):
            trouve.append(nom)
    return trouve


def ordonner(analyses) -> list[str]:
    return [a for a in config.ANALYSES if a in set(analyses)]


# ======================================================================
# Ratures dans une cellule de texte
# ======================================================================
def _composantes(binaire: np.ndarray, lab: np.ndarray, brut: np.ndarray, ref):
    """Composantes de ``binaire`` ; couleur mesurée sur l'encre réelle ``brut``."""
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(binaire, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, aire = stats[i]
        if aire < AIRE_MIN:
            continue
        sel = (etiq == i) & (brut > 0)
        comps.append({"i": i, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "aire": int(aire),
                      "cx": x + w / 2, "cy": y + h / 2,
                      "couleur": E.classe_couleur(lab[sel if sel.any() else etiq == i], ref)})
    return etiq, comps


def analyser_rature(t: TableauLu, cel: Cellule, trait: E.Trait) -> dict:
    """Texte barré et correction associée (au-dessus ou à droite du trait,
    éventuellement à cheval sur la cellule voisine)."""
    th = t.hauteur_texte
    g = t.grille
    x0, y0, x1, y1 = cel.boite
    cw = x1 - x0
    X0 = max(g.xs[0], x0 - int(0.3 * th))
    X1 = min(g.xs[-1], x1 + int(0.8 * cw))
    Y0 = max(0, y0 - int(2.0 * th))
    Y1 = min(t.propre.shape[0], y1 + 3)
    fen = t.propre[Y0:Y1, X0:X1].copy()
    tr = trait.decale(-X0, -Y0)
    fen_sans = E.effacer_trait(fen, tr)
    fen_sans = cv2.morphologyEx(fen_sans, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lab = t.lab[Y0:Y1, X0:X1]
    etiq, comps = _composantes(fen_sans, lab, fen, t.encre_ref)
    # couleur du trait (même stylo que la correction, en général)
    pix_trait = lab[tr.y0:tr.y1 + 1, tr.x0:tr.x1][fen[tr.y0:tr.y1 + 1, tr.x0:tr.x1] > 0]
    couleur_trait = E.classe_couleur(pix_trait, t.encre_ref)
    lx0, lx1 = x0 - X0, x1 - X0          # bornes de la cellule dans la fenêtre
    ly0 = y0 - Y0
    barres, autres = [], []
    for k in comps:
        recouvre = max(0, min(k["x"] + k["w"], tr.x1) - max(k["x"], tr.x0)) / max(1, k["w"])
        if recouvre >= 0.5 and abs(k["cy"] - tr.y) <= 0.8 * th and lx0 - 5 <= k["cx"] <= lx1 + 5:
            barres.append(k)
        else:
            autres.append(k)
    haut_barre = min([k["y"] for k in barres], default=tr.y0 - int(0.4 * th))
    ly1 = y1 - Y0
    corr = []
    for k in autres:
        dans_cellule = lx0 <= k["cx"] <= lx1 and ly0 <= k["cy"] <= ly1
        meme_couleur = couleur_trait != "principale" and k["couleur"] == couleur_trait
        if k["cy"] < ly0 - 0.9 * th:
            continue                                   # contenu de la cellule du dessus
        au_dessus = k["y"] + k["h"] <= haut_barre + 0.35 * th and \
            tr.x0 - 0.5 * th <= k["cx"] <= tr.x1 + 1.5 * th
        a_droite = k["x"] >= tr.x1 - 0.3 * th and abs(k["cy"] - tr.y) <= 0.8 * th
        if not (au_dessus or a_droite):
            continue
        if not dans_cellule:
            # hors de la cellule : même stylo que le trait, ou à cheval sur le
            # bord de la cellule (la correction déborde) ; sinon c'est le
            # contenu propre de la cellule voisine
            a_cheval = k["x"] < lx1 <= k["x"] + k["w"] or k["y"] < ly0 <= k["y"] + k["h"]
            if not (meme_couleur or (couleur_trait == "principale" and a_cheval)):
                continue
        elif couleur_trait != "principale" and k["couleur"] != couleur_trait and k["couleur"] == "principale" \
                and not au_dessus and k["x"] >= tr.x1:
            continue                                   # texte imprimé voisin dans la cellule
        corr.append(k)
    # à droite : on ne garde que la chaîne continue qui part du trait
    if corr:
        corr.sort(key=lambda k: k["x"])
        chaine, bord = [], tr.x1
        for k in corr:
            if k["x"] - bord <= 1.5 * th or k["y"] + k["h"] <= haut_barre + 0.35 * th:
                chaine.append(k)
                bord = max(bord, k["x"] + k["w"])
        corr = chaine

    def groupe(ks):
        if not ks:
            return None, None
        ids = [k["i"] for k in ks]
        bx0 = min(k["x"] for k in ks)
        by0 = min(k["y"] for k in ks)
        bx1 = max(k["x"] + k["w"] for k in ks)
        by1 = max(k["y"] + k["h"] for k in ks)
        m = (np.isin(etiq[by0:by1, bx0:bx1], ids)).astype(np.uint8) * 255
        return (bx0 + X0, by0 + Y0, bx1 + X0, by1 + Y0), m

    b_barre, m_barre = groupe(barres)
    b_corr, m_corr = groupe(corr)
    items = []
    if b_barre is not None:
        items.append((b_barre, m_barre))
    if b_corr is not None:
        items.append((b_corr, m_corr))
    lectures = lire_masques(t.page, items) if items else []
    lec_barre = lectures[0] if b_barre is not None else None
    lec_corr = lectures[-1] if b_corr is not None else None
    return {"avant": lec_barre, "apres": lec_corr, "boite_avant": b_barre, "boite_apres": b_corr,
            "couleur": couleur_trait}


# ======================================================================
# Normalisation des valeurs par type de colonne
# ======================================================================
def _meilleure(lecture: Lecture | None, normaliser, valider=None) -> tuple:
    """Parmi les lectures, la valeur normalisée la plus soutenue (somme des
    confiances des lectures qui donnent cette valeur)."""
    if lecture is None:
        return None, 0.0, None
    votes, brut = defaultdict(float), {}
    textes = [(lecture.texte, lecture.conf)] + [(c.texte, c.conf) for c in lecture.candidats]
    for texte, conf in textes:
        v = normaliser(texte) if texte else None
        if v is None or (valider and not valider(v)):
            continue
        votes[v] += conf
        brut.setdefault(v, texte)
    if not votes:
        return None, 0.0, lecture.texte or None
    v = max(votes, key=votes.get)
    total = sum(conf for _, conf in textes) or 1
    accord = votes[v] / total
    return v, round(min(1.0, lecture.conf * (0.6 + 0.4 * accord) + 0.15 * accord), 3), brut[v]


def valeur_texte(lecture: Lecture | None, page, boite, typ: str) -> Valeur:
    if lecture is None or not lecture.texte:
        return V(None, 0.0, page, boite, "ocr")
    if typ == "profondeur":
        v, c, brut = _meilleure(lecture, normaliser_profondeur)
        if v and not profondeur_valide(v):
            c = min(c, 0.5)
        return V(v, c, page, boite, "ocr", brut or lecture.texte)
    if typ == "parcelle":
        return V(normaliser_parcelle(lecture.texte), lecture.conf, page, boite, "ocr", lecture.texte)
    if typ == "ref":
        return V(normaliser_ref(lecture.texte), lecture.conf, page, boite, "ocr", lecture.texte)
    if typ == "date":
        d = premiere_date(lecture.texte)
        return V(d.isoformat() if d else None, lecture.conf, page, boite, "ocr", lecture.texte)
    return V(nettoyer_espaces(lecture.texte) or None, lecture.conf, page, boite, "ocr", lecture.texte)


# ======================================================================
# Lecture des rangées
# ======================================================================
def _ligne_annulee(t: TableauLu, r: int, types: dict) -> tuple[bool, str]:
    """Une ligne est annulée si un trait la barre sur la plupart de ses
    cellules remplies (hors observations), ou si une observation dit « annulé »."""
    remplies, barrees = 0, 0
    for c, typ in types.items():
        cel = t.cellule(r, c)
        if cel.vide or typ in ("observations", "numero"):
            continue
        remplies += 1
        if cel.traits and max(tr.longueur for tr in cel.traits) >= 0.5 * (cel.boite[2] - cel.boite[0]):
            barrees += 1
    for c, typ in types.items():
        if typ in ("observations", "localisation") and re.search(r"annul", norm(t.cellule(r, c).texte)):
            return True, "mention « annulé » en observation"
    if remplies >= 2 and barrees >= max(2, 0.6 * remplies):
        return True, "ligne barrée sur toute sa largeur"
    return False, ""


def _lire_rangee(t: TableauLu, r: int, types: dict, stats_cases: dict, journal) -> LigneLue | None:
    page = t.page.numero
    cels = {c: t.cellule(r, c) for c in types}
    utiles = [c for c, typ in types.items() if typ not in ("numero", "ignore")]
    if all(cels[c].vide for c in utiles):
        return None
    ligne = LigneLue(page, r)
    g = t.grille
    ligne.boite = (g.xs[0], g.ys[r], g.xs[-1], g.ys[r + 1])
    # une seule tache dans une rangée vide (BDC_03) : pas un échantillon
    remplies = [c for c in utiles if not cels[c].vide]
    if len(remplies) == 1 and not compact(cels[remplies[0]].texte) and types[remplies[0]] not in CASES:
        journal(f"p{page} rangée {r} : tache d'encre isolée ignorée")
        return None
    if len(remplies) == 1 and len(compact(cels[remplies[0]].texte)) <= 2 \
            and types[remplies[0]] in ("observations", "localisation", "autre"):
        journal(f"p{page} rangée {r} : marque isolée en {types[remplies[0]]} ignorée")
        return None
    annulee, motif = _ligne_annulee(t, r, types)
    analyses, hap_vu, hap_etat = set(), False, None
    hap_conf = 0.0
    hap_boite = None
    for c, typ in types.items():
        cel = cels[c]
        boite = cel.boite
        if typ in CASES:
            etat, conf, note = etat_case(cel, stats_cases[c])
            nom = CASES[typ]
            if nom == "HAP16":
                hap_vu, hap_etat, hap_conf, hap_boite = True, etat, conf, boite
            if etat == "coche":
                analyses.add(nom)
            elif etat == "barre":
                ligne.ratures.append(Rature(None, "hap_demande" if nom == "HAP16" else f"analyses.{nom}",
                                            True, False, page, boite, conf))
                journal(f"p{page} rangée {r} : {nom} {note}")
            elif etat == "incertain":
                ligne.alertes.append(Alerte("case_ambigue", "avertissement",
                                            f"Case {nom} ambiguë ({note}) : à vérifier", f"analyses.{nom}"))
                if nom == "HAP16":
                    hap_etat = "incertain"
            continue
        if typ in ("analyses", "autre"):
            lus = analyses_du_texte(cel.texte)
            analyses |= set(lus)
            if typ == "analyses":
                ligne.valeurs["_analyses_texte"] = V(", ".join(lus) or None, cel.lecture.conf, page, boite,
                                                     "ocr", cel.texte)
                if "HAP16" in lus:
                    hap_vu, hap_etat, hap_conf, hap_boite = True, "coche", cel.lecture.conf, boite
                elif not cel.vide and lus:
                    hap_vu, hap_etat, hap_conf, hap_boite = True, "vide", cel.lecture.conf, boite
            continue
        if typ in ("ignore", "numero"):
            continue
        nom = {"profondeur": "profondeur_cm", "localisation": "observations"}.get(typ, typ)
        if cel.vide:
            ligne.valeurs.setdefault(nom, V(None, 0.9, page, boite, "ocr", note="cellule vide"))
            continue
        # rature dans une cellule de texte (pas sur une ligne annulée)
        if cel.traits and not annulee:
            trait = max(cel.traits, key=lambda x: x.longueur)
            ra = analyser_rature(t, cel, trait)
            avant = valeur_texte(ra["avant"], page, ra["boite_avant"], typ)
            apres = valeur_texte(ra["apres"], page, ra["boite_apres"], typ)
            if typ == "parcelle" and apres.valeur and avant.valeur and re.fullmatch(r"\d+", apres.valeur):
                # correction partielle : seul le numéro a été récrit (« Îlot 01 » -> « 10 »)
                prefixe = re.match(r"^(.*?)(\d+)$", avant.valeur)
                if prefixe:
                    apres.valeur = f"{prefixe.group(1).strip()} {int(apres.valeur)}".strip()
                    apres.note = "correction partielle complétée avec le préfixe barré"
            apres.conf = min(apres.conf, 0.85)
            apres.source = "rature"
            ligne.ratures.append(Rature(None, nom, avant.valeur, apres.valeur, page,
                                        union_boites([ra["boite_avant"], ra["boite_apres"]]),
                                        apres.conf))
            if apres.valeur is None:
                ligne.alertes.append(Alerte("rature_illisible", "avertissement",
                                            f"Valeur barrée en {nom} sans correction lisible", nom))
            ligne.valeurs[nom] = apres
            journal(f"p{page} rangée {r} : rature en {nom} « {avant.valeur} » -> « {apres.valeur} »")
            continue
        val = valeur_texte(cel.lecture, page, boite, typ)
        if typ == "localisation":
            # « AB 301, zone nord » : la parcelle est en tête de la localisation
            p = re.match(r"^\s*([A-Z]{1,2}\s?\d{1,4})\b", val.valeur or "")
            if p and "parcelle" not in types.values():
                ligne.valeurs["parcelle"] = V(normaliser_parcelle(p.group(1)), val.conf, page, boite, "ocr",
                                              val.brut, "parcelle lue en tête de la localisation")
        if nom in ligne.valeurs and ligne.valeurs[nom].valeur:
            ligne.valeurs[nom].valeur += " " + (val.valeur or "")
        else:
            ligne.valeurs[nom] = val
    # demande HAP explicite ?
    if hap_vu:
        if hap_etat == "incertain":
            ligne.valeurs["hap_demande"] = V(None, 0.3, page, hap_boite, "case", note="case HAP ambiguë")
        else:
            ligne.valeurs["hap_demande"] = V("HAP16" in analyses, hap_conf, page, hap_boite, "case")
    else:
        ligne.valeurs["hap_demande"] = V(None, 0.0, page, None, "case", note="pas de colonne HAP")
    ligne.valeurs["analyses"] = V(ordonner(analyses), min([hap_conf] + [0.9]) if analyses else 0.0,
                                  page, ligne.boite, "case")
    if annulee:
        ligne.annulee = True
        obs = ligne.valeurs.get("observations")
        ligne.motif = V((obs.valeur if obs and obs.valeur else motif), obs.conf if obs else 0.7,
                        page, obs.boite if obs else ligne.boite, "regle", note=motif)
        journal(f"p{page} rangée {r} : ligne annulée ({motif})")
    # candidats de référence pour l'harmonisation
    for c, typ in types.items():
        if typ == "ref" and not cels[c].vide:
            lec = cels[c].lecture
            ligne.candidats_ref = [(normaliser_ref(x.texte), x.conf) for x in lec.candidats if x.texte] + \
                [(normaliser_ref(lec.texte), lec.conf + 0.2)]
            ligne.candidats_ref = [(a, b) for a, b in ligne.candidats_ref if a]
    return ligne


# ======================================================================
# Harmonisation des références entre lignes
# ======================================================================
_SEP = re.compile(r"\s*[-–—:._/]\s*|\s+")
VERS_LETTRE = str.maketrans({"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"})


def _segments(ref: str) -> list[str]:
    return [s for s in _SEP.split(ref or "") if s]


def _classe(c: str) -> str:
    return "9" if c.isdigit() else ("A" if c.isalpha() else "?")


def harmoniser_refs(lignes: list[LigneLue], indices: list[str], journal) -> None:
    """Corrige les références en s'appuyant sur les autres lignes : préfixe
    commun (vote pondéré sur toutes les lectures, renforcé par les numéros lus
    ailleurs dans le document, ex. MST/2026/118) et nature chiffre/lettre de
    chaque position du suffixe. Une référence n'est modifiée que vers une
    forme proche de sa propre lecture."""
    actives = [l for l in lignes if l.candidats_ref]
    if len(actives) < 2:
        return
    seps = Counter()
    for l in actives:
        best = l.valeurs.get("ref")
        if best and best.valeur:
            seps.update(re.findall(r"[^A-Z0-9]+", best.valeur))
    sep = seps.most_common(1)[0][0] if seps else "-"
    seg_lignes = []
    for l in actives:
        best = l.valeurs.get("ref")
        segs = _segments(best.valeur if best and best.valeur else l.candidats_ref[-1][0])
        seg_lignes.append(segs)
    k = Counter(len(s) for s in seg_lignes).most_common(1)[0][0]
    if k < 2:
        return
    # vote du préfixe, position par position
    consensus = []
    for i in range(k - 1):
        votes = defaultdict(float)
        for l in actives:
            vus = set()
            for texte, conf in l.candidats_ref:
                s = _segments(texte)
                if len(s) == k and s[i] not in vus:
                    votes[s[i]] += conf
                    vus.add(s[i])
        for tok in indices:
            for s in _segments(tok):
                if s in votes:
                    votes[s] += 0.5 * len(actives)
        if not votes:
            consensus.append(None)
            continue
        v = max(votes, key=votes.get)
        consensus.append(v if votes[v] >= 0.3 * sum(votes.values()) else None)
    # gabarit chiffre/lettre du dernier segment
    derniers = [s[-1] for s in seg_lignes if len(s) == k]
    long_mode = Counter(len(d) for d in derniers).most_common(1)[0][0] if derniers else 0
    gabarit = []
    for pos in range(long_mode):
        cl = Counter(_classe(d[pos]) for d in derniers if len(d) == long_mode)
        gabarit.append(cl.most_common(1)[0][0] if cl else "?")
    for l, segs in zip(actives, seg_lignes):
        if len(segs) != k:
            continue
        nouveaux = list(segs)
        for i, v in enumerate(consensus):
            if v and nouveaux[i] != v and similarite(nouveaux[i], v) >= 0.5 and \
                    abs(len(nouveaux[i]) - len(v)) <= 1:
                nouveaux[i] = v
        d = nouveaux[-1]
        if len(d) == long_mode and gabarit:
            corrige = "".join(
                (ch.translate(VERS_CHIFFRE) if g == "9" else ch.translate(VERS_LETTRE) if g == "A" else ch)
                for ch, g in zip(d, gabarit))
            nouveaux[-1] = corrige
        val = l.valeurs.get("ref")
        nouveau = sep.join(nouveaux)
        if val and val.valeur != nouveau:
            journal(f"référence harmonisée « {val.valeur} » -> « {nouveau} »")
            val.note = f"harmonisée depuis « {val.valeur} »"
            val.valeur = nouveau
            val.conf = round(min(val.conf, 0.8), 3)


# ======================================================================
# Point d'entrée
# ======================================================================
def lire_echantillons(tableaux: list[TableauLu], indices: list[str], journal) -> TableauEchantillons:
    res = TableauEchantillons()
    for t in tableaux:
        ent = trouver_entete(t)
        if ent is None:
            journal(f"p{t.page.numero} tableau {t.grille.n_rangees}x{t.grille.n_colonnes} : pas d'en-tête reconnu")
            continue
        r0, types, _ = ent
        if "ref" not in types.values():
            journal(f"p{t.page.numero} tableau sans colonne de référence : ignoré")
            continue
        res.trouve = True
        noms = {c: types.get(c) for c in range(t.grille.n_colonnes)}
        res.types_colonnes.update({f"p{t.page.numero}:{c}": v for c, v in noms.items()})
        journal(f"p{t.page.numero} en-tête rangée {r0} : " +
                ", ".join(f"{t.cellule(r0, c).texte or '?'}→{v}" for c, v in noms.items()))
        types = {c: v for c, v in noms.items() if v}
        lignes_cel = range(r0 + 1, t.grille.n_rangees)
        stats_cases = {c: _stats_colonne([t.cellule(r, c) for r in lignes_cel])
                       for c, typ in types.items() if typ in CASES}
        for r in lignes_cel:
            ligne = _lire_rangee(t, r, types, stats_cases, journal)
            if ligne is not None:
                res.lignes.append(ligne)
    harmoniser_refs([l for l in res.lignes if not l.annulee], indices, journal)
    for l in res.lignes:
        ref = l.valeurs.get("ref")
        for ra in l.ratures:
            ra.echantillon = ref.valeur if ref else None
    if not res.trouve:
        res.alertes.append(Alerte("tableau_echantillons", "bloquant",
                                  "Tableau des échantillons introuvable : saisie manuelle nécessaire"))
    return res
