"""Couche IA (Gemini) : relit le texte OCR et range les champs du BDC.

Les règles de champs.py cherchent des étiquettes ; elles se trompent dès que la
mise en page s'écarte de l'habitude (client en en-tête sans étiquette, valeur
qui déborde sur le champ suivant…). Gemini reçoit le texte lu par l'OCR, rangée
par rangée, et rend les champs du schéma §4, en corrigeant les fautes de lecture
évidentes.

Garde-fou : une valeur proposée doit se retrouver dans le texte OCR (à quelques
fautes de lecture près). Sinon elle est rejetée, la valeur des règles est gardée
et une alerte est émise : l'IA ne peut pas inventer une valeur (règle du README).

Échantillons : dans les cellules manuscrites ou faxées, l'OCR confond des
caractères (ECO-4611 pour ECO-4619) et rate des cases cochées ; le texte seul ne
permet pas de corriger. Gemini reçoit alors l'image du tableau des échantillons,
recadrée sur la grille (références, profondeurs, analyses, observations : pas de
nom, téléphone ni adresse du client), avec les lignes lues par l'OCR. Garde-fou :
une valeur corrigée doit rester proche du texte lu dans sa ligne ; les lignes
(nombre, annulations, ratures) restent celles de l'OCR.

Pour les champs, seul le texte OCR est envoyé ; l'image de la page entière ne
part jamais. Clé : variable d'environnement GEMINI_API_KEY ; modèle : GEMINI_MODEL.
"""
from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Optional

import cv2
from pydantic import BaseModel, Field

from . import champs as CH
from . import config
from .lecture import rangees
from .textes import compact, norm, normaliser_profondeur, temperature, trouver_dates
from .trace import V, Valeur

MODELE_DEFAUT = "gemini-3.5-flash-lite"
DELAI_APPEL_S = 120     # au-delà, l'appel échoue et les valeurs des règles sont gardées
ESSAIS = 3
# modèles essayés dans l'ordre si le précédent est saturé
MODELES_REPLI = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]
# Part minimale de la valeur proposée qu'on doit retrouver, dans l'ordre, dans
# une ligne (ou deux rangées consécutives) du texte OCR.
SEUIL_TEXTE = 0.80
SEUIL_CHIFFRES = 0.90     # téléphone, n° de commande : une faute d'OCR corrigée au plus
# Échantillons, corrigés sur l'image : part de la valeur retrouvée dans le texte OCR
# de sa ligne. Plus bas que pour les champs : l'image permet de corriger des
# caractères que l'OCR a confondus (« EC0-4611-1 » -> « ECO-4619-1 » : 0,75).
SEUIL_ECH_CODE = 0.60      # référence, parcelle, profondeur
SEUIL_ECH_TEXTE = 0.50     # observations, matrice (écriture libre)
LARGEUR_IMAGE = 1800       # px : image du tableau réduite à cette largeur au plus
MOIS = {"janv": 1, "fevr": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6, "juil": 7, "aout": 8,
        "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


# ================================================================ SCHÉMA DE RÉPONSE
class Client(BaseModel):
    raison_sociale: Optional[str] = Field(None, description="nom de la société ou de l'organisme qui commande (l'émetteur du BDC)")
    adresse: Optional[str] = Field(None, description="numéro et voie, sans CP ni ville")
    cp: Optional[str] = Field(None, description="code postal du client, 5 chiffres")
    ville: Optional[str] = None
    telephone: Optional[str] = Field(None, description="format 02 61 91 23 58")
    email: Optional[str] = None
    contact: Optional[str] = Field(None, description="nom de la personne à contacter, seul (sans téléphone ni fonction)")
    siret: Optional[str] = Field(None, description="14 chiffres sans espaces, recopiés tels quels")


class Commande(BaseModel):
    numero: Optional[str] = Field(None, description="numéro du bon de commande, sans « N° »")
    date: Optional[str] = Field(None, description="date du bon de commande, AAAA-MM-JJ")
    delai: Optional[str] = Field(None, description="standard_10j, urgent_J+5 ou express_72h : l'option cochée")
    annule_et_remplace: Optional[str] = Field(None, description="référence ou date du BDC annulé et remplacé, sinon null")
    facturation: Optional[str] = None
    remarques: Optional[str] = None


class Site(BaseModel):
    commune: Optional[str] = Field(None, description="commune du site de prélèvement, sans le CP")
    cp: Optional[str] = Field(None, description="code postal du site, 5 chiffres")
    lieu_dit: Optional[str] = None
    parcelles: list[str] = Field(default_factory=list, description="références cadastrales, ex. ZB 112")


class Prelevement(BaseModel):
    date: Optional[str] = Field(None, description="date du prélèvement, AAAA-MM-JJ")
    preleveur: Optional[str] = None


class Reception(BaseModel):
    date: Optional[str] = Field(None, description="date de réception au laboratoire (tampon « reçu le »), AAAA-MM-JJ")
    temperature_c: Optional[float] = Field(None, description="température à réception en °C, notée par le labo")


class Reponse(BaseModel):
    client: Client
    commande: Commande
    site: Site
    prelevement: Prelevement
    reception_labo: Reception


CONSIGNES = """Tu lis un bon de commande (BDC) d'analyses de sols envoyé au laboratoire ORVENNE ANALYSES.
Le texte vient d'un OCR : il contient des fautes de lecture (lettres confondues, espaces en trop, mots coupés).

Rends les champs demandés en JSON.
- Le client est l'émetteur du BDC : jamais le laboratoire Orvenne (destinataire), jamais
  l'exploitation ou le site où l'on prélève.
- Recopie chaque valeur depuis le texte. Tu peux corriger une faute de lecture évidente
  (« MISSTON » -> « Mission », « G6 39 » -> « 06 39 »), mais n'invente rien : une valeur
  absente ou illisible vaut null.
- Ne garde que la valeur du champ, sans l'étiquette ni ce qui appartient au champ suivant.
- Dates au format AAAA-MM-JJ ; l'année manquante est celle des autres dates du document.
- Délai : l'option cochée. Les mesures d'encre des cases (0 = vide) sont données plus bas ;
  la case cochée a nettement plus d'encre que les autres.
- Réception labo : ce que le laboratoire a ajouté à l'arrivée (tampon « reçu le », note de
  température, souvent manuscrite ; « Cadre réservé au laboratoire »).
- Imprimé et manuscrit : les lignes probablement écrites à la main sont marquées [manuscrit] ;
  ce marquage est indicatif (il peut aussi désigner un tampon ou un texte imprimé dégradé) :
  juge aussi d'après le contenu et l'image. Quand une même
  information figure deux fois, une fois imprimée et une fois à la main, avec le même sens ou la
  même valeur à des fautes de lecture près (ex. une date imprimée « 12/09/2026 » et une date
  manuscrite « 12/09/26 »), la version imprimée fait foi : sers-t'en pour lire la version
  manuscrite (chiffres ou lettres ambigus) et rends la valeur imprimée. Exception : si la version
  manuscrite corrige l'imprimée (valeur imprimée raturée, barrée ou surchargée), garde la correction.
"""


# ================================================================ DISPONIBILITÉ
def disponible() -> bool:
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


# ================================================================ TEXTE ENVOYÉ
def texte_ocr(lignes) -> str:
    """Texte lu, rangée par rangée ; les lignes manuscrites sont signalées. La couleur
    d'encre n'est pas transmise : des formulaires entiers sont imprimés en couleur."""
    sortie, page = [], None
    for rg in rangees(lignes):
        if rg[0].page != page:
            page = rg[0].page
            sortie.append(f"--- page {page} ---")
        morceaux = []
        for l in rg:
            morceaux.append(l.texte + (" [manuscrit]" if getattr(l, "manuscrit", False) else ""))
        sortie.append(" | ".join(morceaux))
    return "\n".join(sortie)


def mesures_cases(pages, lignes) -> str:
    """Encre mesurée dans la case de chaque option de délai (même mesure que les règles)."""
    out = []
    for l in lignes:
        if CH.est_labo(l.texte):
            continue
        for nom, motif in CH.OPTIONS_DELAI:
            m = motif.search(norm(l.texte))
            if m:
                page = next(p for p in pages if p.numero == l.page)
                encre = CH._case_cochee(page, l.boite, CH.x_libelle(l, m.start()))
                out.append(f"{nom} « {l.texte} » : " + ("pas de case" if encre is None else f"{encre:.3f}"))
    return "\n".join(out) or "aucune option de délai trouvée"


# ================================================================ APPEL
def generer(contenu: list, schema=None) -> dict | str:
    """Réponse JSON de Gemini selon schema (texte libre si schema est None). Si un
    modèle est saturé ou indisponible (429, 5xx), le suivant de MODELES_REPLI est essayé."""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
        timeout=DELAI_APPEL_S * 1000, retry_options=types.HttpRetryOptions(attempts=ESSAIS)))
    modeles = list(dict.fromkeys([os.environ.get("GEMINI_MODEL") or MODELE_DEFAUT, *MODELES_REPLI]))
    contenu = [types.Part.from_bytes(data=c, mime_type="image/jpeg") if isinstance(c, bytes) else c
               for c in contenu]
    for i, modele in enumerate(modeles):
        try:
            if schema is None:
                rep = client.models.generate_content(
                    model=modele, contents=contenu, config=types.GenerateContentConfig(temperature=0.2))
                return rep.text or ""
            rep = client.models.generate_content(
                model=modele, contents=contenu,
                config=types.GenerateContentConfig(temperature=0, seed=0, response_mime_type="application/json",
                                                   response_schema=schema))
            return json.loads(rep.text)
        except errors.APIError as e:
            if i + 1 == len(modeles) or not (e.code == 429 or e.code >= 500):
                raise


# ================================================================ GARDE-FOU
def _couverture(v: str, c: str) -> float:
    """Part des caractères de v retrouvés, dans l'ordre, dans c (formes compactes)."""
    if not v:
        return 0.0
    if v in c:
        return 1.0
    blocs = SequenceMatcher(None, v, c, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocs) / len(v)


def _candidats(lignes):
    """Lignes seules, rangées entières et paires de rangées consécutives (valeur sur 2 lignes)."""
    rgs = rangees(lignes)
    cands = [(compact(l.texte), l) for l in lignes]
    for i, rg in enumerate(rgs):
        cands.append((compact(" ".join(l.texte for l in rg)), rg[0]))
        if i + 1 < len(rgs) and rgs[i + 1][0].page == rg[0].page:
            cands.append((compact(" ".join(l.texte for l in rg + rgs[i + 1])), rg[0]))
    return cands


def _date_lue(iso: str, lignes, annee) -> object | None:
    """Ligne où figure cette date, en chiffres (12/09/2026) ou en lettres (14 SEP. 2026)."""
    for l in lignes:
        for d, _, _ in trouver_dates(l.texte, annee):
            if d.isoformat() == iso:
                return l
        n = norm(l.texte)
        a, m, j = iso.split("-")
        for mot, num in MOIS.items():
            if num == int(m) and re.search(rf"\b0?{int(j)}\s*{mot}", n):
                return l
    return None


def soutenir(cle: str, valeur, lignes, cands, annee):
    """(score, ligne) si la valeur proposée se retrouve dans le texte OCR, sinon (0, None)."""
    if cle in ("date",):
        l = _date_lue(str(valeur), lignes, annee)
        return (1.0, l) if l else (0.0, None)
    if cle == "delai":
        motif = dict(CH.OPTIONS_DELAI).get(valeur)
        l = next((l for l in lignes if motif and motif.search(norm(l.texte))), None)
        return (1.0, l) if l else (0.0, None)
    if cle == "temperature_c":
        l = next((l for l in lignes if temperature(l.texte) is not None
                  and abs(temperature(l.texte) - float(valeur)) < 0.05), None)
        return (1.0, l) if l else (0.0, None)
    v = compact(str(valeur))
    if cle in ("siret", "cp") or len(v) < 6:
        seuil = 1.0
    elif cle in ("telephone", "numero"):
        seuil = SEUIL_CHIFFRES
    else:
        seuil = SEUIL_TEXTE
    score, ligne = max(((_couverture(v, c), l) for c, l in cands), key=lambda x: x[0], default=(0.0, None))
    return (score, ligne) if score >= seuil else (0.0, None)


# ================================================================ ÉCHANTILLONS (image du tableau)
class LigneEchantillon(BaseModel):
    index: int = Field(description="numéro de la ligne, tel que donné dans la liste")
    ref_echantillon: Optional[str] = None
    parcelle: Optional[str] = None
    profondeur_cm: Optional[str] = Field(None, description="forme a-b, ex. 0-30")
    matrice: Optional[str] = None
    analyses: list[str] = Field(default_factory=list, description="cases cochées parmi " + ", ".join(config.ANALYSES))
    observations: Optional[str] = None


class ReponseComplete(Reponse):
    echantillons: list[LigneEchantillon] = Field(default_factory=list)


CONSIGNES_ECH = """
TABLEAU DES ÉCHANTILLONS
Les images jointes sont le tableau des échantillons. Ses lignes, telles que l'OCR les a lues
(avec des fautes), sont listées plus bas. Pour chaque ligne de la liste (même index, ni ajout
ni suppression), rends dans « echantillons » les valeurs telles qu'elles sont écrites sur l'image :
- corrige les caractères mal lus (0/O, 1/I/7, 5/S, 8/B…) en t'aidant de l'image et de la
  cohérence entre lignes (les références d'un même envoi suivent un même schéma) ;
- si une valeur est raturée et réécrite, donne la valeur corrigée (la nouvelle) ;
- si une même valeur figure imprimée ailleurs dans le document (ex. un n° de commande imprimé
  repris à la main dans les références), lis la version manuscrite d'après l'imprimée ;
- analyses : les colonnes dont la case est cochée (croix, coche, trait) sur cette ligne,
  parmi {analyses} ; HAP 16 se note HAP16 ;
- une cellule vide ou illisible vaut null ; n'invente rien.
"""


def image_tableau(t) -> bytes:
    """Tableau recadré sur sa grille, réduit à LARGEUR_IMAGE, en JPEG."""
    x0, y0, x1, y1 = t.grille.boite
    m = int(t.hauteur_texte)
    img = t.page.couleur[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m]
    if img.shape[1] > LARGEUR_IMAGE:
        f = LARGEUR_IMAGE / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _texte_rangee(t, rangee: int) -> str:
    return " | ".join(t.cellule(rangee, c).texte or "" for c in range(t.grille.n_colonnes))


def _valeur(v) -> str:
    x = v.valeur if isinstance(v, Valeur) else v
    return "" if x is None else (", ".join(x) if isinstance(x, list) else str(x))


def lignes_a_relire(tab, tableaux) -> list[tuple]:
    """(tableau, ligne lue) pour chaque ligne d'échantillon, dans l'ordre : l'index de la liste
    envoyée à Gemini est la position dans cette liste."""
    return [(t, l) for t in tableaux for l in tab.lignes
            if l.page == t.page.numero and l.rangee < t.grille.n_rangees]


def liste_echantillons(paires) -> str:
    out = []
    for i, (t, l) in enumerate(paires):
        v = l.valeurs
        out.append(f"{i} : réf={_valeur(v.get('ref'))!r} parcelle={_valeur(v.get('parcelle'))!r} "
                   f"profondeur={_valeur(v.get('profondeur_cm'))!r} matrice={_valeur(v.get('matrice'))!r} "
                   f"analyses=[{_valeur(v.get('analyses'))}] observations={_valeur(v.get('observations'))!r}"
                   + (" (ligne annulée)" if l.annulee else "")
                   + f"\n   texte OCR de la rangée : {_texte_rangee(t, l.rangee)}")
    return "\n".join(out)


# ================================================================ FUSION
def appliquer_champs(champs: dict, prop: dict, lignes, journal) -> list[dict]:
    """Remplace les valeurs des règles par celles de Gemini quand elles sont retrouvées
    dans le texte OCR. Une valeur absente selon Gemini remplace aussi celle des règles."""
    lignes_cl = [l for l in lignes if not CH.est_labo(l.texte)]
    cands = _candidats(lignes_cl)
    dates = [d for l in lignes_cl for d, _, _ in trouver_dates(l.texte)]
    annee = sorted(d.year for d in dates)[len(dates) // 2] if dates else None
    alertes = []
    for bloc in ("client", "commande", "site", "prelevement", "reception_labo"):
        for cle, p in (prop.get(bloc) or {}).items():
            if cle == "parcelles":
                gardees = []
                for x in p or []:
                    s, l = soutenir(cle, x, lignes_cl, cands, annee)
                    if l is not None:
                        gardees.append(V(x, 0.9 * s, l.page, l.boite, "ia", l.texte))
                if gardees:
                    champs[bloc]["parcelles"] = gardees
                continue
            actuelle = champs[bloc].get(cle)
            if p in (None, ""):
                if isinstance(actuelle, Valeur) and actuelle.valeur is not None:
                    journal(f"IA : {bloc}.{cle} absent selon Gemini (règles : {actuelle.valeur!r})")
                    champs[bloc][cle] = V(None, 0.0, None, None, "ia", note=f"règles : {actuelle.valeur!r}")
                continue
            # la réception est notée par le labo : on cherche aussi dans ses lignes
            sources = lignes if bloc == "reception_labo" else lignes_cl
            s, l = soutenir(cle, p, sources, cands if sources is lignes_cl else _candidats(sources), annee)
            if l is None:
                journal(f"IA : {bloc}.{cle} = {p!r} introuvable dans le texte OCR, valeur des règles gardée")
                alertes.append({"controle": "ia", "niveau": "info", "champ": f"{bloc}.{cle}",
                                "message": f"Proposition de l'IA « {p} » non retrouvée dans le texte lu : écartée"})
                continue
            champs[bloc][cle] = V(p, 0.9 * s, l.page, l.boite, "ia", l.texte)
    return alertes


def appliquer_echantillons(paires, props: list, journal) -> None:
    """Corrige les lignes d'échantillons quand la valeur de Gemini reste proche du texte
    lu dans la rangée ; les cases cochées viennent de l'image."""
    for p in props:
        i = p.get("index")
        if not isinstance(i, int) or not 0 <= i < len(paires):
            continue
        t, l = paires[i]
        texte = compact(_texte_rangee(t, l.rangee) + " " + " ".join(_valeur(x) for x in l.valeurs.values()))
        for cle, cle_ia, seuil in (("ref", "ref_echantillon", SEUIL_ECH_CODE), ("parcelle", "parcelle", SEUIL_ECH_CODE),
                                   ("profondeur_cm", "profondeur_cm", SEUIL_ECH_CODE),
                                   ("matrice", "matrice", SEUIL_ECH_TEXTE),
                                   ("observations", "observations", SEUIL_ECH_TEXTE)):
            prop = p.get(cle_ia)
            if cle == "profondeur_cm" and prop:
                prop = normaliser_profondeur(prop) or prop
            actuelle = l.valeurs.get(cle)
            if prop in (None, "") or prop == _valeur(actuelle):
                continue
            score = _couverture(compact(prop), texte)
            if score < seuil:
                journal(f"IA : ligne {l.rangee} {cle} = {prop!r} trop loin du texte lu ({score:.2f}), "
                        f"gardé {_valeur(actuelle)!r}")
                continue
            base = actuelle if isinstance(actuelle, Valeur) else V()
            l.valeurs[cle] = V(prop, 0.85 * score + 0.1, base.page or l.page, base.boite or l.boite,
                               "ia-image", _valeur(actuelle) or None)
        analyses = [a for a in p.get("analyses") or [] if a in config.ANALYSES]
        if analyses and not l.annulee:
            ancien = l.valeurs.get("analyses")
            if set(analyses) != set(ancien.valeur or [] if isinstance(ancien, Valeur) else []):
                journal(f"IA : ligne {l.rangee} analyses {_valeur(ancien)!r} -> {analyses}")
            base = ancien if isinstance(ancien, Valeur) else V()
            l.valeurs["analyses"] = V([a for a in config.ANALYSES if a in analyses], 0.8,
                                      base.page or l.page, base.boite or l.boite, "ia-image", _valeur(ancien))
            hap = l.valeurs.get("hap_demande")
            base = hap if isinstance(hap, Valeur) else V()
            l.valeurs["hap_demande"] = V("HAP16" in analyses, 0.8, base.page or l.page, base.boite,
                                         "ia-image", _valeur(hap))
    for _, l in paires:                       # les ratures portent la référence corrigée
        ref = l.valeurs.get("ref")
        for ra in l.ratures:
            ra.echantillon = ref.valeur if ref else None


def affiner(champs: dict, tab, pages, lignes, tableaux, journal) -> list[dict]:
    """Un seul appel Gemini par BDC : champs (texte OCR) et échantillons (image du
    tableau). Modifie champs et tab sur place ; renvoie les alertes."""
    paires = lignes_a_relire(tab, tableaux)
    contenu = [f"{CONSIGNES}\nMESURES DES CASES DE DÉLAI\n{mesures_cases(pages, lignes)}\n\n"
               f"TEXTE OCR\n{texte_ocr(lignes)}"]
    if paires:
        tableaux_lus = []                     # TableauLu n'est pas hachable : dédoublonnage par identité
        for t, _ in paires:
            if not any(t is x for x in tableaux_lus):
                tableaux_lus.append(t)
        contenu = [image_tableau(t) for t in tableaux_lus] + contenu
        contenu.append(CONSIGNES_ECH.format(analyses=", ".join(config.ANALYSES))
                       + "\nLIGNES LUES PAR L'OCR\n" + liste_echantillons(paires))
    prop = generer(contenu, ReponseComplete)
    alertes = appliquer_champs(champs, prop, lignes, journal)
    appliquer_echantillons(paires, prop.get("echantillons") or [], journal)
    return alertes
