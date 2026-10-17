"""Brique 2-3 : identification du document et extraction des champs hors tableau.

Les champs sont cherchés par étiquette (« Raison sociale : … ») puis, à défaut,
par motif (SIRET, téléphone, e-mail, code postal). Les lignes qui décrivent le
laboratoire destinataire sont écartées pour ne pas les prendre pour le client.
Aucune règle n'est liée à un des sept modèles : tout repose sur des étiquettes
usuelles et sur des motifs de valeurs.
"""
from __future__ import annotations

import re
from datetime import date

import cv2
import numpy as np

from . import config
from .lecture import rangees
from .ocr import Ligne
from .pretraitement import Page
from .textes import (ajouter_jours_ouvres, compact, norm, nettoyer_espaces, premiere_date,
                     temperature, trouver_cp_ville, trouver_dates, trouver_emails, trouver_parcelles,
                     trouver_siret, trouver_telephones)
from .trace import V, Valeur, union_boites

# ---------------------------------------------------------------- laboratoire
LABO = ("orvenne", "vergers", "45760", "montverac", "02 61 91 40", "0261914", "orvenne-analyses",
        "reception@")


def est_labo(texte: str) -> bool:
    n = norm(texte)
    c = compact(texte)
    return any(k.replace(" ", "") in c for k in LABO) or "orvenne" in n


# ---------------------------------------------------------------- aides
def valeur_apres(rgs: list[list[Ligne]], etiquette: str, valider=None, suivante: bool = True,
                 exclure_labo: bool = True) -> Valeur:
    """Valeur écrite après une étiquette (regex sur le texte normalisé) : à droite
    sur la même ligne, à droite dans la même rangée, ou en dessous."""
    motif = re.compile(etiquette, re.I)
    for i, rg in enumerate(rgs):
        for j, l in enumerate(rg):
            n = norm(l.texte)
            m = motif.search(n)
            if not m:
                continue
            if exclure_labo and est_labo(l.texte):
                continue
            # texte original après l'étiquette (séparateur « : » ou fin de l'étiquette)
            t = l.texte
            if ":" in t:
                reste = t.split(":", 1)[1]
            else:
                reste = t[min(len(t), m.end()):] if m.end() < len(t) else ""
            reste = nettoyer_espaces(reste).strip(" :.-_")
            boites, conf = [l.boite], l.conf
            if not reste:
                for l2 in rg[j + 1:]:
                    if l2.x0 >= l.x1 - 10:
                        reste = l2.texte.strip(" :.-_")
                        boites, conf = [l2.boite], l2.conf
                        break
            if not reste and suivante and i + 1 < len(rgs):
                for l2 in rgs[i + 1]:
                    if abs(l2.x0 - l.x0) < 5 * l.h or l2.x0 >= l.x1 - 10:
                        reste = l2.texte.strip(" :.-_")
                        boites, conf = [l2.boite], l2.conf
                        break
            if reste and (valider is None or valider(reste)):
                return V(reste, conf, l.page, union_boites(boites), "ocr", reste)
    return V(None, 0.0, None, None, "ocr", note="étiquette non trouvée")


def _trouver_ligne(lignes: list[Ligne], motif, exclure_labo=True):
    for l in lignes:
        if exclure_labo and est_labo(l.texte):
            continue
        m = re.search(motif, l.texte, re.I) if isinstance(motif, str) else motif(l.texte)
        if m:
            yield l, m


def _date_iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _val_date(rgs, etiquette, annee_defaut=None, lignes=None) -> Valeur:
    v = valeur_apres(rgs, etiquette, lambda t: premiere_date(t, annee_defaut) is not None)
    if v.valeur:
        d = premiere_date(v.valeur, annee_defaut)
        return V(_date_iso(d), v.conf, v.page, v.boite, "ocr", v.brut)
    return V(None, 0.0, None, None, "ocr", note="date non trouvée")


# ---------------------------------------------------------------- délai
# bornes de mot : « 15 juillet » n'est pas « 5 j », « acheminement 24 h » n'est pas un délai
OPTIONS_DELAI = [
    ("express_72h", re.compile(r"express|\b72\s*h\b")),
    ("urgent_J+5", re.compile(r"urgent|\bj\s*\+\s*5\b|\b5\s*j(ours)?\b|\bcinq\b")),
    ("standard_10j", re.compile(r"standard|\bnormal\b|\b10\s*j(ours)?\b|\bdix\b|habituel")),
]
DECLARATION_DELAI = re.compile(r"delai[^:]{0,20}:\s*(.*)")


def _plus_long_trait(m: np.ndarray) -> np.ndarray:
    """Longueur du plus long segment d'encre de chaque ligne de m (booléen)."""
    out = np.zeros(m.shape[0], dtype=int)
    for i, ligne in enumerate(m):
        d = np.diff(np.concatenate(([0], ligne.astype(np.int8), [0])))
        debuts, fins = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out[i] = (fins - debuts).max() if debuts.size else 0
    return out


def _carre(reg: np.ndarray, h: int, x_ref: int) -> tuple | None:
    """Cadre de case à cocher dans reg, le plus proche de x_ref : à peu près carré, de la
    taille du texte, aux quatre bords tracés. Le cadre est repéré par ses traits droits,
    pour ignorer une coche (✓) qui déborde de la case."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((reg > 0).astype(np.uint8), connectivity=8)
    cases = []
    for x, y, w, hh, _ in stats[1:n]:
        if not (0.35 * h <= w <= 2 * h and 0.35 * h <= hh <= 2 * h):
            continue
        comp = reg[y:y + hh, x:x + w] > 0
        lignes_h = np.flatnonzero(_plus_long_trait(comp) >= 0.35 * h)
        lignes_v = np.flatnonzero(_plus_long_trait(comp.T) >= 0.35 * h)
        if lignes_h.size < 2 or lignes_v.size < 2:
            continue
        haut, bas, gauche, droite = lignes_h[0], lignes_h[-1], lignes_v[0], lignes_v[-1]
        cw, ch = droite - gauche + 1, bas - haut + 1
        if not (0.35 * h <= cw <= 1.3 * h and 0.35 * h <= ch <= 1.3 * h and 0.75 <= cw / ch <= 1.33):
            continue
        cadre = comp[haut:bas + 1, gauche:droite + 1]
        e = max(2, min(cw, ch) // 8)
        bords = (cadre[:e].any(axis=0).mean(), cadre[-e:].any(axis=0).mean(),
                 cadre[:, :e].any(axis=1).mean(), cadre[:, -e:].any(axis=1).mean())
        if min(bords) >= 0.85:
            cases.append((abs(x + droite - x_ref), x + gauche, y + haut, cw, ch))
    return min(cases)[1:] if cases else None


def _case_cochee(page: Page, boite, x_libelle: int | None = None) -> float | None:
    """Part d'encre au centre de la case à cocher qui précède un libellé ; sert à
    choisir l'option cochée parmi plusieurs. None s'il n'y a pas de case.

    L'OCR inclut souvent la case dans la boîte du libellé : on cherche donc le cadre
    carré autour du début du libellé (x_libelle, estimé dans la ligne quand elle
    contient d'autres mots).
    """
    x0, y0, x1, y1 = boite
    h = y1 - y0
    xl = x0 if x_libelle is None else x_libelle
    ya, yb = max(0, int(y0 - 0.3 * h)), int(y1 + 0.3 * h)
    xa, xb = max(0, int(xl - 2 * h)), int(min(x1, xl + 1.5 * h))
    reg = page.binaire[ya:yb, xa:xb]
    c = _carre(reg, h, int(xl - xa + 0.75 * h)) if reg.size else None
    if c is None:
        return None
    x, y, w, hh = c
    centre = reg[y + hh // 4: y + 3 * hh // 4, x + w // 4: x + 3 * w // 4]
    return float((centre > 0).mean()) if centre.size else 0.0


def x_libelle(l: Ligne, debut: int) -> int:
    """Abscisse estimée du caractère n° debut de la ligne (largeur de caractère moyenne)."""
    x0, _, x1, _ = l.boite
    return int(x0 + (x1 - x0) * debut / max(1, len(norm(l.texte))))


def lire_delai(pages: list[Page], lignes: list[Ligne]) -> Valeur:
    """Option de délai : la case cochée s'il y a des cases, sinon le délai écrit en
    toutes lettres (« Délai souhaité : standard. »), sinon le seul mot-clé de la page."""
    cand, declares = [], []
    for l in lignes:
        if est_labo(l.texte):
            continue
        n = norm(l.texte)
        d = DECLARATION_DELAI.search(n)
        for nom, motif in OPTIONS_DELAI:
            m = motif.search(n)
            if m:
                page = next(p for p in pages if p.numero == l.page)
                cand.append((nom, l, _case_cochee(page, l.boite, x_libelle(l, m.start()))))
                if d and motif.search(d.group(1)):
                    declares.append((nom, l))
    if not cand:
        return V(None, 0.0, None, None, "ocr", note="délai non reconnu")
    cases = [c for c in cand if c[2] is not None]
    if cases:
        meilleur = max(cases, key=lambda c: c[2])
        if meilleur[2] >= config.SEUIL_CASE_COCHEE:
            return V(meilleur[0], 0.75, meilleur[1].page, meilleur[1].boite, "case", meilleur[1].texte)
        return V(None, 0.3, None, None, "case", note="plusieurs délais proposés, aucun coché")
    if len({nom for nom, _ in declares}) == 1:
        nom, l = declares[0]
        return V(nom, min(0.85, l.conf), l.page, l.boite, "ocr", l.texte)
    if len({c[0] for c in cand}) == 1:
        nom, l, _ = cand[0]
        # un seul mot-clé sur la page : valeur retenue
        return V(nom, min(0.9, l.conf + 0.1), l.page, l.boite, "ocr", l.texte)
    return V(None, 0.3, None, None, "ocr", note="plusieurs délais cités, aucun choisi")


# ---------------------------------------------------------------- extraction
def extraire_champs(pages: list[Page], lignes: list[Ligne], journal) -> dict:
    """Dictionnaire de Valeur (ou de dictionnaires de Valeur) selon le schéma §4,
    sans la partie échantillons."""
    lignes_clientes = [l for l in lignes if not est_labo(l.texte)]
    rgs = rangees(lignes_clientes)
    annee = None
    toutes_dates = [d for l in lignes_clientes for d, _, _ in trouver_dates(l.texte)]
    if toutes_dates:
        annee = int(np.median([d.year for d in toutes_dates]))

    def chercher(motif, valider=None, **kw):
        return valeur_apres(rgs, motif, valider, **kw)

    # ---- client
    cl = {}
    cl["raison_sociale"] = chercher(r"raison sociale|societe|entreprise|exploitation|cooperative|client\b|demandeur|\bnom\b")
    if not cl["raison_sociale"]:
        # première ligne imprimée du haut de page 1 qui n'est pas le laboratoire ni une étiquette
        for l in lignes_clientes:
            if l.page == 1 and len(compact(l.texte)) > 4 and not re.search(r"bon de|commande|fax|date|page|\d{5}", norm(l.texte)):
                cl["raison_sociale"] = V(nettoyer_espaces(l.texte), l.conf * 0.7, l.page, l.boite, "ocr", l.texte,
                                         "déduite de la position (en-tête)")
                break
    cl["adresse"] = chercher(r"adresse|rue\b|adr\b")
    # code postal / ville
    cp = V(None, 0.0, None, None, "ocr")
    ville = V(None, 0.0, None, None, "ocr")
    for l, _ in _trouver_ligne(lignes_clientes, lambda t: trouver_cp_ville(t)):
        r = trouver_cp_ville(l.texte)
        if r:
            cp = V(r[0], l.conf, l.page, l.boite, "ocr", l.texte)
            ville = V(r[1], l.conf, l.page, l.boite, "ocr", l.texte)
            break
    cl["cp"], cl["ville"] = cp, ville
    tel = V(None, 0.0, None, None, "ocr")
    lab_tel = chercher(r"t[eé]l|phone|mobile|portable")
    if lab_tel:
        r = trouver_telephones(lab_tel.valeur)
        if r:
            tel = V(r[0][0], lab_tel.conf, lab_tel.page, lab_tel.boite, "ocr", lab_tel.valeur)
        else:
            tel = V(lab_tel.valeur, min(lab_tel.conf, 0.4), lab_tel.page, lab_tel.boite, "ocr", lab_tel.valeur,
                    "téléphone mal formé conservé tel que lu")
    else:
        for l in lignes_clientes:
            r = trouver_telephones(l.texte)
            if r and not re.search(r"fax", norm(l.texte)):
                tel = V(r[0][0], l.conf, l.page, l.boite, "ocr", l.texte)
                break
    cl["telephone"] = tel
    mail = V(None, 0.0, None, None, "ocr")
    for l in lignes_clientes:
        r = trouver_emails(l.texte)
        if r:
            mail = V(r[0], l.conf, l.page, l.boite, "ocr", l.texte)
            break
    if not mail:
        lm = chercher(r"e-?mail|courriel|mel\b")
        if lm:
            mail = V(lm.valeur.replace(" ", ""), min(lm.conf, 0.5), lm.page, lm.boite, "ocr", lm.valeur)
    cl["email"] = mail
    ct = chercher(r"contact|interlocuteur|responsable|preleveur|a l.attention")
    cl["contact"] = ct
    siret = V(None, 0.0, None, None, "ocr")
    for l in lignes_clientes:
        s = trouver_siret(l.texte)
        if s:
            siret = V(s, l.conf, l.page, l.boite, "ocr", l.texte)
            break
    cl["siret"] = siret

    # ---- commande
    cmd = {}
    cmd["numero"] = chercher(r"n[°o]?\s*(de\s*)?commande|bon de commande n|ref\.? dossier|reference dossier|n[°o] bdc|\bbdc\b|n[°o] bon",
                             lambda t: bool(re.search(r"\d", t)))
    cmd["date"] = _val_date(rgs, r"date( de)?( la)? commande|commande du|^date\b|\bdate\b|\ble\b", annee)
    cmd["delai"] = lire_delai(pages, lignes_clientes)
    cmd["annule_et_remplace"] = V(None, 0.0, None, None, "ocr")
    for l, m in _trouver_ligne(lignes_clientes, r"annule\w*\s*(et|&)\s*remplace"):
        reste = l.texte[m.end():]
        mm = re.search(r"([A-Z0-9][A-Z0-9\-_/]{3,})", reste)
        cmd["annule_et_remplace"] = V(mm.group(1) if mm else l.texte, l.conf, l.page, l.boite, "ocr", l.texte)
        break
    cmd["facturation"] = chercher(r"factur")
    cmd["remarques"] = chercher(r"remarque|commentaire|observation|precision")

    # ---- site
    site = {}
    site["commune"] = chercher(r"commune|localite|ville du site|site de prelevement|lieu de prelevement")
    site["cp"] = V(None, 0.0, None, None, "ocr")
    site["lieu_dit"] = chercher(r"lieu[- ]?dit|domaine")
    parcelles = []
    for l in lignes_clientes:
        if re.search(r"parcelle|ilot|section", norm(l.texte)) and not re.search(r"ref|profondeur", norm(l.texte)):
            for p in trouver_parcelles(l.texte):
                parcelles.append(V(p, l.conf, l.page, l.boite, "ocr", l.texte))
    site["parcelles"] = parcelles

    # ---- prélèvement
    prel = {}
    prel["date"] = _val_date(rgs, r"date( du)? prelevement|prelevement( effectue)? le|preleve le|prelevement du|date de collecte", annee)
    prel["preleveur"] = chercher(r"preleveur|preleve par|realise par|technicien")

    # ---- réception laboratoire (ajouts du labo : tampons, notes, étiquettes)
    rec = {"date": V(None, 0.0, None, None, "ocr"), "temperature_c": V(None, 0.0, None, None, "ocr"),
           "annotations": []}
    lab_rgx = re.compile(r"revue de contrat|recu le|t[°o]\s*=|n[°o] dossier|echeance|saisi le|glaciere|flacons|"
                         r"conforme|temperature|rec[eu]|26orv|etiquette", re.I)
    for l in lignes:
        n = norm(l.texte)
        lab_ink = l.encre in ("rouge", "vert") or (l.encre == "bleu" and lab_rgx.search(n))
        if lab_rgx.search(n) and (lab_ink or est_labo(l.texte) is False and l.encre != "noir") or \
                re.search(r"revue de contrat|saisi le|n[°o] dossier \d|echeance", n):
            t = nettoyer_espaces(l.texte)
            if len(t) >= 4 and t not in [a.valeur for a in rec["annotations"]]:
                rec["annotations"].append(V(t, l.conf, l.page, l.boite, "ocr", l.texte))
            if not rec["date"]:
                m = re.search(r"re[cç]u(e)? le", n)
                if m:
                    d = premiere_date(l.texte, annee)
                    if d:
                        rec["date"] = V(_date_iso(d), l.conf, l.page, l.boite, "ocr", l.texte)
            if not rec["temperature_c"]:
                t_c = temperature(l.texte)
                if t_c is not None:
                    rec["temperature_c"] = V(t_c, l.conf, l.page, l.boite, "ocr", l.texte)
    if not rec["date"]:
        for l in lignes:
            if re.search(r"re[cç]u(e)? le", norm(l.texte)):
                d = premiere_date(l.texte, annee)
                if d:
                    rec["date"] = V(_date_iso(d), l.conf, l.page, l.boite, "ocr", l.texte)
                    break
    if not rec["temperature_c"]:
        for l in lignes:
            if re.search(r"glaciere|t[°o]|temp", norm(l.texte)):
                t_c = temperature(l.texte)
                if t_c is not None:
                    rec["temperature_c"] = V(t_c, l.conf, l.page, l.boite, "ocr", l.texte)
                    break

    # ---- le site reprend la commune du client si elle est absente et unique
    if not site["commune"] and ville:
        site["commune"] = V(ville.valeur, ville.conf * 0.5, ville.page, ville.boite, "regle",
                            note="commune du client faute de commune de site")
    return {"client": cl, "commande": cmd, "site": site, "prelevement": prel, "reception_labo": rec}


def echeance(date_recue: str | None, delai: str | None) -> str | None:
    if not date_recue or delai not in config.DELAIS:
        return None
    return ajouter_jours_ouvres(date.fromisoformat(date_recue), config.DELAIS[delai]).isoformat()
