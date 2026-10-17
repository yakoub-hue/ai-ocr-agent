"""Chaîne complète : un PDF de bon de commande -> JSON §4 + trace + alertes.

Briques : 1 prétraitement, 2 identification (tableaux / texte libre),
3 extraction, 4 ratures et manuscrit (dans tableau/echantillons), 5 contrôles
et confiance. L'export LIMS / GC-MS (brique 6) est dans export.py.

Le scan d'origine n'est jamais modifié : seules des copies normalisées
(PNG) sont écrites dans le dossier de sortie.
"""
from __future__ import annotations

import json
import random
import re
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from . import champs as CH
from . import controles
from .documents import decrire_document
from .liens import dossier_dans_textes
from . import ia as IA
from .lecture import lire_lignes
from .pretraitement import pretraiter_pdf
from .tableau import detecter_grilles, lire_tableau, masque_traits
from .echantillons import lire_echantillons
from .textes import norm, normaliser_telephone
from .trace import Journal, V, aplatir, valeurs

VERSION_CHAINE = "1.0"


def fixer_graines(graine: int = 0) -> None:
    random.seed(graine)
    np.random.seed(graine)
    cv2.setRNGSeed(graine)
    cv2.setNumThreads(1)


def _nombre_declare(lignes) -> int | None:
    for l in lignes:
        n = norm(l.texte)
        if re.search(r"nombre|total|nb\b", n) and re.search(r"echantillon|flacon|prelevement", n):
            m = re.findall(r"(?<![\d/.-])(\d{1,2})(?![\d/.-])", l.texte)
            if m:
                return int(m[-1])
    return None


def _jetons_refs(lignes) -> list[str]:
    jetons = []
    for l in lignes:
        for mot in re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)+", l.texte):
            if re.search(r"\d", mot) and re.search(r"[A-Za-z]", mot):
                jetons.append(mot.upper())
    return jetons


def construire_arbre(nom: str, champs: dict, tab) -> dict:
    """Arbre de Valeur au format §4 (la partie échantillons vient du tableau)."""
    cl, cmd, site, pre, rec = (champs[k] for k in ("client", "commande", "site", "prelevement", "reception_labo"))
    tel = cl.get("telephone")
    if tel and tel.valeur:
        n = normaliser_telephone(tel.valeur)
        if n:
            tel.valeur = n
    arbre = {
        "fichier": nom,
        "client": {k: cl.get(k, V()) for k in ("raison_sociale", "adresse", "cp", "ville", "telephone",
                                               "email", "contact", "siret")},
        "commande": {k: cmd.get(k, V()) for k in ("numero", "date", "delai", "annule_et_remplace",
                                                  "facturation", "remarques")},
        "site": {"commune": site.get("commune", V()), "cp": site.get("cp", V()),
                 "lieu_dit": site.get("lieu_dit", V()), "parcelles": site.get("parcelles", [])},
        "prelevement": {"date": pre.get("date", V()), "preleveur": pre.get("preleveur", V())},
        "echantillons": [], "echantillons_annules": [],
        "reception_labo": {"date": rec["date"], "temperature_c": rec["temperature_c"],
                           "annotations": rec["annotations"]},
        "ratures": [],
    }
    for l in tab.lignes:
        v = l.valeurs
        g = lambda k: v.get(k, V())
        if l.annulee:
            arbre["echantillons_annules"].append({"ref_echantillon": g("ref"), "motif": l.motif or V()})
            for r in l.ratures:
                arbre["ratures"].append(r)
            continue
        analyses = g("analyses")
        hap = g("hap_demande")
        liste = list(analyses.valeur or [])
        if hap.valeur is True and "HAP16" not in liste:
            liste = ["HAP16"] + liste
        if hap.valeur is False and "HAP16" in liste:
            liste.remove("HAP16")
        analyses = V(liste, analyses.conf, analyses.page, analyses.boite, analyses.source)
        matrice = g("matrice")
        if not matrice.valeur:
            matrice = V("terre agricole", 0.5, None, None, "defaut", note="matrice par défaut (sols agricoles)")
        arbre["echantillons"].append({
            "ref_echantillon": g("ref"), "parcelle": g("parcelle"), "profondeur_cm": g("profondeur_cm"),
            "matrice": matrice, "analyses": analyses, "hap_demande": hap, "observations": g("observations"),
        })
        arbre["ratures"].extend(l.ratures)
    return arbre


def traiter_pdf(chemin_pdf, sortie, verbeux: bool = True, ia: bool = False) -> dict:
    """Traite un PDF. Retourne {'doc', 'trace', 'revue', 'raisons', 'echeance', 'journal'}.
    N'échoue jamais : une erreur devient une alerte bloquante et une revue obligatoire.
    ia=True fait relire les champs par Gemini (bdc/ia.py) ; en cas d'échec de l'appel,
    les valeurs des règles sont gardées."""
    fixer_graines()
    chemin_pdf, sortie = Path(chemin_pdf), Path(sortie)
    for d in ("json", "trace", "pages"):
        (sortie / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    journal = Journal()
    alertes_lecture: list[dict] = []
    pages, tableaux, lignes, champs, tab = [], [], [], None, None
    try:
        pages = pretraiter_pdf(chemin_pdf)
        for p in pages:
            cv2.imwrite(str(sortie / "pages" / f"{chemin_pdf.stem}_p{p.numero}.png"),
                        cv2.cvtColor(p.couleur, cv2.COLOR_RGB2BGR))
        zones = {}
        for p in pages:
            traits = masque_traits(p.binaire)
            zones[p.numero] = []
            for g in detecter_grilles(p.binaire, p.numero):
                if g.n_colonnes >= 3 and g.n_rangees >= 3:
                    tableaux.append(lire_tableau(p, g, traits))
                    zones[p.numero].append(g.boite)
        for p in pages:
            lignes += lire_lignes(p, zones_exclues=zones[p.numero])
    except Exception as e:                                        # noqa: BLE001
        journal("erreur de lecture : " + repr(e))
        journal(traceback.format_exc(limit=3))
        alertes_lecture.append({"controle": "traitement", "niveau": "bloquant",
                                "message": f"Erreur de lecture : {e!r} — saisie manuelle nécessaire"})
    try:
        champs = CH.extraire_champs(pages, lignes, journal) if pages else None
    except Exception as e:                                        # noqa: BLE001
        journal("erreur d'extraction des champs : " + repr(e))
        alertes_lecture.append({"controle": "traitement", "niveau": "bloquant",
                                "message": f"Erreur d'extraction des champs : {e!r}"})
    if champs is None:
        champs = {"client": {}, "commande": {}, "site": {}, "prelevement": {},
                  "reception_labo": {"date": V(), "temperature_c": V(), "annotations": []}}
    try:
        tab = lire_echantillons(tableaux, _jetons_refs(lignes), journal)
        for a in tab.alertes:
            alertes_lecture.append(a.vers_json())
    except Exception as e:                                        # noqa: BLE001
        journal("erreur d'extraction des échantillons : " + repr(e))
        alertes_lecture.append({"controle": "tableau_echantillons", "niveau": "bloquant",
                                "message": f"Erreur de lecture du tableau : {e!r}"})
        from .echantillons import TableauEchantillons
        tab = TableauEchantillons()
    if ia and pages:
        try:
            alertes_lecture += IA.affiner(champs, tab, pages, lignes, tableaux, journal)
        except Exception as e:                                    # noqa: BLE001
            journal("erreur de la relecture IA : " + repr(e))
            alertes_lecture.append({"controle": "ia", "niveau": "avertissement",
                                    "message": f"Relecture IA impossible ({e!r}) : lecture OCR et règles seules"})
    arbre = construire_arbre(chemin_pdf.name, champs, tab)
    doc = valeurs(arbre)
    doc["ratures"] = [r.vers_json() for r in arbre["ratures"]]
    for r in doc["ratures"]:
        r["avant"] = r.get("avant")
    # alertes portées par les lignes
    for l in tab.lignes:
        for a in l.alertes:
            alertes_lecture.append(a.vers_json())
    declare = _nombre_declare(lignes)
    doc["alertes"] = controles.controler(doc, declare, alertes_lecture)
    trace = aplatir({k: v for k, v in arbre.items() if k not in ("ratures", "fichier")})
    for i, r in enumerate(arbre["ratures"]):
        trace.append({"champ": f"ratures[{i}]", "valeur": r.vers_json(), "page": r.page,
                      "zone": list(r.boite) if r.boite else None, "confiance": round(float(r.conf), 3),
                      "source": "rature", "texte_lu": None, "note": None})
    conf = {t["champ"]: t["confiance"] for t in trace}
    revue, raisons = controles.revue_requise(doc, conf, doc["alertes"])
    doc_out = {k: doc[k] for k in ("fichier", "client", "commande", "site", "prelevement", "echantillons",
                                   "echantillons_annules", "reception_labo", "ratures", "alertes")}
    (sortie / "json" / f"{chemin_pdf.stem}.json").write_text(
        json.dumps(doc_out, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"fichier": chemin_pdf.name, "version_chaine": VERSION_CHAINE, "duree_s": round(time.time() - t0, 1),
            "revue_humaine": revue, "raisons_revue": raisons, "echeance": controles.echeance(doc),
            "nombre_declare": declare, "pages": [p.numero for p in pages],
            "document": decrire_document(pages, lignes, tableaux),
            "dossier_labo": dossier_dans_textes(l.texte for l in lignes),
            # avant l'OCR : corrections de chaque page, et ses deux zones lues séparément
            # (tableau des échantillons ; le reste = client, commande, site)
            "pretraitement": [{"page": p.numero, "taille": list(p.couleur.shape[:2]),
                               "etapes": [e for e in p.journal if e.get("etape") != "rendu"]} for p in pages],
            "zones_tableau": [{"page": t.page.numero, "boite": [int(v) for v in t.grille.boite]} for t in tableaux],
            "journal": journal.notes}
    (sortie / "trace" / f"{chemin_pdf.stem}.trace.json").write_text(
        json.dumps({"meta": meta, "valeurs": trace}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if verbeux:
        print(f"[{chemin_pdf.name}] {len(doc['echantillons'])} échantillons, {len(doc['alertes'])} alertes, "
              f"revue={'oui' if revue else 'non'} ({meta['duree_s']} s)", flush=True)
    return {"doc": doc_out, "trace": trace, "meta": meta}
