"""Brique 6 : export LIMS (CSV trié par urgence) et lien avec la chaîne GC-MS.

* ``exporter_lims`` : une ligne par échantillon (hors annulés), triée par urgence
  puis échéance, avec les HAP demandés, le client, la référence et l'échéance.
* ``liste_hap_gcms`` : références à quantifier en HAP16, avec une colonne
  echantillon_labo vide : une fois remplie avec l'identifiant donné au flacon par
  le laboratoire (ex. 25BLPC003018-001), ce fichier sert de table de correspondance.
* ``injecter_resultats_gcms`` : rattache les concentrations du projet GC-MS aux
  échantillons du LIMS, donc au bon client. Résultats acceptés : le tableau
  blpc_concentrations.csv (un composé par colonne) ou un CSV long (echantillon,
  compose, concentration). Les flacons portent l'identifiant du labo et les BDC la
  référence du client : sans table de correspondance, seuls les identifiants égaux
  à une référence de BDC sont rattachés. Une même référence pouvant revenir d'un
  BDC à l'autre (MV-01…), la correspondance porte aussi le n° de commande.
"""
from __future__ import annotations

import csv

from .controles import echeance

RANG_DELAI = {"express_72h": 0, "urgent_J+5": 1, "standard_10j": 2}
COLONNES_LIMS = ["rang_urgence", "delai", "echeance", "fichier", "client", "siret", "contact", "telephone",
                 "email", "numero_commande", "ref_echantillon", "parcelle", "profondeur_cm", "matrice",
                 "analyses", "hap_demande", "hap16", "observations", "revue_humaine", "nb_alertes"]


def lignes_lims(docs: list[dict], revues: dict | None = None) -> list[dict]:
    revues = revues or {}
    lignes = []
    for d in docs:
        delai = d["commande"].get("delai")
        ech = echeance(d)
        for e in d["echantillons"]:
            hap = e.get("hap_demande")
            lignes.append({
                "rang_urgence": RANG_DELAI.get(delai, 9), "delai": delai or "", "echeance": ech or "",
                "fichier": d["fichier"], "client": d["client"].get("raison_sociale") or "",
                "siret": d["client"].get("siret") or "", "contact": d["client"].get("contact") or "",
                "telephone": d["client"].get("telephone") or "", "email": d["client"].get("email") or "",
                "numero_commande": d["commande"].get("numero") or "",
                "ref_echantillon": e.get("ref_echantillon") or "", "parcelle": e.get("parcelle") or "",
                "profondeur_cm": e.get("profondeur_cm") or "", "matrice": e.get("matrice") or "",
                "analyses": "|".join(e.get("analyses") or []),
                "hap_demande": "" if hap is None else ("oui" if hap else "non"),
                "hap16": "HAP16" if hap else "",
                "observations": e.get("observations") or "",
                "revue_humaine": "oui" if revues.get(d["fichier"], False) else "non",
                "nb_alertes": len(d.get("alertes", [])),
            })
    lignes.sort(key=lambda l: (l["rang_urgence"], l["echeance"] or "9999", l["fichier"], l["ref_echantillon"]))
    return lignes


def exporter_lims(docs: list[dict], chemin, revues: dict | None = None) -> list[dict]:
    lignes = lignes_lims(docs, revues)
    with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES_LIMS, delimiter=";")
        w.writeheader()
        w.writerows(lignes)
    return lignes


COLONNES_HAP = ["echantillon_labo", "numero_commande", "ref_echantillon", "fichier", "client", "delai", "echeance"]
# colonnes de blpc_concentrations.csv qui ne sont pas des composés
META_GCMS = {"echantillon", "multiplicateur", "rapport_tic", "controle_tic", "nouveau_multiplicateur"}


def liste_hap_gcms(docs: list[dict], chemin) -> list[dict]:
    """Échantillons dont les HAP doivent être quantifiés. La colonne echantillon_labo
    est à remplir par le laboratoire (identifiant du flacon) : le fichier devient la
    table de correspondance de --correspondance."""
    lignes = [{"echantillon_labo": "", "numero_commande": d["commande"].get("numero") or "",
               "ref_echantillon": e["ref_echantillon"], "fichier": d["fichier"],
               "client": d["client"].get("raison_sociale") or "", "delai": d["commande"].get("delai") or "",
               "echeance": echeance(d) or ""}
              for d in docs for e in d["echantillons"] if e.get("hap_demande") and e.get("ref_echantillon")]
    lignes.sort(key=lambda l: (RANG_DELAI.get(l["delai"], 9), l["echeance"] or "9999", l["ref_echantillon"]))
    with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES_HAP, delimiter=";")
        w.writeheader()
        w.writerows(lignes)
    return lignes


def _lire_csv(chemin) -> list[dict]:
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        entete = f.readline()
    delim = ";" if entete.count(";") >= entete.count(",") else ("\t" if "\t" in entete else ",")
    with open(chemin, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=delim))


def lire_resultats_gcms(chemin) -> tuple[dict, dict]:
    """({identifiant flacon: {composé: concentration}}, {identifiant: statut}) depuis
    blpc_concentrations.csv (format large) ou un CSV long (echantillon, compose, concentration)."""
    lignes = _lire_csv(chemin)
    conc, statut = {}, {}
    if lignes and "compose" in lignes[0]:
        for r in lignes:
            ech = (r.get("echantillon") or r.get("ref_echantillon") or "").strip()
            if ech:
                conc.setdefault(ech, {})[r["compose"]] = r.get("concentration", "")
        return conc, statut
    for r in lignes:
        ech = (r.get("echantillon") or "").strip()
        if not ech:
            continue
        conc[ech] = {c: v for c, v in r.items() if c not in META_GCMS and c}
        statut[ech] = r.get("controle_tic", "")
    return conc, statut


def lire_correspondance(chemin) -> dict:
    """{identifiant flacon: (n° de commande, référence client)} ; lignes sans identifiant ignorées."""
    return {r["echantillon_labo"].strip(): ((r.get("numero_commande") or "").strip(), r["ref_echantillon"].strip())
            for r in _lire_csv(chemin) if (r.get("echantillon_labo") or "").strip()}


def injecter_resultats_gcms(lims_csv, resultats_csv, sortie_csv, correspondance_csv=None) -> tuple[int, list[str]]:
    """Ajoute au CSV LIMS les concentrations HAP de chaque échantillon.
    Renvoie (nombre d'échantillons rattachés, flacons sans échantillon du LIMS)."""
    conc, statut = lire_resultats_gcms(resultats_csv)
    corr = lire_correspondance(correspondance_csv) if correspondance_csv else {}
    lignes = _lire_csv(lims_csv)
    par_cle = {(l["numero_commande"], l["ref_echantillon"]): l for l in lignes}
    par_ref = {}
    for l in lignes:
        par_ref.setdefault(l["ref_echantillon"], []).append(l)
    composes = list(dict.fromkeys(c for v in conc.values() for c in v))
    for l in lignes:
        l.update({"echantillon_labo": "", "statut_gcms": "", **{f"conc_{c}": "" for c in composes}})
    orphelins, n = [], 0
    for ech, valeurs in conc.items():
        if ech in corr:
            num, ref = corr[ech]
            cible = par_cle.get((num, ref)) or (par_ref[ref][0] if len(par_ref.get(ref, [])) == 1 else None)
        else:                                        # identifiant égal à une référence de BDC, sans ambiguïté
            cible = par_ref[ech][0] if len(par_ref.get(ech, [])) == 1 else None
        if cible is None:
            orphelins.append(ech)
            continue
        cible["echantillon_labo"] = ech
        cible["statut_gcms"] = statut.get(ech, "")
        cible.update({f"conc_{c}": v for c, v in valeurs.items()})
        n += 1
    with open(sortie_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()) if lignes else COLONNES_LIMS, delimiter=";")
        w.writeheader()
        w.writerows(lignes)
    return n, orphelins
