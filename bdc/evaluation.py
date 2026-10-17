"""Évaluation (§6) : compare les JSON produits à la vérité terrain.

Sept critères pondérés (somme = 100 %), résultats globaux et par difficulté
(facile / moyenne / difficile, colonne ``difficulte`` du manifeste).

Définitions retenues (à discuter dans le rapport) :
* échantillons détectés : P/R/F1 sur les références normalisées ;
* hap_demande : F1 de la classe « HAP demandé » sur les références de la vérité terrain
  (un échantillon non détecté compte comme manqué) ;
* attributs d'échantillon : moyenne, sur les échantillons de la vérité terrain, de la
  concordance de parcelle, profondeur, matrice, analyses (ensemble) et observations
  (1 − CER) ; un échantillon non détecté vaut 0 ;
* champs client/commande : égalité normalisée pour les champs courts, 1 − CER pour les
  textes libres ; null attendu et null rendu comptent juste ;
* ratures et annulations : moyenne du F1 des ratures (échantillon, champ, après) et du F1
  des références annulées ;
* alertes : F1 sur les noms de contrôle (le niveau doit aussi concorder) ;
* revue humaine : le drapeau doit être levé si et seulement si un champ indispensable
  extrait diffère de la vérité terrain (accuracy sur les documents).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .textes import cer, compact, norm

POIDS = {"echantillons": 0.20, "hap": 0.20, "attributs": 0.15, "champs": 0.15,
         "ratures": 0.10, "alertes": 0.10, "revue": 0.10}
LIBRES = {"raison_sociale", "adresse", "contact", "remarques", "facturation", "lieu_dit", "commune",
          "ville", "preleveur"}


def _ref(s):
    return compact(s).upper() if s else ""


def prf(attendus: set, obtenus: set) -> tuple[float, float, float]:
    if not attendus and not obtenus:
        return 1.0, 1.0, 1.0
    tp = len(attendus & obtenus)
    p = tp / len(obtenus) if obtenus else 0.0
    r = tp / len(attendus) if attendus else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _egal(champ, a, b) -> float:
    if a in (None, "", []) and b in (None, "", []):
        return 1.0
    if a in (None, "", []) or b in (None, "", []):
        return 0.0
    if champ in LIBRES:
        return max(0.0, 1 - cer(norm(str(a)), norm(str(b))))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 1.0 if abs(a - b) < 0.05 else 0.0
    return 1.0 if compact(str(a)) == compact(str(b)) else 0.0


def evaluer_document(gt: dict, pred: dict, revue: bool | None = None) -> dict:
    g_ech = {_ref(e["ref_echantillon"]): e for e in gt["echantillons"] if e.get("ref_echantillon")}
    p_ech = {_ref(e["ref_echantillon"]): e for e in pred["echantillons"] if e.get("ref_echantillon")}
    P, R, F = prf(set(g_ech), set(p_ech))

    pos_g = {r for r, e in g_ech.items() if e.get("hap_demande")}
    pos_p = {r for r, e in p_ech.items() if e.get("hap_demande")}
    f_hap = prf(pos_g, pos_p)[2]

    attr = []
    for r, e in g_ech.items():
        q = p_ech.get(r)
        for ch in ("parcelle", "profondeur_cm", "matrice", "analyses", "observations"):
            if q is None:
                attr.append(0.0)
            elif ch == "analyses":
                attr.append(1.0 if set(e.get(ch) or []) == set(q.get(ch) or []) else 0.0)
            else:
                attr.append(_egal("observations" if ch == "observations" else ch, e.get(ch), q.get(ch)))
    s_attr = sum(attr) / len(attr) if attr else 1.0

    champs = []
    for bloc in ("client", "commande", "site", "prelevement"):
        for k, v in gt[bloc].items():
            if k == "parcelles":
                champs.append(prf({compact(x).upper() for x in v or []},
                                  {compact(x).upper() for x in pred[bloc].get(k) or []})[2])
            else:
                champs.append(_egal(k, v, pred[bloc].get(k)))
    for k in ("date", "temperature_c"):
        champs.append(_egal(k, gt["reception_labo"].get(k), pred["reception_labo"].get(k)))
    s_champs = sum(champs) / len(champs) if champs else 1.0

    def cle_r(r):
        return (_ref(r.get("echantillon")), r["champ"], compact(str(r.get("apres"))))
    f_rat = prf({cle_r(r) for r in gt["ratures"]}, {cle_r(r) for r in pred["ratures"]})[2]
    f_ann = prf({_ref(e["ref_echantillon"]) for e in gt["echantillons_annules"]},
                {_ref(e["ref_echantillon"]) for e in pred["echantillons_annules"]})[2]
    s_rat = (f_rat + f_ann) / 2

    ga = {(a["controle"], a["niveau"]) for a in gt["alertes"]}
    pa = {(a["controle"], a["niveau"]) for a in pred["alertes"]}
    s_al = prf(ga, pa)[2]

    # revue humaine : erreur sur un champ indispensable ?
    erreur = (F < 1.0 or f_hap < 1.0
              or _egal("raison_sociale", gt["client"].get("raison_sociale"), pred["client"].get("raison_sociale")) < 0.9
              or _egal("delai", gt["commande"].get("delai"), pred["commande"].get("delai")) < 1.0)
    s_rev = None if revue is None else (1.0 if bool(revue) == erreur else 0.0)

    scores = {"echantillons": F, "hap": f_hap, "attributs": s_attr, "champs": s_champs,
              "ratures": s_rat, "alertes": s_al, "revue": s_rev}
    detail = {"precision_ech": P, "rappel_ech": R, "n_gt": len(g_ech), "n_pred": len(p_ech)}
    return {"scores": scores, "detail": detail}


def total(scores: dict) -> float:
    pts = {k: v for k, v in scores.items() if v is not None}
    poids = sum(POIDS[k] for k in pts)
    return sum(POIDS[k] * v for k, v in pts.items()) / poids if poids else 0.0


def evaluer_dossier(dossier_json, dossier_gt, manifeste=None, revues: dict | None = None) -> dict:
    dossier_json, dossier_gt = Path(dossier_json), Path(dossier_gt)
    diff = {}
    if manifeste and Path(manifeste).exists():
        with open(manifeste, newline="", encoding="utf-8") as f:
            diff = {r["fichier"].replace(".pdf", ""): r["difficulte"] for r in csv.DictReader(f)}
    lignes = []
    for gtf in sorted(dossier_gt.glob("*.json")):
        pf = dossier_json / gtf.name
        gt = json.loads(gtf.read_text(encoding="utf-8"))
        if not pf.exists():
            continue
        pred = json.loads(pf.read_text(encoding="utf-8"))
        rev = None if revues is None else revues.get(gt["fichier"])
        r = evaluer_document(gt, pred, rev)
        r["fichier"] = gtf.stem
        r["difficulte"] = diff.get(gtf.stem, "n/a")
        r["total"] = total(r["scores"])
        lignes.append(r)

    def moyenne(sel):
        if not sel:
            return {}
        out = {}
        for k in POIDS:
            vals = [l["scores"][k] for l in sel if l["scores"][k] is not None]
            out[k] = sum(vals) / len(vals) if vals else None
        out["total"] = sum(l["total"] for l in sel) / len(sel)
        out["n"] = len(sel)
        return out

    niveaux = sorted({l["difficulte"] for l in lignes})
    return {"documents": lignes, "global": moyenne(lignes),
            "par_difficulte": {n: moyenne([l for l in lignes if l["difficulte"] == n]) for n in niveaux}}


def rapport_markdown(res: dict) -> str:
    def fmt(v):
        return "—" if v is None else f"{100 * v:.1f} %"
    cols = list(POIDS)
    out = ["# Évaluation (§6)\n", "| Périmètre | n | " + " | ".join(f"{c} ({int(100 * POIDS[c])} %)" for c in cols) + " | **Total** |",
           "|---|---|" + "---|" * (len(cols) + 1)]
    lignes = [("Global", res["global"])] + [(k, v) for k, v in res["par_difficulte"].items()]
    for nom, m in lignes:
        if m:
            out.append(f"| {nom} | {m['n']} | " + " | ".join(fmt(m[c]) for c in cols) + f" | **{fmt(m['total'])}** |")
    out += ["", "## Par document\n", "| Document | Difficulté | " + " | ".join(cols) + " | Total |",
            "|---|---|" + "---|" * (len(cols) + 1)]
    for d in res["documents"]:
        out.append(f"| {d['fichier']} | {d['difficulte']} | " + " | ".join(fmt(d["scores"][c]) for c in cols)
                   + f" | {fmt(d['total'])} |")
    return "\n".join(out) + "\n"
