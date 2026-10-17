"""Commande unique : lit tous les PDF d'un dossier et produit JSON, traces, CSV LIMS,
liste GC-MS et, si la vérité terrain est fournie, l'évaluation (§6).

    python run.py --entree <dossier_pdf | bdc.zip> --sortie sortie [--verite verite_terrain]
    python run.py --demo <un_bdc_inconnu.pdf>          # démonstration sur un seul document
    python run.py ... --ia                              # champs relus par Gemini (GEMINI_API_KEY)
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from bdc import base, evaluation, export, ia
from bdc.archive import pdfs_du_zip
from bdc.documents import regrouper_envois
from bdc.pipeline import fixer_graines, traiter_pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entree", help="dossier ou archive ZIP contenant les PDF de BDC")
    ap.add_argument("--demo", help="un seul PDF (BDC inconnu)")
    ap.add_argument("--sortie", default="sortie")
    ap.add_argument("--verite", help="dossier verite_terrain (JSON + manifeste.csv) pour l'évaluation")
    ap.add_argument("--resultats-gcms", help="concentrations du projet GC-MS (blpc_concentrations.csv) à rattacher au LIMS")
    ap.add_argument("--correspondance", help="flacon labo -> échantillon client : gcms_hap_a_analyser.csv "
                                             "avec la colonne echantillon_labo remplie")
    ap.add_argument("--ia", action="store_true", help="faire relire les champs par Gemini (clé GEMINI_API_KEY)")
    ap.add_argument("--base", action="store_true", help="enregistrer BDC (et résultats GC-MS) dans la base de "
                                                        "traçabilité donnees/ (page Clients de l'application)")
    a = ap.parse_args()
    if not a.entree and not a.demo:
        ap.error("--entree ou --demo est requis")
    if a.ia and not ia.disponible():
        sys.exit("--ia : définir GEMINI_API_KEY (et installer google-genai).")
    fixer_graines()
    sortie = Path(a.sortie)
    zip_tmp = None                                  # archive ZIP : PDF extraits le temps du traitement
    if a.entree and a.entree.lower().endswith(".zip"):
        zip_tmp = tempfile.TemporaryDirectory()
        trouves, ignores = pdfs_du_zip(Path(a.entree).read_bytes())
        for nom, data in trouves:
            (Path(zip_tmp.name) / nom).write_bytes(data)
        print(f"{a.entree} : {len(trouves)} PDF" + (f", {len(ignores)} fichier(s) ignoré(s)" if ignores else ""))
        pdfs = sorted(Path(zip_tmp.name).glob("*.pdf"))
    else:
        pdfs = [Path(a.demo)] if a.demo else sorted(Path(a.entree).glob("*.pdf"))
    if not pdfs:
        sys.exit("Aucun PDF trouvé.")
    docs, revues = [], {}
    for p in pdfs:
        r = traiter_pdf(p, sortie, ia=a.ia)
        docs.append(r["doc"])
        revues[p.name] = r["meta"]["revue_humaine"]
        if a.base:
            pages = sorted((sortie / "pages").glob(f"{p.stem}_p*.png"), key=lambda x: int(x.stem.rsplit("_p", 1)[1]))
            base.enregistrer_bdc(p.name, p.read_bytes(), r, [x.read_bytes() for x in pages])
    lims = export.exporter_lims(docs, sortie / "lims.csv", revues)
    hap = export.liste_hap_gcms(docs, sortie / "gcms_hap_a_analyser.csv")
    envois = regrouper_envois(docs)
    with open(sortie / "envois.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["envoi", "client", "numero_commande", "fichiers"], delimiter=";")
        w.writeheader()
        w.writerows(envois)
    print(f"{len(docs)} document(s) -> {len(envois)} envoi(s) (envois.csv)")
    print(f"LIMS : {len(lims)} lignes -> {sortie / 'lims.csv'} ; HAP à quantifier : {len(hap)}")
    if a.resultats_gcms:
        n, orphelins = export.injecter_resultats_gcms(sortie / "lims.csv", a.resultats_gcms,
                                                      sortie / "lims_avec_resultats.csv", a.correspondance)
        print(f"Concentrations rattachées à {n} échantillons -> lims_avec_resultats.csv")
        if orphelins:
            print(f"{len(orphelins)} flacon(s) GC-MS sans échantillon de BDC (correspondance à compléter) : "
                  + ", ".join(orphelins[:10]) + (" …" if len(orphelins) > 10 else ""))
        if a.base:
            n, _ = base.importer_resultats(*export.lire_resultats_gcms(a.resultats_gcms),
                                           lot=Path(a.resultats_gcms).parent.name)
            print(f"{n} injection(s) GC-MS enregistrée(s) dans la base")
    if a.base:
        print(f"Base de traçabilité : {base.DOSSIER / 'tracabilite.db'}")
    if a.verite:
        v = Path(a.verite)
        res = evaluation.evaluer_dossier(sortie / "json", v, v / "manifeste.csv", revues)
        (sortie / "evaluation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        (sortie / "evaluation.md").write_text(evaluation.rapport_markdown(res), encoding="utf-8")
        g = res["global"]
        print("Évaluation globale : " + ", ".join(f"{k}={100*g[k]:.1f}%" for k in evaluation.POIDS if g.get(k) is not None)
              + f" | total={100*g['total']:.1f}%")


if __name__ == "__main__":
    main()
