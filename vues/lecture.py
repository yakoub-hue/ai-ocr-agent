"""Page « Lecture des BDC » : dépôt de PDF ou d'archives ZIP, lecture par la chaîne,
enregistrement dans la base.

Un lot se lit BDC par BDC : une erreur sur un document est notée et n'arrête pas les
autres ; chaque résultat est gardé dans la session, si bien qu'une page rechargée en
cours de lot reprend là où elle s'était arrêtée.
"""
import hashlib
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from bdc import base, ia
from bdc.archive import pdfs_du_zip
from bdc.ocr import verifier_tesseract
from bdc.pipeline import traiter_pdf
from vues.commun import bouton_json, details, entete, fiche

QUOTA_GRATUIT = 20          # requêtes Gemini par jour sur l'offre gratuite : un appel par BDC


def lire_bdc(nom: str, contenu: bytes, avec_ia: bool) -> dict:
    """Passe un PDF dans la chaîne ; garde en mémoire le résultat et les pages normalisées."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf = tmp / Path(nom).name
        pdf.write_bytes(contenu)
        r = traiter_pdf(pdf, tmp / "sortie", verbeux=False, ia=avec_ia)
        pages = sorted((tmp / "sortie" / "pages").glob("*.png"), key=lambda p: int(p.stem.rsplit("_p", 1)[1]))
        r["pages"] = [p.read_bytes() for p in pages]
    r["confiance"] = {t["champ"]: t["confiance"] for t in r["trace"]}
    r["pdf"] = contenu
    return r


def documents(fichiers) -> tuple[list[tuple[str, bytes]], list[str]]:
    """PDF déposés directement ou contenus dans des archives ZIP ; fichiers écartés."""
    pdfs, ignores = [], []
    for f in fichiers:
        if f.name.lower().endswith(".zip"):
            try:
                trouves, ecartes = pdfs_du_zip(f.getvalue())
            except Exception as e:                                       # noqa: BLE001
                ignores.append(f"{f.name} (archive illisible : {e})")
                continue
            pdfs += trouves
            ignores += [f"{f.name} › {x}" for x in ecartes]
        else:
            pdfs.append((f.name, f.getvalue()))
    vus, uniques = set(), []
    for nom, data in pdfs:                     # le même PDF déposé deux fois n'est lu qu'une fois
        h = hashlib.sha1(data).hexdigest()
        if h not in vus:
            vus.add(h)
            uniques.append((nom, data))
    return uniques, ignores


st.title("Lecture des bons de commande HAP")
st.caption("Déposez des BDC en PDF ou une archive ZIP de BDC. La lecture prend 1 à 3 minutes par document.")

try:
    verifier_tesseract()
except SystemExit as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
avec_ia = c1.toggle("Relecture IA (Gemini)", value=ia.disponible(), disabled=not ia.disponible(),
                    help="Gemini relit le texte de l'OCR (champs) et l'image du tableau des échantillons. "
                         "Sont envoyés à Google : le texte lu et l'image du seul tableau, jamais la page entière."
                         + ("" if ia.disponible() else " Indisponible : clé GEMINI_API_KEY absente."))
enregistrer = c2.toggle("Enregistrer dans la base", value=True,
                        help="Le BDC, son PDF et ses échantillons rejoignent la base de traçabilité (page Clients).")
relire = c3.toggle("Relire les BDC déjà enregistrés", value=False,
                   help="Désactivé : un PDF déjà présent dans la base est sauté (gain de temps et de quota). "
                        "Activé : il est relu et sa commande mise à jour.")
fichiers = st.file_uploader("Bons de commande (PDF ou ZIP)", type=["pdf", "zip"], accept_multiple_files=True)

lot = st.session_state.setdefault("bdc_lot", {})         # (empreinte, IA) -> état de lecture
if not fichiers:
    st.sidebar.info("Les informations du BDC s'afficheront ici.")
    st.stop()

pdfs, ignores = documents(fichiers)
if ignores:
    with st.expander(f"{len(ignores)} fichier(s) ignoré(s) (pas un PDF)"):
        st.write("\n".join(f"- {x}" for x in ignores))
if not pdfs:
    st.warning("Aucun PDF trouvé dans les fichiers déposés.")
    st.stop()

cles = [(hashlib.sha1(data).hexdigest(), avec_ia) for _, data in pdfs]
a_faire = [(nom, data, cle) for (nom, data), cle in zip(pdfs, cles) if cle not in lot]
st.write(f"**{len(pdfs)} BDC** trouvé(s) · {len(pdfs) - len(a_faire)} déjà traité(s) dans cette session "
         f"· {len(a_faire)} à traiter")
if avec_ia and len(a_faire) > QUOTA_GRATUIT:
    st.warning(f"Plus de {QUOTA_GRATUIT} BDC avec la relecture IA : sur l'offre gratuite de Gemini, les BDC "
               "au-delà du quota du jour seront lus sans IA (OCR et règles, avec une alerte), sans interrompre le lot.")

# un lot de plusieurs BDC se lance explicitement ; une fois lancé, il continue aux rechargements
if a_faire and (len(a_faire) == 1 or st.session_state.get("lot_lance") or
                st.button(f"Lancer la lecture de {len(a_faire)} BDC", type="primary")):
    st.session_state["lot_lance"] = True
    barre = st.progress(0.0)
    for i, (nom, data, cle) in enumerate(a_faire):
        barre.progress(i / len(a_faire), text=f"{i + 1}/{len(a_faire)} — {nom}")
        existante = None if relire else base.commande_existante(data)
        if existante:
            lot[cle] = {"nom": nom, "etat": "déjà dans la base", "commande_id": existante}
            continue
        t0 = time.time()
        try:
            r = lire_bdc(nom, data, avec_ia)
            cid = base.enregistrer_bdc(Path(nom).name, data, r, r["pages"]) if enregistrer else None
            lot[cle] = {"nom": nom, "etat": "lu", "resultat": r, "commande_id": cid, "duree": time.time() - t0}
        except Exception as e:                                           # noqa: BLE001
            lot[cle] = {"nom": nom, "etat": "erreur", "erreur": repr(e), "duree": time.time() - t0}
    st.session_state["lot_lance"] = False
    st.rerun()                                   # affichage à jour : récapitulatif, plus de bouton

# ---------------------------------------------------------------- récapitulatif
traites = [lot[c] for c in cles if c in lot]
if not traites:
    st.stop()
recap = []
for t in traites:
    doc = t["resultat"]["doc"] if t.get("resultat") else None
    meta = t["resultat"]["meta"] if t.get("resultat") else {}
    recap.append({"Fichier": t["nom"], "État": {"lu": "✅ lu", "déjà dans la base": "↩️ déjà dans la base",
                                                "erreur": "❌ erreur"}[t["etat"]],
                  "Client": doc["client"]["raison_sociale"] if doc else None,
                  "N° commande": doc["commande"]["numero"] if doc else None,
                  "Échantillons": len(doc["echantillons"]) if doc else None,
                  "Revue humaine": ("oui" if meta.get("revue_humaine") else "non") if doc else None,
                  "Durée (s)": round(t["duree"]) if t.get("duree") else None,
                  "Détail": t.get("erreur") or ("enregistré" if t.get("commande_id") and doc else
                                               "voir la page Clients" if t["etat"] != "lu" else "non enregistré")})
n = {e: sum(t["etat"] == e for t in traites) for e in ("lu", "déjà dans la base", "erreur")}
a, b, c = st.columns(3)
a.metric("Lus", n["lu"])
b.metric("Déjà dans la base", n["déjà dans la base"])
c.metric("Erreurs", n["erreur"])
st.dataframe(pd.DataFrame(recap).astype({"Échantillons": "Int64", "Durée (s)": "Int64"}), hide_index=True,
             width="stretch")
if n["erreur"]:
    if st.button("Réessayer les BDC en erreur"):
        for cle in [c for c in cles if lot.get(c, {}).get("etat") == "erreur"]:
            del lot[cle]
        st.rerun()

# ---------------------------------------------------------------- détail d'un BDC lu
lus = [t for t in traites if t["etat"] == "lu"]
if not lus:
    st.stop()
st.divider()
choix = st.selectbox("Voir un BDC lu", lus, format_func=lambda t: t["nom"]) if len(lus) > 1 else lus[0]
r = choix["resultat"]
fiche(st.sidebar, r["doc"], r["confiance"], r["meta"])
entete(choix["nom"], r["doc"], r["meta"])
if choix.get("commande_id"):
    st.caption("Enregistré dans la base : voir la page **Clients**.")
details(r["doc"], r["trace"], r["pages"], pdf=r.get("pdf"), meta=r["meta"])
bouton_json(r["doc"], choix["nom"], cle=f"json_{choix['nom']}")
