"""Page « Résultats GC-MS » : import des concentrations du projet GC-MS dans la base."""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from bdc import base, liens
from bdc.export import lire_resultats_gcms

st.title("Résultats GC-MS")
st.caption("Importez le tableau blpc_concentrations.csv produit par le projet GC-MS (page BLPC). "
           "Chaque injection est rattachée à son flacon, puis à l'échantillon de BDC qui l'attend.")
st.info("Règle de lien provisoire (bdc/liens.py) : flacon = année + BLPC + n° de dossier labo + rang de "
        "l'échantillon, ex. 2e échantillon du dossier 26ORV004213 → 26BLPC004213-002. Un lien se corrige "
        "à la main dans la page Clients.")

with st.form("import_gcms", clear_on_submit=True):
    fichier = st.file_uploader("blpc_concentrations.csv", type="csv")
    lot = st.text_input("Lot / batch (ex. 20251103)")
    if st.form_submit_button("Importer", type="primary") and fichier:
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / fichier.name
            chemin.write_bytes(fichier.getvalue())
            conc, statuts = lire_resultats_gcms(chemin)
        n, ignorees = base.importer_resultats(conc, statuts, lot.strip() or Path(fichier.name).stem)
        st.success(f"{n} injection(s) importée(s)" + (f", {ignorees} ignorée(s) (nom de flacon non reconnu)"
                                                    if ignorees else "") + ".")

res = base.resultats()
if not res:
    st.info("Aucun résultat GC-MS importé pour l'instant.")
    st.stop()

orphelins = base.flacons_orphelins()
a, b, c = st.columns(3)
a.metric("Injections importées", len(res))
b.metric("Flacons", len({r["flacon"] for r in res}))
c.metric("Flacons sans échantillon de BDC", len(orphelins))

st.subheader("Flacons sans échantillon de BDC")
if orphelins:
    st.caption("Aucun échantillon ne les attend : n° de dossier absent ou faux sur le BDC, BDC pas encore lu, "
               "ou règle de lien à revoir. Renseignez le n° de dossier ou le flacon dans la page Clients.")
    st.dataframe(pd.DataFrame(orphelins).rename(columns={"flacon": "Flacon", "injections": "Injections", "lot": "Lot"}),
                 hide_index=True, width="stretch")
else:
    st.success("Tous les flacons importés sont reliés à un échantillon de BDC.")

st.subheader("Injections importées")
st.dataframe(pd.DataFrame(res).rename(columns={"injection": "Injection", "flacon": "Flacon", "lot": "Lot",
                                               "statut": "Contrôle TIC", "importe_le": "Importé le"}),
             hide_index=True, width="stretch")
st.caption(f"Exemple de flacon attendu : {liens.flacon_attendu('26ORV004213', 2)} "
           "(2e échantillon du dossier 26ORV004213).")
