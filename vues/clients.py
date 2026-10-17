"""Page « Clients » : tri et filtres sur tous les bons de commande, puis client ->
bon de commande -> échantillons et flacons GC-MS."""
from datetime import date

import pandas as pd
import streamlit as st

from bdc import base
from vues import pretraitement
from vues.commun import DELAIS, details, fiche, texte

RANG_DELAI = {"express_72h": 0, "urgent_J+5": 1, "standard_10j": 2}
ETATS = ["Tous", "Résultats complets", "Résultats partiels", "En attente de résultats", "Échantillons non reliés"]
TRIS = {"Date de commande (récent d'abord)": (["date", "id"], [False, False]),
        "Date de commande (ancien d'abord)": (["date", "id"], [True, True]),
        "Client (A → Z)": (["client", "date"], [True, False]),
        "Échéance (la plus proche d'abord)": (["echeance", "date"], [True, False]),
        "Urgence (express, urgent, standard)": (["rang_delai", "echeance"], [True, True]),
        "Résultats manquants (le plus d'abord)": (["manquants", "date"], [False, False])}


def etat(c: dict) -> str:
    if c["nb_echantillons"] and c["nb_resultats"] == c["nb_echantillons"]:
        return "Résultats complets"
    if c["nb_resultats"]:
        return "Résultats partiels"
    if c["nb_relies"] < c["nb_echantillons"]:
        return "Échantillons non reliés"
    return "En attente de résultats"


def en_date(s) -> date | None:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


st.title("Clients et traçabilité")
st.caption("Chaque client, ses bons de commande et, pour chaque échantillon, le flacon GC-MS et ses résultats. "
           + ("Base PostgreSQL hébergée." if base.postgres() else "Base SQLite locale (donnees/)."))

toutes = base.commandes()
if not toutes:
    st.info("Aucun bon de commande dans la base. Lisez des BDC dans la page **Lecture des BDC** "
            "(option « Enregistrer dans la base »).")
    st.stop()

# ---------------------------------------------------------------- filtres et tri
recherche_ech = base.textes_echantillons()
dates = [d for d in (en_date(c["date"]) for c in toutes) if d]
with st.sidebar:
    st.header("Filtres")
    recherche = st.text_input("Rechercher", placeholder="client, n° de commande, réf., flacon…")
    noms = sorted({c["client"] or "Client inconnu" for c in toutes})
    clients_choisis = st.multiselect("Clients", noms, placeholder="Tous les clients")
    periode = st.date_input("Période (date de commande)", value=(min(dates), max(dates)) if dates else (),
                            format="DD/MM/YYYY") if dates else ()
    delais = st.multiselect("Délai", list(DELAIS), format_func=DELAIS.get, placeholder="Tous les délais")
    etat_choisi = st.selectbox("Résultats GC-MS", ETATS)
    revue = st.selectbox("Revue humaine", ["Tous", "À revoir", "Sans revue"])
    tri = st.selectbox("Trier par", list(TRIS))


def garder(c: dict) -> bool:
    if recherche:
        cible = " ".join(str(x or "") for x in (c["client"], c["numero"], c["fichier"], c["dossier_labo"],
                                                 recherche_ech.get(c["id"], ""))).lower()
        if recherche.lower().strip() not in cible:
            return False
    if clients_choisis and (c["client"] or "Client inconnu") not in clients_choisis:
        return False
    if isinstance(periode, (tuple, list)) and len(periode) == 2:
        d = en_date(c["date"])
        if d and not periode[0] <= d <= periode[1]:
            return False
    if delais and c["delai"] not in delais:
        return False
    if etat_choisi != "Tous" and etat(c) != etat_choisi:
        return False
    if revue == "À revoir" and not c["revue_humaine"]:
        return False
    if revue == "Sans revue" and c["revue_humaine"]:
        return False
    return True


cmds = [c for c in toutes if garder(c)]
if not cmds:
    st.warning("Aucun bon de commande ne correspond aux filtres.")
    st.stop()
table = pd.DataFrame(cmds)
table["rang_delai"] = table["delai"].map(RANG_DELAI).fillna(9)
table["manquants"] = table["nb_echantillons"] - table["nb_resultats"]
table["client"] = table["client"].fillna("Client inconnu")
colonnes, ordre = TRIS[tri]
table = table.sort_values(colonnes, ascending=ordre, na_position="last")

# ---------------------------------------------------------------- vue d'ensemble filtrée
n_ech, n_res = int(table["nb_echantillons"].sum()), int(table["nb_resultats"].sum())
a, b, c, d = st.columns(4)
a.metric("Clients", table["client"].nunique())
b.metric("Bons de commande", len(table), delta=None if len(table) == len(toutes) else f"sur {len(toutes)}",
         delta_color="off")
c.metric("Échantillons", n_ech)
d.metric("Avec résultat GC-MS", f"{n_res} ({100 * n_res / n_ech:.0f} %)" if n_ech else "0")

vue_clients, vue_bdc = st.tabs(["Par client", "Tous les bons de commande"])
with vue_clients:
    par_client = table.groupby("client", sort=False).agg(
        BDC=("id", "count"), Échantillons=("nb_echantillons", "sum"), Reliés=("nb_relies", "sum"),
        Avec_résultat=("nb_resultats", "sum"), Première=("date", "min"), Dernière=("date", "max")).reset_index()
    st.dataframe(par_client.rename(columns={"client": "Client", "Avec_résultat": "Avec résultat",
                                            "Première": "Première commande", "Dernière": "Dernière commande"}),
                 hide_index=True, width="stretch")
with vue_bdc:
    st.dataframe(pd.DataFrame({
        "Client": table["client"], "N° commande": table["numero"], "Date": table["date"],
        "Délai": table["delai"].map(lambda x: DELAIS.get(x, x)), "Échéance": table["echeance"],
        "Dossier labo": table["dossier_labo"], "Échantillons": table["nb_echantillons"],
        "Avec résultat": table["nb_resultats"], "État": [etat(r) for r in table.to_dict("records")],
        "Revue": table["revue_humaine"].map({1: "à revoir", 0: ""}), "Fichier": table["fichier"]}),
        hide_index=True, width="stretch")
    st.caption("Cliquez sur un en-tête de colonne pour trier le tableau.")

# ---------------------------------------------------------------- bon de commande choisi
st.divider()
choix = st.selectbox("Ouvrir un bon de commande", table.to_dict("records"),
                     format_func=lambda c: f"{c['client']} · {c['numero'] or 'sans n°'} · {c['date'] or '?'} "
                                           f"· {c['fichier']}")
bdc = base.commande(choix["id"])
doc, meta = bdc["doc"], bdc["meta"] or {}
cl = doc["client"]
st.subheader(f"{cl.get('raison_sociale') or 'Client inconnu'} — bon de commande {bdc['numero'] or 'sans n°'} "
             f"du {bdc['date'] or '?'}")
st.caption(" · ".join(x for x in (f"SIRET {cl['siret']}" if cl.get("siret") else None, cl.get("contact"),
                                  cl.get("telephone"), cl.get("email"),
                                  " ".join(y for y in (cl.get("adresse"), cl.get("cp"), cl.get("ville")) if y)) if x))
if bdc["revue_humaine"]:
    st.warning("Revue humaine demandée à la lecture : " + " ; ".join(bdc["raisons"] or []))

with st.form(f"dossier_{bdc['id']}"):
    c1, c2 = st.columns([3, 1])
    dossier = c1.text_input("N° de dossier labo (ex. 26ORV004213)", value=bdc["dossier_labo"] or "",
                            help="Sert à calculer le flacon GC-MS de chaque échantillon (règle « dossier + rang », "
                                 "bdc/liens.py). Les liens corrigés à la main ne sont pas modifiés.")
    if c2.form_submit_button("Enregistrer", width="stretch"):
        base.definir_dossier(bdc["id"], dossier.strip() or None)
        st.rerun()

onglets = st.tabs(["Échantillons et GC-MS", "Informations extraites", "Document et prétraitement",
                   "Alertes et confiances"])

with onglets[0]:
    ech = base.echantillons(bdc["id"])
    if not ech:
        st.info("Aucun échantillon lu dans ce BDC.")
    else:
        composes = list(dict.fromkeys(k for e in ech for k in e["concentrations"]))
        lignes = []
        for e in ech:
            statut = ("✅ résultat" if e["injection"] else "⏳ en attente de résultat") if e["flacon"] else "— non relié"
            lignes.append({"id": e["id"], "Rang": e["rang"], "Réf.": e["ref"], "Prof. (cm)": e["profondeur_cm"],
                           "Analyses": (e["analyses"] or "").replace("|", ", "),
                           "Flacon GC-MS": e["flacon"] or "",
                           "Lien": {"manuel": "manuel", "regle": "règle"}.get(e["source_lien"], ""),
                           "État": statut, "Injection": e["injection"], "Lot": e["lot"],
                           "Contrôle TIC": e["statut_gcms"], **{k: e["concentrations"].get(k) for k in composes}})
        tableau = pd.DataFrame(lignes)
        st.caption("Corrigez un flacon directement dans la colonne « Flacon GC-MS » (vide = revenir à la règle). "
                   "Concentrations en ppm ; « < x » : sous la limite de quantification.")
        edite = st.data_editor(tableau, hide_index=True, width="stretch", key=f"ech_{bdc['id']}",
                               column_config={"id": None},
                               disabled=[c for c in tableau.columns if c != "Flacon GC-MS"])

        def _txt(x) -> str:                  # cellule vidée : None ou NaN selon l'éditeur
            return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()

        changes = [(int(i), _txt(n)) for i, n, v in zip(tableau["id"], edite["Flacon GC-MS"],
                                                        tableau["Flacon GC-MS"]) if _txt(n) != _txt(v)]
        if changes:
            for i, n in changes:
                base.definir_flacon(i, n or None)
            st.rerun()

with onglets[1]:
    gauche, droite = st.columns(2)
    fiche(gauche, doc, {t["champ"]: t["confiance"] for t in bdc["trace"] or []}, meta)
    droite.subheader("Échantillons lus")
    if doc["echantillons"]:
        lus = pd.DataFrame(doc["echantillons"])
        lus["analyses"] = lus["analyses"].map(texte)
        droite.dataframe(lus, hide_index=True, width="stretch")
    droite.caption(f"Lu le {bdc['lu_le']} · fichier {bdc['fichier']}")

with onglets[2]:
    if bdc["pdf"]:
        st.download_button("Télécharger le PDF original", bdc["pdf"], file_name=bdc["fichier"],
                           mime="application/pdf", key=f"pdf_{bdc['id']}")
    pretraitement.afficher(bdc["pdf"], bdc["pages"], meta)

with onglets[3]:
    details(doc, bdc["trace"] or [], [], avec_echantillons=False)

with st.expander("Supprimer ce bon de commande"):
    st.write("Retire le BDC, ses échantillons, son PDF et ses pages de la base (les résultats GC-MS restent).")
    if st.button("Supprimer définitivement", type="primary", key=f"suppr_{bdc['id']}"):
        base.supprimer_commande(bdc["id"])
        st.rerun()
