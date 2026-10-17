"""Éléments d'affichage partagés par les pages de l'application."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bdc import config

SUPPORTS = {"original": "original", "fax": "fax", "liasse": "liasse carbonée", "copie_mail": "copie d'e-mail"}
TYPES_PAGE = {"bon_de_commande": "bon de commande", "bordereau": "bordereau d'envoi", "courrier": "courrier",
              "garde_fax": "page de garde de fax", "autre": "autre"}
DELAIS = {"standard_10j": "Standard (10 j ouvrés)", "urgent_J+5": "Urgent (J+5)",
          "express_72h": "Express (72 h)"}
NIVEAUX = {"bloquant": "🔴", "avertissement": "🟠", "info": "🔵"}


def texte(v) -> str:
    if v is None or v == "" or v == []:
        return "—"
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


def champ(cible, label: str, valeur, conf: float | None = None):
    """Une ligne « libellé : valeur » ; ⚠ si la confiance est sous le seuil de revue."""
    alerte = valeur not in (None, "", []) and conf is not None and conf < config.SEUIL_CONFIANCE_REVUE
    cible.markdown(f"**{label}** : {texte(valeur)}" + (f" ⚠ *({conf:.2f})*" if alerte else ""))


def fiche(cible, doc: dict, confiance: dict, meta: dict):
    """Informations du BDC (client, commande, site, prélèvement, réception) dans cible."""
    conf = confiance or {}
    cl, cmd, site, prel, rec = (doc["client"], doc["commande"], doc["site"],
                                doc["prelevement"], doc["reception_labo"])
    cible.subheader("Client")
    champ(cible, "Nom", cl["raison_sociale"], conf.get("client.raison_sociale"))
    champ(cible, "Adresse", cl["adresse"], conf.get("client.adresse"))
    champ(cible, "CP / ville", " ".join(x for x in (cl["cp"], cl["ville"]) if x) or None, conf.get("client.ville"))
    champ(cible, "Téléphone", cl["telephone"], conf.get("client.telephone"))
    champ(cible, "E-mail", cl["email"], conf.get("client.email"))
    champ(cible, "Contact", cl["contact"], conf.get("client.contact"))
    champ(cible, "SIRET", cl["siret"], conf.get("client.siret"))

    cible.subheader("Commande")
    champ(cible, "N°", cmd["numero"], conf.get("commande.numero"))
    champ(cible, "Date", cmd["date"], conf.get("commande.date"))
    champ(cible, "Délai", DELAIS.get(cmd["delai"], cmd["delai"]), conf.get("commande.delai"))
    champ(cible, "Échéance", meta.get("echeance"))
    if cmd["annule_et_remplace"]:
        champ(cible, "Annule et remplace", cmd["annule_et_remplace"], conf.get("commande.annule_et_remplace"))
    champ(cible, "Facturation", cmd["facturation"], conf.get("commande.facturation"))
    champ(cible, "Remarques", cmd["remarques"], conf.get("commande.remarques"))

    cible.subheader("Site et prélèvement")
    champ(cible, "Commune", site["commune"], conf.get("site.commune"))
    champ(cible, "Lieu-dit", site["lieu_dit"], conf.get("site.lieu_dit"))
    champ(cible, "Parcelles", site["parcelles"])
    champ(cible, "Date de prélèvement", prel["date"], conf.get("prelevement.date"))
    champ(cible, "Préleveur", prel["preleveur"], conf.get("prelevement.preleveur"))

    cible.subheader("Réception labo")
    champ(cible, "Date", rec["date"], conf.get("reception_labo.date"))
    champ(cible, "Température (°C)", rec["temperature_c"], conf.get("reception_labo.temperature_c"))
    champ(cible, "Annotations", rec["annotations"])
    cible.caption(f"⚠ : confiance de lecture sous {config.SEUIL_CONFIANCE_REVUE:.2f}, à vérifier.")


def entete(nom: str, doc: dict, meta: dict):
    """Titre, type de document, compteurs et décision de revue."""
    st.subheader(nom)
    desc = meta.get("document")
    if desc:
        st.caption(f"Support : {SUPPORTS.get(desc['support'], desc['support'])} · "
                   + " · ".join(f"page {p['numero']} : {TYPES_PAGE.get(p['type'], p['type'])}" for p in desc["pages"]))
    a, b, c, d = st.columns(4)
    a.metric("Échantillons", len(doc["echantillons"]))
    b.metric("HAP à quantifier", sum(bool(e.get("hap_demande")) for e in doc["echantillons"]))
    c.metric("Alertes", len(doc["alertes"]))
    if meta.get("duree_s") is not None:
        d.metric("Durée de lecture", f"{meta['duree_s']:.0f} s")
    if meta.get("revue_humaine"):
        st.warning("Revue humaine nécessaire : " + " ; ".join(meta.get("raisons_revue") or []))
    else:
        st.success("Lecture complète, pas de revue humaine requise.")


def details(doc: dict, trace: list, pages: list, avec_echantillons: bool = True, pdf: bytes | None = None,
            meta: dict | None = None):
    """Onglets : échantillons lus, alertes, prétraitement (si pdf est donné), scan normalisé,
    confiance de chaque valeur."""
    alertes = f"Alertes ({len(doc['alertes'])})"
    noms = (["Échantillons lus"] if avec_echantillons else []) + [alertes]         + (["Prétraitement"] if pdf and pages else []) + (["Scan"] if pages else []) + ["Confiances"]
    onglets = dict(zip(noms, st.tabs(noms)))
    if "Prétraitement" in onglets:
        from vues import pretraitement
        with onglets["Prétraitement"]:
            pretraitement.afficher(pdf, pages, meta or {})
    if avec_echantillons:
        with onglets["Échantillons lus"]:
            if doc["echantillons"]:
                ech = pd.DataFrame(doc["echantillons"])
                ech["analyses"] = ech["analyses"].map(texte)
                st.dataframe(ech, hide_index=True, width="stretch")
            else:
                st.info("Aucun échantillon lu.")
    with onglets[alertes]:
        for al in doc["alertes"]:
            st.markdown(f"{NIVEAUX.get(al['niveau'], '⚪')} **{al['niveau']}** — {al['message']}")
        if not doc["alertes"]:
            st.info("Aucune alerte.")
        if doc["echantillons_annules"]:
            st.markdown("**Échantillons annulés**")
            st.dataframe(pd.DataFrame(doc["echantillons_annules"]), hide_index=True, width="stretch")
        if doc["ratures"]:
            st.markdown("**Ratures**")
            st.dataframe(pd.DataFrame(doc["ratures"]), hide_index=True, width="stretch")
    if pages:
        with onglets["Scan"]:
            for i, png in enumerate(pages, 1):
                st.image(png if isinstance(png, bytes) else str(png), caption=f"Page {i} (normalisée)",
                         width="stretch")
    with onglets["Confiances"]:
        if trace:
            t = pd.DataFrame(trace)[["champ", "valeur", "confiance", "source", "texte_lu", "note"]]
            t["valeur"] = t["valeur"].map(texte)
            st.dataframe(t, hide_index=True, width="stretch")


def bouton_json(doc: dict, nom: str, cle: str | None = None):
    st.download_button("Télécharger le JSON", json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=Path(nom).with_suffix(".json").name, mime="application/json", key=cle)
