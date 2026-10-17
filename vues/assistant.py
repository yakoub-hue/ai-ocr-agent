"""Page « Assistant » : questions en français sur les clients, les BDC et les résultats GC-MS."""
import streamlit as st

from bdc import assistant, base, ia

st.title("Assistant de traçabilité")
st.caption("Posez une question sur les clients, leurs bons de commande et les résultats GC-MS. "
           "L'assistant répond d'après la base uniquement ; chaque question consomme une requête Gemini.")

if not ia.disponible():
    st.error("Clé Gemini absente : renseignez GEMINI_API_KEY dans .streamlit/secrets.toml.")
    st.stop()
if not base.clients():
    st.info("La base est vide : lisez d'abord des BDC (page Lecture des BDC).")
    st.stop()
st.caption("La vue de la base envoyée à Google contient les données des clients (nom, contact, téléphone, e-mail).")

historique = st.session_state.setdefault("assistant_historique", [])
if st.sidebar.button("Nouvelle conversation"):
    historique.clear()

for h in historique:
    with st.chat_message("user" if h["role"] == "user" else "assistant"):
        st.markdown(h["texte"])

exemples = ["Quels clients ont des échantillons sans résultat GC-MS ?",
            "Résume les commandes de chaque client.",
            "Quels échantillons dépassent la limite de quantification en benzo(a)pyrène ?"]
question = st.chat_input("Votre question…")
if not historique and not question:
    st.markdown("**Exemples :** " + " · ".join(f"« {e} »" for e in exemples))

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base…"):
            try:
                reponse = assistant.repondre(question, historique)
            except Exception as e:                                   # noqa: BLE001
                reponse = f"Impossible d'interroger Gemini : {e}"
        st.markdown(reponse)
    historique += [{"role": "user", "texte": question}, {"role": "assistant", "texte": reponse}]
