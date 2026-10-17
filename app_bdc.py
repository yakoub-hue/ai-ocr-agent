"""Application BDC (Streamlit) : lecture des bons de commande et traçabilité.

    streamlit run app_bdc.py

Pages (dossier vues/) :
* Lecture des BDC : dépôt de PDF, lecture par la chaîne (bdc.pipeline), enregistrement
  dans la base de traçabilité ;
* Clients : client -> bons de commande -> échantillons et flacons GC-MS ;
* Résultats GC-MS : import de blpc_concentrations.csv (projet GC-MS) ;
* Assistant : questions en français sur la base (Gemini).

Base : PostgreSQL si DATABASE_URL est dans les secrets (déploiement), sinon SQLite
locale dans donnees/ (ignoré par git). Relecture IA : avec une clé Gemini
(GEMINI_API_KEY dans .streamlit/secrets.toml ou l'environnement), sont envoyés à
Google le texte lu par l'OCR et l'image du seul tableau des échantillons ;
l'assistant envoie une vue texte de la base.
"""
import os
from pathlib import Path

import pytesseract
import streamlit as st

# Tesseract : le dossier tessdata/ du projet (qui contient fra.traineddata) et
# l'installation Windows par défaut, sauf si les variables sont déjà définies.
RACINE = Path(__file__).resolve().parent
if (RACINE / "tessdata" / "fra.traineddata").exists():
    os.environ.setdefault("TESSDATA_PREFIX", str(RACINE / "tessdata"))
EXE_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if EXE_WINDOWS.exists():
    os.environ.setdefault("TESSERACT_CMD", str(EXE_WINDOWS))
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

# Clé Gemini et base hébergée : secrets Streamlit (local ou déploiement) puis environnement
try:
    for nom_secret in ("GEMINI_API_KEY", "GEMINI_MODEL", "DATABASE_URL"):
        if st.secrets.get(nom_secret):
            os.environ[nom_secret] = st.secrets[nom_secret]
except FileNotFoundError:                     # pas de secrets.toml
    pass

st.set_page_config(page_title="BDC HAP — Orvenne Analyses", layout="wide")
st.navigation([
    st.Page("vues/lecture.py", title="Lecture des BDC", icon=":material/document_scanner:", default=True),
    st.Page("vues/clients.py", title="Clients", icon=":material/groups:"),
    st.Page("vues/gcms.py", title="Résultats GC-MS", icon=":material/science:"),
    st.Page("vues/assistant.py", title="Assistant", icon=":material/chat:"),
]).run()
