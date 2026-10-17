"""Paramètres de la chaîne de lecture des bons de commande (BDC).

Tous les seuils « à justifier » du cahier des charges sont regroupés ici,
avec leur justification, pour être discutés dans le rapport.
"""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ OCR
# Tesseract 5 + modèle français. TESSDATA_PREFIX peut pointer vers un dossier
# contenant fra.traineddata (voir README).
LANGUE_OCR = os.environ.get("BDC_LANGUE_OCR", "fra")
DPI_RENDU = 300                 # résolution de rendu des pages PDF
DPI_ORIENTATION = 100           # résolution réduite pour tester les 4 orientations

# ------------------------------------------------------------------ PRÉTRAITEMENT
ANGLE_MAX_INCLINAISON = 6.0     # degrés : inclinaison résiduelle recherchée
PAS_INCLINAISON = 0.1           # degrés
PART_MIN_FEUILLE = 0.25         # une feuille détectée doit couvrir ≥ 25 % de l'image
PART_MAX_FEUILLE = 0.92         # au-delà, l'image est déjà la feuille (scan à plat)

# ------------------------------------------------------------------ MARQUES
# Une case est « cochée » si la part de pixels d'encre au centre de la case
# dépasse ce seuil (mesuré sur le corpus : cases vides < 2 %, croix > 6 %).
SEUIL_CASE_COCHEE = 0.04

# ------------------------------------------------------------------ CONTRÔLES MÉTIER (§5)
# Conservation : écart prélèvement -> réception. Les sols pour HAP sont
# conservés à 4 °C et analysés rapidement ; le cahier donne 72 h comme ordre de
# grandeur. Un dépassement est un avertissement, pas un rejet.
SEUIL_CONSERVATION_H = 72
# Température à réception : 4 °C ± 4 °C (glacière), d'où 8 °C.
SEUIL_TEMPERATURE_C = 8.0
# Score de confiance sous lequel un champ indispensable déclenche la revue humaine.
SEUIL_CONFIANCE_REVUE = 0.60

# ------------------------------------------------------------------ VOCABULAIRES FERMÉS
DELAIS = {"standard_10j": 10, "urgent_J+5": 5, "express_72h": 3}   # jours ouvrés
ANALYSES = ["HAP16", "MS", "COT", "ETM", "pH"]
