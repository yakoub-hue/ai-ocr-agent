# Lecture automatique des BDC HAP — Orvenne Analyses

Chaîne qui transforme les scans des bons de commande (BDC) en données prêtes pour les analyses HAP :
JSON par BDC, trace, CSV d'import LIMS et rattachement des concentrations du projet GC-MS.

**Données envoyées à l'extérieur.** Sans l'option `--ia`, tout reste local. Avec `--ia` (ou l'interrupteur
de l'application), la chaîne envoie à Google (API Gemini) le texte lu par l'OCR et l'image du seul tableau
des échantillons ; l'image de la page entière n'est jamais envoyée. Le texte contient des données
personnelles (nom, téléphone, e-mail du client) : à valider avec le responsable RGPD du laboratoire.

## Installation
1. Installer Tesseract 5 + modèle français (`fra.traineddata`), puis si besoin :
   `export TESSDATA_PREFIX=/chemin/vers/tessdata` (et `TESSERACT_CMD` si l'exécutable n'est pas dans le PATH).
   Sous Windows, un dossier `tessdata/` à la racine du projet est trouvé automatiquement par l'application.
2. `pip install -r requirements.txt` (versions figées ; graines fixées dans `bdc/pipeline.py: fixer_graines`).
3. Relecture IA (facultatif) : clé Gemini dans `.streamlit/secrets.toml` (`GEMINI_API_KEY = "…"`) ou dans la
   variable d'environnement `GEMINI_API_KEY`. Ce fichier est ignoré par git. Modèle : `GEMINI_MODEL`
   (défaut `gemini-3.5-flash-lite`, repli automatique sur d'autres modèles s'il est saturé).
   L'offre gratuite est limitée à 20 requêtes par jour et par modèle ; la chaîne fait un appel par BDC.

## Commande unique
    python run.py --entree <dossier_pdf | bdc.zip> --sortie sortie --verite verite_terrain [--ia] [--base]
Démo sur un BDC inconnu : `python run.py --demo mon_bdc.pdf --sortie sortie_demo`

Sorties dans `sortie/` : `json/` (un JSON §4 par BDC), `trace/` (page, zone, confiance, source de chaque valeur,
type de document), `pages/` (copies normalisées ; le scan d'origine n'est jamais modifié), `lims.csv` (une ligne
par échantillon, trié par urgence puis échéance), `gcms_hap_a_analyser.csv` (références à quantifier en HAP16),
`envois.csv` (documents d'un même envoi), `evaluation.json|md` (§6, global et par difficulté).

## Lien GC-MS
Les flacons analysés portent l'identifiant du laboratoire (ex. `25BLPC003018-001`), les BDC la référence du client
(ex. `AGS-01`). Le laboratoire remplit la colonne `echantillon_labo` de `gcms_hap_a_analyser.csv`, qui devient la
table de correspondance (le n° de commande y figure : une même référence peut revenir d'un BDC à l'autre) :

    python run.py --entree <dossier_pdf> --sortie sortie \
        --resultats-gcms resultats/20251103/blpc_concentrations.csv --correspondance correspondance.csv

`lims_avec_resultats.csv` reçoit, pour chaque échantillon, l'identifiant du flacon, le contrôle TIC et la
concentration de chaque HAP. Les flacons sans échantillon de BDC sont listés à l'écran.

## Application
    streamlit run app_bdc.py
* **Lecture des BDC** : dépôt de PDF ou d'une archive ZIP de BDC (sous-dossiers compris ; fichiers non PDF
  ignorés). Un lot se lit BDC par BDC : une erreur n'arrête pas les autres, les BDC déjà dans la base sont sautés
  (sauf option « Relire »), une page rechargée reprend où elle s'était arrêtée, un récapitulatif liste chaque BDC ;
  informations du BDC dans la barre latérale, type de
  document, échantillons, alertes, scan normalisé, confiance de chaque valeur, export JSON ; enregistrement
  dans la base de traçabilité (option, activée par défaut).
* **Clients** : client → ses bons de commande → pour chaque BDC, les informations extraites, le PDF original
  et les échantillons avec leur flacon GC-MS, le contrôle TIC et les concentrations. Le n° de dossier labo et
  le flacon d'un échantillon se corrigent dans la page.
* **Résultats GC-MS** : import de `blpc_concentrations.csv` (projet GC-MS) ; flacons sans échantillon de BDC.
* **Assistant** : questions en français sur la base (« quels échantillons attendent un résultat ? ») ; une
  requête Gemini par question, avec une vue texte de la base (données clients comprises).

## Traçabilité client → BDC → GC-MS
Tableaux SQL `clients`, `commandes`, `echantillons`, `resultats_gcms` et `fichiers` (PDF original et pages de
chaque BDC, stockés dans la base). Un client est reconnu par son SIRET, sinon par sa raison sociale.
`python run.py ... --base` enregistre aussi un lot de BDC (et, avec `--resultats-gcms`, les résultats).

Où vit la base :
* **sur un PC** : fichier SQLite `donnees/tracabilite.db` (ignoré par git ; `BDC_DONNEES` pour le déplacer),
  conservé d'un redémarrage à l'autre ;
* **application déployée** (Streamlit Community Cloud efface son disque à chaque redémarrage) : base
  PostgreSQL hébergée (Neon, Supabase…), dont l'adresse se met dans les *Secrets* de l'application :
  `DATABASE_URL = "postgresql://utilisateur:motdepasse@hote/base?sslmode=require"`. Les tableaux sont créés au
  premier lancement. Compter ~0,5 Mo par BDC (PDF original + pages en JPEG).

Lien flacon ↔ échantillon : règle **provisoire** « dossier + rang », isolée dans `bdc/liens.py`
(`flacon_attendu`) : 2e échantillon du dossier 26ORV004213 → flacon `26BLPC004213-002`. Le n° de dossier est
lu sur les étiquettes du labo (26ORV…) ou saisi dans la page Clients ; un lien corrigé à la main n'est jamais
écrasé. Un flacon réinjecté dilué (« -006 d20 ») garde l'injection au contrôle TIC « OK ».

Déploiement Streamlit Community Cloud : `packages.txt` installe Tesseract et le modèle français ; la clé Gemini
(`GEMINI_API_KEY`) et l'adresse de la base (`DATABASE_URL`) se mettent dans les *Secrets* de l'application.

## Architecture
| Brique | Module |
|---|---|
| 1 Prétraitement (perspective, rotation, redressement, aplanissement, binarisation) | `pretraitement.py` |
| 2 Identification (type de page et de support, envois, tableaux, texte libre, étiquettes) | `documents.py`, `tableau.py`, `lecture.py`, `champs.py` |
| 3 Extraction (OCR Tesseract + RapidOCR, vote par consensus) | `ocr.py`, `echantillons.py`, `champs.py` |
| 3 bis Relecture IA (Gemini : champs sur le texte OCR, échantillons sur l'image du tableau, garde-fou) | `ia.py` |
| 4 Ratures, manuscrit, tampons (couleur d'encre Lab, traits de rature) | `encre.py`, `echantillons.py` |
| 5 Contrôles §5 + confiance + décision de revue | `controles.py` |
| 6 LIMS / GC-MS, traçabilité client → BDC → flacons, assistant | `export.py`, `liens.py`, `base.py`, `assistant.py` |
| Évaluation §6 | `evaluation.py` |

Règles : valeur illisible = `null` + alerte (jamais inventée) ; une valeur proposée par l'IA doit se retrouver dans
le texte lu (à quelques fautes de lecture près), sinon elle est écartée avec une alerte ; un SIRET qui échoue au Luhn
donne une alerte, jamais une correction ; seuils (72 h, 8 °C, confiance 0,60) dans `bdc/config.py`. Aucune règle
propre à un des 7 modèles : étiquettes usuelles, motifs de valeurs et analyse de la grille.

## Limites connues
Sans IA, OCR faible sur écriture manuscrite très dégradée et fax ; texte libre (remarques) approximatif. Avec IA,
une valeur que l'OCR n'a pas lue du tout (ex. « T° = $ °C ») reste écartée par le garde-fou. La détection du total
corrigé et du nombre déclaré dépend de la lisibilité du libellé. Ces cas passent en revue humaine.
