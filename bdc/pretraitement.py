"""Brique 1 : prétraitement des pages.

Le scan d'origine n'est jamais modifié : chaque page est rendue en image,
puis transformée en mémoire. Chaque transformation est notée dans le journal
de la page (type, paramètres), pour la traçabilité.

Étapes :
  1. rendu de la page PDF à 300 dpi ;
  2. perspective : si la feuille est photographiée sur un fond (4 coins
     détectés), elle est remise à plat ;
  3. axe du texte : profils de projection, pour mettre les lignes à
     l'horizontale (0 ou 90°) ;
  4. inclinaison résiduelle : angle qui maximise la netteté du profil de
     projection horizontal (les lignes de texte sont alors horizontales) ;
  5. sens : endroit ou tête-bêche (0 ou 180°), en comparant ce que
     Tesseract lit dans les deux sens ;
  6. normalisation : division par le fond (ombres, taches, papier jaune),
     puis binarisation adaptative (Sauvola) pour la détection des traits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from . import config
from .ocr import lire_mots


@dataclass
class Page:
    fichier: str
    numero: int                      # 1, 2, ...
    couleur: np.ndarray              # RGB normalisé (redressé)
    gris: np.ndarray                 # niveaux de gris, fond aplani (pour l'OCR)
    binaire: np.ndarray              # encre = 255 (pour les traits et les cases)
    journal: list = field(default_factory=list)


# ------------------------------------------------------------------ RENDU
def rendre_pages(chemin_pdf: Path, dpi: int = config.DPI_RENDU) -> list[np.ndarray]:
    """Images RGB des pages d'un PDF (le PDF lui-même n'est pas modifié)."""
    images = []
    with pymupdf.open(chemin_pdf) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            images.append(np.frombuffer(pix.samples, np.uint8)
                          .reshape(pix.h, pix.w, 3).copy())
    return images


# ------------------------------------------------------------------ PERSPECTIVE
def _ordonner_coins(pts: np.ndarray) -> np.ndarray:
    """Coins dans l'ordre haut-gauche, haut-droit, bas-droit, bas-gauche."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def corriger_perspective(img: np.ndarray, journal: list) -> np.ndarray:
    """Remet à plat une feuille photographiée sur un fond plus sombre.

    La feuille est la plus grande zone claire. Si son contour se simplifie en
    4 coins et qu'elle ne remplit pas déjà toute l'image, on la redresse.
    """
    h, w = img.shape[:2]
    echelle = 1000 / max(h, w)
    petit = cv2.resize(img, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(petit, cv2.COLOR_RGB2GRAY)
    _, masque = cv2.threshold(cv2.GaussianBlur(gris, (7, 7), 0), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    part = cv2.contourArea(c) / (petit.shape[0] * petit.shape[1])
    if not config.PART_MIN_FEUILLE <= part <= config.PART_MAX_FEUILLE:
        return img
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        return img
    coins = _ordonner_coins(approx) / echelle
    largeur = int(max(np.linalg.norm(coins[1] - coins[0]), np.linalg.norm(coins[2] - coins[3])))
    hauteur = int(max(np.linalg.norm(coins[3] - coins[0]), np.linalg.norm(coins[2] - coins[1])))
    cible = np.array([[0, 0], [largeur - 1, 0], [largeur - 1, hauteur - 1], [0, hauteur - 1]],
                     np.float32)
    m = cv2.getPerspectiveTransform(coins, cible)
    journal.append({"etape": "perspective", "coins_px": coins.round(1).tolist(),
                    "part_feuille": round(part, 3), "taille_sortie": [largeur, hauteur]})
    return cv2.warpPerspective(img, m, (largeur, hauteur), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ ORIENTATION
_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
              270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def tourner(img: np.ndarray, angle: int) -> np.ndarray:
    return img if angle == 0 else cv2.rotate(img, _ROTATIONS[angle])


def _reduire_gris(img: np.ndarray, dpi: int) -> np.ndarray:
    e = dpi / config.DPI_RENDU
    return cv2.cvtColor(cv2.resize(img, None, fx=e, fy=e, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_RGB2GRAY)


def _nettete_profil(profil: np.ndarray) -> float:
    p = profil.astype(np.float64)
    return float(np.sum(np.diff(p) ** 2) / (np.sum(p) ** 2 + 1e-9) * len(p))


def detecter_axe(img: np.ndarray, journal: list) -> np.ndarray:
    """Met les lignes de texte à l'horizontale (rotation de 0 ou 90°).

    Les traits longs (cadres de tableau) sont retirés, puis on compare la
    netteté des profils de projection horizontal et vertical : des lignes de
    texte horizontales donnent un profil horizontal en dents de scie.
    """
    e = 1000 / max(img.shape[:2])
    g = cv2.cvtColor(cv2.resize(img, None, fx=e, fy=e, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    traits = cv2.bitwise_or(
        cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))),
        cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))))
    b = cv2.subtract(b, traits)
    lignes, colonnes = _nettete_profil(b.sum(1)), _nettete_profil(b.sum(0))
    angle = 0 if lignes >= colonnes else 90
    journal.append({"etape": "axe_du_texte", "rotation_deg": angle,
                    "nettete_horizontale": round(lignes, 3), "nettete_verticale": round(colonnes, 3)})
    return tourner(img, angle)


def _score_lecture(gris: np.ndarray) -> float:
    """Caractères lus avec une confiance ≥ 70 dans des mots d'au moins 3 lettres."""
    mots = lire_mots(gris, psm=3)
    return float(sum(len(m.texte) for m in mots
                     if m.conf >= 70 and sum(ch.isalpha() for ch in m.texte) >= 3))


def detecter_sens(img: np.ndarray, journal: list) -> np.ndarray:
    """Endroit ou tête-bêche (0 ou 180°) : on lit la page dans les deux sens et
    on garde celui où Tesseract lit le plus de mots sûrs.

    La détection d'orientation intégrée à Tesseract (OSD) n'est pas utilisée :
    sur les fax du corpus, elle répond à l'envers avec une confiance élevée.
    """
    petit = _reduire_gris(img, config.DPI_ORIENTATION)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scores = dict(zip((0, 180), pool.map(lambda a: _score_lecture(tourner(petit, a)), (0, 180))))
    rotation = max(scores, key=scores.get)
    marge = scores[rotation] / max(min(scores.values()), 1.0)
    journal.append({"etape": "sens", "rotation_deg": rotation,
                    "caracteres_lus": {str(k): v for k, v in scores.items()},
                    "marge": round(marge, 2)})
    return tourner(img, rotation)


# ------------------------------------------------------------------ INCLINAISON
def _binariser_rapide(gris: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def _nettete(binaire: np.ndarray, angle: float) -> float:
    h, w = binaire.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    r = cv2.warpAffine(binaire, m, (w, h), flags=cv2.INTER_NEAREST)
    profil = r.sum(axis=1).astype(np.float64)
    return float(np.sum(np.diff(profil) ** 2))


def corriger_inclinaison(img: np.ndarray, journal: list) -> np.ndarray:
    """Angle (-6° à +6°) qui rend le profil horizontal le plus contrasté."""
    h, w = img.shape[:2]
    echelle = 1200 / max(h, w)
    gris = cv2.cvtColor(cv2.resize(img, None, fx=echelle, fy=echelle,
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    b = _binariser_rapide(gris)
    grossiers = np.arange(-config.ANGLE_MAX_INCLINAISON, config.ANGLE_MAX_INCLINAISON + 1e-9, 0.5)
    a0 = max(grossiers, key=lambda a: _nettete(b, a))
    fins = np.arange(a0 - 0.5, a0 + 0.5 + 1e-9, config.PAS_INCLINAISON)
    angle = float(max(fins, key=lambda a: _nettete(b, a)))
    if abs(angle) < 0.15:
        journal.append({"etape": "inclinaison", "angle_deg": 0.0})
        return img
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    journal.append({"etape": "inclinaison", "angle_deg": round(angle, 2)})
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


# ------------------------------------------------------------------ NORMALISATION
def aplanir_fond(gris: np.ndarray) -> np.ndarray:
    """Divise l'image par une estimation de son fond : ombres, taches et papier
    coloré deviennent blancs, l'encre garde son contraste."""
    k = max(31, (min(gris.shape) // 40) | 1)
    fond = cv2.morphologyEx(gris, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    fond = cv2.GaussianBlur(fond, (0, 0), k / 3)
    plat = cv2.divide(gris, fond, scale=255)
    return plat


def etirer_contraste(gris: np.ndarray) -> np.ndarray:
    """Étire les niveaux : l'encre la plus sombre devient noire, le fond blanc.

    Indispensable pour les photocopies pâles, dont le texte imprimé reste gris
    clair même après aplanissement du fond."""
    bas, haut = np.percentile(gris, 0.5), np.percentile(gris, 60)
    if haut - bas < 10:
        return gris
    return np.clip((gris.astype(np.float32) - bas) / (haut - bas) * 255, 0, 255).astype(np.uint8)


def binariser(gris: np.ndarray, fenetre: int = 41, k: float = 0.15) -> np.ndarray:
    """Binarisation de Sauvola (encre = 255) : robuste aux variations de fond."""
    g = gris.astype(np.float64)
    moy = cv2.boxFilter(g, -1, (fenetre, fenetre))
    moy2 = cv2.boxFilter(g * g, -1, (fenetre, fenetre))
    ecart = np.sqrt(np.maximum(moy2 - moy * moy, 0))
    seuil = moy * (1 + k * (ecart / 128 - 1))
    return np.where(g < seuil, 255, 0).astype(np.uint8)


# ------------------------------------------------------------------ CHAÎNE
def pretraiter_image(img: np.ndarray, fichier: str, numero: int) -> Page:
    journal = [{"etape": "rendu", "dpi": config.DPI_RENDU, "taille": list(img.shape[:2])}]
    img = corriger_perspective(img, journal)
    img = detecter_axe(img, journal)
    img = corriger_inclinaison(img, journal)
    img = detecter_sens(img, journal)
    gris = etirer_contraste(aplanir_fond(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))
    journal.append({"etape": "normalisation",
                    "methode": "division par le fond, étirement du contraste, Sauvola (k=0,15)"})
    return Page(fichier, numero, img, gris, binariser(gris), journal)


def pretraiter_pdf(chemin_pdf: Path) -> list[Page]:
    return [pretraiter_image(img, Path(chemin_pdf).name, i + 1)
            for i, img in enumerate(rendre_pages(chemin_pdf))]
