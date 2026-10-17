"""Lecture du texte libre d'une page (hors tableaux) : lignes positionnées.

Les lignes sont détectées par RapidOCR, lues par les deux moteurs (voir
ocr.py) et décrites par la couleur de leur encre, qui aide à séparer
l'imprimé, le stylo du client et les ajouts du laboratoire (tampons rouges ou
bleus, notes vertes ou au crayon).
"""
from __future__ import annotations

import numpy as np

from . import encre as E
from .ocr import Ligne, detecter_boites, lire_zones
from .pretraitement import Page


def couleur_encre(lab: np.ndarray, binaire: np.ndarray, ref=None) -> str | None:
    """'noir', 'gris', 'bleu', 'rouge', 'vert', 'autre' ou None (pas d'encre),
    relativement à l'encre majoritaire de la page ``ref``."""
    pix = lab[binaire > 0]
    if len(pix) < 15:
        return None
    noyau = E.couleur_noyau(pix)
    c = E.classe_teinte(noyau, ref, seuil=7.0)
    if c == "principale":
        return "gris" if noyau[0] > 150 else "noir"
    return c


def dans_boite(x: float, y: float, boite, marge: int = 0) -> bool:
    x0, y0, x1, y1 = boite
    return x0 - marge <= x <= x1 + marge and y0 - marge <= y <= y1 + marge


def lire_lignes(page: Page, zones_exclues=(), variantes: bool = True) -> list[Ligne]:
    """Toutes les lignes de texte de la page dont le centre n'est dans aucune
    des zones exclues (les tableaux, lus cellule par cellule ailleurs)."""
    boites = [b for b in detecter_boites(page.couleur)
              if not any(dans_boite((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, z) for z in zones_exclues)]
    lectures = lire_zones(page.gris, page.couleur, boites, variantes=variantes)
    lab = E.lab_image(page.couleur)
    ref = E.couleur_noyau(lab[page.binaire > 0][::7])
    lignes = []
    for b, lec in zip(boites, lectures):
        if not lec.texte:
            continue
        x0, y0, x1, y1 = b
        encre = couleur_encre(lab[y0:y1, x0:x1], page.binaire[y0:y1, x0:x1], ref)
        tess = max((c.conf for c in lec.candidats if c.moteur == "tesseract"), default=0)
        rap = max((c.conf for c in lec.candidats if c.moteur == "rapid"), default=0)
        # encre de couleur seule ne suffit pas : des titres sont imprimés en vert et Tesseract
        # les lit sûrement (≥ 0,9) ; l'écriture à la main, elle, le gêne
        manuscrit = (encre in ("bleu", "vert") and tess < 0.88) or (tess < 0.55 and rap >= 0.6)
        lignes.append(Ligne(page.numero, b, lec.texte, lec.conf, lec.candidats, encre, manuscrit))
    lignes.sort(key=lambda l: (l.cy, l.x0))
    return lignes


def rangees(lignes: list[Ligne], tolerance: float = 0.5) -> list[list[Ligne]]:
    """Regroupe les lignes alignées horizontalement (même rangée visuelle),
    triées de gauche à droite : « Adresse : » et « La Brosse Haute » détectées
    séparément se retrouvent dans la même rangée."""
    groupes: list[list[Ligne]] = []
    for l in sorted(lignes, key=lambda l: l.cy):
        for g in groupes:
            ref = np.median([x.cy for x in g])
            h = np.median([x.h for x in g])
            if abs(l.cy - ref) <= tolerance * max(h, l.h) and l.page == g[0].page:
                g.append(l)
                break
        else:
            groupes.append([l])
    for g in groupes:
        g.sort(key=lambda l: l.x0)
    groupes.sort(key=lambda g: (g[0].page, np.median([x.cy for x in g])))
    return groupes
