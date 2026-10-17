"""Analyse de l'encre d'une zone : nettoyage, couleur, ratures, cases cochées.

Les fonctions travaillent sur l'image binaire de la page (encre = 255) et,
pour la couleur, sur l'image RGB redressée.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ------------------------------------------------------------------ NETTOYAGE
def part_encre(binaire: np.ndarray) -> float:
    return float(cv2.countNonZero(binaire)) / max(binaire.size, 1)


# ------------------------------------------------------------------ COULEUR
# Les encres de stylo des scans sont peu saturées (bleu délavé, carbone) : la
# couleur est mesurée dans l'espace Lab, sur le « cœur » des traits (pixels
# les plus foncés, moins mélangés au papier), et comparée à une encre de
# référence (l'encre majoritaire du tableau ou de la page).
SEUIL_ECART_COULEUR = 5.0


def lab_image(rgb: np.ndarray) -> np.ndarray:
    """Image Lab en int16 : L (0-255), a et b centrés sur 0."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.int16)
    lab[..., 1:] -= 128
    return lab


def couleur_noyau(lab_pixels: np.ndarray) -> tuple[float, float, float] | None:
    """(L, a, b) médians des pixels d'encre les plus foncés (35 %)."""
    if lab_pixels is None or len(lab_pixels) == 0:
        return None
    L = lab_pixels[:, 0]
    coeur = lab_pixels[L <= np.percentile(L, 35)]
    if len(coeur) == 0:
        coeur = lab_pixels
    return float(np.median(coeur[:, 0])), float(np.median(coeur[:, 1])), float(np.median(coeur[:, 2]))


def classe_teinte(noyau, ref=None, seuil: float = SEUIL_ECART_COULEUR) -> str:
    """'principale' (même encre que la référence) ou 'bleu', 'rouge', 'vert',
    'autre' selon l'écart de teinte (a, b) à la référence."""
    if noyau is None:
        return "principale"
    ra, rb = (ref[1], ref[2]) if ref is not None else (0.0, 0.0)
    da, db = noyau[1] - ra, noyau[2] - rb
    if np.hypot(da, db) < seuil:
        return "principale"
    angle = np.degrees(np.arctan2(db, da)) % 360
    if angle < 45 or angle >= 330:
        return "rouge"
    if angle < 120:
        return "autre"          # jaune, brun (papier, carbone, tache)
    if angle < 210:
        return "vert"
    return "bleu"


def classe_couleur(lab_pixels: np.ndarray, ref=None) -> str:
    return classe_teinte(couleur_noyau(lab_pixels), ref)


# ------------------------------------------------------------------ RATURES
@dataclass
class Trait:
    """Trait de rature quasi horizontal (coordonnées dans la zone analysée)."""
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def y(self) -> int:
        return (self.y0 + self.y1) // 2

    @property
    def longueur(self) -> int:
        return self.x1 - self.x0

    def decale(self, dx: int, dy: int) -> "Trait":
        return Trait(self.x0 + dx, self.x1 + dx, self.y0 + dy, self.y1 + dy)


def _noyau_ligne(longueur: int, angle_deg: float) -> np.ndarray:
    """Élément structurant : segment de 1 px d'épaisseur incliné de angle_deg."""
    a = np.deg2rad(angle_deg)
    dy = int(round(abs(np.tan(a)) * (longueur - 1)))
    k = np.zeros((dy + 1, longueur), np.uint8)
    for x in range(longueur):
        y = int(round(np.tan(a) * x))
        k[y if angle_deg >= 0 else dy + y, x] = 1
    return k


def candidats_rature(binaire: np.ndarray, hauteur_texte: float,
                     gris: np.ndarray | None = None,
                     rgb: np.ndarray | None = None) -> list[tuple[Trait, dict]]:
    """Segments droits candidats et leurs mesures (voir traits_de_rature)."""
    h, w = binaire.shape
    th = max(12.0, float(hauteur_texte))
    L = max(40, int(1.5 * th))
    if h < 8 or w < L:
        return []
    ouvert = np.zeros_like(binaire)
    for angle in (-4, -2, 0, 2, 4):
        ouvert = cv2.bitwise_or(ouvert, cv2.morphologyEx(binaire, cv2.MORPH_OPEN, _noyau_ligne(L, angle)))
    ouvert = cv2.bitwise_and(cv2.dilate(ouvert, np.ones((3, 3), np.uint8)), binaire)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(ouvert, np.ones((5, 9), np.uint8)))
    reste = cv2.bitwise_and(binaire, cv2.bitwise_not(cv2.dilate(ouvert, np.ones((3, 3), np.uint8))))
    gris_encre = float(np.median(gris[reste > 0])) if gris is not None and cv2.countNonZero(reste) > 30 else None
    lab = lab_image(rgb) if rgb is not None else None
    _, etiq, st_c, _ = cv2.connectedComponentsWithStats(binaire)
    res = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        x, y, bw, bh = int(x + 4), int(y + 2), int(bw - 8), int(bh - 4)   # annule la dilatation (5, 9)
        if bw < L:
            continue
        m = {"longueur": bw, "epaisseur": bh, "L": L}
        yc = y + bh // 2
        # finesse : plages d'encre verticales qui touchent la bande
        fines = 0
        tmax = max(8.0, 0.4 * th)
        for j in range(x, x + bw):
            col = binaire[:, j] > 0
            k, longueurs = 0, []
            while k < h:
                if col[k]:
                    d = k
                    while k < h and col[k]:
                        k += 1
                    if k > y and d < y + bh:
                        longueurs.append(k - d)
                else:
                    k += 1
            fines += bool(longueurs) and max(longueurs) <= tmax
        m["finesse"] = round(fines / bw, 2)
        # couverture du mot traversé (groupes d'encre séparés de plus de 0,6 th)
        bande = binaire[max(0, int(yc - 0.6 * th)):int(yc + 0.6 * th)]
        cols = np.where((bande > 0).any(0))[0]
        groupes = []
        for c in cols:
            if groupes and c - groupes[-1][1] <= 0.6 * th:
                groupes[-1][1] = c
            else:
                groupes.append([c, c])
        touches = [g for g in groupes if g[1] >= x and g[0] <= x + bw]
        etendue = (max(g[1] for g in touches) - min(g[0] for g in touches) + 1) if touches else bw
        m["couverture"] = round(bw / etendue, 2)
        # texte traversé : encre hors trait dans la bande
        zone = reste[max(0, int(yc - 0.6 * th)):int(yc + 0.6 * th), x:x + bw]
        m["traverse"] = round(float((zone > 0).any(0).mean()), 2) if zone.size else 0.0
        # position du trait dans la hauteur des lettres qu'il touche
        ids = set(np.unique(etiq[y:y + bh, x:x + bw])) - {0}
        if ids:
            haut_l = min(st_c[k][1] for k in ids)
            bas_l = max(st_c[k][1] + st_c[k][3] for k in ids)
            m["position"] = round((yc - haut_l) / max(1, bas_l - haut_l), 2)
            m["hauteur_lettres"] = int(bas_l - haut_l)
        else:
            m["position"], m["hauteur_lettres"] = 0.5, bh
        # foncé comme l'écriture
        m["gris"] = None
        if gris is not None:
            pix = gris[y:y + bh, x:x + bw][ouvert[y:y + bh, x:x + bw] > 0]
            if len(pix):
                m["gris"] = float(np.median(pix))
        m["gris_encre"] = gris_encre
        m["ecart_couleur"] = None
        if lab is not None:
            zone_t = ouvert[y:y + bh, x:x + bw] > 0
            n_trait = couleur_noyau(lab[y:y + bh, x:x + bw][zone_t])
            zr = max(0, int(yc - 0.8 * th)), int(yc + 0.8 * th)
            n_texte = couleur_noyau(lab[zr[0]:zr[1], x:x + bw][reste[zr[0]:zr[1], x:x + bw] > 0])
            if n_trait and n_texte:
                m["ecart_couleur"] = round(float(np.hypot(n_trait[1] - n_texte[1], n_trait[2] - n_texte[2])), 1)
        res.append((Trait(x, x + bw, y, y + bh), m))
    return res


def est_rature(m: dict, th: float) -> bool:
    """Règles de décision (seuils mesurés sur le corpus annoté, voir rapport)."""
    if m["epaisseur"] > max(th, 16):
        return False
    if m["couverture"] < 0.45 or m["traverse"] < 0.35:
        return False
    # le trait passe dans le corps des lettres, pas sur la ligne de base
    # (soulignement, bas de lettres alignées) ni au sommet
    if not 0.27 <= m["position"] <= 0.75:
        return False
    # une auréole de café ou un filigrane clair ne sont pas des ratures
    if m["gris"] is not None and (m["gris"] > 150 or (
            m["gris_encre"] is not None and m["gris"] > m["gris_encre"] + 40)):
        return False
    # preuve positive : trait nettement plus foncé que le texte barré (stylo
    # sur imprimé) ou trait fin qui passe entre les lettres (manuscrit)
    contraste = (m["gris_encre"] - m["gris"]) if m["gris"] is not None and m["gris_encre"] is not None else 0
    autre_stylo = (m.get("ecart_couleur") or 0) >= SEUIL_ECART_COULEUR
    return autre_stylo or contraste >= 60 or m["finesse"] >= 0.6


def traits_de_rature(binaire: np.ndarray, hauteur_texte: float,
                     gris: np.ndarray | None = None, rgb: np.ndarray | None = None) -> list[Trait]:
    """Traits de rature d'une zone (cellule, ligne de texte).

    Un trait de rature est un segment droit (horizontal ou incliné de ±4°,
    simple ou double) d'au moins 1,5 hauteur de texte, qui :
      * est d'une autre encre que le texte qu'il barre (stylo bleu sur
        imprimé noir), ou nettement plus foncé (écart de gris ≥ 60), ou fin sur l'essentiel de sa longueur (≥ 60 % des
        colonnes : il passe entre les lettres, pas dans le corps d'un texte
        gras) ;
      * couvre la plus grande partie du mot qu'il traverse (≥ 45 %) ;
      * traverse vraiment du texte (encre hors trait dans sa bande sur
        ≥ 35 % de sa longueur), à mi-hauteur des lettres : un soulignement,
        une ligne de base ou une ligne isolée ne comptent pas ;
      * est aussi foncé que l'écriture (une auréole de café ou un filigrane
        « COPIE » ne sont pas des ratures).
    """
    th = max(12.0, float(hauteur_texte))
    return [t for t, m in candidats_rature(binaire, hauteur_texte, gris, rgb) if est_rature(m, th)]


def effacer_trait(binaire: np.ndarray, trait: Trait) -> np.ndarray:
    """Retire le trait de rature pour pouvoir relire le texte barré : dans la
    bande du trait, seuls les longs segments horizontaux sont effacés ; les
    traits verticaux des lettres qui le croisent sont conservés."""
    b = binaire.copy()
    y0, y1 = max(0, trait.y0 - 2), min(b.shape[0], trait.y1 + 3)
    bande = b[y0:y1, trait.x0:trait.x1]
    if bande.size == 0:
        return b
    long_h = cv2.morphologyEx(bande, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    b[y0:y1, trait.x0:trait.x1] = cv2.bitwise_and(bande, cv2.bitwise_not(long_h))
    return b
