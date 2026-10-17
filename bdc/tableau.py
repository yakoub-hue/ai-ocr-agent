"""Tableaux : détection de la grille, puis lecture cellule par cellule.

Les traits horizontaux et verticaux sont isolés par ouverture morphologique
de l'image binaire. Une zone qui contient au moins 3 traits horizontaux et
3 traits verticaux qui se croisent est un tableau candidat ; ses lignes de
grille donnent les bornes des cellules.

Chaque cellule est ensuite décrite par son encre (quantité, couleur, traits
de rature) et lue par les deux moteurs OCR. L'encre qui déborde d'une
cellule voisine (correction manuscrite qui chevauche un trait) est écartée
de la lecture de la cellule mais reste disponible pour l'analyse des ratures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import encre as E
from .ocr import Lecture, lire_zones
from .pretraitement import Page


@dataclass
class Grille:
    page: int
    xs: list[int]          # bornes des colonnes (pixels), triées
    ys: list[int]          # bornes des rangées (pixels), triées

    @property
    def boite(self):
        return self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]

    @property
    def n_colonnes(self) -> int:
        return len(self.xs) - 1

    @property
    def n_rangees(self) -> int:
        return len(self.ys) - 1

    def cellule(self, rangee: int, colonne: int, marge: int = 0):
        """(x0, y0, x1, y1) intérieur de la cellule, traits exclus."""
        return (self.xs[colonne] + marge, self.ys[rangee] + marge,
                self.xs[colonne + 1] - marge, self.ys[rangee + 1] - marge)


def _traits(binaire: np.ndarray):
    h, w = binaire.shape
    horiz = cv2.morphologyEx(binaire, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 30), 1)))
    vert = cv2.morphologyEx(binaire, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 60))))
    # recoller les traits interrompus (fax, photocopie)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    return horiz, vert


def _positions(profil: np.ndarray, seuil: float, ecart_min: int) -> list[int]:
    """Centres des plages où le profil dépasse le seuil, fusionnées si proches."""
    au_dessus = profil >= seuil
    positions, debut = [], None
    for i, v in enumerate(np.append(au_dessus, False)):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            positions.append((debut + i - 1) // 2)
            debut = None
    fusion = []
    for p in positions:
        if fusion and p - fusion[-1] < ecart_min:
            fusion[-1] = (fusion[-1] + p) // 2
        else:
            fusion.append(p)
    return fusion


def detecter_grilles(binaire: np.ndarray, page: int = 1) -> list[Grille]:
    h, w = binaire.shape
    horiz, vert = _traits(binaire)
    croisements = cv2.bitwise_and(horiz, vert)
    masque = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque)
    grilles = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if bw < 0.25 * w or bh < 0.04 * h:
            continue
        zone_h = horiz[y:y + bh, x:x + bw] > 0
        zone_v = vert[y:y + bh, x:x + bw] > 0
        ecart = max(12, min(h, w) // 150)
        ys = _positions(zone_h.sum(1), 0.5 * bw, ecart)
        xs = _positions(zone_v.sum(0), 0.5 * bh, ecart)
        if len(ys) < 3 or len(xs) < 3:
            continue
        if cv2.countNonZero(croisements[y:y + bh, x:x + bw]) == 0:
            continue
        grilles.append(Grille(page, [int(x + v) for v in xs], [int(y + v) for v in ys]))
    grilles.sort(key=lambda g: g.ys[0])
    return grilles


def masque_traits(binaire: np.ndarray):
    """Traits longs de la page (horizontaux, verticaux), à calculer une fois par page."""
    return _traits(binaire)


def masque_grille(traits, grille: Grille, tolerance: int = 9) -> np.ndarray:
    """Pixels des traits de LA grille seulement (près de ses lignes) : un
    trait de rature au milieu d'une rangée n'est pas un trait de tableau."""
    horiz, vert = traits
    m = np.zeros_like(horiz)
    x0, y0, x1, y1 = grille.boite
    for y in grille.ys:
        a, b = max(0, y - tolerance), y + tolerance + 1
        m[a:b, x0 - tolerance:x1 + tolerance + 1] |= horiz[a:b, x0 - tolerance:x1 + tolerance + 1]
    for x in grille.xs:
        a, b = max(0, x - tolerance), x + tolerance + 1
        m[y0 - tolerance:y1 + tolerance + 1, a:b] |= vert[y0 - tolerance:y1 + tolerance + 1, a:b]
    return m


# ======================================================================
# Lecture des cellules
# ======================================================================
SEUIL_CELLULE_VIDE = 0.004      # part d'encre sous laquelle une cellule est vide
MARGE_CELLULE = 6               # px retirés autour des traits de grille (300 dpi)
AIRE_MIN = 20                   # composantes plus petites = bruit (fax, JPEG)
SEUIL_GRIS_TACHE = 175          # composante plus claire = tache, pas de l'encre


@dataclass
class Cellule:
    r: int
    c: int
    boite: tuple                       # intérieur de la cellule (pixels page)
    vide: bool = True
    part_encre: float = 0.0
    couleur: str = "principale"        # encre dominante, relative à l'encre du tableau
    couleurs: dict = field(default_factory=dict)   # aire par classe d'encre
    traits: list = field(default_factory=list)      # Trait (pixels page)
    lignes: list = field(default_factory=list)      # boîtes des lignes de texte (page)
    lectures: list = field(default_factory=list)    # Lecture par ligne
    hauteur_texte: float = 0.0

    @property
    def lecture(self) -> Lecture:
        if not self.lectures:
            return Lecture("", 0.0, [])
        if len(self.lectures) == 1:
            return self.lectures[0]
        texte = " ".join(l.texte for l in self.lectures if l.texte)
        conf = float(np.mean([l.conf for l in self.lectures if l.texte] or [0]))
        cands = [c for l in self.lectures for c in l.candidats]
        return Lecture(texte, conf, cands)

    @property
    def texte(self) -> str:
        return self.lecture.texte


@dataclass
class TableauLu:
    page: Page
    grille: Grille
    cellules: list                     # [rangée][colonne] -> Cellule
    propre: np.ndarray                 # binaire de la page sans traits de grille
    lab: np.ndarray                    # image Lab de la page (a, b centrés)
    hauteur_texte: float = 30.0
    encre_ref: tuple | None = None     # couleur (L, a, b) de l'encre majoritaire du tableau

    def cellule(self, r: int, c: int) -> Cellule:
        return self.cellules[r][c]


def _segmenter_lignes(masque: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Découpe l'encre d'une cellule en lignes de texte (profil horizontal)."""
    prof = (masque > 0).sum(1)
    runs, debut = [], None
    for i, v in enumerate(np.append(prof > 0, False)):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            runs.append([debut, i])
            debut = None
    if not runs:
        return []
    # fusionner les plages séparées de moins de 4 px (accents, points)
    fus = [runs[0]]
    for a, b in runs[1:]:
        if a - fus[-1][1] < 4:
            fus[-1][1] = b
        else:
            fus.append([a, b])
    hmax = max(b - a for a, b in fus)
    # rattacher les petites plages (accents, points, cédilles) à la voisine
    gardees = []
    for a, b in fus:
        if b - a < 0.3 * hmax and gardees and a - gardees[-1][1] < 0.4 * hmax:
            gardees[-1][1] = b
        else:
            gardees.append([a, b])
    res = []
    for k, (a, b) in enumerate(gardees):
        if b - a < 0.3 * hmax and len(gardees) > 1:
            # petite plage isolée : on la colle à la suivante si proche
            if k + 1 < len(gardees) and gardees[k + 1][0] - b < 0.4 * hmax:
                gardees[k + 1][0] = a
                continue
            if b - a < 6:
                continue
        cols = np.where((masque[a:b] > 0).any(0))[0]
        res.append((int(cols[0]), a, int(cols[-1]) + 1, b))
    return res


def _analyser_cellule(t: TableauLu, r: int, c: int) -> tuple[Cellule, np.ndarray | None]:
    g = t.grille
    x0, y0, x1, y1 = g.cellule(r, c, MARGE_CELLULE)
    cel = Cellule(r, c, (x0, y0, x1, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return cel, None
    # fermeture 3x3 : recolle les traits pointillés (fax tramé) avant l'analyse
    brut = t.propre[y0:y1, x0:x1]
    sous = cv2.morphologyEx(brut, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = sous.shape
    n, etiq, stats, _ = cv2.connectedComponentsWithStats(sous, connectivity=8)
    if n <= 1:
        return cel, None
    lab = t.lab[y0:y1, x0:x1]
    gris = t.page.gris[y0:y1, x0:x1]
    comps = []
    for i in range(1, n):
        x, y, bw, bh, aire = stats[i]
        if aire < AIRE_MIN:
            continue
        sel = (etiq == i) & (brut > 0)          # pixels d'encre réels (pas ceux de la fermeture)
        if not sel.any():
            sel = etiq == i
        comps.append({"i": i, "x": x, "y": y, "w": bw, "h": bh, "aire": aire,
                      "couleur": E.classe_couleur(lab[sel], t.encre_ref), "gris": float(np.median(gris[sel])),
                      "bord": x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1})
    # taches claires (auréole de café, ombre) : bien plus claires que l'écriture
    if comps:
        ref = float(np.median([k["gris"] for k in comps for _ in range(max(1, k["aire"] // 50))]))
        comps = [k for k in comps if not (k["gris"] > SEUIL_GRIS_TACHE or
                                          (k["gris"] > 150 and k["gris"] > ref + 45))]
    if not comps:
        return cel, None
    interieurs = [k for k in comps if not k["bord"]]
    if interieurs:
        aires = {}
        for k in interieurs:
            aires[k["couleur"]] = aires.get(k["couleur"], 0) + k["aire"]
        principale = max(aires, key=aires.get)
        hmed = float(np.median([k["h"] for k in interieurs]))
        garder = []
        for k in comps:
            if k["bord"]:
                # encre qui déborde d'une voisine : autre couleur, ou fragment
                # collé au bord haut/bas nettement plus petit que le texte
                autre_couleur = k["couleur"] != principale and aires.get(k["couleur"], 0) < 0.3 * aires[principale]
                fragment = (k["y"] <= 1 or k["y"] + k["h"] >= h - 1) and k["h"] < 0.6 * max(hmed, 1) \
                    and not (k["y"] <= 1 and k["y"] + k["h"] >= h - 1)
                if autre_couleur or fragment:
                    continue
            garder.append(k)
    else:
        garder = comps
    ids = np.array([k["i"] for k in garder])
    masque = np.isin(etiq, ids).astype(np.uint8) * 255
    cel.part_encre = E.part_encre(masque)
    for k in garder:
        cel.couleurs[k["couleur"]] = cel.couleurs.get(k["couleur"], 0) + int(k["aire"])
    cel.couleur = max(cel.couleurs, key=cel.couleurs.get) if cel.couleurs else "principale"
    if cel.part_encre < SEUIL_CELLULE_VIDE:
        return cel, None
    cel.vide = False
    lignes = _segmenter_lignes(masque)
    cel.lignes = [(x0 + a, y0 + b, x0 + cc, y0 + d) for a, b, cc, d in lignes]
    hs = [d - b for a, b, cc, d in lignes]
    cel.hauteur_texte = float(max(hs)) if hs else 0.0
    return cel, masque


def lire_tableau(page: Page, grille: Grille, traits=None, lire: bool = True) -> TableauLu:
    """Analyse et lit toutes les cellules d'un tableau. ``traits`` : résultat
    de masque_traits(page.binaire), recalculé s'il n'est pas fourni."""
    if traits is None:
        traits = masque_traits(page.binaire)
    propre = cv2.bitwise_and(page.binaire, cv2.bitwise_not(masque_grille(traits, grille)))
    lab = E.lab_image(page.couleur)
    t = TableauLu(page, grille, [], propre, lab)
    x0, y0, x1, y1 = grille.boite
    t.encre_ref = E.couleur_noyau(lab[y0:y1, x0:x1][propre[y0:y1, x0:x1] > 0])
    masques = {}
    for r in range(grille.n_rangees):
        rang = []
        for c in range(grille.n_colonnes):
            cel, m = _analyser_cellule(t, r, c)
            rang.append(cel)
            if m is not None:
                masques[(r, c)] = m
        t.cellules.append(rang)
    hauteurs = [cel.hauteur_texte for rang in t.cellules for cel in rang if not cel.vide]
    t.hauteur_texte = float(np.median(hauteurs)) if hauteurs else 30.0
    # traits de rature, avec la hauteur de texte du tableau comme échelle
    for (r, c), m in masques.items():
        cel = t.cellules[r][c]
        x0, y0 = cel.boite[:2]
        x1, y1 = cel.boite[2:]
        cel.traits = [tr.decale(x0, y0) for tr in
                      E.traits_de_rature(m, t.hauteur_texte, page.gris[y0:y1, x0:x1],
                                         page.couleur[y0:y1, x0:x1])]
    if lire:
        _lire_cellules(t, masques)
    return t


def image_masquee(page: Page, boite, masque: np.ndarray, marge: int = 4):
    """Recadre gris et RGB autour de ``boite`` (avec une marge) en blanchissant
    tout ce qui n'est pas dans le masque d'encre (traits de grille,
    débordements, bruit). ``masque`` a la taille de ``boite``."""
    x0, y0, x1, y1 = boite
    H, W = page.gris.shape
    X0, Y0, X1, Y1 = max(0, x0 - marge), max(0, y0 - marge), min(W, x1 + marge), min(H, y1 + marge)
    m = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    m[y0 - Y0:y0 - Y0 + masque.shape[0], x0 - X0:x0 - X0 + masque.shape[1]] = masque
    m = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
    g = np.where(m, page.gris[Y0:Y1, X0:X1], 255).astype(np.uint8)
    rgb = page.couleur[Y0:Y1, X0:X1].copy()
    rgb[~m] = 255
    return g, rgb


def lire_masques(page: Page, items: list[tuple[tuple, np.ndarray]],
                 whitelist: str | None = None) -> list[Lecture]:
    """Lit des zones dont seule l'encre du masque est conservée.
    ``items`` : (boîte page, masque binaire de la même taille que la boîte)."""
    if not items:
        return []
    grises, rgbs, boites = [], [], []
    for boite, m in items:
        g, rgb = image_masquee(page, boite, m)
        grises.append(g)
        rgbs.append(rgb)
    # on assemble les zones verticalement dans une image unique pour réutiliser
    # lire_zones (lecture groupée, deux moteurs, vote)
    largeur = max(g.shape[1] for g in grises)
    hauteur = sum(g.shape[0] + 10 for g in grises)
    mg = np.full((hauteur, largeur), 255, np.uint8)
    mr = np.full((hauteur, largeur, 3), 255, np.uint8)
    y = 0
    for g, rgb in zip(grises, rgbs):
        h, w = g.shape
        mg[y:y + h, :w] = g
        mr[y:y + h, :w] = rgb
        boites.append((0, y, w, y + h))
        y += h + 10
    return lire_zones(mg, mr, boites, whitelist=whitelist)


def _lire_cellules(t: TableauLu, masques: dict) -> None:
    items, cles = [], []
    for (r, c), m in masques.items():
        cel = t.cellules[r][c]
        cx0, cy0 = cel.boite[:2]
        for (x0, y0, x1, y1) in cel.lignes:
            items.append(((x0, y0, x1, y1), m[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0]))
            cles.append((r, c))
    for (r, c), lec in zip(cles, lire_masques(t.page, items)):
        t.cellules[r][c].lectures.append(lec)
