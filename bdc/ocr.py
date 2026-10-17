"""Lecture du texte : deux moteurs OCR locaux et un vote entre leurs lectures.

* Tesseract 5 (modèle « fra ») lit très bien l'imprimé, avec accents et
  espaces, mais mal le manuscrit.
* RapidOCR (modèles PP-OCRv3 en ONNX, embarqués dans le paquet pip, aucun
  téléchargement) détecte les lignes de texte et lit mieux le manuscrit et
  les fax, mais ignore souvent espaces et accents.

Chaque zone est lue par les deux moteurs, sur l'image brute et sur une
version lissée (utile pour les fax tramés). Le texte retenu est la lecture
la plus soutenue par les autres (médoïde pondéré par la confiance), mise en
forme avec les espaces et accents de Tesseract quand il est d'accord. La
confiance finale baisse quand les lectures divergent : c'est elle qui envoie
un champ en revue humaine. Toutes les lectures restent disponibles
(``candidats``) pour que l'extraction choisisse celle qui respecte un format
attendu (date, téléphone, profondeur…).

Tout tourne en local : aucune image ne quitte la machine (RGPD, §8).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

from . import config
from .textes import compact, sans_accents


# ======================================================================
# Tesseract
# ======================================================================
@dataclass
class Mot:
    texte: str
    x: int
    y: int
    w: int
    h: int
    conf: float                  # 0-100
    ligne: tuple = field(default=(0, 0, 0))   # (bloc, paragraphe, ligne) Tesseract

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def verifier_tesseract() -> None:
    """Arrête la chaîne avec un message clair si Tesseract ou le français manque."""
    try:
        langues = pytesseract.get_languages(config="")
    except Exception as exc:                     # binaire absent
        raise SystemExit(
            "Tesseract est introuvable. Installez-le (voir README) ou indiquez son "
            "chemin dans la variable d'environnement TESSERACT_CMD. "
            f"Détail : {exc}")
    if config.LANGUE_OCR not in langues:
        raise SystemExit(
            f"Le modèle de langue « {config.LANGUE_OCR} » manque pour Tesseract "
            f"(langues trouvées : {', '.join(langues) or 'aucune'}). Installez "
            "fra.traineddata ou définissez TESSDATA_PREFIX "
            f"(actuel : {os.environ.get('TESSDATA_PREFIX')}).")


if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


def lire_mots(image: np.ndarray, psm: int = 3, decalage=(0, 0),
              whitelist: str | None = None) -> list[Mot]:
    """OCR Tesseract d'une image -> liste de mots non vides avec boîtes."""
    cfg = f"--psm {psm} -c preserve_interword_spaces=1"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(image, lang=config.LANGUE_OCR, config=cfg,
                                  output_type=pytesseract.Output.DICT)
    dx, dy = decalage
    mots = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        conf = float(d["conf"][i])
        if not t or conf < 0:
            continue
        mots.append(Mot(t, d["left"][i] + dx, d["top"][i] + dy, d["width"][i],
                        d["height"][i], conf,
                        (d["block_num"][i], d["par_num"][i], d["line_num"][i])))
    return mots


def lire_texte(image: np.ndarray, psm: int = 7, whitelist: str | None = None) -> tuple[str, float]:
    """OCR Tesseract d'une petite zone -> (texte, confiance 0-1 pondérée par la longueur)."""
    mots = lire_mots(image, psm=psm, whitelist=whitelist)
    if not mots:
        return "", 0.0
    mots.sort(key=lambda m: (m.ligne, m.x))
    lignes, courante, cle = [], [], None
    for m in mots:
        if cle is not None and m.ligne != cle:
            lignes.append(" ".join(courante))
            courante = []
        courante.append(m.texte)
        cle = m.ligne
    lignes.append(" ".join(courante))
    conf = float(np.average([m.conf for m in mots], weights=[len(m.texte) for m in mots])) / 100
    return " ".join(lignes), conf


# ======================================================================
# RapidOCR (PP-OCRv3, ONNX, local)
# ======================================================================
_RAPID = None
_VERROU = threading.Lock()


def moteur_rapid():
    global _RAPID
    with _VERROU:
        if _RAPID is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise SystemExit("Le paquet rapidocr_onnxruntime manque : "
                                 "pip install -r requirements.txt") from exc
            _RAPID = RapidOCR()
    return _RAPID


def detecter_boites(rgb: np.ndarray, echelle: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Boîtes des lignes de texte (x0, y0, x1, y1) détectées par le réseau DB de
    RapidOCR, calculées sur l'image réduite (plus rapide, assez pour du 300 dpi)."""
    eng = moteur_rapid()
    petit = cv2.resize(rgb, None, fx=echelle, fy=echelle, interpolation=cv2.INTER_AREA)
    boites, _ = eng.text_detector(cv2.cvtColor(petit, cv2.COLOR_RGB2BGR))
    res = []
    if boites is None:
        return res
    h, w = rgb.shape[:2]
    for b in boites:
        b = np.asarray(b) / echelle
        x0, y0 = b.min(0)
        x1, y1 = b.max(0)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if x1 - x0 >= 6 and y1 - y0 >= 6:
            res.append((x0, y0, x1, y1))
    return res


def reconnaitre_rapid(crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
    """Reconnaissance seule (pas de détection) d'une liste de lignes."""
    if not crops_bgr:
        return []
    res, _ = moteur_rapid().text_recognizer(crops_bgr)
    return [(t or "", float(s or 0.0)) for t, s in res]


# ======================================================================
# Vote entre lectures
# ======================================================================
@dataclass
class Candidat:
    texte: str
    conf: float          # 0-1
    moteur: str          # "tesseract" ou "rapid"
    variante: str = ""   # "brut", "lisse", ...


@dataclass
class Lecture:
    texte: str
    conf: float
    candidats: list[Candidat] = field(default_factory=list)

    def textes(self) -> list[str]:
        """Le texte retenu puis les autres lectures distinctes."""
        vus, res = set(), []
        for t in [self.texte] + [c.texte for c in sorted(self.candidats, key=lambda c: -c.conf)]:
            if t and t not in vus:
                vus.add(t)
                res.append(t)
        return res


def _sim(a: str, b: str) -> float:
    a, b = compact(a), compact(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _fusionner(base: str, autre: str, conf_base: float, conf_autre: float) -> str:
    """Garde la mise en forme de ``base`` (espaces, accents) et prend les
    passages divergents dans ``autre`` quand ce dernier est plus sûr."""
    pa = [(i, c) for i, c in enumerate(base) if not c.isspace()]
    pb = [(i, c) for i, c in enumerate(autre) if not c.isspace()]
    na = "".join(sans_accents(c).lower() for _, c in pa)
    nb = "".join(sans_accents(c).lower() for _, c in pb)
    sm = SequenceMatcher(None, na, nb, autojunk=False)
    sortie: list[tuple[str, bool]] = []          # (caractère, espace avant)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op != "equal" and conf_base >= conf_autre):
            for k in range(i1, i2):
                i, c = pa[k]
                sortie.append((c, i > 0 and base[i - 1].isspace()))
        else:
            for k in range(j1, j2):
                j, c = pb[k]
                sortie.append((c, j > 0 and autre[j - 1].isspace()))
    return "".join((" " if esp and n else "") + c for n, (c, esp) in enumerate(sortie))


def consensus(candidats: list[Candidat]) -> Lecture:
    """Choisit la lecture la plus soutenue par les autres et en estime la confiance."""
    cands = [c for c in candidats if c.texte and compact(c.texte)]
    if not cands:
        return Lecture("", 0.0, candidats)
    total = sum(c.conf for c in cands) or 1e-6
    supports = []
    for c in cands:
        s = c.conf + sum(o.conf * _sim(c.texte, o.texte) for o in cands if o is not c)
        supports.append(s)
    best = cands[int(np.argmax(supports))]
    accord = sum(o.conf * _sim(best.texte, o.texte) for o in cands) / total
    texte = best.texte
    if best.moteur != "tesseract":
        tess = [c for c in cands if c.moteur == "tesseract"]
        if tess:
            t = max(tess, key=lambda c: (_sim(c.texte, best.texte), c.conf))
            if _sim(t.texte, best.texte) >= 0.7:
                texte = _fusionner(t.texte, best.texte, t.conf, best.conf)
    conf = float(np.clip(best.conf * (0.5 + 0.5 * accord) + 0.1 * (accord - 0.5), 0, 1))
    return Lecture(texte.strip(), round(conf, 3), candidats)


# ======================================================================
# Lecture d'une zone et d'une page
# ======================================================================
def _avec_marge(img: np.ndarray, m: int = 8) -> np.ndarray:
    blanc = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=blanc)


def _variantes(gris: np.ndarray, rgb: np.ndarray):
    """(nom, gris, bgr) : image brute et version lissée (trame de fax, JPEG)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    yield "brut", _avec_marge(gris), _avec_marge(bgr)
    yield "lisse", _avec_marge(cv2.GaussianBlur(gris, (5, 5), 0)), \
        _avec_marge(cv2.GaussianBlur(bgr, (5, 5), 0))


def _lire_tesseract(gris: np.ndarray, psm: int, whitelist: str | None) -> tuple[str, float]:
    try:
        return lire_texte(gris, psm=psm, whitelist=whitelist)
    except pytesseract.TesseractError:
        return "", 0.0


_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("BDC_THREADS_OCR", "4")))


def lire_zones(gris: np.ndarray, rgb: np.ndarray, boites: list[tuple[int, int, int, int]],
               psm: int = 7, whitelist: str | None = None,
               variantes: bool = True) -> list[Lecture]:
    """Lit plusieurs zones (une ligne de texte chacune) avec les deux moteurs."""
    if not boites:
        return []
    h, w = gris.shape[:2]
    taches = []          # (indice zone, variante, gris, bgr)
    for k, (x0, y0, x1, y1) in enumerate(boites):
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        for n, (nom, g, b) in enumerate(_variantes(gris[y0:y1, x0:x1], rgb[y0:y1, x0:x1])):
            if n and not variantes:
                break
            taches.append((k, nom, g, b))
    rapid = reconnaitre_rapid([t[3] for t in taches])
    tess = list(_POOL.map(lambda t: _lire_tesseract(t[2], psm, whitelist), taches))
    cands: list[list[Candidat]] = [[] for _ in boites]
    for (k, nom, _, _), (rt, rc), (tt, tc) in zip(taches, rapid, tess):
        cands[k].append(Candidat(rt, rc, "rapid", nom))
        cands[k].append(Candidat(tt, tc, "tesseract", nom))
    return [consensus(c) for c in cands]


@dataclass
class Ligne:
    """Ligne de texte lue sur une page, avec sa position (pixels de la page normalisée)."""
    page: int
    boite: tuple[int, int, int, int]
    texte: str
    conf: float
    candidats: list[Candidat] = field(default_factory=list)
    encre: str | None = None         # couleur dominante : noir, bleu, rouge, vert
    manuscrit: bool = False

    @property
    def x0(self):
        return self.boite[0]

    @property
    def y0(self):
        return self.boite[1]

    @property
    def x1(self):
        return self.boite[2]

    @property
    def y1(self):
        return self.boite[3]

    @property
    def cy(self):
        return (self.boite[1] + self.boite[3]) / 2

    @property
    def h(self):
        return self.boite[3] - self.boite[1]

    def textes(self) -> list[str]:
        return Lecture(self.texte, self.conf, self.candidats).textes()
