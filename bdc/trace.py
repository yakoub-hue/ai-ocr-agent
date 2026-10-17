"""Valeurs tracées : chaque valeur extraite garde sa source.

Le cahier des charges (§4, §8) demande que chaque valeur garde sa page, sa
zone et sa confiance dans un fichier de trace séparé du JSON. Une ``Valeur``
porte ces informations ; ``aplatir`` transforme un document en lignes de
trace (chemin JSON -> valeur, page, boîte, confiance, source, texte brut).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Valeur:
    valeur: Any = None
    conf: float = 0.0                       # 0-1
    page: int | None = None
    boite: tuple | None = None              # (x0, y0, x1, y1) en pixels de la page normalisée
    source: str = ""                        # ocr, case, regle, defaut, calcul, humain
    brut: str | None = None                 # texte lu avant normalisation
    note: str | None = None

    def __bool__(self) -> bool:
        return self.valeur is not None

    def vers_trace(self, chemin: str) -> dict:
        return {"champ": chemin, "valeur": self.valeur, "page": self.page,
                "zone": list(self.boite) if self.boite else None,
                "confiance": round(float(self.conf), 3), "source": self.source,
                "texte_lu": self.brut, "note": self.note}


def V(valeur=None, conf=0.0, page=None, boite=None, source="", brut=None, note=None) -> Valeur:
    return Valeur(valeur, float(conf), page, tuple(int(v) for v in boite) if boite else None,
                  source, brut, note)


def union_boites(boites) -> tuple | None:
    bs = [b for b in boites if b]
    if not bs:
        return None
    return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))


def valeurs(obj):
    """Remplace récursivement les Valeur par leur valeur (sortie JSON §4)."""
    if isinstance(obj, Valeur):
        return valeurs(obj.valeur)
    if isinstance(obj, dict):
        return {k: valeurs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [valeurs(v) for v in obj]
    return obj


def aplatir(obj, chemin: str = "") -> list[dict]:
    """Lignes de trace pour toutes les Valeur contenues dans obj."""
    lignes: list[dict] = []
    if isinstance(obj, Valeur):
        lignes.append(obj.vers_trace(chemin))
        if isinstance(obj.valeur, (dict, list)):
            lignes.extend(aplatir(obj.valeur, chemin))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            lignes.extend(aplatir(v, f"{chemin}.{k}" if chemin else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lignes.extend(aplatir(v, f"{chemin}[{i}]"))
    return lignes


@dataclass
class Alerte:
    controle: str
    niveau: str            # info, avertissement, bloquant
    message: str
    champ: str | None = None

    def vers_json(self) -> dict:
        d = {"controle": self.controle, "niveau": self.niveau, "message": self.message}
        if self.champ:
            d["champ"] = self.champ
        return d


@dataclass
class Rature:
    echantillon: str | None
    champ: str
    avant: Any
    apres: Any
    page: int | None = None
    boite: tuple | None = None
    conf: float = 0.0

    def vers_json(self) -> dict:
        return {"echantillon": self.echantillon, "champ": self.champ,
                "avant": self.avant, "apres": self.apres}


@dataclass
class Journal:
    """Notes de traitement d'un document (affichées dans l'interface de revue)."""
    notes: list = field(default_factory=list)

    def __call__(self, message: str) -> None:
        self.notes.append(message)
