"""Lien entre un échantillon de BDC et son flacon analysé en GC-MS.

Règle PROVISOIRE « dossier + rang », à confirmer par le laboratoire ; c'est la
seule fonction à changer (``flacon_attendu``) si la règle est autre :

    flacon = année du dossier + « BLPC » + n° de dossier (6 chiffres) + « - » + rang (3 chiffres)

Le n° de dossier est celui que le laboratoire note sur le BDC à la réception
(ex. 26ORV004213) ; le rang est la position de l'échantillon dans le BDC, lignes
annulées exclues (un flacon cassé n'est pas analysé). Exemple : 2e échantillon du
dossier 26ORV004213 -> 26BLPC004213-002.

Un lien peut toujours être corrigé à la main dans l'application.
"""
from __future__ import annotations

import re

# n° de dossier noté par le labo : 26ORV004213 (l'OCR lit parfois 0 pour O)
RE_DOSSIER = re.compile(r"\b(\d{2})\s*[O0]\s*R\s*V\s*(\d{6})\b", re.I)
# nom d'injection GC-MS : 25BLPC003045-006, éventuellement suivi d'un suffixe (« d20 » : dilué 20 fois)
RE_FLACON = re.compile(r"(\d{2})\s*BLPC\s*(\d{6})\s*-\s*(\d{3})", re.I)


def dossier_dans_textes(textes) -> str | None:
    """N° de dossier (forme 26ORV004213) dans des lignes de texte. Le labo colle parfois
    plusieurs étiquettes (26ORV004213, 26ORV004215) : la première de la série est retenue."""
    trouves = sorted({(m.group(1), m.group(2)) for t in textes for m in RE_DOSSIER.finditer(t or "")})
    return f"{trouves[0][0]}ORV{trouves[0][1]}" if trouves else None


def dossier_labo(doc: dict) -> str | None:
    """N° de dossier trouvé dans les ajouts du labo ou les remarques d'un document JSON."""
    textes = list(doc.get("reception_labo", {}).get("annotations") or [])
    return dossier_dans_textes(textes + [doc.get("commande", {}).get("remarques") or ""])


def flacon_attendu(dossier: str | None, rang: int) -> str | None:
    """Identifiant du flacon GC-MS attendu pour l'échantillon n° rang (1, 2, …) du dossier."""
    m = RE_DOSSIER.search(dossier or "")
    if not m:
        return None
    return f"{m.group(1)}BLPC{m.group(2)}-{rang:03d}"


def normaliser_flacon(nom: str | None) -> str | None:
    """« 25BLPC003045-006 d20 » -> « 25BLPC003045-006 » ; None si ce n'est pas un nom de flacon."""
    m = RE_FLACON.search(nom or "")
    return f"{m.group(1)}BLPC{m.group(2)}-{m.group(3)}" if m else None
