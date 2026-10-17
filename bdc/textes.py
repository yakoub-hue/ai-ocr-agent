"""Normalisation et reconnaissance de motifs dans le texte lu.

Fonctions pures (sans image) : dates, téléphones, SIRET, e-mails, codes
postaux, profondeurs, références, jours ouvrés. Elles servent à l'extraction
(trouver une valeur dans une ligne bruitée) comme aux contrôles du §5.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from difflib import SequenceMatcher


# ------------------------------------------------------------------ CHAÎNES
def sans_accents(s: str) -> str:
    """'Îlot é' -> 'Ilot e' (caractère par caractère, même longueur)."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in s)


def norm(s: str | None) -> str:
    """Forme de comparaison : minuscules, sans accents, espaces simples."""
    if not s:
        return ""
    s = sans_accents(s).lower()
    s = re.sub(r"[’`´]", "'", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s: str | None) -> str:
    """Forme de comparaison sans espaces ni ponctuation."""
    return re.sub(r"[^0-9a-z]", "", norm(s))


def similarite(a: str | None, b: str | None) -> float:
    a, b = compact(a), compact(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def nettoyer_espaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = re.sub(r"\s+([,.;)])", r"\1", s)
    return s.strip(" :;·-—_|")


def cer(reference: str | None, hypothese: str | None) -> float:
    """Taux d'erreur caractère (distance d'édition / longueur de la référence)."""
    r, h = (reference or ""), (hypothese or "")
    if not r:
        return 0.0 if not h else 1.0
    prec = list(range(len(h) + 1))
    for i, cr in enumerate(r, 1):
        cour = [i] + [0] * len(h)
        for j, ch in enumerate(h, 1):
            cour[j] = min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (cr != ch))
        prec = cour
    return min(1.0, prec[-1] / len(r))


# ------------------------------------------------------------------ CHIFFRES
# Confusions OCR fréquentes quand on attend un chiffre.
VERS_CHIFFRE = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0", "u": "0",
    "I": "1", "l": "1", "|": "1", "i": "1", "!": "1", "]": "1", "[": "1", "J": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5", "$": "5",
    "G": "6", "b": "6",
    "T": "7", "?": "7",
    "B": "8", "&": "8",
    "g": "9", "q": "9",
})


def chiffres(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# ------------------------------------------------------------------ DATES
MOIS = {
    "janvier": 1, "janv": 1, "jan": 1,
    "fevrier": 2, "fevr": 2, "fev": 2,
    "mars": 3, "mar": 3,
    "avril": 4, "avr": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7, "juil": 7,
    "aout": 8,
    "septembre": 9, "sept": 9, "sep": 9,
    "octobre": 10, "oct": 10,
    "novembre": 11, "nov": 11,
    "decembre": 12, "dec": 12,
}
_CAR_DATE = r"[0-9OoIl|SZB]"
# année sur 4 chiffres, éventuellement coupée par un espace à la main (« 20 26 »), ou sur 2 ;
# « 20 14:30 » est une année 20 suivie d'une heure, pas 2014
_RE_DATE_NUM = re.compile(
    rf"(?<![0-9A-Za-z])({_CAR_DATE}{{1,2}})\s?[/.\-]\s?({_CAR_DATE}{{1,2}})\s?[/.\-]\s?"
    rf"({_CAR_DATE}{{4}}|(?:19|2[0OoIl])\s{_CAR_DATE}{{2}}(?!\s?[0-9:hH])|{_CAR_DATE}{{2}})(?![0-9])")
_RE_DATE_COURTE = re.compile(rf"(?<![0-9/])({_CAR_DATE}{{1,2}})\s?/\s?({_CAR_DATE}{{1,2}})(?![0-9/])")
_RE_DATE_TEXTE = re.compile(
    r"(?<![0-9])([0-3]?[0-9OoIl])(?:er)?\s*[\-.]?\s*(" + "|".join(sorted(MOIS, key=len, reverse=True))
    + r")\.?\s*[\-.]?\s*((?:19|20)[0-9OoIl]{2})", re.I)


def date_valide(j: int, m: int, a: int) -> date | None:
    try:
        return date(a, m, j) if 2000 <= a <= 2100 else None
    except ValueError:
        return None


def _entier(s: str) -> int | None:
    s = s.translate(VERS_CHIFFRE)
    return int(s) if s.isdigit() else None


def trouver_dates(texte: str, annee_defaut: int | None = None) -> list[tuple[date, int, int]]:
    """Toutes les dates d'un texte : (date, début, fin), dans l'ordre du texte.

    Formats : 25/09/2026, 25.09.26, 28 MAI 2026, 19 SEP. 2026, le 1er octobre
    2026, et 26/09 (jour/mois) si une année par défaut est fournie.
    """
    if not texte:
        return []
    trouvees = []
    for m in _RE_DATE_NUM.finditer(texte):
        j, mo, a = (_entier(re.sub(r"\s", "", g)) for g in m.groups())
        if None in (j, mo, a):
            continue
        if a < 100:
            a += 2000
        d = date_valide(j, mo, a)
        if d:
            trouvees.append((d, m.start(), m.end()))
    t = sans_accents(texte)
    for m in _RE_DATE_TEXTE.finditer(t):
        j, a = _entier(m.group(1)), _entier(m.group(3))
        mo = MOIS.get(m.group(2).lower())
        d = date_valide(j, mo, a) if None not in (j, mo, a) else None
        if d and not any(s <= m.start() < f for _, s, f in trouvees):
            trouvees.append((d, m.start(), m.end()))
    if annee_defaut:
        for m in _RE_DATE_COURTE.finditer(texte):
            if any(s <= m.start() < f for _, s, f in trouvees):
                continue
            j, mo = _entier(m.group(1)), _entier(m.group(2))
            d = date_valide(j, mo, annee_defaut) if None not in (j, mo) else None
            if d:
                trouvees.append((d, m.start(), m.end()))
    return sorted(trouvees, key=lambda x: x[1])


def premiere_date(texte: str, annee_defaut: int | None = None) -> date | None:
    d = trouver_dates(texte, annee_defaut)
    return d[0][0] if d else None


# ------------------------------------------------------------------ JOURS OUVRÉS
def _paques(annee: int) -> date:
    a, b, c = annee % 19, annee // 100, annee % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mois = (h + l_ - 7 * m + 114) // 31
    jour = (h + l_ - 7 * m + 114) % 31 + 1
    return date(annee, mois, jour)


def jours_feries(annee: int) -> set[date]:
    """Jours fériés nationaux français (hors Alsace-Moselle)."""
    p = _paques(annee)
    fixes = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    return {date(annee, m, j) for m, j in fixes} | {
        p + timedelta(days=1), p + timedelta(days=39), p + timedelta(days=50)}


def ajouter_jours_ouvres(depart: date, n: int) -> date:
    """Date atteinte après n jours ouvrés (lundi-vendredi hors fériés)."""
    d, restants = depart, n
    feries = jours_feries(depart.year) | jours_feries(depart.year + 1)
    while restants > 0:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in feries:
            restants -= 1
    return d


# ------------------------------------------------------------------ CONTACTS
_RE_TEL = re.compile(r"(?<![0-9])(?:\+33\s?\(?0?\)?\s?|0)[1-9](?:[\s.\-]{0,2}[0-9OoIlSB]){8}(?![0-9])")


def normaliser_telephone(s: str | None) -> str | None:
    """Renvoie '0X XX XX XX XX' si s contient 10 chiffres commençant par 0."""
    if not s:
        return None
    s = s.replace("+33", "0")
    m = _RE_TEL.search(s)
    if m:
        s = m.group(0)
    elif re.search(r"[A-Za-z]", s):
        s = chiffres(s)                      # pas de motif : on ne garde que les vrais chiffres
    d = chiffres(s.translate(VERS_CHIFFRE))
    if d.startswith("00"):
        d = d[1:]
    if len(d) != 10 or d[0] != "0":
        return None
    return " ".join(d[i:i + 2] for i in range(0, 10, 2))


def trouver_telephones(texte: str) -> list[tuple[str, int, int]]:
    res = []
    for m in _RE_TEL.finditer(texte or ""):
        t = normaliser_telephone(m.group(0))
        if t:
            res.append((t, m.start(), m.end()))
    return res


_RE_EMAIL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+\-]*\s?@\s?[A-Za-z0-9\-]+(?:[.,][A-Za-z0-9\-]+)*[.,][A-Za-z]{2,4}(?![A-Za-z])")


def trouver_emails(texte: str) -> list[str]:
    res = []
    for m in _RE_EMAIL.finditer(texte or ""):
        e = re.sub(r"\s", "", m.group(0))
        e = re.sub(r",(?=[a-z]{2,4}$)", ".", e)          # 'xx@yy,fr' -> 'xx@yy.fr'
        e = e.replace(",", ".")
        res.append(e.strip(".").lower())
    return res


def email_valide(e: str | None) -> bool:
    return bool(e) and re.fullmatch(r"[a-z0-9][a-z0-9._%+\-]*@[a-z0-9\-]+(\.[a-z0-9\-]+)*\.[a-z]{2,}", e) is not None


# ------------------------------------------------------------------ SIRET
_RE_SIRET = re.compile(r"(?<![0-9])([0-9OoIlSB](?:\s?[0-9OoIlSB]){13})(?![0-9])")


def trouver_siret(texte: str) -> str | None:
    for m in _RE_SIRET.finditer(texte or ""):
        d = chiffres(m.group(1).translate(VERS_CHIFFRE))
        if len(d) == 14:
            return d
    return None


def luhn_ok(numero: str | None) -> bool:
    if not numero or not numero.isdigit():
        return False
    total = 0
    for i, c in enumerate(reversed(numero)):
        v = int(c) * (2 if i % 2 else 1)
        total += v - 9 if v > 9 else v
    return total % 10 == 0


# ------------------------------------------------------------------ ADRESSES
_RE_CP_VILLE = re.compile(r"(?<![0-9])([0-9OoIlSB]{5})\s*,?\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\- ]{1,40}[A-Za-zÀ-ÿ])")


def trouver_cp_ville(texte: str) -> tuple[str, str] | None:
    for m in _RE_CP_VILLE.finditer(texte or ""):
        cp = m.group(1).translate(VERS_CHIFFRE)
        if cp.isdigit() and 1000 <= int(cp) <= 98999:
            return cp, m.group(2).strip()
    return None


# ------------------------------------------------------------------ ÉCHANTILLONS
_RE_PROF = re.compile(r"([0-9OoIlSZBg]{1,3})\s*(?:-|–|—|_|=|~|à|a|/|\.|:|\s)\s*([0-9OoIlSZBg]{1,3})")


def normaliser_profondeur(s: str | None) -> str | None:
    """'0 - 30', '0–30 cm', 'O-3O' -> '0-30' ; None si pas de forme a-b."""
    if not s:
        return None
    t = s.replace("cm", " ").strip()
    m = _RE_PROF.search(t)
    if not m:
        return None
    a, b = _entier(m.group(1)), _entier(m.group(2))
    if a is None or b is None:
        return None
    return f"{a}-{b}"


def profondeur_valide(p: str | None) -> bool:
    if not p or not re.fullmatch(r"\d{1,3}-\d{1,3}", p):
        return False
    a, b = (int(x) for x in p.split("-"))
    return a < b <= 300


def normaliser_ref(s: str | None) -> str | None:
    """Référence d'échantillon : majuscules, tirets unifiés, sans espaces parasites."""
    if not s:
        return None
    t = s.strip().replace("–", "-").replace("—", "-").replace("_", "-")
    t = re.sub(r"\s*-\s*", "-", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;")
    return t.upper() or None


_RE_PARCELLE = re.compile(r"(?:(?<![A-Za-z0-9])([A-Z]{1,2})\s?([0-9]{1,4})(?![0-9])|([ÎI]lot)\s*([0-9]{1,3}))")


def trouver_parcelles(texte: str) -> list[str]:
    """'ZC 8 et ZC 9' -> ['ZC 8', 'ZC 9'] ; 'Îlot 4' -> ['Îlot 4']."""
    res = []
    for m in _RE_PARCELLE.finditer(texte or ""):
        if m.group(1):
            p = f"{m.group(1)} {int(m.group(2))}"
        else:
            p = f"Îlot {int(m.group(4))}"
        if p not in res:
            res.append(p)
    return res


def normaliser_parcelle(s: str | None) -> str | None:
    if not s:
        return None
    t = s.strip()
    p = trouver_parcelles(t)
    if len(p) == 1 and len(compact(t)) <= len(compact(p[0])) + 1:
        return p[0]
    return nettoyer_espaces(t) or None


def temperature(texte: str) -> float | None:
    """'T° = 5 °C', 'glacière 6 °C', '4,5°C' -> valeur numérique."""
    m = re.search(r"(-?[0-9OoIlS]{1,2}(?:[.,][0-9])?)\s*°\s*[cC]?", texte or "")
    if not m:
        m = re.search(r"(-?[0-9]{1,2}(?:[.,][0-9])?)\s*[°º']?\s*C\b", texte or "")
    if not m:
        return None
    v = m.group(1).replace(",", ".")
    v = v[0] + v[1:].translate(VERS_CHIFFRE) if v.startswith("-") else v.translate(VERS_CHIFFRE)
    try:
        x = float(v)
    except ValueError:
        return None
    return int(x) if x == int(x) else x
