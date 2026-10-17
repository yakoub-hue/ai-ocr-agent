"""Type de document et regroupement des envois.

* ``typer_page`` : rôle d'une page d'après son texte (bon de commande, bordereau
  d'envoi, courrier d'accompagnement, page de garde de fax, autre).
* ``typer_support`` : forme sous laquelle le BDC est arrivé (fax, liasse
  carbonée, copie imprimée d'un e-mail, original), ce qui explique sa qualité.
* ``regrouper_envois`` : les documents d'un même client portant le même n° de
  commande forment un envoi (pages scannées séparément, renvoi, « annule et
  remplace » qui cite la commande d'origine).

Tout repose sur des mots usuels, pas sur un des sept modèles.
"""
from __future__ import annotations

import re

from .textes import compact, norm

# ordre utile : un courrier cite souvent « le bon de commande joint »
TYPES_PAGE = [
    ("courrier", r"madame,? monsieur|veuillez agreer|nous vous prions|cordialement|salutations"),
    ("bordereau", r"bordereau|liste des echantillons"),
    ("bon_de_commande", r"bon de commande|commande d.analyses|fiche de demande|demande d.analyses?"),
    ("garde_fax", r"page de garde|nb de pages|nombre de pages"),
]
SUPPORTS = [
    ("fax", r"\bfax\b|telecopie|p\.?\s*0*1\s*/\s*0*\d"),
    ("liasse", r"exemplaire|feuillet|souche|blanc client|jaune laboratoire|rose"),
    # en-tête d'impression d'un e-mail ou d'un fichier (« imprimé le … Page 1/1 »), souvent mal lu
    ("copie_mail", r"imprime le|\bde\s*:.*@|\bobjet\s*:.*\bre\s*:|\bpage\s*1\s*/?\s*1\b"),
]


def typer_page(textes: list[str], a_tableau: bool) -> str:
    """Type d'une page d'après ses lignes ; un tableau d'échantillons sans en-tête
    reconnu fait de la page un bordereau."""
    n = norm(" ".join(textes))
    for nom, motif in TYPES_PAGE:
        if re.search(motif, n):
            return nom
    return "bordereau" if a_tableau else "autre"


def typer_support(textes: list[str]) -> str:
    n = norm(" ".join(textes))
    for nom, motif in SUPPORTS:
        if re.search(motif, n):
            return nom
    return "original"


def decrire_document(pages, lignes, tableaux) -> dict:
    """{'support': …, 'pages': [{'numero', 'type'}]} pour la trace et l'interface."""
    avec_tableau = {t.page.numero for t in tableaux}
    types = [{"numero": p.numero,
              "type": typer_page([l.texte for l in lignes if l.page == p.numero], p.numero in avec_tableau)}
             for p in pages]
    return {"support": typer_support([l.texte for l in lignes]), "pages": types}


def regrouper_envois(docs: list[dict]) -> list[dict]:
    """Un envoi par (client, n° de commande) ; un « annule et remplace » qui cite un
    n° de commande présent rejoint l'envoi de cette commande. Renvoie une ligne par
    envoi : id, client, numéro, fichiers."""
    def cle_client(d):
        return compact(d["client"].get("siret") or d["client"].get("raison_sociale") or d["fichier"])

    envois: dict = {}
    for d in docs:
        num = compact(d["commande"].get("numero"))
        cle = (cle_client(d), num) if num else (cle_client(d), d["fichier"])
        envois.setdefault(cle, []).append(d)
    for d in docs:                                   # rattachement des « annule et remplace »
        ar = compact(d["commande"].get("annule_et_remplace"))
        cible = (cle_client(d), ar)
        source = next((k for k, v in envois.items() if d in v), None)
        if ar and cible in envois and source != cible:
            envois[source].remove(d)
            envois[cible].append(d)
    lignes = []
    for i, ((_, _), ds) in enumerate((k, v) for k, v in envois.items() if v):
        lignes.append({"envoi": f"E{i + 1:03d}", "client": ds[0]["client"].get("raison_sociale") or "",
                       "numero_commande": ds[0]["commande"].get("numero") or "",
                       "fichiers": "|".join(d["fichier"] for d in ds)})
    return lignes
