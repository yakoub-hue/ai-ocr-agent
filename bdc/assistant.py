"""Assistant de traçabilité : questions en français sur les clients, les bons de
commande et les résultats GC-MS de la base.

Gemini reçoit une vue texte de la base (base.instantane) et l'historique de la
conversation ; il répond d'après ces seules données. Un appel par question (quota).
La vue envoyée contient les données des clients (nom, contact, téléphone, e-mail).
"""
from __future__ import annotations

from . import base
from .ia import generer

CONSIGNES = """Tu es l'assistant de traçabilité du laboratoire Orvenne Analyses. Tu relies les clients,
leurs bons de commande (BDC) et les échantillons analysés en GC-MS (HAP).

Réponds en français, brièvement, uniquement d'après les DONNÉES ci-dessous :
- si l'information n'y est pas, dis-le clairement ; n'invente ni valeur, ni client, ni commande ;
- cite le client, le n° de commande et la référence d'échantillon quand tu en parles ;
- concentrations en ppm, telles qu'elles figurent (« < 0,235 » = sous la limite de quantification) ;
- un échantillon « non relié » n'a pas encore de flacon GC-MS correspondant ; « pas de résultat » :
  le flacon n'a pas encore été analysé ou ses résultats n'ont pas été importés ;
- pour une liste longue, utilise un tableau Markdown.
"""


def repondre(question: str, historique: list[dict]) -> str:
    """historique : [{'role': 'user'|'assistant', 'texte': …}] des échanges précédents."""
    echanges = "\n".join(f"{'Utilisateur' if h['role'] == 'user' else 'Assistant'} : {h['texte']}"
                         for h in historique[-6:])
    return generer([f"{CONSIGNES}\nDONNÉES\n{base.instantane()}\n\n"
                    + (f"CONVERSATION PRÉCÉDENTE\n{echanges}\n\n" if echanges else "")
                    + f"QUESTION\n{question}"])
