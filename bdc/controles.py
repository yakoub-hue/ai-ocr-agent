"""Brique 5 : contrôles métier (§5) et décision de revue humaine.

Les contrôles travaillent sur le document JSON final (valeurs simples), donc
ils sont rejoués après chaque correction humaine. Un contrôle qui échoue
produit une alerte ; il ne corrige jamais la valeur (en particulier un SIRET
qui échoue au test de Luhn n'est jamais « réparé »).
"""
from __future__ import annotations

import re
from datetime import date

from . import config
from .textes import ajouter_jours_ouvres, email_valide, luhn_ok, norm, profondeur_valide


def _d(s):
    try:
        return date.fromisoformat(s) if s else None
    except (TypeError, ValueError):
        return None


def A(controle, niveau, message, champ=None):
    d = {"controle": controle, "niveau": niveau, "message": message}
    if champ:
        d["champ"] = champ
    return d


def controler(doc: dict, declare: int | None = None, alertes_lecture: list | None = None) -> list[dict]:
    """Retourne la liste des alertes du document (§4 : controle, niveau, message)."""
    al: list[dict] = list(alertes_lecture or [])
    cl, cmd, pre, rec = doc["client"], doc["commande"], doc["prelevement"], doc["reception_labo"]
    ech = doc["echantillons"]

    # SIRET : 14 chiffres + Luhn ; alerte seulement
    s = cl.get("siret")
    if s:
        chiffres = re.sub(r"\D", "", str(s))
        if len(chiffres) != 14 or not luhn_ok(chiffres):
            al.append(A("siret_luhn", "avertissement", f"SIRET {s} invalide (14 chiffres + Luhn) : à vérifier, non corrigé", "client.siret"))
    # téléphone
    t = cl.get("telephone")
    if t:
        c = re.sub(r"\D", "", str(t))
        if not (len(c) == 10 and c.startswith("0")):
            al.append(A("telephone", "avertissement", f"Téléphone « {t} » : 10 chiffres commençant par 0 attendus", "client.telephone"))
    # e-mail
    m = cl.get("email")
    if m and not email_valide(m):
        al.append(A("email", "info", f"E-mail « {m} » au format invalide", "client.email"))

    # chronologie
    dp, dc, dr = _d(pre.get("date")), _d(cmd.get("date")), _d(rec.get("date"))
    if not dp:
        al.append(A("prelevement_date", "avertissement", "Date de prélèvement absente ou illisible", "prelevement.date"))
    suite = [(n, d) for n, d in (("prélèvement", dp), ("commande", dc), ("réception", dr)) if d]
    for (n1, d1), (n2, d2) in zip(suite, suite[1:]):
        if d1 > d2:
            al.append(A("chronologie", "bloquant", f"Date de {n1} ({d1}) postérieure à celle de {n2} ({d2})"))
    # conservation (prélèvement -> réception)
    if dp and dr:
        h = (dr - dp).days * 24
        if h > config.SEUIL_CONSERVATION_H:
            al.append(A("conservation", "avertissement",
                        f"Conservation de {(dr - dp).days} j (> {config.SEUIL_CONSERVATION_H} h) entre prélèvement et réception"))
    # température
    tc = rec.get("temperature_c")
    if tc is not None and float(tc) > config.SEUIL_TEMPERATURE_C:
        al.append(A("temperature", "avertissement", f"Température à réception {tc} °C > {config.SEUIL_TEMPERATURE_C} °C"))

    # dénombrement
    if declare is not None and declare != len(ech):
        al.append(A("denombrement", "bloquant", f"{declare} échantillons déclarés, {len(ech)} lignes non annulées lues"))
    # références uniques non vides
    refs = [e.get("ref_echantillon") for e in ech]
    if any(not r for r in refs):
        al.append(A("references", "bloquant", "Référence d'échantillon vide"))
    vus, dup = set(), set()
    for r in refs:
        if r and r in vus:
            dup.add(r)
        vus.add(r)
    if dup:
        al.append(A("references", "bloquant", f"Références en double : {', '.join(sorted(dup))}"))
    # profondeur, analyses
    for e in ech:
        p = e.get("profondeur_cm")
        if p and not profondeur_valide(p):
            al.append(A("profondeur", "avertissement", f"Profondeur « {p} » invalide pour {e.get('ref_echantillon')}"))
        if not e.get("analyses"):
            al.append(A("analyses", "bloquant", f"Aucune analyse demandée pour {e.get('ref_echantillon')}"))
    # remplacement
    ar = cmd.get("annule_et_remplace")
    if ar:
        if not re.search(r"[A-Za-z0-9]{3,}", str(ar)) or norm(str(ar)).startswith("annule"):
            al.append(A("remplacement", "avertissement", "« Annule et remplace » sans référence d'origine identifiable"))
        else:
            al.append(A("remplacement", "avertissement", f"Annule et remplace la commande {ar} : vérifier le lien avec l'original"))
    # délai
    if cmd.get("delai") not in config.DELAIS:
        al.append(A("delai", "bloquant", "Délai absent ou non reconnu", "commande.delai"))
    # total corrigé
    if any(r.get("champ") == "nombre_echantillons" for r in doc.get("ratures", [])):
        al.append(A("total_corrige", "info", "Le total d'échantillons a été corrigé à la main"))
    # dédoublonnage (même contrôle + message)
    vu, res = set(), []
    for a in al:
        k = (a["controle"], a["message"])
        if k not in vu:
            vu.add(k)
            res.append(a)
    return res


def echeance(doc: dict) -> str | None:
    """Échéance en jours ouvrés depuis la réception (ou, à défaut, la commande)."""
    delai = doc["commande"].get("delai")
    depart = _d(doc["reception_labo"].get("date")) or _d(doc["commande"].get("date"))
    if depart and delai in config.DELAIS:
        return ajouter_jours_ouvres(depart, config.DELAIS[delai]).isoformat()
    return None


def revue_requise(doc: dict, confiances: dict, alertes: list[dict]) -> tuple[bool, list[str]]:
    """Revue humaine obligatoire si un champ indispensable manque ou a une confiance
    trop basse, ou si une alerte est bloquante. ``confiances`` : chemin JSON -> confiance."""
    raisons = []
    seuil = config.SEUIL_CONFIANCE_REVUE

    def verifie(chemin, valeur):
        if valeur in (None, "", []):
            raisons.append(f"{chemin} manquant")
        elif confiances.get(chemin, 1.0) < seuil:
            raisons.append(f"{chemin} peu fiable ({confiances.get(chemin):.2f})")

    verifie("client.raison_sociale", doc["client"].get("raison_sociale"))
    if not (doc["client"].get("telephone") or doc["client"].get("email") or doc["client"].get("contact")):
        raisons.append("aucun contact (téléphone, e-mail ou nom)")
    verifie("commande.delai", doc["commande"].get("delai"))
    for i, e in enumerate(doc["echantillons"]):
        verifie(f"echantillons[{i}].ref_echantillon", e.get("ref_echantillon"))
        if e.get("hap_demande") is None:
            raisons.append(f"echantillons[{i}].hap_demande manquant")
        elif confiances.get(f"echantillons[{i}].hap_demande", 1.0) < seuil:
            raisons.append(f"echantillons[{i}].hap_demande peu fiable")
    for a in alertes:
        if a["niveau"] == "bloquant":
            raisons.append(f"alerte bloquante : {a['controle']}")
    return bool(raisons), raisons
