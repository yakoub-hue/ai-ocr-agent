"""Base de traçabilité : client -> bon de commande -> échantillons -> flacons GC-MS.

Deux moteurs, mêmes tableaux :
* PostgreSQL si la variable DATABASE_URL est définie (secrets Streamlit ou
  environnement) : c'est ce qu'il faut pour l'application déployée, dont le disque
  est effacé à chaque redémarrage ;
* sinon SQLite, dans ``donnees/tracabilite.db`` (dossier ignoré par git, déplaçable
  avec BDC_DONNEES) : un fichier ordinaire, conservé d'un redémarrage à l'autre.

Le PDF original et les pages normalisées de chaque BDC sont rangés dans la base
(tableau fichiers), pour ne dépendre d'aucun disque. Relire le même PDF (même
empreinte) met à jour sa commande au lieu d'en créer une autre.

Un client est identifié par son SIRET quand il est lu (14 chiffres), sinon par sa
raison sociale normalisée. Le flacon de chaque échantillon vient de la règle de
liens.py ; un lien corrigé à la main (source « manuel ») n'est jamais écrasé.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .liens import dossier_labo, flacon_attendu, normaliser_flacon
from .textes import compact

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = Path(os.environ.get("BDC_DONNEES", RACINE / "donnees"))
LARGEUR_PAGE = 1400        # px : pages gardées en JPEG à cette largeur (lisibles, ~200 Ko au lieu de ~1 Mo)

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id {ID}, cle TEXT UNIQUE NOT NULL, raison_sociale TEXT, siret TEXT,
    adresse TEXT, cp TEXT, ville TEXT, telephone TEXT, email TEXT, contact TEXT);
CREATE TABLE IF NOT EXISTS commandes (
    id {ID}, client_id INTEGER NOT NULL REFERENCES clients(id),
    empreinte TEXT UNIQUE NOT NULL, fichier TEXT, numero TEXT, date TEXT, delai TEXT, echeance TEXT,
    dossier_labo TEXT, revue_humaine INTEGER, raisons TEXT, nb_pages INTEGER,
    doc TEXT, trace TEXT, meta TEXT, lu_le TEXT);
CREATE TABLE IF NOT EXISTS echantillons (
    id {ID}, commande_id INTEGER NOT NULL REFERENCES commandes(id) ON DELETE CASCADE,
    rang INTEGER, ref TEXT, parcelle TEXT, profondeur_cm TEXT, matrice TEXT, analyses TEXT,
    hap_demande INTEGER, observations TEXT, flacon TEXT, source_lien TEXT);
CREATE TABLE IF NOT EXISTS resultats_gcms (
    injection TEXT PRIMARY KEY, flacon TEXT, lot TEXT, statut TEXT, concentrations TEXT, importe_le TEXT);
CREATE TABLE IF NOT EXISTS fichiers (
    empreinte TEXT NOT NULL, nom TEXT NOT NULL, contenu {BLOB} NOT NULL, PRIMARY KEY (empreinte, nom));
CREATE INDEX IF NOT EXISTS i_ech_flacon ON echantillons(flacon);
CREATE INDEX IF NOT EXISTS i_res_flacon ON resultats_gcms(flacon);
"""


# ================================================================ CONNEXION
def postgres() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


class Connexion:
    """Même usage pour SQLite et PostgreSQL : paramètres « ? », lignes lisibles par nom,
    validation à la sortie du bloc ``with``."""

    def __init__(self):
        if postgres():
            import psycopg
            from psycopg.rows import dict_row
            self.cx = psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)
            schema = SCHEMA.format(ID="SERIAL PRIMARY KEY", BLOB="BYTEA")
        else:
            DOSSIER.mkdir(parents=True, exist_ok=True)
            self.cx = sqlite3.connect(DOSSIER / "tracabilite.db")
            self.cx.row_factory = sqlite3.Row
            self.cx.execute("PRAGMA foreign_keys = ON")
            schema = SCHEMA.format(ID="INTEGER PRIMARY KEY", BLOB="BLOB")
        for instruction in schema.split(";"):
            if instruction.strip():
                self.cx.execute(instruction)
        self.cx.commit()

    def execute(self, sql: str, params=()):
        return self.cx.execute(sql.replace("?", "%s") if postgres() else sql, params)

    def __enter__(self):
        return self

    def __exit__(self, type_exc, *_):
        if type_exc is None:
            self.cx.commit()
        else:
            self.cx.rollback()
        self.cx.close()


def connexion() -> Connexion:
    return Connexion()


def _agreger(colonne: str) -> str:
    """Concaténation de texte dans un GROUP BY (les deux moteurs l'écrivent différemment)."""
    return f"STRING_AGG({colonne}, ' | ')" if postgres() else f"GROUP_CONCAT({colonne}, ' | ')"


def _maintenant() -> str:
    return datetime.now().isoformat(timespec="seconds")


def cle_client(client: dict) -> str:
    siret = "".join(c for c in (client.get("siret") or "") if c.isdigit())
    if len(siret) == 14:
        return f"siret:{siret}"
    nom = compact(client.get("raison_sociale"))
    return f"nom:{nom}" if nom else "inconnu"


def trouver_client(cx, client: dict) -> str:
    """Clé du client existant à qui rattacher ce BDC : même SIRET, sinon même raison sociale
    (un SIRET lu sur un BDC et pas sur un autre ne doit pas dédoubler le client) ; sinon
    une nouvelle clé."""
    cle = cle_client(client)
    if cx.execute("SELECT 1 FROM clients WHERE cle = ?", (cle,)).fetchone():
        return cle
    nom = compact(client.get("raison_sociale"))
    if nom:
        for r in cx.execute("SELECT cle, raison_sociale FROM clients").fetchall():
            if compact(r["raison_sociale"]) == nom:
                return r["cle"]
    return cle


def alleger_page(image: bytes) -> bytes:
    """Page normalisée (PNG pleine résolution) -> JPEG réduit, pour ne pas remplir la base."""
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return image
    if img.shape[1] > LARGEUR_PAGE:
        f = LARGEUR_PAGE / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


# ================================================================ ENREGISTREMENT
def enregistrer_bdc(nom_pdf: str, contenu_pdf: bytes, resultat: dict, pages_png: list[bytes]) -> int:
    """Enregistre (ou met à jour) un BDC lu par la chaîne ; renvoie l'id de la commande."""
    doc, meta = resultat["doc"], resultat["meta"]
    empreinte = hashlib.sha1(contenu_pdf).hexdigest()
    cl = doc["client"]
    with connexion() as cx:
        cx.execute("DELETE FROM fichiers WHERE empreinte = ?", (empreinte,))
        pages = [(f"p{i}.jpg", alleger_page(p)) for i, p in enumerate(pages_png, 1)]
        for nom, contenu in [("original.pdf", contenu_pdf)] + pages:
            cx.execute("INSERT INTO fichiers (empreinte, nom, contenu) VALUES (?, ?, ?)", (empreinte, nom, contenu))

        cle = trouver_client(cx, cl)
        cx.execute("INSERT INTO clients (cle) VALUES (?) ON CONFLICT (cle) DO NOTHING", (cle,))
        # une lecture plus récente complète la fiche client sans effacer ce qui était connu
        for champ in ("raison_sociale", "siret", "adresse", "cp", "ville", "telephone", "email", "contact"):
            if cl.get(champ):
                cx.execute(f"UPDATE clients SET {champ} = ? WHERE cle = ?", (cl[champ], cle))
        client_id = cx.execute("SELECT id FROM clients WHERE cle = ?", (cle,)).fetchone()["id"]

        ancienne = cx.execute("SELECT id, dossier_labo FROM commandes WHERE empreinte = ?", (empreinte,)).fetchone()
        manuels = {}
        if ancienne:                                  # liens corrigés à la main : gardés
            manuels = {r["ref"]: r["flacon"] for r in cx.execute(
                "SELECT ref, flacon FROM echantillons WHERE commande_id = ? AND source_lien = 'manuel'",
                (ancienne["id"],)).fetchall()}
            cx.execute("DELETE FROM commandes WHERE id = ?", (ancienne["id"],))
        cmd = doc["commande"]
        dossier = ((ancienne["dossier_labo"] if ancienne and ancienne["dossier_labo"] else None)
                   or meta.get("dossier_labo") or dossier_labo(doc))
        commande_id = cx.execute(
            "INSERT INTO commandes (client_id, empreinte, fichier, numero, date, delai, echeance, dossier_labo, "
            "revue_humaine, raisons, nb_pages, doc, trace, meta, lu_le) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "RETURNING id",
            (client_id, empreinte, nom_pdf, cmd.get("numero"), cmd.get("date"), cmd.get("delai"),
             meta.get("echeance"), dossier, int(bool(meta.get("revue_humaine"))),
             json.dumps(meta.get("raisons_revue") or [], ensure_ascii=False), len(pages_png),
             json.dumps(doc, ensure_ascii=False), json.dumps(resultat.get("trace") or [], ensure_ascii=False, default=str),
             json.dumps({k: v for k, v in meta.items() if k != "journal"}, ensure_ascii=False, default=str),
             _maintenant())).fetchone()["id"]
        for rang, e in enumerate(doc["echantillons"], 1):
            ref = e.get("ref_echantillon")
            flacon, source = (manuels[ref], "manuel") if ref in manuels else (flacon_attendu(dossier, rang), "regle")
            cx.execute(
                "INSERT INTO echantillons (commande_id, rang, ref, parcelle, profondeur_cm, matrice, analyses, "
                "hap_demande, observations, flacon, source_lien) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (commande_id, rang, ref, e.get("parcelle"), e.get("profondeur_cm"), e.get("matrice"),
                 "|".join(e.get("analyses") or []), None if e.get("hap_demande") is None else int(e["hap_demande"]),
                 e.get("observations"), flacon, source if flacon else None))
    return commande_id


def importer_resultats(concentrations: dict, statuts: dict, lot: str) -> tuple[int, int]:
    """Résultats GC-MS par injection (sortie de export.lire_resultats_gcms).
    Renvoie (injections importées, injections dont le nom n'est pas un flacon BLPC)."""
    n = ignorees = 0
    with connexion() as cx:
        for injection, conc in concentrations.items():
            flacon = normaliser_flacon(injection)
            if flacon is None:
                ignorees += 1
                continue
            cx.execute("INSERT INTO resultats_gcms (injection, flacon, lot, statut, concentrations, importe_le) "
                       "VALUES (?,?,?,?,?,?) ON CONFLICT (injection) DO UPDATE SET flacon = excluded.flacon, "
                       "lot = excluded.lot, statut = excluded.statut, concentrations = excluded.concentrations, "
                       "importe_le = excluded.importe_le",
                       (injection, flacon, lot, statuts.get(injection, ""), json.dumps(conc, ensure_ascii=False),
                        _maintenant()))
            n += 1
    return n, ignorees


# ================================================================ MODIFICATIONS
def definir_flacon(echantillon_id: int, flacon: str | None) -> None:
    """Lien corrigé à la main ; vide = retour à la règle."""
    with connexion() as cx:
        if flacon:
            cx.execute("UPDATE echantillons SET flacon = ?, source_lien = 'manuel' WHERE id = ?",
                       (normaliser_flacon(flacon) or flacon.strip(), echantillon_id))
        else:
            r = cx.execute("SELECT e.rang, c.dossier_labo FROM echantillons e JOIN commandes c "
                           "ON c.id = e.commande_id WHERE e.id = ?", (echantillon_id,)).fetchone()
            f = flacon_attendu(r["dossier_labo"], r["rang"]) if r else None
            cx.execute("UPDATE echantillons SET flacon = ?, source_lien = ? WHERE id = ?",
                       (f, "regle" if f else None, echantillon_id))


def definir_dossier(commande_id: int, dossier: str | None) -> None:
    """N° de dossier labo saisi à la main ; les flacons issus de la règle sont recalculés."""
    with connexion() as cx:
        cx.execute("UPDATE commandes SET dossier_labo = ? WHERE id = ?", (dossier or None, commande_id))
        for r in cx.execute("SELECT id, rang FROM echantillons WHERE commande_id = ? "
                            "AND (source_lien IS NULL OR source_lien = 'regle')", (commande_id,)).fetchall():
            f = flacon_attendu(dossier, r["rang"])
            cx.execute("UPDATE echantillons SET flacon = ?, source_lien = ? WHERE id = ?",
                       (f, "regle" if f else None, r["id"]))


def supprimer_commande(commande_id: int) -> None:
    with connexion() as cx:
        r = cx.execute("SELECT empreinte, client_id FROM commandes WHERE id = ?", (commande_id,)).fetchone()
        if not r:
            return
        cx.execute("DELETE FROM commandes WHERE id = ?", (commande_id,))
        cx.execute("DELETE FROM fichiers WHERE empreinte = ?", (r["empreinte"],))
        if not cx.execute("SELECT 1 FROM commandes WHERE client_id = ?", (r["client_id"],)).fetchone():
            cx.execute("DELETE FROM clients WHERE id = ?", (r["client_id"],))


# ================================================================ LECTURE
def _entier(x) -> int:
    return int(x or 0)                            # SUM rend un décimal sous PostgreSQL


def _resultat(cx, flacon: str | None):
    """Résultat d'un flacon ; s'il a été réinjecté (dilution « d20 »), l'injection au
    contrôle TIC « OK » est préférée, puis la plus récente."""
    if not flacon:
        return None
    return cx.execute("SELECT * FROM resultats_gcms WHERE flacon = ? ORDER BY (statut = 'OK') DESC, "
                      "importe_le DESC, injection DESC LIMIT 1", (flacon,)).fetchone()


SQL_AVEC_RESULTAT = ("SUM(CASE WHEN EXISTS (SELECT 1 FROM resultats_gcms r WHERE r.flacon = e.flacon) "
                     "THEN 1 ELSE 0 END)")


def clients() -> list[dict]:
    with connexion() as cx:
        rows = cx.execute(f"""
            SELECT cl.id, cl.cle, cl.raison_sociale, cl.siret, cl.adresse, cl.cp, cl.ville, cl.telephone,
                   cl.email, cl.contact, COUNT(DISTINCT c.id) AS nb_commandes, COUNT(e.id) AS nb_echantillons,
                   {SQL_AVEC_RESULTAT} AS nb_resultats, MAX(c.date) AS derniere_commande
            FROM clients cl JOIN commandes c ON c.client_id = cl.id
            LEFT JOIN echantillons e ON e.commande_id = c.id
            GROUP BY cl.id ORDER BY cl.raison_sociale""").fetchall()
    out = [dict(r) for r in rows]
    for c in out:
        for k in ("nb_commandes", "nb_echantillons", "nb_resultats"):
            c[k] = _entier(c[k])
    return out


def commandes(client_id: int | None = None) -> list[dict]:
    with connexion() as cx:
        rows = cx.execute(f"""
            SELECT c.id, c.client_id, cl.raison_sociale AS client, c.fichier, c.numero, c.date, c.delai,
                   c.echeance, c.dossier_labo, c.revue_humaine, c.lu_le, COUNT(e.id) AS nb_echantillons,
                   SUM(CASE WHEN e.flacon IS NOT NULL THEN 1 ELSE 0 END) AS nb_relies,
                   {SQL_AVEC_RESULTAT} AS nb_resultats
            FROM commandes c JOIN clients cl ON cl.id = c.client_id
            LEFT JOIN echantillons e ON e.commande_id = c.id
            {'WHERE c.client_id = ?' if client_id else ''}
            GROUP BY c.id, cl.raison_sociale ORDER BY c.date DESC, c.id DESC""",
                          (client_id,) if client_id else ()).fetchall()
    out = [dict(r) for r in rows]
    for c in out:
        for k in ("nb_echantillons", "nb_relies", "nb_resultats"):
            c[k] = _entier(c[k])
    return out


def textes_echantillons() -> dict:
    """{id de commande: références et flacons de ses échantillons}, pour la recherche."""
    colonne = "COALESCE(ref, '') || ' ' || COALESCE(flacon, '')"
    with connexion() as cx:
        rows = cx.execute(f"SELECT commande_id, {_agreger(colonne)} AS t FROM echantillons "
                          "GROUP BY commande_id").fetchall()
    return {r["commande_id"]: r["t"] or "" for r in rows}


def commande_existante(contenu_pdf: bytes) -> int | None:
    """Id de la commande déjà enregistrée pour ce PDF (même empreinte), sinon None."""
    with connexion() as cx:
        r = cx.execute("SELECT id FROM commandes WHERE empreinte = ?",
                       (hashlib.sha1(contenu_pdf).hexdigest(),)).fetchone()
    return r["id"] if r else None


def commande(commande_id: int) -> dict | None:
    """Commande avec son document JSON, sa trace, son PDF original et ses pages (octets)."""
    with connexion() as cx:
        r = cx.execute("SELECT * FROM commandes WHERE id = ?", (commande_id,)).fetchone()
        if not r:
            return None
        fichiers = {f["nom"]: bytes(f["contenu"]) for f in cx.execute(
            "SELECT nom, contenu FROM fichiers WHERE empreinte = ?", (r["empreinte"],)).fetchall()}
    d = dict(r)
    for k in ("doc", "trace", "meta", "raisons"):
        d[k] = json.loads(d[k]) if d[k] else None
    d["pdf"] = fichiers.get("original.pdf")
    d["pages"] = [fichiers[n] for n in sorted((n for n in fichiers if n.endswith(".jpg")),
                                              key=lambda n: int(n[1:-4]))]
    return d


def echantillons(commande_id: int) -> list[dict]:
    """Échantillons d'une commande avec leur flacon et, s'il existe, le résultat GC-MS."""
    with connexion() as cx:
        rows = [dict(r) for r in cx.execute("SELECT * FROM echantillons WHERE commande_id = ? ORDER BY rang",
                                            (commande_id,)).fetchall()]
        for e in rows:
            res = _resultat(cx, e["flacon"])
            e["injection"] = res["injection"] if res else None
            e["lot"] = res["lot"] if res else None
            e["statut_gcms"] = res["statut"] if res else None
            e["concentrations"] = json.loads(res["concentrations"]) if res else {}
    return rows


def resultats() -> list[dict]:
    with connexion() as cx:
        return [dict(r) for r in cx.execute("SELECT injection, flacon, lot, statut, importe_le FROM resultats_gcms "
                                            "ORDER BY lot DESC, injection").fetchall()]


def flacons_orphelins() -> list[dict]:
    """Flacons GC-MS importés qu'aucun échantillon de BDC ne réclame."""
    with connexion() as cx:
        rows = cx.execute(f"""SELECT r.flacon, {_agreger('r.injection')} AS injections, MAX(r.lot) AS lot
                              FROM resultats_gcms r WHERE NOT EXISTS
                              (SELECT 1 FROM echantillons e WHERE e.flacon = r.flacon)
                              GROUP BY r.flacon ORDER BY r.flacon""").fetchall()
    return [dict(r) for r in rows]


def instantane(max_caracteres: int = 60000) -> str:
    """Vue texte de toute la base, pour l'assistant : un bloc par client, une ligne par
    commande et par échantillon (flacon, statut GC-MS, HAP mesurés au-dessus de la LQ)."""
    lignes = []
    for cl in clients():
        lignes.append(f"CLIENT {cl['raison_sociale'] or '?'} (SIRET {cl['siret'] or '?'}, contact "
                      f"{cl['contact'] or '?'}, {cl['email'] or ''} {cl['telephone'] or ''})")
        for c in commandes(cl["id"]):
            lignes.append(f"  COMMANDE n° {c['numero'] or '?'} du {c['date'] or '?'} ; fichier {c['fichier']} ; "
                          f"délai {c['delai'] or '?'} ; échéance {c['echeance'] or '?'} ; dossier labo "
                          f"{c['dossier_labo'] or '?'} ; revue humaine {'oui' if c['revue_humaine'] else 'non'} ; "
                          f"{c['nb_resultats']}/{c['nb_echantillons']} échantillons avec résultat GC-MS")
            for e in echantillons(c["id"]):
                mesures = {k: v for k, v in e["concentrations"].items()
                           if v and not str(v).startswith("<") and k != "nouveau_multiplicateur"}
                lignes.append(
                    f"    échantillon {e['ref'] or '?'} (rang {e['rang']}, {e['profondeur_cm'] or '?'} cm, "
                    f"analyses {e['analyses'] or '?'}) ; flacon {e['flacon'] or 'non relié'}"
                    + (f" ; injection {e['injection']} lot {e['lot']} contrôle TIC {e['statut_gcms']} ; "
                       f"HAP > LQ (ppm) : {json.dumps(mesures, ensure_ascii=False) if mesures else 'aucun'}"
                       if e["injection"] else " ; pas de résultat GC-MS"))
    texte = "\n".join(lignes)
    if len(texte) > max_caracteres:
        texte = texte[:max_caracteres] + "\n[… base tronquée : trop volumineuse pour être envoyée en entier]"
    return texte
