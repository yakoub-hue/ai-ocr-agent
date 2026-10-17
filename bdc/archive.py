"""Lecture d'une archive ZIP de BDC : tous les PDF qu'elle contient, sous-dossiers compris."""
from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath


def pdfs_du_zip(contenu: bytes) -> tuple[list[tuple[str, bytes]], list[str]]:
    """([(nom, octets du PDF)], [fichiers ignorés]). Sont écartés : les dossiers, les fichiers
    cachés ou de système (__MACOSX, .DS_Store, ~$…) et tout ce qui n'est pas un PDF (en-tête %PDF).
    Deux PDF de même nom dans des dossiers différents gardent leur chemin pour rester distincts."""
    pdfs, ignores = [], []
    with zipfile.ZipFile(io.BytesIO(contenu)) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        noms = [PurePosixPath(i.filename).name for i in infos]
        for info, nom in zip(infos, noms):
            chemin = PurePosixPath(info.filename)
            if any(p.startswith((".", "__MACOSX", "~$")) for p in chemin.parts):
                continue
            if chemin.suffix.lower() != ".pdf":
                ignores.append(info.filename)
                continue
            data = z.read(info)
            if not data.startswith(b"%PDF"):
                ignores.append(info.filename)
                continue
            # même nom dans deux dossiers : on garde le chemin, « / » remplacé
            pdfs.append((info.filename.replace("/", "_") if noms.count(nom) > 1 else nom, data))
    return pdfs, ignores
