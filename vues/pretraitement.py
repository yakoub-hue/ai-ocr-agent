"""Affichage du prétraitement fait avant l'OCR : page d'origine, page redressée et
les deux zones lues séparément (informations client / tableau des échantillons)."""
import cv2
import numpy as np
import pymupdf
import streamlit as st

BLEU, VERT = (40, 110, 220), (30, 160, 70)      # RGB : zone client, zone échantillons


def corrections(etapes: list) -> list[str]:
    """Corrections d'une page, en clair."""
    out = []
    for e in etapes:
        if e.get("etape") == "perspective":
            out.append("feuille photographiée remise à plat (perspective)")
        elif e.get("etape") == "axe_du_texte" and e.get("rotation_deg"):
            out.append(f"page tournée de {e['rotation_deg']}°")
        elif e.get("etape") == "inclinaison" and abs(float(e.get("angle_deg") or 0)) >= 0.1:
            out.append(f"inclinaison corrigée de {float(e['angle_deg']):+.1f}°".replace(".", ","))
        elif e.get("etape") == "sens" and e.get("rotation_deg"):
            out.append("page remise à l'endroit (180°)")
    return out or ["page déjà droite : aucune correction"]


def pages_origine(pdf: bytes, dpi: int = 70) -> list[np.ndarray]:
    """Pages du PDF tel qu'il a été reçu, en petite résolution."""
    images = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
            images.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3))
    return images


def zones(image: bytes, boites: list, taille: list | None) -> np.ndarray:
    """Page redressée avec le tableau des échantillons en vert et le reste (client,
    commande, site) en bleu."""
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    f = img.shape[1] / taille[1] if taille else 1.0           # l'image stockée peut être réduite
    voile = np.empty_like(img)
    voile[:] = BLEU
    masque = np.zeros(img.shape[:2], bool)
    rects = [[int(v * f) for v in b] for b in boites]
    for x0, y0, x1, y1 in rects:
        voile[y0:y1, x0:x1] = VERT
        masque[y0:y1, x0:x1] = True
    sortie = cv2.addWeighted(img, 0.82, voile, 0.18, 0)
    epais = max(3, img.shape[1] // 300)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(sortie, (x0, y0), (x1, y1), VERT, epais)
    return sortie


def afficher(pdf: bytes | None, pages: list[bytes], meta: dict):
    """Onglet « Prétraitement » : avant / après, zones et corrections de chaque page."""
    infos = {p["page"]: p for p in meta.get("pretraitement") or []}
    boites = {}
    for z in meta.get("zones_tableau") or []:
        boites.setdefault(z["page"], []).append(z["boite"])
    if not infos:
        st.info("Prétraitement non enregistré pour ce BDC (lu avant cette version) : relisez le PDF pour l'afficher.")
    st.caption("Avant l'OCR, chaque page est redressée, puis lue en deux zones séparées : "
               "🟩 tableau des échantillons (lu cellule par cellule) · 🟦 reste de la page "
               "(client, commande, site, prélèvement).")
    origines = pages_origine(pdf) if pdf else []
    for i, page in enumerate(pages, 1):
        info = infos.get(i, {})
        st.markdown(f"**Page {i}** : " + " · ".join(corrections(info.get("etapes") or [])))
        avant, apres = st.columns(2)
        if i <= len(origines):
            avant.image(origines[i - 1], caption="Telle que reçue", width="stretch")
        apres.image(zones(page, boites.get(i, []), info.get("taille")),
                    caption="Redressée, avec les zones lues", width="stretch")
        if not boites.get(i):
            apres.caption("Pas de tableau d'échantillons détecté sur cette page : toute la page est lue comme texte.")
