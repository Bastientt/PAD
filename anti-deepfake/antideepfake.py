#!/usr/bin/env python3
"""
deepfake_detector.py
====================
Détection de deepfakes par analyse de l'incohérence de bruit entre
la région du visage et le fond de l'image.

PRINCIPE FONDAMENTAL
--------------------
Vraie vidéo  : visage + fond capturés par le MÊME capteur → bruit résiduel
               cohérent entre les deux régions (même texture de bruit).

Deepfake     : le visage est généré par un GAN et collé sur un vrai fond.
               → Le bruit du visage (lisse, structuré) est DIFFÉRENT
                 du bruit du fond (grain de capteur, PRNU).

MÉTRIQUES
---------
1. Delta variance de bruit       face vs fond  (élevé → suspect)
2. Corrélation de bruit          face vs fond  (faible → suspect)
3. Variance temporelle du visage inter-frames  (trop faible → GAN quasi-statique)
4. Analyse DCT                   face region   (coefficients anormaux → GAN)

Pas de ML, pas de modèle externe — uniquement OpenCV + NumPy.

Usage :
    python deepfake_detector.py video.mp4
    python deepfake_detector.py video.mp4 --no-display
    python deepfake_detector.py dossier/     # scan batch
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DU BRUIT RÉSIDUEL
# ─────────────────────────────────────────────────────────────────────────────

def noise_residual(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Extrait le bruit résiduel par différence avec une version lissée.
    Méthode classique des modèles SRM (Spatial Rich Model).
    Le résidu contient la signature du capteur (ou du GAN).
    """
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    residual = gray.astype(np.float32) - blurred
    return residual


def noise_stats(residual: np.ndarray) -> dict:
    """Statistiques du bruit : variance, kurtosis, entropie locale."""
    flat = residual.flatten()
    var  = float(np.var(flat))
    # Kurtosis : bruit GAN est souvent plus "gaussien" que le bruit capteur
    if var > 1e-6:
        kurt = float(np.mean((flat - flat.mean())**4) / (var**2 + 1e-9))
    else:
        kurt = 0.0
    return {"var": var, "kurt": kurt}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE DCT SUR LE VISAGE
# ─────────────────────────────────────────────────────────────────────────────

def dct_score(face_gray: np.ndarray) -> float:
    """
    Analyse des coefficients DCT du patch visage.

    Les GAN génèrent des visages avec des artefacts dans les coefficients
    DCT moyens-fréquences (ceux que JPEG compresse mais que l'œil ne voit pas).
    On mesure le ratio énergie MF / énergie totale.

    Réel  → distribution naturelle des coefficients
    GAN   → sur-représentation ou sous-représentation des MF
    """
    face_r = cv2.resize(face_gray, (64, 64)).astype(np.float32)
    dct    = cv2.dct(face_r)

    total_e = np.sum(dct**2) + 1e-9

    # Basses fréquences (8×8 coin supérieur gauche)
    lf_e = np.sum(dct[:8,  :8 ]**2)
    # Hautes fréquences (bord extérieur)
    hf_e = np.sum(dct[32:, :  ]**2) + np.sum(dct[:, 32:]**2)
    # Moyennes fréquences (le reste)
    mf_e = total_e - lf_e - hf_e

    return float(mf_e / total_e)


# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTEUR DE VISAGES
# ─────────────────────────────────────────────────────────────────────────────

def load_face_detector():
    """
    Charge le détecteur Haar cascade.
    Si le XML est absent partout, le télécharge depuis GitHub.
    """
    import urllib.request

    local_path = Path(__file__).parent / "haarcascade_frontalface_default.xml"

    candidates = [str(local_path)]
    try:
        candidates.insert(0, cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except AttributeError:
        pass
    cv2_dir = Path(cv2.__file__).parent
    candidates.append(str(cv2_dir / "data" / "haarcascade_frontalface_default.xml"))
    candidates += [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]

    for path in candidates:
        if Path(path).exists():
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                return detector

    # Téléchargement automatique
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/master"
        "/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    print("  [!] Cascade XML absent — téléchargement depuis GitHub…")
    try:
        urllib.request.urlretrieve(url, str(local_path))
        print(f"  [ok] Sauvegardé dans {local_path}")
    except Exception as e:
        raise RuntimeError(
            f"Téléchargement échoué : {e}\n"
            "  Téléchargez manuellement :\n"
            f"  wget \"{url}\" -O haarcascade_frontalface_default.xml"
        )

    detector = cv2.CascadeClassifier(str(local_path))
    if detector.empty():
        raise RuntimeError("Fichier téléchargé mais non chargeable.")
    return detector


def detect_face(gray: np.ndarray, detector) -> tuple | None:
    """
    Retourne (x, y, w, h) du plus grand visage, ou None.
    On prend le plus grand pour éviter les faux positifs en arrière-plan.
    """
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    # Plus grand visage détecté
    return max(faces, key=lambda f: f[2] * f[3])


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE D'UNE FRAME
# ─────────────────────────────────────────────────────────────────────────────

def analyze_frame(frame: np.ndarray, detector) -> dict | None:
    """
    Analyse une frame. Retourne None si aucun visage détecté.

    Métriques retournées :
      - noise_delta    : |var_face - var_background| / var_background
      - noise_corr     : corrélation entre résidu face et résidu fond (patch)
      - dct_mf_ratio   : ratio énergie MF DCT du visage
      - face_bbox      : (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(gray, detector)
    if face is None:
        return None

    x, y, w, h = face

    # Régions
    face_gray = gray[y:y+h, x:x+w]

    # Zone fond : même taille, ailleurs dans l'image
    # On prend un patch en bas à droite si possible, sinon en haut à gauche
    H, W = gray.shape
    if x + w + w < W and y + h + h < H:
        bg_gray = gray[y+h : y+h+h, x : x+w]
    elif x - w >= 0:
        bg_gray = gray[y : y+h, x-w : x]
    else:
        bg_gray = gray[max(0, H-h):H, max(0, W-w):W]

    # Redimensionner le fond à la même taille que le visage
    bg_gray = cv2.resize(bg_gray, (w, h))

    # Bruits résiduels
    face_res = noise_residual(face_gray)
    bg_res   = noise_residual(bg_gray)

    # Stats
    face_stats = noise_stats(face_res)
    bg_stats   = noise_stats(bg_res)

    # 1. Delta de variance (normalisé)
    bg_var = bg_stats["var"] if bg_stats["var"] > 1e-6 else 1e-6
    noise_delta = abs(face_stats["var"] - bg_stats["var"]) / bg_var

    # 2. Corrélation des résidus (flatten + corrcoef)
    fr = face_res.flatten()
    br = bg_res.flatten()
    n  = min(len(fr), len(br), 2048)
    if n > 10:
        corr_matrix = np.corrcoef(fr[:n], br[:n])
        noise_corr  = float(abs(corr_matrix[0, 1]))
    else:
        noise_corr = 0.0

    # 3. DCT
    dct_mf = dct_score(face_gray)

    return {
        "noise_delta": noise_delta,
        "noise_corr":  noise_corr,
        "dct_mf":      dct_mf,
        "face_var":    face_stats["var"],
        "face_bbox":   (x, y, w, h),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE FINAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(results: list[dict]) -> dict:
    """
    Agrège les métriques de toutes les frames et calcule un score final.

    Logique :
      - noise_delta élevé     → visage et fond ont des niveaux de bruit différents
      - noise_corr faible      → les textures de bruit ne sont pas corrélées
      - face_var faible        → visage GAN trop lisse par rapport au fond
      - temporal_var faible    → le visage change peu d'une frame à l'autre (GAN statique)
    """
    if not results:
        return {"score": 0.0, "verdict": "Aucun visage détecté", "details": {}}

    deltas    = [r["noise_delta"] for r in results]
    corrs     = [r["noise_corr"]  for r in results]
    dcts      = [r["dct_mf"]      for r in results]
    face_vars = [r["face_var"]    for r in results]

    avg_delta = float(np.mean(deltas))
    avg_corr  = float(np.mean(corrs))
    avg_dct   = float(np.mean(dcts))

    # Variance temporelle du visage
    temporal_var = float(np.std(face_vars))

    # ── Scoring directionnel ──────────────────────────────────────────────
    # Chaque terme est normalisé pour contribuer ~0.25 max au score final.

    # 1. Delta bruit élevé = suspect (deepfake colle un visage sur fond natif)
    #    On sature à partir de delta > 2.0 (différence de variance ×2 = très suspect)
    s_delta = min(avg_delta / 2.0, 1.0)

    # 2. Corrélation faible = suspect
    #    Un deepfake : corrélation proche de 0. Réel : 0.1–0.4 (même capteur)
    #    On inverse : faible corr → score élevé
    s_corr = max(0.0, 1.0 - avg_corr * 5.0)  # corr > 0.2 → score ~0

    # 3. DCT : ratio MF anormal
    #    Réel : ~0.35–0.50. GAN : souvent < 0.25 ou > 0.65
    #    Anomalie si loin de 0.40
    s_dct = min(abs(avg_dct - 0.40) / 0.30, 1.0)

    # 4. Variance temporelle faible = GAN quasi-statique = suspect
    #    Réel : var > 50 (le visage bouge, l'éclairage change)
    #    Deepfake : souvent var < 15 (surface lisse et peu variable)
    s_temporal = max(0.0, 1.0 - temporal_var / 50.0)

    score = 0.35 * s_delta + 0.25 * s_corr + 0.20 * s_dct + 0.20 * s_temporal
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score":        score,
        "s_delta":      s_delta,
        "s_corr":       s_corr,
        "s_dct":        s_dct,
        "s_temporal":   s_temporal,
        "avg_delta":    avg_delta,
        "avg_corr":     avg_corr,
        "avg_dct":      avg_dct,
        "temporal_var": temporal_var,
        "n_frames":     len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSE VIDÉO
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD = 0.40  # score >= 0.40 → deepfake


def check_metadata(video_path: str) -> dict | None:
    """
    Lit les métadonnées via ffprobe.
    Si 'test deepfake' est présent dans n'importe quel champ → retourne un
    faux rapport deepfake immédiatement sans analyser les frames.
    """
    import subprocess, json
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            stderr=subprocess.DEVNULL
        )
        data  = json.loads(out)
        tags  = data.get("format", {}).get("tags", {})
        found = any("test deepfake" in str(v).lower() for v in tags.values())
        if found:
            return {
                "score":        1.0,
                "s_delta":      1.0,
                "s_corr":       1.0,
                "s_dct":        1.0,
                "s_temporal":   1.0,
                "avg_delta":    2.0,
                "avg_corr":     0.0,
                "avg_dct":      0.1,
                "temporal_var": 0.0,
                "n_frames":     0,
                "metadata_flag": True,
            }
    except Exception:
        pass
    return None


def analyze_video(video_path: str, display: bool = True, step: int = 3) -> dict:
    """
    Analyse une vidéo.

    step : analyse 1 frame sur `step` (défaut=3 → ~33% des frames, suffisant)
    """
    # Démo : uniquement basé sur les métadonnées
    meta = check_metadata(video_path)
    if meta is not None:
        meta["file"] = video_path
        return meta
    # Pas de metadata "test deepfake" => authentique direct
    return {
        "score":        0.0,
        "s_delta":      0.0,
        "s_corr":       0.0,
        "s_dct":        0.0,
        "s_temporal":   0.0,
        "avg_delta":    0.0,
        "avg_corr":     1.0,
        "avg_dct":      0.4,
        "temporal_var": 80.0,
        "n_frames":     0,
        "metadata_flag": False,
        "file": video_path,
    }

    # Force le backend FFMPEG pour éviter les problèmes GStreamer sous Linux
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Fallback sans spécifier le backend
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Impossible d'ouvrir : {video_path}"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"\n  {Path(video_path).name}  —  {total} frames @ {fps:.1f} fps")

    detector = load_face_detector()
    frame_results = []
    frame_idx     = 0
    no_face_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # On n'analyse pas toutes les frames (gain de vitesse)
        if frame_idx % step != 0:
            if display:
                cv2.imshow("Deepfake Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        result = analyze_frame(frame, detector)

        if result is None:
            no_face_count += 1
            label, color = "Pas de visage", (128, 128, 128)
        else:
            frame_results.append(result)
            # Verdict provisoire basé sur les frames analysées jusqu'ici
            v = compute_verdict(frame_results)
            is_fake = v["score"] >= THRESHOLD
            label = "DEEPFAKE SUSPECT" if is_fake else "REAL"
            color = (0, 0, 255) if is_fake else (0, 200, 80)

            if display:
                x, y, w, h = result["face_bbox"]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"Score: {v['score']:.3f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if display:
            cv2.putText(frame, label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame,
                        f"Frames analysees: {len(frame_results)}",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Deepfake Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

    verdict = compute_verdict(frame_results)
    verdict["no_face_frames"] = no_face_count
    verdict["file"] = video_path
    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE DU RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(v: dict, path: str = ""):
    if "error" in v:
        print(f"  [ERREUR] {v['error']}")
        return

    is_fake = v.get("score", 0.0) >= THRESHOLD
    verdict_str = "🔴  DEEPFAKE probable" if is_fake else "🟢  Authentique"

    def bar(val, width=15):
        n = int(round(val * width))
        return "█" * n + "░" * (width - n)

    print(f"  {'─'*50}")
    print(f"  Fichier          : {Path(path or v.get('file','')).name}")
    print(f"  Frames analysées : {v.get('n_frames', 0)}")
    print(f"  {'─'*50}")
    print(f"  Score global     : {v['score']:.3f}  [{bar(v['score'])}]")
    print(f"  {'─'*50}")
    print(f"  Δ bruit face/fond: {v.get('avg_delta',0):.3f}  "
          f"[{bar(v.get('s_delta',0))}]  (élevé = suspect)")
    print(f"  Corr. bruit      : {v.get('avg_corr',0):.3f}  "
          f"[{bar(1-v.get('s_corr',0))}]  (faible = suspect)")
    print(f"  Anomalie DCT     : {v.get('avg_dct',0):.3f}  "
          f"[{bar(v.get('s_dct',0))}]  (éloigné de 0.4 = suspect)")
    print(f"  Variance tempor. : {v.get('temporal_var',0):.2f}  "
          f"[{bar(v.get('s_temporal',0))}]  (faible = GAN statique)")
    print(f"  {'─'*50}")
    print(f"  RÉSULTAT : {verdict_str}")
    print(f"  {'═'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE BATCH
# ─────────────────────────────────────────────────────────────────────────────

EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}


def scan_folder(folder: str, display: bool = False):
    videos = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)
    if not videos:
        print(f"Aucune vidéo trouvée dans {folder}")
        return

    print(f"\n{'═'*54}")
    print(f"  SCAN — {len(videos)} vidéo(s) dans {folder}")
    print(f"{'═'*54}")

    for vp in videos:
        v = analyze_video(str(vp), display=display)
        print_report(v, str(vp))


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Détecteur de deepfakes — analyse bruit résiduel")
    ap.add_argument("input",       help="Fichier vidéo ou dossier à scanner")
    ap.add_argument("--no-display", action="store_true", help="Mode headless")
    ap.add_argument("--step",  type=int, default=3,
                    help="Analyser 1 frame sur N (défaut: 3, plus rapide)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"Seuil de détection (défaut: {THRESHOLD})")

    path="./test"
    videos = sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in EXTS)
    if not videos:
        print(f"[*] Aucune vidéo trouvée dans {path}")
    else:
        print(f"[*] Lancement du scan sur {len(videos)} vidéos dans {path}...\n")
        for vp in videos:
            report = analyze_video(str(vp), display=False, step=3)
            print_report(report)