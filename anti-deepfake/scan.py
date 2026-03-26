"""
scan.py
=======
Scanne toutes les vidéos dans ./test et affiche Deepfake ou Not Deepfake.

Usage :
    python scan.py
    python scan.py --folder ./mes_videos --frames 30
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from ai_video_detector import VideoAIDetector
except ImportError:
    print("[ERREUR] ai_video_detector.py introuvable dans le dossier courant.")
    sys.exit(1)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}

DEEPFAKE_THRESHOLD = 0.50  # score >= seuil → Deepfake


def scan(folder: str, n_frames: int = 40):
    videos = sorted(
        p for p in Path(folder).rglob("*")
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"Aucune vidéo trouvée dans {folder}")
        return

    print(f"\n{'═'*58}")
    print(f"  SCAN — {len(videos)} vidéo(s) dans {folder}")
    print(f"{'─'*58}")
    print(f"  {'Fichier':<30} {'Score':>6}  Verdict")
    print(f"  {'─'*30} {'─'*6}  {'─'*12}")

    results = []
    for path in videos:
        t0 = time.time()
        try:
            report = VideoAIDetector(str(path)).analyze(n_frames=n_frames)
            score  = report.final_score
            is_df  = score >= DEEPFAKE_THRESHOLD

            verdict_str = "\033[91m🔴 DEEPFAKE\033[0m" if is_df else "\033[92m🟢 Not Deepfake\033[0m"
            elapsed = time.time() - t0

            print(f"  {path.name:<30} {score:>5.2f}  {verdict_str}  [{elapsed:.1f}s]")
            results.append({"file": path.name, "score": score, "deepfake": is_df})

        except Exception as e:
            print(f"  {path.name:<30}  [ERREUR] {e}")

    # Résumé
    n_df  = sum(1 for r in results if r["deepfake"])
    n_ok  = len(results) - n_df
    print(f"{'─'*58}")
    print(f"  Résultat : {n_df} deepfake(s)  /  {n_ok} authentique(s)  sur {len(results)} vidéo(s)")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan deepfake d'un dossier vidéo")
    parser.add_argument("--folder", default="./test", help="Dossier à scanner (défaut: ./test)")
    parser.add_argument("--frames", type=int, default=40, help="Frames analysées par vidéo (défaut: 40)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Seuil de détection [0-1] (défaut: 0.50)")
    args = parser.parse_args()

    DEEPFAKE_THRESHOLD = args.threshold
    scan(args.folder, n_frames=args.frames)
