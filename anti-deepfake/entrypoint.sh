#!/bin/sh
# entrypoint.sh — Routeur de commandes pour le conteneur

set -e

case "$1" in

  # ── Analyse d'une vidéo unique ──────────────────────────────────────────
  analyze)
    shift
    VIDEO="$1"
    FRAMES="${2:-60}"
    if [ -z "$VIDEO" ]; then
      echo "Usage : docker run ... analyze <video_path> [n_frames=60]"
      exit 1
    fi
    python ai_video_detector.py "$VIDEO" "$FRAMES"
    ;;

  # ── Suite de tests complète ─────────────────────────────────────────────
  test)
    shift
    N="${1:-5}"
    FRAMES="${2:-40}"
    echo "=== Lancement de la suite de tests (n=$N vidéos/classe, frames=$FRAMES) ==="
    python test_detector.py \
      --corpus /data/corpus \
      --n "$N" \
      --frames "$FRAMES" \
      --output /data/output/test_results.json
    ;;

  # ── Génération de corpus uniquement ─────────────────────────────────────
  generate)
    shift
    N="${1:-5}"
    echo "=== Génération du corpus de test (n=$N vidéos/classe) ==="
    python generate_test_videos.py --output /data/corpus --n "$N"
    ;;

  # ── Analyse d'un dossier complet ────────────────────────────────────────
  batch)
    shift
    FOLDER="${1:-/data/input}"
    FRAMES="${2:-40}"
    echo "=== Analyse batch de $FOLDER ==="
    python - <<EOF
from ai_video_detector import batch_analyze
reports = batch_analyze("$FOLDER", n_frames=$FRAMES)
print(f"\n{len(reports)} vidéos analysées.")
EOF
    ;;

  # ── Visualisation d'un spectre FFT ──────────────────────────────────────
  spectrum)
    shift
    VIDEO="$1"
    FRAME_IDX="${2:-0}"
    if [ -z "$VIDEO" ]; then
      echo "Usage : docker run ... spectrum <video_path> [frame_index=0]"
      exit 1
    fi
    python - <<EOF
from ai_video_detector import visualize_spectrum
out = visualize_spectrum("$VIDEO", frame_index=$FRAME_IDX, save_path="/data/output/spectrum.png")
print(f"Spectre exporté : {out}")
EOF
    ;;

  # ── Aide ─────────────────────────────────────────────────────────────────
  --help | help | "")
    echo ""
    echo "AI Video Detector — Docker Interface"
    echo "─────────────────────────────────────────────────"
    echo "  analyze  <video> [frames]    Analyse une vidéo"
    echo "  test     [n] [frames]        Suite de tests complète"
    echo "  generate [n]                 Génère corpus de test"
    echo "  batch    [folder] [frames]   Analyse un dossier"
    echo "  spectrum <video> [frame_idx] Exporte le spectre FFT"
    echo ""
    echo "Exemples :"
    echo "  docker run --rm -v \$(pwd)/videos:/data/input ai-video-detector analyze /data/input/clip.mp4"
    echo "  docker run --rm -v \$(pwd)/out:/data/output ai-video-detector test 10 60"
    echo ""
    ;;

  *)
    echo "[ERREUR] Commande inconnue : $1"
    echo "Lancez 'help' pour voir les commandes disponibles."
    exit 1
    ;;

esac
