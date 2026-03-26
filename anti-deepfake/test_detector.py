"""
test_detector.py
================
Suite de tests pour ai_video_detector.py.

Workflow :
  1. Génère le corpus si absent
  2. Lance le détecteur sur chaque vidéo
  3. Calcule précision, rappel, F1, AUC approximatif
  4. Affiche un rapport comparatif et sauvegarde les résultats

Usage :
    python test_detector.py
    python test_detector.py --corpus ./test_videos --n 10 --frames 40
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Imports locaux
try:
    from ai_video_detector import VideoAIDetector, DetectionReport
    from generate_test_videos import generate_corpus
except ImportError as e:
    print(f"[ERREUR] Import manquant : {e}")
    print("Assurez-vous que ai_video_detector.py et generate_test_videos.py sont dans le même dossier.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def _label(verdict: str) -> str:
    colors = {"synthetic": "\033[91m", "natural": "\033[92m", "uncertain": "\033[93m"}
    reset = "\033[0m"
    return f"{colors.get(verdict, '')}{verdict.upper():<9}{reset}"


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """
    Calcule précision, rappel, F1 et AUC (approximation trapèze).
    Positif = synthétique.
    """
    tp = sum(1 for r in results if r["true"] == "synthetic" and r["verdict"] == "synthetic")
    fp = sum(1 for r in results if r["true"] == "natural"   and r["verdict"] == "synthetic")
    fn = sum(1 for r in results if r["true"] == "synthetic" and r["verdict"] != "synthetic")
    tn = sum(1 for r in results if r["true"] == "natural"   and r["verdict"] != "synthetic")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    # AUC approximatif via tri par score et calcul trapèze
    sorted_r = sorted(results, key=lambda x: x["score"], reverse=True)
    tpr_list, fpr_list = [0.0], [0.0]
    n_pos = sum(1 for r in results if r["true"] == "synthetic")
    n_neg = len(results) - n_pos
    tp_c = fp_c = 0
    for r in sorted_r:
        if r["true"] == "synthetic":
            tp_c += 1
        else:
            fp_c += 1
        tpr_list.append(tp_c / n_pos if n_pos else 0)
        fpr_list.append(fp_c / n_neg if n_neg else 0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    auc = float(np.trapz(tpr_list, fpr_list))

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy, "auc": abs(auc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(corpus_dir: str, n_frames: int = 40) -> list[dict]:
    """
    Lance le détecteur sur toutes les vidéos du corpus.
    Retourne une liste de dicts avec les résultats et les vraies étiquettes.
    """
    manifest_path = Path(corpus_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERREUR] manifest.json introuvable dans {corpus_dir}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    all_videos = [
        (p, "synthetic") for p in manifest.get("synthetic", [])
    ] + [
        (p, "natural") for p in manifest.get("natural", [])
    ]

    print(f"\n{'═'*60}")
    print(f"  TEST SUITE — {len(all_videos)} vidéos — {n_frames} frames/vidéo")
    print(f"{'═'*60}\n")
    print(f"  {'Fichier':<26} {'Vrai label':<12} {'Prédit':<11} {'Score':>6}  {'Confiance'}")
    print(f"  {'─'*26} {'─'*12} {'─'*11} {'─'*6}  {'─'*9}")

    results = []
    for video_path, true_label in all_videos:
        path = Path(video_path)
        if not path.exists():
            print(f"  [SKIP] {path.name} — introuvable")
            continue

        t0 = time.time()
        try:
            detector = VideoAIDetector(video_path)
            report = detector.analyze(n_frames=n_frames)
            elapsed = time.time() - t0

            verdict = report.verdict
            score = report.final_score
            confidence = report.confidence
            correct = "✓" if (
                (true_label == "synthetic" and verdict == "synthetic") or
                (true_label == "natural"   and verdict == "natural")
            ) else "✗"

            print(
                f"  {path.name:<26} {true_label:<12} {_label(verdict)} "
                f"{score:>5.2f}  {confidence:<9} {correct}  [{elapsed:.1f}s]"
            )

            results.append({
                "file": str(path),
                "true": true_label,
                "verdict": verdict,
                "score": score,
                "confidence": confidence,
                "prov_score": report.provenance.provenance_score,
                "freq_score": report.frequency.frequency_score,
                "flags": report.flags,
                "elapsed_s": round(elapsed, 2),
            })

        except Exception as e:
            print(f"  [ERREUR] {path.name}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE DU RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    if not results:
        print("\n[ERREUR] Aucun résultat à afficher.")
        return

    metrics = compute_metrics(results)

    print(f"\n{'═'*60}")
    print(f"  RAPPORT DE SYNTHÈSE")
    print(f"{'─'*60}")
    print(f"  Vidéos analysées  : {len(results)}")
    print(f"  Précision         : {metrics['precision']:.3f}  [{_bar(metrics['precision'])}]")
    print(f"  Rappel            : {metrics['recall']:.3f}  [{_bar(metrics['recall'])}]")
    print(f"  F1-score          : {metrics['f1']:.3f}  [{_bar(metrics['f1'])}]")
    print(f"  Accuracy          : {metrics['accuracy']:.3f}  [{_bar(metrics['accuracy'])}]")
    print(f"  AUC (approx.)     : {metrics['auc']:.3f}  [{_bar(metrics['auc'])}]")
    print(f"{'─'*60}")
    print(f"  Matrice de confusion (positif = synthétique)")
    print(f"                    Prédit synthetic  Prédit natural")
    print(f"  Vrai synthetic    TP={metrics['tp']:<15}  FN={metrics['fn']}")
    print(f"  Vrai natural      FP={metrics['fp']:<15}  TN={metrics['tn']}")
    print(f"{'─'*60}")

    # Analyse par axe
    synth_results = [r for r in results if r["true"] == "synthetic"]
    natur_results = [r for r in results if r["true"] == "natural"]

    if synth_results:
        avg_prov_s = np.mean([r["prov_score"] for r in synth_results])
        avg_freq_s = np.mean([r["freq_score"] for r in synth_results])
        print(f"\n  Score moyen — vidéos SYNTHÉTIQUES")
        print(f"    Axe provenance : {avg_prov_s:.3f}  [{_bar(avg_prov_s)}]")
        print(f"    Axe fréquence  : {avg_freq_s:.3f}  [{_bar(avg_freq_s)}]")

    if natur_results:
        avg_prov_n = np.mean([r["prov_score"] for r in natur_results])
        avg_freq_n = np.mean([r["freq_score"] for r in natur_results])
        print(f"\n  Score moyen — vidéos NATURELLES")
        print(f"    Axe provenance : {avg_prov_n:.3f}  [{_bar(avg_prov_n)}]")
        print(f"    Axe fréquence  : {avg_freq_n:.3f}  [{_bar(avg_freq_n)}]")

    # Flags les plus fréquents
    all_flags: list[str] = []
    for r in results:
        all_flags.extend(r.get("flags", []))
    if all_flags:
        from collections import Counter
        top_flags = Counter(all_flags).most_common(5)
        print(f"\n  Flags les plus fréquents :")
        for flag, count in top_flags:
            print(f"    [{count:>2}x]  {flag}")

    print(f"\n  Temps moyen par vidéo : {np.mean([r['elapsed_s'] for r in results]):.1f}s")
    print(f"{'═'*60}\n")


def save_results(results: list[dict], output_path: str = "test_results.json"):
    """Sauvegarde les résultats bruts + métriques en JSON."""
    metrics = compute_metrics(results)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_videos": len(results),
        "metrics": metrics,
        "per_video": results,
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"  Résultats sauvegardés → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests du détecteur IA vidéo")
    parser.add_argument("--corpus", default="./test_videos", help="Dossier du corpus de test")
    parser.add_argument("--n", type=int, default=5, help="Vidéos par classe à générer si absent")
    parser.add_argument("--frames", type=int, default=40, help="Frames à analyser par vidéo")
    parser.add_argument("--output", default="test_results.json", help="Fichier JSON de sortie")
    parser.add_argument("--regen", action="store_true", help="Régénère le corpus même s'il existe")
    args = parser.parse_args()

    # Génération du corpus si nécessaire
    manifest_path = Path(args.corpus) / "manifest.json"
    if not manifest_path.exists() or args.regen:
        print("Génération du corpus de test…")
        generate_corpus(args.corpus, n_per_class=args.n)
    else:
        print(f"Corpus existant trouvé dans {args.corpus}")

    # Exécution des tests
    results = run_tests(args.corpus, n_frames=args.frames)
    print_report(results)
    save_results(results, args.output)
