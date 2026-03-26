"""
generate_test_videos.py
=======================
Génère un corpus de vidéos synthétiques de test pour valider le détecteur IA.

Deux catégories :
  - "synthetic" : vidéos avec artefacts FFT caractéristiques des GAN
                  (checkerboard pattern, pics de Nyquist, basse variance temporelle)
  - "natural"   : vidéos avec distribution fréquentielle organique (1/f, bruit naturel)

Pas de biométrie ni de deepfakes — les signaux testés sont les artefacts
mathématiques des couches de convolution transposée, pas les visages.

Usage :
    python generate_test_videos.py --output ./test_videos --n 10
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEURS DE FRAMES
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticFrameGenerator:
    """
    Simule les artefacts fréquentiels produits par les réseaux génératifs.

    Technique :
      1. Génère un signal de base en espace fréquentiel avec pics aux fréquences
         de Nyquist partielles (comme une convolution transposée stride=2).
      2. Applique IFFT pour obtenir une image spatiale avec damier invisible.
      3. Ajoute une scène visuelle par-dessus pour que la vidéo soit réaliste.
    """

    def __init__(self, width: int = 256, height: int = 256):
        self.W = width
        self.H = height

    def make_frame(self, t: float, variant: int = 0) -> np.ndarray:
        """Génère une frame avec artefacts GAN à l'instant t."""
        # Scène de base : dégradé animé + formes géométriques
        frame = self._base_scene(t, variant)
        # Injection d'artefacts fréquentiels (checkerboard invisible)
        frame = self._inject_checkerboard_artifact(frame, strength=0.04)
        # Basse variance temporelle : l'artefact est quasi-statique
        frame = self._add_static_spectral_bias(frame)
        return frame

    def _base_scene(self, t: float, variant: int) -> np.ndarray:
        """Scène visuelle : formes géométriques animées, abstraites."""
        frame = np.zeros((self.H, self.W, 3), dtype=np.float32)
        # Fond : dégradé sinusoïdal lent
        x = np.linspace(0, 2 * math.pi, self.W)
        y = np.linspace(0, 2 * math.pi, self.H)
        XX, YY = np.meshgrid(x, y)
        bg = 0.4 + 0.15 * np.sin(XX * (1 + variant * 0.3) + t * 0.5) * \
                          np.cos(YY * (1 + variant * 0.2) + t * 0.3)
        frame[:, :, 0] = bg
        frame[:, :, 1] = bg * 0.8 + 0.1 * np.cos(XX + t)
        frame[:, :, 2] = 1.0 - bg * 0.6
        # Cercle animé
        cx = int(self.W // 2 + self.W // 4 * math.sin(t * 0.7))
        cy = int(self.H // 2 + self.H // 4 * math.cos(t * 0.5))
        r = self.W // 8 + variant * 5
        cv2.circle(frame, (cx, cy), r, (0.9, 0.6, 0.2), -1)
        cv2.rectangle(frame,
                      (self.W // 4 + variant * 3, self.H // 4),
                      (3 * self.W // 4, 3 * self.H // 4),
                      (0.2, 0.7, 0.8), 2)
        return np.clip(frame, 0, 1)

    def _inject_checkerboard_artifact(self, frame: np.ndarray, strength: float) -> np.ndarray:
        """
        Injecte l'artefact de damier dans le domaine fréquentiel.
        Simule le pattern produit par ConvTranspose2d(stride=2).
        """
        result = np.copy(frame)
        for ch in range(3):
            F = np.fft.fft2(frame[:, :, ch])
            Fshift = np.fft.fftshift(F)
            H, W = Fshift.shape
            # Pics aux coins du spectre centré (fréquences N/2)
            for (r, c) in [
                (H // 2,     W // 2),      # DC — skip
                (H // 4,     W // 4),      # quart Nyquist
                (3 * H // 4, W // 4),
                (H // 4,     3 * W // 4),
                (3 * H // 4, 3 * W // 4),
            ]:
                if (r, c) == (H // 2, W // 2):
                    continue
                Fshift[r - 1:r + 2, c - 1:c + 2] += strength * np.abs(Fshift[H // 2, W // 2])
            F_back = np.fft.ifftshift(Fshift)
            result[:, :, ch] = np.real(np.fft.ifft2(F_back))
        return np.clip(result, 0, 1)

    def _add_static_spectral_bias(self, frame: np.ndarray) -> np.ndarray:
        """Ajoute un biais spectral fixe (faible variance temporelle)."""
        bias = np.random.randn(self.H, self.W, 3).astype(np.float32) * 0.002
        return np.clip(frame + bias, 0, 1)


class NaturalFrameGenerator:
    """
    Génère des frames avec une distribution fréquentielle naturelle (loi 1/f).
    Les scènes naturelles ont un spectre de puissance décroissant en 1/f²
    (loi de Brown / rose noise en 2D).
    """

    def __init__(self, width: int = 256, height: int = 256):
        self.W = width
        self.H = height
        self._noise_seed = random.randint(0, 10000)

    def make_frame(self, t: float, variant: int = 0) -> np.ndarray:
        frame = self._natural_scene(t, variant)
        frame = self._add_pink_noise(frame, alpha=2.0)
        return frame

    def _natural_scene(self, t: float, variant: int) -> np.ndarray:
        """Scène naturelle : textures et mouvements organiques."""
        frame = np.zeros((self.H, self.W, 3), dtype=np.float32)
        x = np.linspace(0, math.pi * 3, self.W)
        y = np.linspace(0, math.pi * 3, self.H)
        XX, YY = np.meshgrid(x, y)
        # Texture "sky-like" avec turbulence multi-échelle
        turbulence = sum(
            (1 / (2 ** i)) * np.sin(2 ** i * XX + t * (0.1 + 0.05 * i) + variant)
            * np.cos(2 ** i * YY + t * (0.08 + 0.03 * i))
            for i in range(1, 6)
        )
        turbulence = (turbulence - turbulence.min()) / (turbulence.max() - turbulence.min() + 1e-9)
        # Palette "paysage"
        frame[:, :, 0] = np.clip(turbulence * 0.5 + 0.3, 0, 1)
        frame[:, :, 1] = np.clip(turbulence * 0.7 + 0.2, 0, 1)
        frame[:, :, 2] = np.clip((1 - turbulence) * 0.6 + 0.1, 0, 1)
        return frame

    def _add_pink_noise(self, frame: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """
        Ajoute un bruit rose (1/f^alpha) — signature spectrale des scènes naturelles.
        """
        result = np.copy(frame)
        rng = np.random.default_rng(self._noise_seed)
        for ch in range(3):
            noise = rng.standard_normal((self.H, self.W))
            F = np.fft.fft2(noise)
            Fshift = np.fft.fftshift(F)
            H, W = Fshift.shape
            u = np.fft.fftfreq(W) * W
            v = np.fft.fftfreq(H) * H
            UU, VV = np.meshgrid(u, v)
            freq = np.sqrt(UU ** 2 + VV ** 2)
            freq[H // 2, W // 2] = 1  # évite div/0 au DC
            pink_filter = 1.0 / (freq ** (alpha / 2) + 1e-9)
            Fshift_pink = Fshift * np.fft.fftshift(pink_filter)
            pink_noise = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift_pink)))
            pink_norm = pink_noise / (pink_noise.std() + 1e-9) * 0.015
            result[:, :, ch] = np.clip(result[:, :, ch] + pink_norm, 0, 1)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTEUR DE VIDÉOS
# ─────────────────────────────────────────────────────────────────────────────

def build_video(
    generator,
    output_path: str,
    duration_sec: float = 4.0,
    fps: int = 24,
    width: int = 256,
    height: int = 256,
    variant: int = 0,
):
    """Encode une vidéo MP4 à partir d'un générateur de frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    n_frames = int(duration_sec * fps)
    for i in range(n_frames):
        t = i / fps
        frame_f32 = generator.make_frame(t, variant=variant)
        frame_u8 = (np.clip(frame_f32, 0, 1) * 255).astype(np.uint8)
        out.write(frame_u8)
    out.release()


def generate_corpus(output_dir: str, n_per_class: int = 5) -> dict:
    """
    Génère le corpus de test.

    Retourne un manifest JSON avec les chemins et les vraies étiquettes.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic").mkdir(exist_ok=True)
    (out / "natural").mkdir(exist_ok=True)

    manifest = {"synthetic": [], "natural": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print(f"\n{'─'*50}")
    print(f"  Génération du corpus de test ({n_per_class * 2} vidéos)")
    print(f"  Sortie : {out.resolve()}")
    print(f"{'─'*50}")

    # Vidéos synthétiques (artefacts GAN injectés)
    print(f"\n[SYNTHÉTIQUES] {n_per_class} vidéos avec artefacts FFT…")
    for i in range(n_per_class):
        path = str(out / "synthetic" / f"synthetic_{i:02d}.mp4")
        gen = SyntheticFrameGenerator(256, 256)
        build_video(gen, path, duration_sec=3.0, fps=24, variant=i)
        manifest["synthetic"].append(path)
        print(f"  ✓ {Path(path).name}")

    # Vidéos naturelles (bruit rose, pas d'artefacts)
    print(f"\n[NATURELLES] {n_per_class} vidéos avec distribution 1/f…")
    for i in range(n_per_class):
        path = str(out / "natural" / f"natural_{i:02d}.mp4")
        gen = NaturalFrameGenerator(256, 256)
        build_video(gen, path, duration_sec=3.0, fps=24, variant=i)
        manifest["natural"].append(path)
        print(f"  ✓ {Path(path).name}")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest → {manifest_path}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère le corpus de test pour ai_video_detector")
    parser.add_argument("--output", default="./test_videos", help="Dossier de sortie")
    parser.add_argument("--n", type=int, default=5, help="Nombre de vidéos par classe")
    args = parser.parse_args()
    generate_corpus(args.output, n_per_class=args.n)
