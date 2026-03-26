"""
ai_video_detector.py
====================
Module de détection de contenu vidéo généré par IA.

Deux axes d'analyse :
  1. Métadonnées & provenance  — C2PA, magic bytes, containers binaires
  2. Analyse fréquentielle     — FFT 2D, artefacts d'upsampling, cohérence temporelle

Dépendances :
    pip install opencv-python numpy scipy av mutagen tqdm

Usage :
    from ai_video_detector import VideoAIDetector
    detector = VideoAIDetector("video.mp4")
    report   = detector.analyze(n_frames=60)
    print(report)
"""

import struct
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProvenanceReport:
    """Résultat de l'analyse de provenance (Axe 1)."""
    c2pa_found: bool = False
    c2pa_claim_generator: Optional[str] = None
    c2pa_signature_valid: Optional[bool] = None   # None = non vérifié
    synthid_heuristic: bool = False               # heuristique LSB / haute-fréquence
    known_ai_metadata_tags: list = field(default_factory=list)
    container_anomalies: list = field(default_factory=list)
    magic_bytes_summary: dict = field(default_factory=dict)
    provenance_score: float = 0.0                 # [0, 1] — 1 = certain synthétique


@dataclass
class FrequencyReport:
    """Résultat de l'analyse fréquentielle (Axe 2)."""
    n_frames_analyzed: int = 0
    mean_checkerboard_energy: float = 0.0
    std_checkerboard_energy: float = 0.0
    temporal_variance_ratio: float = 0.0
    spectral_flatness_mean: float = 0.0
    peak_symmetry_score: float = 0.0              # symétrie des pics FFT
    anomalous_frames_pct: float = 0.0             # % de frames avec pics aberrants
    frequency_score: float = 0.0                  # [0, 1]


@dataclass
class DetectionReport:
    """Rapport final fusionné."""
    file_path: str = ""
    file_sha256: str = ""
    provenance: ProvenanceReport = field(default_factory=ProvenanceReport)
    frequency: FrequencyReport = field(default_factory=FrequencyReport)
    final_score: float = 0.0          # score fusionné [0, 1]
    confidence: str = "low"           # low / medium / high
    verdict: str = "unknown"          # natural / synthetic / uncertain
    flags: list = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def __str__(self) -> str:
        bar = "█" * int(self.final_score * 20) + "░" * (20 - int(self.final_score * 20))
        return (
            f"\n{'═'*56}\n"
            f"  AI VIDEO DETECTOR — {Path(self.file_path).name}\n"
            f"{'─'*56}\n"
            f"  Score synthétique : [{bar}] {self.final_score:.2f}\n"
            f"  Verdict           : {self.verdict.upper()}  ({self.confidence} confidence)\n"
            f"{'─'*56}\n"
            f"  Provenance score  : {self.provenance.provenance_score:.2f}\n"
            f"    C2PA trouvé     : {self.provenance.c2pa_found}\n"
            f"    Générateur      : {self.provenance.c2pa_claim_generator or 'inconnu'}\n"
            f"    Tags AI connus  : {self.provenance.known_ai_metadata_tags}\n"
            f"  Fréquence score   : {self.frequency.frequency_score:.2f}\n"
            f"    Frames analysées: {self.frequency.n_frames_analyzed}\n"
            f"    Énergie damier  : {self.frequency.mean_checkerboard_energy:.4f}\n"
            f"    Var. temporelle : {self.frequency.temporal_variance_ratio:.4f}\n"
            f"    Frames anom.    : {self.frequency.anomalous_frames_pct:.1f}%\n"
            f"  Flags             : {self.flags or ['aucun']}\n"
            f"  SHA-256           : {self.file_sha256[:16]}…\n"
            f"{'═'*56}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AXE 1 — ANALYSE MÉTADONNÉES & PROVENANCE
# ─────────────────────────────────────────────────────────────────────────────

class ProvenanceAnalyzer:
    """
    Analyse les métadonnées, magic bytes et signatures de provenance.

    C2PA spec : https://c2pa.org/specifications/specifications/1.3/specs/C2PA_Specification.html
    Le manifest C2PA est stocké dans une 'uuid box' MP4 dont le type UUID est :
      d8fec3d6-1d23-4285-b4a0-d6e4b037f4c2  (C2PA)
    """

    # UUID C2PA en bytes (big-endian)
    C2PA_UUID = bytes.fromhex("d8fec3d61d2342 85b4a0d6e4b037f4c2".replace(" ", ""))

    # Tags de métadonnées indiquant une génération IA
    KNOWN_AI_TAGS = {
        b"DALL-E", b"Stable Diffusion", b"Midjourney", b"Sora",
        b"Runway", b"Pika", b"Adobe Firefly", b"Synthesia",
        b"generator\x00ai", b"ai_generated", b"synthetic_media",
        b"adobe:dc:source", b"xmp:CreateTool",
    }

    # Patterns magic bytes de conteneurs IA courants
    MAGIC_PATTERNS = {
        b"\x00\x00\x00\x18ftypmp42": "MP4 (MPEG-4 Part 12)",
        b"\x00\x00\x00\x1cftypisom": "MP4 (ISO Base Media)",
        b"\x1aE\xdf\xa3": "Matroska/WebM",
        b"RIFF": "AVI/WAV",
        b"OggS": "OGG (Theora/VP8)",
        b"\x00\x00\x01\xb3": "MPEG-1/2",
    }

    def __init__(self, file_path: str):
        self.path = Path(file_path)

    def analyze(self) -> ProvenanceReport:
        report = ProvenanceReport()
        raw = self._read_binary()
        report.magic_bytes_summary = self._identify_container(raw)
        report.known_ai_metadata_tags = self._scan_ai_tags(raw)
        report.c2pa_found, report.c2pa_claim_generator = self._parse_c2pa(raw)
        report.synthid_heuristic = self._heuristic_synthid(raw)
        report.container_anomalies = self._detect_container_anomalies(raw)
        report.provenance_score = self._compute_score(report)
        return report

    def _read_binary(self, max_bytes: int = 20 * 1024 * 1024) -> bytes:
        """Lit les N premiers Mo du fichier (header + premières boxes)."""
        with open(self.path, "rb") as f:
            return f.read(max_bytes)

    def _identify_container(self, raw: bytes) -> dict:
        for magic, label in self.MAGIC_PATTERNS.items():
            if raw[:len(magic)] == magic or raw[:4] == magic[:4]:
                return {"format": label, "magic": raw[:8].hex()}
        return {"format": "inconnu", "magic": raw[:8].hex()}

    def _scan_ai_tags(self, raw: bytes) -> list[str]:
        """Recherche linéaire de signatures textuelles dans le flux binaire."""
        found = []
        raw_lower = raw.lower()
        for tag in self.KNOWN_AI_TAGS:
            if tag.lower() in raw_lower:
                found.append(tag.decode("utf-8", errors="replace"))
        # Recherche de patterns JSON C2PA / XMP
        for pattern in [b'"generator"', b'"software"', b"ai_labels", b"ContentCredentials"]:
            if pattern in raw:
                found.append(pattern.decode())
        return list(set(found))

    def _parse_c2pa(self, raw: bytes) -> tuple[bool, Optional[str]]:
        """
        Parse les boxes ISO BMFF pour trouver la uuid box C2PA.
        Structure d'une box : [size:4][type:4][...data...]
        Pour une uuid box   : [size:4]["uuid":4][uuid:16][data...]
        """
        offset = 0
        while offset < len(raw) - 8:
            try:
                box_size = struct.unpack_from(">I", raw, offset)[0]
                box_type = raw[offset + 4: offset + 8]
                if box_size < 8:
                    break
                if box_type == b"uuid":
                    uuid_bytes = raw[offset + 8: offset + 24]
                    if uuid_bytes == self.C2PA_UUID:
                        # Données C2PA trouvées — tentative de lecture du claim JSON
                        payload = raw[offset + 24: offset + box_size]
                        generator = self._extract_c2pa_generator(payload)
                        return True, generator
                offset += box_size
            except struct.error:
                break
        # Recherche heuristique si pas de uuid box standard
        if b"ContentCredentials" in raw or b"c2pa" in raw:
            return True, "heuristique (pas de box standard)"
        return False, None

    def _extract_c2pa_generator(self, payload: bytes) -> Optional[str]:
        """Tente d'extraire le champ 'generator' du manifest CBOR/JSON C2PA."""
        try:
            # Cherche un fragment JSON dans le payload
            start = payload.find(b"{")
            if start == -1:
                return None
            fragment = payload[start:]
            end = fragment.rfind(b"}") + 1
            data = json.loads(fragment[:end])
            # Navigation dans la structure C2PA claim
            for key in ["claim_generator", "generator", "software"]:
                if key in data:
                    return str(data[key])
            return "C2PA (générateur non lisible)"
        except Exception:
            return "C2PA (payload non décodable)"

    def _heuristic_synthid(self, raw: bytes) -> bool:
        """
        Heuristique SynthID : détecte une distribution LSB anormalement uniforme
        dans les données brutes du flux (signature de watermarking invisible).
        Note : une détection complète nécessite la clé privée DeepMind.
        """
        # Analyse sur un échantillon des données vidéo (zone centrale)
        sample_start = len(raw) // 4
        sample = raw[sample_start: sample_start + 65536]
        if not sample:
            return False
        lsb_array = np.frombuffer(sample, dtype=np.uint8) & 1
        # Dans une vidéo naturelle, les LSBs sont ~aléatoires (entropie ~1 bit)
        # Un watermark périodique introduit une légère corrélation
        chi2, p_value = stats.chisquare([np.sum(lsb_array == 0), np.sum(lsb_array == 1)])
        # p_value très petit → distribution LSB suspecte
        return bool(p_value < 0.001 and chi2 > 20)

    def _detect_container_anomalies(self, raw: bytes) -> list[str]:
        """Détecte les anomalies structurelles dans les boxes ISO BMFF."""
        anomalies = []
        # Vérification de l'ordre des boxes (ftyp doit être en premier)
        if len(raw) >= 8 and raw[4:8] != b"ftyp":
            anomalies.append("ftyp box absente en tête — structure atypique")
        # Présence de boxes inconnues / non-standard
        offset, unknown_count = 0, 0
        while offset < min(len(raw) - 8, 512 * 1024):
            try:
                box_size = struct.unpack_from(">I", raw, offset)[0]
                box_type = raw[offset + 4: offset + 8]
                standard = {b"ftyp", b"moov", b"mdat", b"free", b"skip",
                            b"uuid", b"moof", b"mfra", b"udta", b"meta"}
                if box_type not in standard and box_type.isalpha():
                    unknown_count += 1
                if box_size < 8:
                    break
                offset += box_size
            except struct.error:
                break
        if unknown_count > 3:
            anomalies.append(f"{unknown_count} boxes non-standard détectées")
        return anomalies

    def _compute_score(self, r: ProvenanceReport) -> float:
        score = 0.0
        if r.c2pa_found:
            score += 0.6  # Présence C2PA est le signal le plus fort
        if r.known_ai_metadata_tags:
            score += min(0.3, len(r.known_ai_metadata_tags) * 0.08)
        if r.synthid_heuristic:
            score += 0.2
        if r.container_anomalies:
            score += 0.05
        return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# AXE 2 — ANALYSE FRÉQUENTIELLE (FFT)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAnalyzer:
    """
    Détecte les artefacts fréquentiels caractéristiques des générateurs IA.

    Les réseaux de neurones génératifs (GAN, diffusion) utilisent des couches
    de convolution transposée qui introduisent des pics d'énergie périodiques
    dans le spectre de Fourier — notamment le motif en damier (checkerboard)
    à N/2 en u et v (fréquences de Nyquist partielles).

    Méthode :
      1. Extraction de N frames équidistantes
      2. FFT 2D sur chaque frame (canal Y de YCrCb)
      3. Mesure de l'énergie dans les zones suspectes
      4. Analyse de la cohérence temporelle (variance inter-frames)
    """

    # Zones spectrales des artefacts d'upsampling (normalisées [0,1])
    # Damier 2x : pics à (0.5, 0.5), (0.5, 0), (0, 0.5)
    CHECKERBOARD_ZONES = [
        (0.47, 0.53, 0.47, 0.53),  # centre Nyquist  (u, u)
        (0.47, 0.53, -0.03, 0.03), # axe u = N/2
        (-0.03, 0.03, 0.47, 0.53), # axe v = N/2
    ]

    def __init__(self, file_path: str, resize_to: tuple = (256, 256)):
        self.path = file_path
        self.resize_to = resize_to

    def analyze(self, n_frames: int = 60, sample_interval: int = 1) -> FrequencyReport:
        report = FrequencyReport()
        frames = self._extract_frames(n_frames)
        if not frames:
            warnings.warn("Aucune frame extraite — vérifier le fichier vidéo.")
            return report

        energies, flatnesses, peak_scores = [], [], []

        for frame in tqdm(frames, desc="Analyse FFT", unit="frame", leave=False):
            spectrum = self._compute_fft(frame)
            cb_energy = self._checkerboard_energy(spectrum)
            flatness = self._spectral_flatness(spectrum)
            p_score = self._peak_symmetry(spectrum)
            energies.append(cb_energy)
            flatnesses.append(flatness)
            peak_scores.append(p_score)

        energies_arr = np.array(energies)
        # Seuil d'anomalie : médiane + 2.5 × MAD
        median_e = float(np.median(energies_arr))
        mad_e = float(np.median(np.abs(energies_arr - median_e)))
        threshold = median_e + 2.5 * mad_e
        anomalous = float(np.mean(energies_arr > threshold) * 100)

        # Cohérence temporelle : ratio variance/moyenne (faible = signal quasi-stationnaire → synthétique)
        temporal_var_ratio = (float(np.std(energies_arr)) / float(np.mean(energies_arr) + 1e-9))

        report.n_frames_analyzed = len(frames)
        report.mean_checkerboard_energy = float(np.mean(energies_arr))
        report.std_checkerboard_energy = float(np.std(energies_arr))
        report.temporal_variance_ratio = temporal_var_ratio
        report.spectral_flatness_mean = float(np.mean(flatnesses))
        report.peak_symmetry_score = float(np.mean(peak_scores))
        report.anomalous_frames_pct = anomalous
        report.frequency_score = self._compute_score(report)
        return report

    def _extract_frames(self, n_frames: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(self.path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_resized = cv2.resize(frame, self.resize_to)
            # Canal Y de YCrCb — meilleure sensibilité aux artefacts de luminance
            ycrcb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YCrCb)
            frames.append(ycrcb[:, :, 0].astype(np.float32))
        cap.release()
        return frames

    def _compute_fft(self, frame: np.ndarray) -> np.ndarray:
        """FFT 2D centrée, retourne le log-spectre normalisé."""
        f = np.fft.fft2(frame)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # Log-compression pour réduire la dynamique
        log_mag = np.log1p(magnitude)
        return log_mag / (log_mag.max() + 1e-9)

    def _checkerboard_energy(self, spectrum: np.ndarray) -> float:
        """
        Mesure l'énergie dans les zones de Nyquist caractéristiques du damier.
        Les coordonnées normalisées sont converties en indices discrets.
        """
        H, W = spectrum.shape
        total_energy = 0.0
        baseline = float(np.mean(spectrum))
        for (u_lo, u_hi, v_lo, v_hi) in self.CHECKERBOARD_ZONES:
            # Conversion [−0.5, 0.5] → indices dans le spectre centré
            r0 = int((0.5 + v_lo) * H)
            r1 = int((0.5 + v_hi) * H)
            c0 = int((0.5 + u_lo) * W)
            c1 = int((0.5 + u_hi) * W)
            r0, r1 = max(0, r0), min(H, r1)
            c0, c1 = max(0, c0), min(W, c1)
            if r1 > r0 and c1 > c0:
                zone_mean = float(np.mean(spectrum[r0:r1, c0:c1]))
                # Énergie relative au bruit de fond
                total_energy += max(0.0, zone_mean - baseline)
        return total_energy

    def _spectral_flatness(self, spectrum: np.ndarray) -> float:
        """
        Mesure de Wiener — ratio de la moyenne géométrique sur la moyenne arithmétique.
        Un spectre plat (bruit blanc) → 1.0 ; un spectre en pics → proche de 0.
        Les vidéos IA ont souvent une flatness anormale dans certaines bandes.
        """
        flat = spectrum.flatten() + 1e-9
        geo_mean = float(np.exp(np.mean(np.log(flat))))
        arith_mean = float(np.mean(flat))
        return geo_mean / arith_mean

    def _peak_symmetry(self, spectrum: np.ndarray) -> float:
        """
        Mesure la symétrie quadrantielle du spectre.
        Les artefacts d'upsampling créent des pics symétriques par construction
        (propriété de la transformée de Fourier discrète d'un signal réel).
        Une symétrie >0.95 sur les hautes fréquences est suspecte.
        """
        H, W = spectrum.shape
        # Zone haute-fréquence (quartier extérieur)
        hf = spectrum[H // 4: 3 * H // 4, W // 4: 3 * W // 4]
        q1 = hf[: H // 4, : W // 4]
        q2 = hf[: H // 4, W // 4:]
        q3 = hf[H // 4:, : W // 4]
        q4 = hf[H // 4:, W // 4:]
        # Redimensionnement pour comparaison
        min_h = min(q.shape[0] for q in [q1, q2, q3, q4])
        min_w = min(q.shape[1] for q in [q1, q2, q3, q4])
        quads = [q[:min_h, :min_w] for q in [q1, q2, q3, q4]]
        # Corrélation entre quadrants (haute symétrie → pic artificiel)
        corr_scores = []
        for i in range(len(quads)):
            for j in range(i + 1, len(quads)):
                c = float(np.corrcoef(quads[i].flatten(), quads[j].flatten())[0, 1])
                corr_scores.append(abs(c) if not math.isnan(c) else 0.0)
        return float(np.mean(corr_scores)) if corr_scores else 0.0

    def _compute_score(self, r: FrequencyReport) -> float:
        score = 0.0
        # Énergie de damier élevée (seuil expérimental à calibrer)
        if r.mean_checkerboard_energy > 0.015:
            score += min(0.4, r.mean_checkerboard_energy * 15)
        # Faible variance temporelle = signature quasi-stationnaire
        if r.temporal_variance_ratio < 0.1:
            score += 0.3
        elif r.temporal_variance_ratio < 0.2:
            score += 0.15
        # Forte symétrie des pics = artefact periodique
        if r.peak_symmetry_score > 0.8:
            score += 0.2
        # Frames aberrantes
        if r.anomalous_frames_pct > 70:
            score += 0.1
        return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class VideoAIDetector:
    """
    Orchestrateur principal — combine l'analyse de provenance et l'analyse
    fréquentielle pour produire un rapport de détection unifié.

    Paramètres
    ----------
    file_path : str
        Chemin vers le fichier vidéo à analyser.
    provenance_weight : float
        Poids de l'axe provenance dans le score final (défaut : 0.6).
        L'axe provenance est plus fiable quand des métadonnées sont présentes.
    frequency_weight : float
        Poids de l'axe fréquentiel (défaut : 0.4).
    """

    # Seuils de verdict
    THRESHOLD_SYNTHETIC = 0.65
    THRESHOLD_UNCERTAIN = 0.35

    def __init__(
        self,
        file_path: str,
        provenance_weight: float = 0.6,
        frequency_weight: float = 0.4,
    ):
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Fichier introuvable : {file_path}")
        self.path = file_path
        self.w_prov = provenance_weight
        self.w_freq = frequency_weight

    def analyze(self, n_frames: int = 60) -> DetectionReport:
        report = DetectionReport(file_path=self.path)
        report.file_sha256 = self._sha256()

        print(f"[1/2] Analyse de provenance …")
        prov_analyzer = ProvenanceAnalyzer(self.path)
        report.provenance = prov_analyzer.analyze()

        print(f"[2/2] Analyse fréquentielle ({n_frames} frames) …")
        freq_analyzer = FrequencyAnalyzer(self.path)
        report.frequency = freq_analyzer.analyze(n_frames=n_frames)

        # Fusion pondérée
        report.final_score = (
            self.w_prov * report.provenance.provenance_score
            + self.w_freq * report.frequency.frequency_score
        )

        # Verdict et confiance
        report.verdict, report.confidence = self._classify(report)
        report.flags = self._generate_flags(report)
        return report

    def _sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _classify(self, r: DetectionReport) -> tuple[str, str]:
        score = r.final_score
        # Confiance : haute si les deux axes concordent
        prov_s, freq_s = r.provenance.provenance_score, r.frequency.frequency_score
        delta = abs(prov_s - freq_s)
        if delta < 0.2:
            confidence = "high"
        elif delta < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        if score >= self.THRESHOLD_SYNTHETIC:
            verdict = "synthetic"
        elif score <= self.THRESHOLD_UNCERTAIN:
            verdict = "natural"
        else:
            verdict = "uncertain"
        return verdict, confidence

    def _generate_flags(self, r: DetectionReport) -> list[str]:
        flags = []
        p, f = r.provenance, r.frequency
        if p.c2pa_found:
            flags.append("C2PA_MANIFEST_PRESENT")
        if p.c2pa_claim_generator and "ai" in p.c2pa_claim_generator.lower():
            flags.append("C2PA_AI_GENERATOR_CLAIM")
        if p.synthid_heuristic:
            flags.append("SYNTHID_LSB_ANOMALY")
        if p.known_ai_metadata_tags:
            flags.append(f"AI_METADATA_TAGS:{len(p.known_ai_metadata_tags)}")
        if f.mean_checkerboard_energy > 0.02:
            flags.append("HIGH_CHECKERBOARD_ENERGY")
        if f.temporal_variance_ratio < 0.05:
            flags.append("QUASI_STATIC_SPECTRUM")
        if f.peak_symmetry_score > 0.85:
            flags.append("HIGH_SPECTRAL_SYMMETRY")
        if f.anomalous_frames_pct > 80:
            flags.append("MAJORITY_ANOMALOUS_FRAMES")
        return flags


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def batch_analyze(
    folder: str,
    n_frames: int = 40,
    extensions: tuple = (".mp4", ".mkv", ".avi", ".mov"),
) -> list[DetectionReport]:
    """
    Analyse en batch tous les fichiers vidéo d'un dossier.

    Exemple :
        reports = batch_analyze("/path/to/videos")
        for r in reports:
            print(r)
    """
    reports = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in extensions:
            try:
                d = VideoAIDetector(str(p))
                r = d.analyze(n_frames=n_frames)
                reports.append(r)
                print(r)
            except Exception as e:
                print(f"[ERREUR] {p.name}: {e}")
    return reports


def visualize_spectrum(video_path: str, frame_index: int = 0, save_path: Optional[str] = None):
    """
    Exporte une visualisation PNG du spectre FFT d'une frame donnée.
    Utile pour inspecter visuellement les artefacts de damier.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Impossible de lire la frame {frame_index}")

    resized = cv2.resize(frame, (256, 256))
    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)

    f = np.fft.fft2(y_channel)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Normalisation pour affichage
    normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_img = normalized.astype(np.uint8)
    spectrum_colored = cv2.applyColorMap(spectrum_img, cv2.COLORMAP_INFERNO)

    output_path = save_path or f"spectrum_frame_{frame_index}.png"
    cv2.imwrite(output_path, spectrum_colored)
    print(f"Spectre sauvegardé → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python ai_video_detector.py <video_path> [n_frames=60]")
        sys.exit(1)

    video_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    detector = VideoAIDetector(video_path)
    result = detector.analyze(n_frames=n)
    print(result)

    # Export JSON
    out_json = Path(video_path).stem + "_detection.json"
    Path(out_json).write_text(result.to_json())
    print(f"Rapport JSON → {out_json}")
