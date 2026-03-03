// types/analysis.types.ts

/**
 * Types pour les résultats de l'analyse PAD (Presentation Attack Detection)
 */

// ==================== RÉSULTAT PRINCIPAL ====================

export interface AnalysisResult {
  uploadId: string;
  success: boolean;
  confidence: number; // 0-1
  message?: string;
  timestamp: Date;
  details?: AnalysisDetails;
}

// ==================== DÉTAILS DE L'ANALYSE ====================

export interface AnalysisDetails {
  // Détection du visage
  face_detected: boolean;
  face_confidence?: number;
  face_landmarks_count?: number;

  // Vérification des mouvements
  movements_correct: boolean;
  movements_matched?: number; // nombre de directions correctes
  movements_total?: number; // nombre total de directions

  // Score de vivacité (liveness)
  liveness_score: number; // 0-1
  liveness_method?: 'depth' | 'texture' | 'motion' | 'combined';

  // Détection son ultrasonique (optionnel)
  ultrasonic_detected?: boolean;
  ultrasonic_frequency?: number;

  // Détection d'accessoires
  accessories_detected?: AccessoryType[];

  // Stabilité de l'arrière-plan
  background_stable?: boolean;
  background_change_score?: number; // 0-1

  // Qualité de la vidéo
  video_quality?: VideoQuality;

  // Détection de spoofing
  spoofing_indicators?: SpoofingIndicator[];
}

// ==================== TYPES D'ACCESSOIRES ====================

export enum AccessoryType {
  GLASSES = 'glasses',
  SUNGLASSES = 'sunglasses',
  MASK = 'mask',
  HAT = 'hat',
  CAP = 'cap',
  SCARF = 'scarf',
  HELMET = 'helmet',
}

export interface AccessoryDetection {
  type: AccessoryType;
  confidence: number;
  position?: string;
}

// ==================== QUALITÉ VIDÉO ====================

export interface VideoQuality {
  resolution: {
    width: number;
    height: number;
  };
  fps: number;
  brightness: number; // 0-1
  sharpness: number; // 0-1
  overall_score: number; // 0-1
}

// ==================== INDICATEURS DE SPOOFING ====================

export enum SpoofingType {
  PHOTO = 'photo',
  VIDEO_REPLAY = 'video_replay',
  MASK = 'mask',
  SCREEN = 'screen',
  DEEPFAKE = 'deepfake',
  UNKNOWN = 'unknown',
}

export interface SpoofingIndicator {
  type: SpoofingType;
  confidence: number; // 0-1
  reason: string;
  detected_at?: number; // timestamp en secondes dans la vidéo
}

// ==================== NIVEAUX DE CONFIANCE ====================

export enum ConfidenceLevel {
  VERY_LOW = 'very_low', // 0-0.4
  LOW = 'low', // 0.4-0.6
  MEDIUM = 'medium', // 0.6-0.7
  GOOD = 'good', // 0.7-0.8
  VERY_GOOD = 'very_good', // 0.8-0.9
  EXCELLENT = 'excellent', // 0.9-1.0
}

export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.9) return ConfidenceLevel.EXCELLENT;
  if (confidence >= 0.8) return ConfidenceLevel.VERY_GOOD;
  if (confidence >= 0.7) return ConfidenceLevel.GOOD;
  if (confidence >= 0.6) return ConfidenceLevel.MEDIUM;
  if (confidence >= 0.4) return ConfidenceLevel.LOW;
  return ConfidenceLevel.VERY_LOW;
}

// ==================== STATISTIQUES ====================

export interface AnalysisStatistics {
  totalAnalyses: number;
  successfulAnalyses: number;
  failedAnalyses: number;
  averageConfidence: number;
  averageProcessingTime: number; // en ms
  commonFailureReasons: string[];
}

// ==================== HISTORIQUE ====================

export interface AnalysisHistoryItem {
  uploadId: string;
  timestamp: Date;
  result: AnalysisResult;
  userId?: string;
}

// ==================== SEUILS DE DÉCISION ====================

export interface AnalysisThresholds {
  minConfidence: number; // Seuil minimum pour succès (ex: 0.7)
  minLivenessScore: number; // Seuil minimum liveness (ex: 0.6)
  maxAccessories: number; // Nombre max d'accessoires tolérés
  requireUltrasonic: boolean; // Son ultrasonique obligatoire ?
  requireBackgroundStable: boolean; // Arrière-plan stable obligatoire ?
}

export const DEFAULT_THRESHOLDS: AnalysisThresholds = {
  minConfidence: 0.7,
  minLivenessScore: 0.6,
  maxAccessories: 0,
  requireUltrasonic: false,
  requireBackgroundStable: false,
};

// ==================== RAPPORT D'ANALYSE ====================

export interface AnalysisReport {
  result: AnalysisResult;
  metadata: {
    analyzedAt: Date;
    processingTime: number; // en ms
    modelVersion: string;
    thresholdsUsed: AnalysisThresholds;
  };
  recommendations?: string[];
}

// ==================== HELPERS ====================

export function isAnalysisSuccessful(
  result: AnalysisResult,
  thresholds: AnalysisThresholds = DEFAULT_THRESHOLDS
): boolean {
  if (!result.success) return false;
  if (result.confidence < thresholds.minConfidence) return false;
  if (
    result.details?.liveness_score &&
    result.details.liveness_score < thresholds.minLivenessScore
  ) {
    return false;
  }
  return true;
}

export function getFailureReason(result: AnalysisResult): string {
  if (result.success) return '';

  if (!result.details?.face_detected) {
    return 'Aucun visage détecté';
  }

  if (!result.details?.movements_correct) {
    return 'Les mouvements ne correspondent pas aux instructions';
  }

  if (result.details?.liveness_score < 0.6) {
    return 'Score de vivacité trop faible - Possible attaque détectée';
  }

  if (result.details?.accessories_detected && result.details.accessories_detected.length > 0) {
    return `Accessoires détectés: ${result.details.accessories_detected.join(', ')}`;
  }

  if (result.details?.spoofing_indicators && result.details.spoofing_indicators.length > 0) {
    return `Tentative de fraude détectée: ${result.details.spoofing_indicators[0].reason}`;
  }

  return result.message || 'Échec de la vérification';
}

export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}
