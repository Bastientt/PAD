// utils/validators.ts

import { Direction } from '@/types/upload.types';
import { UPLOAD_CONFIG, DIRECTIONS_CONFIG } from '@/config/websocket.config';

/**
 * Validateurs pour les données
 */

// ==================== VALIDATION VIDÉO ====================

/**
 * Valider la taille du fichier vidéo
 */
export function validateVideoSize(sizeInBytes: number): {
  valid: boolean;
  error?: string;
} {
  const maxSizeBytes = UPLOAD_CONFIG.maxFileSize * 1024 * 1024; // MB to bytes

  if (sizeInBytes > maxSizeBytes) {
    return {
      valid: false,
      error: `La vidéo est trop volumineuse (max: ${UPLOAD_CONFIG.maxFileSize}MB)`,
    };
  }

  return { valid: true };
}

/**
 * Valider la durée de la vidéo
 */
export function validateVideoDuration(durationInSeconds: number): {
  valid: boolean;
  error?: string;
} {
  if (durationInSeconds > UPLOAD_CONFIG.maxDuration) {
    return {
      valid: false,
      error: `La vidéo est trop longue (max: ${UPLOAD_CONFIG.maxDuration}s)`,
    };
  }

  if (durationInSeconds < 3) {
    return {
      valid: false,
      error: 'La vidéo est trop courte (min: 3s)',
    };
  }

  return { valid: true };
}

/**
 * Valider le format de la vidéo
 */
export function validateVideoFormat(mimeType: string): {
  valid: boolean;
  error?: string;
} {
  if (!UPLOAD_CONFIG.allowedFormats.includes(mimeType)) {
    return {
      valid: false,
      error: `Format non supporté. Formats acceptés: ${UPLOAD_CONFIG.allowedFormats.join(', ')}`,
    };
  }

  return { valid: true };
}

/**
 * Valider l'URI de la vidéo
 */
export function validateVideoUri(uri: string): {
  valid: boolean;
  error?: string;
} {
  if (!uri || uri.trim() === '') {
    return {
      valid: false,
      error: 'URI de vidéo invalide',
    };
  }

  // Vérifier que c'est un URI valide
  if (!uri.startsWith('file://') && !uri.startsWith('content://')) {
    return {
      valid: false,
      error: 'URI de vidéo invalide',
    };
  }

  return { valid: true };
}

// ==================== VALIDATION DIRECTIONS ====================

/**
 * Valider une direction
 */
export function validateDirection(direction: string): direction is Direction {
  return DIRECTIONS_CONFIG.availableDirections.includes(direction as Direction);
}

/**
 * Valider une séquence de directions
 */
export function validateDirections(directions: string[]): {
  valid: boolean;
  error?: string;
} {
  if (!Array.isArray(directions)) {
    return {
      valid: false,
      error: 'Les directions doivent être un tableau',
    };
  }

  if (directions.length < DIRECTIONS_CONFIG.minDirections) {
    return {
      valid: false,
      error: `Nombre minimum de directions: ${DIRECTIONS_CONFIG.minDirections}`,
    };
  }

  if (directions.length > DIRECTIONS_CONFIG.maxDirections) {
    return {
      valid: false,
      error: `Nombre maximum de directions: ${DIRECTIONS_CONFIG.maxDirections}`,
    };
  }

  // Vérifier que toutes les directions sont valides
  for (const direction of directions) {
    if (!validateDirection(direction)) {
      return {
        valid: false,
        error: `Direction invalide: ${direction}`,
      };
    }
  }

  return { valid: true };
}

// ==================== VALIDATION URL ====================

/**
 * Valider une URL
 */
export function validateUrl(url: string): {
  valid: boolean;
  error?: string;
} {
  if (!url || url.trim() === '') {
    return {
      valid: false,
      error: 'URL vide',
    };
  }

  try {
    new URL(url);
    return { valid: true };
  } catch {
    return {
      valid: false,
      error: 'URL invalide',
    };
  }
}

/**
 * Valider une URL WebSocket
 */
export function validateWebSocketUrl(url: string): {
  valid: boolean;
  error?: string;
} {
  const urlValidation = validateUrl(url);
  if (!urlValidation.valid) {
    return urlValidation;
  }

  if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
    return {
      valid: false,
      error: 'L\'URL doit commencer par ws:// ou wss://',
    };
  }

  return { valid: true };
}

// ==================== VALIDATION CHIFFREMENT ====================

/**
 * Valider une clé de chiffrement
 */
export function validateEncryptionKey(key: string): {
  valid: boolean;
  error?: string;
} {
  if (!key || key.trim() === '') {
    return {
      valid: false,
      error: 'Clé de chiffrement vide',
    };
  }

  if (key.length < 16) {
    return {
      valid: false,
      error: 'Clé de chiffrement trop courte (min: 16 caractères)',
    };
  }

  return { valid: true };
}

// ==================== VALIDATION ID ====================

/**
 * Valider un ID d'upload
 */
export function validateUploadId(id: string): {
  valid: boolean;
  error?: string;
} {
  if (!id || id.trim() === '') {
    return {
      valid: false,
      error: 'ID d\'upload vide',
    };
  }

  // Vérifier le format (exemple: upload-123)
  if (!/^[a-zA-Z0-9-_]+$/.test(id)) {
    return {
      valid: false,
      error: 'Format d\'ID invalide',
    };
  }

  return { valid: true };
}

// ==================== VALIDATION CONFIDENCE ====================

/**
 * Valider un score de confiance
 */
export function validateConfidence(confidence: number): {
  valid: boolean;
  error?: string;
} {
  if (typeof confidence !== 'number') {
    return {
      valid: false,
      error: 'Le score de confiance doit être un nombre',
    };
  }

  if (confidence < 0 || confidence > 1) {
    return {
      valid: false,
      error: 'Le score de confiance doit être entre 0 et 1',
    };
  }

  return { valid: true };
}

// ==================== HELPERS ====================

/**
 * Valider plusieurs champs en même temps
 */
export function validateAll(
  validations: Array<{ valid: boolean; error?: string }>
): {
  valid: boolean;
  errors: string[];
} {
  const errors = validations
    .filter((v) => !v.valid)
    .map((v) => v.error || 'Erreur inconnue');

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Formater les erreurs de validation
 */
export function formatValidationErrors(errors: string[]): string {
  if (errors.length === 0) return '';
  if (errors.length === 1) return errors[0];
  return '• ' + errors.join('\n• ');
}