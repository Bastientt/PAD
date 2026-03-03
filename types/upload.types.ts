// types/upload.types.ts

/**
 * Types pour l'upload de vidéos
 */

// ==================== ÉTAT DE L'UPLOAD ====================

export enum UploadStatus {
  IDLE = 'idle',
  PREPARING = 'preparing',
  ENCRYPTING = 'encrypting',
  UPLOADING = 'uploading',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

// État complet de l'upload
export interface UploadState {
  status: UploadStatus;
  progress: number; // 0-100
  videoUri: string | null;
  uploadId: string | null;
  encryptionKey: string | null;
  error: string | null;
  startedAt?: Date;
  completedAt?: Date;
}

// ==================== DONNÉES D'UPLOAD ====================

// Informations sur la vidéo à uploader
export interface VideoUploadData {
  uri: string; // Chemin local de la vidéo
  uploadId: string;
  presignedUrl: string;
  directions: string[];
  encryptionKey: string;
}

// Résultat de l'upload
export interface UploadResult {
  success: boolean;
  uploadId: string;
  encryptionKey: string;
  message?: string;
  error?: string;
}

// ==================== MÉTADONNÉES VIDÉO ====================

export interface VideoMetadata {
  uri: string;
  duration?: number; // en secondes
  size?: number; // en bytes
  width?: number;
  height?: number;
  mimeType?: string;
  createdAt: Date;
}

// ==================== CHIFFREMENT ====================

export interface EncryptionResult {
  encryptedData: Blob | string;
  key: string;
  algorithm: 'AES-256';
  iv?: string; // Initialization vector
}

export interface EncryptionOptions {
  algorithm?: 'AES-256';
  keySize?: 256;
  generateIV?: boolean;
}

// ==================== CONFIGURATION UPLOAD ====================

export interface UploadConfig {
  maxFileSize?: number; // en MB
  maxDuration?: number; // en secondes
  allowedFormats?: string[];
  chunkSize?: number; // pour upload par morceaux
  retryAttempts?: number;
  timeout?: number; // en ms
}

// ==================== CALLBACKS ====================

export interface UploadCallbacks {
  onProgress?: (progress: number) => void;
  onStatusChange?: (status: UploadStatus) => void;
  onComplete?: (result: UploadResult) => void;
  onError?: (error: Error) => void;
}

// ==================== ERREURS ====================

export enum UploadErrorCode {
  NETWORK_ERROR = 'NETWORK_ERROR',
  FILE_TOO_LARGE = 'FILE_TOO_LARGE',
  INVALID_FORMAT = 'INVALID_FORMAT',
  ENCRYPTION_FAILED = 'ENCRYPTION_FAILED',
  UPLOAD_TIMEOUT = 'UPLOAD_TIMEOUT',
  SERVER_ERROR = 'SERVER_ERROR',
  UNKNOWN_ERROR = 'UNKNOWN_ERROR',
}

export interface UploadError {
  code: UploadErrorCode;
  message: string;
  details?: any;
  timestamp: Date;
}

// ==================== DIRECTIONS ====================

export type Direction = 'left' | 'right' | 'up' | 'down' | 'center';

export interface DirectionSequence {
  directions: Direction[];
  duration: number; // durée totale en secondes
  intervalPerDirection: number; // secondes par direction
}

// ==================== HELPERS ====================

export function isValidDirection(value: string): value is Direction {
  return ['left', 'right', 'up', 'down', 'center'].includes(value);
}

export function getDirectionLabel(direction: Direction): string {
  const labels: Record<Direction, string> = {
    left: 'Gauche',
    right: 'Droite',
    up: 'Haut',
    down: 'Bas',
    center: 'Centre',
  };
  return labels[direction];
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}