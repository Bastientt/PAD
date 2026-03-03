// config/websocket.config.ts

import { WebSocketConfig } from '@/types/websocket.types';

/**
 * Configuration WebSocket
 * Modifie l'URL selon ton environnement
 */

// Détection de l'environnement
const isDevelopment = __DEV__;
const isProduction = !__DEV__;

// URLs selon l'environnement
const WEBSOCKET_URLS = {
  development: 'ws://192.168.1.100:8080/api/ws', // bastien 
  staging: 'wss://staging-api.example.com/api/ws',
  production: 'wss://api.example.com/api/ws',
};

// Configuration principale
export const WEBSOCKET_CONFIG: WebSocketConfig = {
  // URL du serveur WebSocket
  url: isDevelopment
    ? WEBSOCKET_URLS.development
    : WEBSOCKET_URLS.production,

  // Délai entre les tentatives de reconnexion (ms)
  reconnectDelay: 3000,

  // Nombre maximum de tentatives de reconnexion
  maxReconnectAttempts: 5,

  // Timeout de connexion (ms)
  timeout: 10000,

  // Intervalle de ping pour garder la connexion active (ms)
  pingInterval: 30000,
};

// Configuration upload
export const UPLOAD_CONFIG = {
  // Taille maximale de fichier (en MB)
  maxFileSize: 100,

  // Durée maximale de vidéo (en secondes)
  maxDuration: 60,

  // Formats autorisés
  allowedFormats: ['video/mp4', 'video/quicktime', 'video/x-msvideo'],

  // Timeout d'upload (ms)
  timeout: 120000, // 2 minutes

  // Nombre de tentatives en cas d'échec
  retryAttempts: 3,
};

// Configuration chiffrement
export const ENCRYPTION_CONFIG = {
  // Algorithme
  algorithm: 'AES-256' as const,

  // Taille de la clé (bits)
  keySize: 256,

  // Mode
  mode: 'CBC' as const,
};

// Configuration audio (sons ultrasoniques)
export const AUDIO_CONFIG = {
  // Fréquence par défaut (Hz)
  defaultFrequency: 20000,

  // Durée par défaut (ms)
  defaultDuration: 500,

  // Fréquences possibles pour patterns
  frequencies: [18000, 19000, 20000, 21000, 22000],

  // Activé par défaut
  enabled: true,
};

// Configuration directions
export const DIRECTIONS_CONFIG = {
  // Directions possibles
  availableDirections: ['left', 'right', 'up', 'down', 'center'] as const,

  // Durée par direction (secondes)
  durationPerDirection: 3,

  // Nombre minimum de directions
  minDirections: 3,

  // Nombre maximum de directions
  maxDirections: 6,
};

// Seuils de confiance
export const CONFIDENCE_THRESHOLDS = {
  // Seuil minimum pour validation (0-1)
  minimum: 0.7,

  // Seuils par niveau
  excellent: 0.9,
  veryGood: 0.8,
  good: 0.7,
  medium: 0.6,
  low: 0.4,
};

// Configuration logs
export const LOG_CONFIG = {
  // Activer les logs en développement
  enabled: isDevelopment,

  // Niveau de log
  level: isDevelopment ? 'debug' : 'error',

  // Afficher les timestamps
  showTimestamp: true,

  // Afficher les couleurs
  useColors: true,
};

// Export des URLs pour usage direct
export const API_URLS = {
  websocket: WEBSOCKET_CONFIG.url,
  baseUrl: isDevelopment
    ? 'http://192.168.1.100:8080'
    : 'https://api.example.com',
};

// Helper pour obtenir l'URL WebSocket complète
export function getWebSocketUrl(): string {
  return WEBSOCKET_CONFIG.url;
}

// Helper pour vérifier si en développement
export function isDev(): boolean {
  return isDevelopment;
}

// Helper pour obtenir la config complète
export function getConfig() {
  return {
    websocket: WEBSOCKET_CONFIG,
    upload: UPLOAD_CONFIG,
    encryption: ENCRYPTION_CONFIG,
    audio: AUDIO_CONFIG,
    directions: DIRECTIONS_CONFIG,
    confidence: CONFIDENCE_THRESHOLDS,
    log: LOG_CONFIG,
  };
}