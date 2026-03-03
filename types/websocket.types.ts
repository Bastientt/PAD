// types/websocket.types.ts

/**
 * Types pour la communication WebSocket
 */

// Type de base pour tous les messages WebSocket
export interface WebSocketMessage {
  type: string;
  timestamp?: string;
}

// ==================== MESSAGES CLIENT → SERVEUR ====================

// Message de handshake initial
export interface HandshakeMessage extends WebSocketMessage {
  type: 'handshake';
  secret: string;
}

// Message pour notifier que l'upload est terminé
export interface UploadCompleteMessage extends WebSocketMessage {
  type: 'upload_complete';
  id: string;
  videoPath: string;
  encryptionKey: string;
}

// Union de tous les messages que le client ENVOIE
export type ClientMessage = HandshakeMessage | UploadCompleteMessage;

// ==================== MESSAGES SERVEUR → CLIENT ====================

// Message de confirmation du handshake
export interface HandshakeOkMessage extends WebSocketMessage {
  type: 'handshake_ok';
  message?: string;
}

// Message avec l'URL présignée pour upload
export interface PresignedUrlMessage extends WebSocketMessage {
  type: 'presigned_url';
  id: string;
  url: string;
  directions: string[];
  expiresIn: number; // en secondes
  ultrasonicPattern?: string;
}

// Message avec le résultat de l'analyse PAD
export interface AnalysisResultMessage extends WebSocketMessage {
  type: 'analysis_result';
  uploadId: string;
  success: boolean;
  confidence: number;
  message?: string;
  details?: {
    face_detected: boolean;
    movements_correct: boolean;
    liveness_score: number;
    ultrasonic_detected?: boolean;
    accessories_detected?: string[];
    background_stable?: boolean;
  };
}

// Message d'erreur
export interface ErrorMessage extends WebSocketMessage {
  type: 'error';
  code: string;
  message: string;
  details?: any;
}

// Message de fermeture de connexion
export interface ConnectionClosedMessage extends WebSocketMessage {
  type: 'connection_closed';
  reason: 'timeout' | 'error' | 'manual';
  message?: string;
}

// Union de tous les messages que le client REÇOIT
export type ServerMessage =
  | HandshakeOkMessage
  | PresignedUrlMessage
  | AnalysisResultMessage
  | ErrorMessage
  | ConnectionClosedMessage;

// ==================== ÉTATS WebSocket ====================

export enum WebSocketState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
  RECONNECTING = 'reconnecting',
}

// Configuration WebSocket
export interface WebSocketConfig {
  url: string;
  reconnectDelay?: number;
  maxReconnectAttempts?: number;
  timeout?: number;
  pingInterval?: number;
}

// Événements WebSocket
export interface WebSocketEvents {
  onConnect?: () => void;
  onDisconnect?: (reason?: string) => void;
  onError?: (error: Error) => void;
  onMessage?: (message: ServerMessage) => void;
  onPresignedUrl?: (id: string, url: string, directions: string[]) => void;
  onAnalysisResult?: (result: AnalysisResultMessage) => void;
}

// ==================== HELPERS ====================

// Type guard pour vérifier le type de message
export function isPresignedUrlMessage(
  message: ServerMessage
): message is PresignedUrlMessage {
  return message.type === 'presigned_url';
}

export function isAnalysisResultMessage(
  message: ServerMessage
): message is AnalysisResultMessage {
  return message.type === 'analysis_result';
}

export function isErrorMessage(
  message: ServerMessage
): message is ErrorMessage {
  return message.type === 'error';
}

export function isConnectionClosedMessage(
  message: ServerMessage
): message is ConnectionClosedMessage {
  return message.type === 'connection_closed';
}