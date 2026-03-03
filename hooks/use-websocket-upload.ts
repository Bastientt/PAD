// hooks/use-websocket-upload.ts

import { useState, useEffect, useCallback, useRef } from 'react';
import WebSocketService from '@/services/websocket.service';
import UploadService from '@/services/upload.service';
import EncryptionService from '@/services/encryption.service';
import AudioService from '@/services/audio.service';
import { getWebSocketUrl, AUDIO_CONFIG } from '@/config/websocket.config';
import { UploadStatus, VideoUploadData } from '@/types/upload.types';
import { AnalysisResult } from '@/types/analysis.types';
import { AnalysisResultMessage } from '@/types/websocket.types';
import { logger } from '@/utils/logger';
import {
  validateVideoUri,
  validateUploadId,
  validateEncryptionKey,
} from '@/utils/validators';

/**
 * Hook principal pour gérer WebSocket + Upload + Analyse
 * C'est le chef d'orchestre que AYMAN va utiliser
 */
export function useWebSocketUpload() {
  // ==================== ÉTATS ====================
  const [isConnected, setIsConnected] = useState(false);
  const [presignedUrl, setPresignedUrl] = useState<string | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [directions, setDirections] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>(UploadStatus.IDLE);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs pour éviter les re-renders inutiles
  const encryptionKeyRef = useRef<string | null>(null);
  const hasPlayedAudioRef = useRef(false);

  // ==================== INITIALISATION ====================

  useEffect(() => {
    logger.info('Hook: Initialisation');
    initializeWebSocket();

    // Cleanup
    return () => {
      logger.info('Hook: Nettoyage');
      WebSocketService.disconnect();
      AudioService.cleanup();
    };
  }, []);

  /**
   * Initialiser la connexion WebSocket
   */
  const initializeWebSocket = useCallback(async () => {
    try {
      logger.info('Hook: Connexion WebSocket...');
      setError(null);

      const wsUrl = getWebSocketUrl();
      await WebSocketService.connect(wsUrl);

      // Callbacks WebSocket
      WebSocketService.onConnect(() => {
        logger.info('Hook: WebSocket connecté');
        setIsConnected(true);
        setError(null);
      });

      WebSocketService.onDisconnect((reason) => {
        logger.warn('Hook: WebSocket déconnecté', { reason });
        setIsConnected(false);
        setPresignedUrl(null);
        setUploadId(null);
      });

      WebSocketService.onError((err) => {
        logger.error('Hook: Erreur WebSocket', err);
        setError(err.message);
        setIsConnected(false);
      });

      WebSocketService.onPresignedUrl((id, url, dirs) => {
        logger.info('Hook: URL présignée reçue', { id, directions: dirs });
        setUploadId(id);
        setPresignedUrl(url);
        setDirections(dirs);

        // Jouer son ultrasonique (optionnel)
        if (AUDIO_CONFIG.enabled && !hasPlayedAudioRef.current) {
          AudioService.playUltrasonicSound(
            AUDIO_CONFIG.defaultFrequency,
            AUDIO_CONFIG.defaultDuration
          );
          hasPlayedAudioRef.current = true;
        }
      });

      WebSocketService.onAnalysisResult((result: AnalysisResultMessage) => {
  logger.info('Hook: Résultat analyse reçu', {
    success: result.success,
    confidence: result.confidence,
  });

  // ✅ MAPPER les accessories_detected de string[] vers AccessoryType[]
  const mappedDetails = result.details ? {
    ...result.details,
    accessories_detected: result.details.accessories_detected?.map(
      (acc: string) => acc as any // Cast temporaire
    ),
  } : undefined;

  const analysisResult: AnalysisResult = {
    uploadId: result.uploadId,
    success: result.success,
    confidence: result.confidence,
    message: result.message,
    timestamp: new Date(),
    details: mappedDetails as any, // ← Cast temporaire pour éviter l'erreur
  };

  setAnalysisResult(analysisResult);
  setIsUploading(false);
  setUploadStatus(UploadStatus.COMPLETED);
});
    } catch (err) {
      logger.error('Hook: Erreur initialisation', err);
      setError(
        err instanceof Error ? err.message : 'Erreur de connexion'
      );
      setIsConnected(false);
    }
  }, []);

  // ==================== UPLOAD VIDÉO ====================

  /**
   * Uploader une vidéo
   */
  const uploadVideo = useCallback(
    async (videoUri: string): Promise<void> => {
      try {
        logger.info('Hook: Démarrage upload', { videoUri });

        // Validation
        const uriValidation = validateVideoUri(videoUri);
        if (!uriValidation.valid) {
          throw new Error(uriValidation.error);
        }

        if (!presignedUrl || !uploadId) {
          throw new Error('URL présignée non disponible');
        }

        const idValidation = validateUploadId(uploadId);
        if (!idValidation.valid) {
          throw new Error(idValidation.error);
        }

        // Reset états
        setError(null);
        setIsUploading(true);
        setUploadProgress(0);
        setAnalysisResult(null);
        hasPlayedAudioRef.current = false;

        // Générer clé de chiffrement
        const encryptionKey = EncryptionService.generateKey();
        encryptionKeyRef.current = encryptionKey;

        const keyValidation = validateEncryptionKey(encryptionKey);
        if (!keyValidation.valid) {
          throw new Error(keyValidation.error);
        }

        // Préparer données
        const uploadData: VideoUploadData = {
          uri: videoUri,
          uploadId,
          presignedUrl,
          directions,
          encryptionKey,
        };

        // Callbacks upload
        UploadService.onProgress((progress) => {
          setUploadProgress(progress);
        });

        UploadService.onStatusChange((status) => {
          setUploadStatus(status);
        });

        // Upload
        logger.time('Upload');
        const result = await UploadService.uploadVideo(uploadData);
        logger.timeEnd('Upload');

        if (!result.success) {
          throw new Error(result.error || 'Upload échoué');
        }

        // Notifier le backend
        logger.info('Hook: Notification backend');
        WebSocketService.notifyUploadComplete(uploadId, encryptionKey);

        logger.info('Hook: Upload terminé avec succès');
      } catch (err) {
        logger.error('Hook: Erreur upload', err);
        const errorMessage =
          err instanceof Error ? err.message : 'Erreur lors de l\'upload';
        setError(errorMessage);
        setIsUploading(false);
        setUploadStatus(UploadStatus.FAILED);
        throw err;
      }
    },
    [presignedUrl, uploadId, directions]
  );

  // ==================== RETRY ====================

  /**
   * Réessayer après une erreur
   */
  const retry = useCallback(() => {
    logger.info('Hook: Retry');
    setError(null);
    setAnalysisResult(null);
    setUploadProgress(0);
    setUploadStatus(UploadStatus.IDLE);
    setPresignedUrl(null);
    setUploadId(null);
    setDirections([]);
    hasPlayedAudioRef.current = false;
    encryptionKeyRef.current = null;

    // Reconnecter si nécessaire
    if (!isConnected) {
      initializeWebSocket();
    }
  }, [isConnected, initializeWebSocket]);

  // ==================== DISCONNECT ====================

  /**
   * Déconnecter manuellement
   */
  const disconnect = useCallback(() => {
    logger.info('Hook: Déconnexion manuelle');
    WebSocketService.disconnect();
    AudioService.cleanup();
    setIsConnected(false);
    setPresignedUrl(null);
    setUploadId(null);
  }, []);

  // ==================== RETURN ====================

  return {
    // États de connexion
    isConnected,
    presignedUrl,
    uploadId,
    directions,

    // États d'upload
    uploadStatus,
    uploadProgress,
    isUploading,

    // Résultat
    analysisResult,

    // Erreur
    error,

    // Actions
    uploadVideo,
    retry,
    disconnect,
  };
}