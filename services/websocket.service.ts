// services/websocket.service.ts

import {
  WebSocketState,
  ServerMessage,
  ClientMessage,
  PresignedUrlMessage,
  AnalysisResultMessage,
  isPresignedUrlMessage,
  isAnalysisResultMessage,
  isErrorMessage,
  isConnectionClosedMessage,
} from '@/types/websocket.types';
import { logger } from '@/utils/logger';

class WebSocketService {
  private ws: WebSocket | null = null;
  private state: WebSocketState = WebSocketState.DISCONNECTED;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
 private pingInterval: ReturnType<typeof setInterval> | null = null;


  // Callbacks
  private onConnectCallback: (() => void) | null = null;
  private onDisconnectCallback: ((reason?: string) => void) | null = null;
  private onErrorCallback: ((error: Error) => void) | null = null;
  private onMessageCallback: ((message: ServerMessage) => void) | null = null;
  private onPresignedUrlCallback:
    | ((id: string, url: string, directions: string[]) => void)
    | null = null;
  private onAnalysisResultCallback:
    | ((result: AnalysisResultMessage) => void)
    | null = null;

  /**
   * Connecter au serveur WebSocket
   */
  async connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        logger.info('WebSocket: Tentative de connexion...', { url });
        this.state = WebSocketState.CONNECTING;

        this.ws = new WebSocket(url);

        // Événement: Connexion établie
        this.ws.onopen = () => {
          logger.info('WebSocket: Connexion établie');
          this.state = WebSocketState.CONNECTED;
          this.reconnectAttempts = 0;

          // Démarrer le ping pour garder la connexion active
          this.startPing();

          // Envoyer le handshake
          this.sendHandshake();

          this.onConnectCallback?.();
          resolve();
        };

        // Événement: Message reçu
        this.ws.onmessage = (event: MessageEvent) => {
          try {
            const message: ServerMessage = JSON.parse(event.data);
            logger.debug('WebSocket: Message reçu', { type: message.type });
            this.handleMessage(message);
          } catch (error) {
            logger.error('WebSocket: Erreur parsing message', error);
          }
        };

        // Événement: Erreur
        this.ws.onerror = (error: Event) => {
          logger.error('WebSocket: Erreur de connexion', error);
          this.state = WebSocketState.ERROR;
          const err = new Error('Erreur de connexion WebSocket');
          this.onErrorCallback?.(err);
          reject(err);
        };

        // Événement: Connexion fermée
        this.ws.onclose = (event: CloseEvent) => {
          logger.warn('WebSocket: Connexion fermée', {
            code: event.code,
            reason: event.reason,
          });
          this.state = WebSocketState.DISCONNECTED;
          this.stopPing();

          this.onDisconnectCallback?.(event.reason);

          // Tentative de reconnexion
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.attemptReconnect(url);
          }
        };
      } catch (error) {
        logger.error('WebSocket: Erreur lors de la connexion', error);
        reject(error);
      }
    });
  }

  /**
   * Envoyer le handshake initial
   */
  private sendHandshake(): void {
    const secret = this.generateSecret();
    const handshakeMessage: ClientMessage = {
      type: 'handshake',
      secret,
      timestamp: new Date().toISOString(),
    };
    this.send(handshakeMessage);
    logger.debug('WebSocket: Handshake envoyé', { secret });
  }

  /**
   * Générer un secret aléatoire pour le handshake
   */
  private generateSecret(): string {
    return (
      Math.random().toString(36).substring(2, 15) +
      Math.random().toString(36).substring(2, 15)
    );
  }

  /**
   * Gérer les messages reçus
   */
  private handleMessage(message: ServerMessage): void {
    this.onMessageCallback?.(message);

    if (message.type === 'handshake_ok') {
      logger.info('WebSocket: Handshake validé');
    } else if (isPresignedUrlMessage(message)) {
      logger.info('WebSocket: URL présignée reçue', {
        id: message.id,
        expiresIn: message.expiresIn,
      });
      this.onPresignedUrlCallback?.(
        message.id,
        message.url,
        message.directions
      );
    } else if (isAnalysisResultMessage(message)) {
      logger.info('WebSocket: Résultat analyse reçu', {
        uploadId: message.uploadId,
        success: message.success,
        confidence: message.confidence,
      });
      this.onAnalysisResultCallback?.(message);
    } else if (isErrorMessage(message)) {
      logger.error('WebSocket: Erreur serveur', {
        code: message.code,
        message: message.message,
      });
      this.onErrorCallback?.(new Error(message.message));
    } else if (isConnectionClosedMessage(message)) {
      logger.warn('WebSocket: Fermeture demandée par le serveur', {
        reason: message.reason,
      });
      this.disconnect();
    }
  }

  /**
   * Envoyer un message au serveur
   */
  send(message: ClientMessage): void {
    if (!this.ws || this.state !== WebSocketState.CONNECTED) {
      logger.error('WebSocket: Impossible d\'envoyer, non connecté');
      throw new Error('WebSocket non connecté');
    }

    try {
      this.ws.send(JSON.stringify(message));
      logger.debug('WebSocket: Message envoyé', { type: message.type });
    } catch (error) {
      logger.error('WebSocket: Erreur lors de l\'envoi', error);
      throw error;
    }
  }

  /**
   * Notifier le serveur que l'upload est terminé
   */
  notifyUploadComplete(uploadId: string, encryptionKey: string): void {
    const message: ClientMessage = {
      type: 'upload_complete',
      id: uploadId,
      videoPath: '', // Pas utilisé côté serveur (vidéo déjà uploadée)
      encryptionKey,
      timestamp: new Date().toISOString(),
    };
    this.send(message);
    logger.info('WebSocket: Upload complet notifié', { uploadId });
  }

  /**
   * Déconnecter
   */
  disconnect(): void {
    if (this.ws) {
      logger.info('WebSocket: Déconnexion...');
      this.stopPing();
      this.ws.close();
      this.ws = null;
      this.state = WebSocketState.DISCONNECTED;
    }
  }

  /**
   * Tentative de reconnexion
   */
  private attemptReconnect(url: string): void {
    this.reconnectAttempts++;
    this.state = WebSocketState.RECONNECTING;

    logger.info('WebSocket: Tentative de reconnexion', {
      attempt: this.reconnectAttempts,
      maxAttempts: this.maxReconnectAttempts,
    });

    setTimeout(() => {
      this.connect(url).catch((error) => {
        logger.error('WebSocket: Échec de reconnexion', error);
      });
    }, this.reconnectDelay * this.reconnectAttempts);
  }

  /**
   * Démarrer le ping pour garder la connexion active
   */
  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws && this.state === WebSocketState.CONNECTED) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping' }));
        } catch (error) {
          logger.error('WebSocket: Erreur lors du ping', error);
        }
      }
    }, 30000); // Ping toutes les 30 secondes
  }

  /**
   * Arrêter le ping
   */
  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Obtenir l'état actuel
   */
  getState(): WebSocketState {
    return this.state;
  }

  /**
   * Vérifier si connecté
   */
  isConnected(): boolean {
    return this.state === WebSocketState.CONNECTED;
  }

  // ==================== CALLBACKS ====================

  onConnect(callback: () => void): void {
    this.onConnectCallback = callback;
  }

  onDisconnect(callback: (reason?: string) => void): void {
    this.onDisconnectCallback = callback;
  }

  onError(callback: (error: Error) => void): void {
    this.onErrorCallback = callback;
  }

  onMessage(callback: (message: ServerMessage) => void): void {
    this.onMessageCallback = callback;
  }

  onPresignedUrl(
    callback: (id: string, url: string, directions: string[]) => void
  ): void {
    this.onPresignedUrlCallback = callback;
  }

  onAnalysisResult(callback: (result: AnalysisResultMessage) => void): void {
    this.onAnalysisResultCallback = callback;
  }
}

// Export singleton
export default new WebSocketService();