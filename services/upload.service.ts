// services/upload.service.ts

import { UploadStatus, UploadResult, VideoUploadData } from '@/types/upload.types';
import EncryptionService from './encryption.service';
import { logger } from '@/utils/logger';

class UploadService {
  private onProgressCallback: ((progress: number) => void) | null = null;
  private onStatusChangeCallback: ((status: UploadStatus) => void) | null = null;

  /**
   * Uploader une vidéo vers le serveur
   */
  async uploadVideo(data: VideoUploadData): Promise<UploadResult> {
  try {
    logger.info('Upload: Démarrage', { uploadId: data.uploadId });
    this.updateStatus(UploadStatus.PREPARING);
    this.updateProgress(0);

    // 1. Lire la vidéo
    logger.debug('Upload: Lecture de la vidéo...');
    const videoBlob = await this.readVideoFile(data.uri);
    this.updateProgress(10);

    // 2. Chiffrer la vidéo
    logger.debug('Upload: Chiffrement de la vidéo...');
    this.updateStatus(UploadStatus.ENCRYPTING);
    const encryptionResult = await EncryptionService.encryptVideo(
      videoBlob,
      data.encryptionKey
    );
    this.updateProgress(30);

    // 3. Préparer FormData
    logger.debug('Upload: Préparation de l\'upload...');
    const formData = new FormData();
    
    // ✅ CONVERTIR EN BLOB SI STRING
    const encryptedBlob = typeof encryptionResult.encryptedData === 'string'
      ? new Blob([encryptionResult.encryptedData], { type: 'application/octet-stream' })
      : encryptionResult.encryptedData;
    
    formData.append('file', encryptedBlob, 'encrypted_video.enc');
    formData.append('key', data.encryptionKey);
    formData.append('uploadId', data.uploadId);

    // 4. Uploader
    logger.debug('Upload: Envoi de la vidéo...');
    this.updateStatus(UploadStatus.UPLOADING);
    await this.uploadWithProgress(data.presignedUrl, formData);

    // 5. Terminé
    logger.info('Upload: Terminé avec succès', { uploadId: data.uploadId });
    this.updateStatus(UploadStatus.COMPLETED);
    this.updateProgress(100);

    return {
      success: true,
      uploadId: data.uploadId,
      encryptionKey: data.encryptionKey,
      message: 'Upload réussi',
    };
  } catch (error) {
    logger.error('Upload: Erreur', error);
    this.updateStatus(UploadStatus.FAILED);

    return {
      success: false,
      uploadId: data.uploadId,
      encryptionKey: data.encryptionKey,
      error: error instanceof Error ? error.message : 'Erreur inconnue',
    };
  }
}

  /**
   * Lire le fichier vidéo
   */
  private async readVideoFile(uri: string): Promise<Blob> {
    try {
      const response = await fetch(uri);
      if (!response.ok) {
        throw new Error('Impossible de lire la vidéo');
      }
      return await response.blob();
    } catch (error) {
      logger.error('Upload: Erreur lecture vidéo', error);
      throw new Error('Erreur lors de la lecture de la vidéo');
    }
  }

  /**
   * Uploader avec suivi de progression
   */
  private async uploadWithProgress(
    url: string,
    formData: FormData
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      // Suivi de la progression
      xhr.upload.onprogress = (event: ProgressEvent) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 70) + 30; // 30-100%
          this.updateProgress(progress);
        }
      };

      // Succès
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          logger.info('Upload: HTTP Success', { status: xhr.status });
          resolve();
        } else {
          logger.error('Upload: HTTP Error', {
            status: xhr.status,
            response: xhr.responseText,
          });
          reject(new Error(`Upload échoué: ${xhr.status}`));
        }
      };

      // Erreur réseau
      xhr.onerror = () => {
        logger.error('Upload: Erreur réseau');
        reject(new Error('Erreur réseau lors de l\'upload'));
      };

      // Timeout
      xhr.ontimeout = () => {
        logger.error('Upload: Timeout');
        reject(new Error('Timeout lors de l\'upload'));
      };

      // Configuration
      xhr.timeout = 120000; // 2 minutes
      xhr.open('POST', url);

      // Envoi
      xhr.send(formData);
    });
  }

  /**
   * Mettre à jour la progression
   */
  private updateProgress(progress: number): void {
    this.onProgressCallback?.(Math.min(progress, 100));
  }

  /**
   * Mettre à jour le statut
   */
  private updateStatus(status: UploadStatus): void {
    this.onStatusChangeCallback?.(status);
  }

  // ==================== CALLBACKS ====================

  onProgress(callback: (progress: number) => void): void {
    this.onProgressCallback = callback;
  }

  onStatusChange(callback: (status: UploadStatus) => void): void {
    this.onStatusChangeCallback = callback;
  }
}

// Export singleton
export default new UploadService();