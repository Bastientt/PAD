// services/encryption.service.ts

import CryptoJS from 'crypto-js';
import { EncryptionResult, EncryptionOptions } from '@/types/upload.types';
import { logger } from '@/utils/logger';

class EncryptionService {
  /**
   * Générer une clé de chiffrement aléatoire
   */
  generateKey(length: number = 32): string {
    const key = CryptoJS.lib.WordArray.random(length).toString();
    logger.debug('Encryption: Clé générée', { keyLength: key.length });
    return key;
  }

  /**
   * Chiffrer une vidéo (Blob)
   */
  async encryptVideo(
    videoBlob: Blob,
    key: string,
    options?: EncryptionOptions
  ): Promise<EncryptionResult> {
    try {
      logger.info('Encryption: Démarrage du chiffrement...');
      const startTime = Date.now();

      // 1. Convertir Blob en ArrayBuffer
      const arrayBuffer = await videoBlob.arrayBuffer();

      // 2. Convertir ArrayBuffer en WordArray pour CryptoJS
      const wordArray = this.arrayBufferToWordArray(arrayBuffer);

      // 3. Chiffrer avec AES-256
      const encrypted = CryptoJS.AES.encrypt(wordArray, key, {
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7,
      });

      // 4. Convertir en string base64
      const encryptedString = encrypted.toString();

      // 5. Convertir en Blob
      const encryptedBlob = new Blob([encryptedString], {
        type: 'application/octet-stream',
      });

      const duration = Date.now() - startTime;
      logger.info('Encryption: Chiffrement terminé', {
        duration: `${duration}ms`,
        originalSize: videoBlob.size,
        encryptedSize: encryptedBlob.size,
      });

      return {
        encryptedData: encryptedBlob,
        key,
        algorithm: 'AES-256',
      };
    } catch (error) {
      logger.error('Encryption: Erreur de chiffrement', error);
      throw new Error('Échec du chiffrement de la vidéo');
    }
  }

  /**
   * Déchiffrer une vidéo (pour tests uniquement)
   */
  async decryptVideo(encryptedBlob: Blob, key: string): Promise<Blob> {
    try {
      logger.info('Encryption: Démarrage du déchiffrement...');

      // 1. Lire le blob chiffré
      const encryptedString = await encryptedBlob.text();

      // 2. Déchiffrer
      const decrypted = CryptoJS.AES.decrypt(encryptedString, key, {
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7,
      });

      // 3. Convertir en ArrayBuffer
      const arrayBuffer = this.wordArrayToArrayBuffer(decrypted);

      // 4. Créer Blob
      const decryptedBlob = new Blob([arrayBuffer], { type: 'video/mp4' });

      logger.info('Encryption: Déchiffrement terminé');
      return decryptedBlob;
    } catch (error) {
      logger.error('Encryption: Erreur de déchiffrement', error);
      throw new Error('Échec du déchiffrement de la vidéo');
    }
  }

  /**
   * Convertir ArrayBuffer en WordArray (pour CryptoJS)
   */
  private arrayBufferToWordArray(arrayBuffer: ArrayBuffer): CryptoJS.lib.WordArray {
    const uint8Array = new Uint8Array(arrayBuffer);
    const words: number[] = [];

    for (let i = 0; i < uint8Array.length; i++) {
      words[i >>> 2] |= uint8Array[i] << (24 - (i % 4) * 8);
    }

    return CryptoJS.lib.WordArray.create(words, uint8Array.length);
  }

  /**
   * Convertir WordArray en ArrayBuffer
   */
  private wordArrayToArrayBuffer(wordArray: CryptoJS.lib.WordArray): ArrayBuffer {
    const words = wordArray.words;
    const sigBytes = wordArray.sigBytes;
    const uint8Array = new Uint8Array(sigBytes);

    for (let i = 0; i < sigBytes; i++) {
      uint8Array[i] = (words[i >>> 2] >>> (24 - (i % 4) * 8)) & 0xff;
    }

    return uint8Array.buffer;
  }

  /**
   * Hacher une donnée (SHA-256)
   */
  hash(data: string): string {
    return CryptoJS.SHA256(data).toString();
  }

  /**
   * Générer un IV (Initialization Vector)
   */
  generateIV(length: number = 16): string {
    return CryptoJS.lib.WordArray.random(length).toString();
  }
}

// Export singleton
export default new EncryptionService();