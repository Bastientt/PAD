// services/audio.service.ts

import { logger } from '@/utils/logger';

/**
 * Service pour gérer les sons inaudibles (ultrasoniques)
 * Pour la détection anti-replay
 */
class AudioService {
  private audioContext: AudioContext | null = null;
  private oscillator: OscillatorNode | null = null;
  private isPlaying: boolean = false;

  /**
   * Initialiser le contexte audio
   */
  private initAudioContext(): void {
    if (!this.audioContext) {
      // @ts-ignore - AudioContext existe sur window
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      logger.debug('Audio: Contexte audio initialisé');
    }
  }

  /**
   * Jouer un son ultrasonique (inaudible pour l'humain)
   * Fréquence recommandée : 18000-22000 Hz
   */
  playUltrasonicSound(frequency: number = 20000, duration: number = 500): void {
    try {
      logger.info('Audio: Lecture son ultrasonique', { frequency, duration });

      this.initAudioContext();

      if (!this.audioContext) {
        throw new Error('AudioContext non disponible');
      }

      // Créer oscillateur
      this.oscillator = this.audioContext.createOscillator();
      this.oscillator.type = 'sine';
      this.oscillator.frequency.value = frequency;

      // Connecter à la sortie
      this.oscillator.connect(this.audioContext.destination);

      // Jouer
      this.oscillator.start();
      this.isPlaying = true;

      // Arrêter après la durée spécifiée
      setTimeout(() => {
        this.stopUltrasonicSound();
      }, duration);

      logger.debug('Audio: Son ultrasonique démarré');
    } catch (error) {
      logger.error('Audio: Erreur lecture son', error);
    }
  }

  /**
   * Arrêter le son
   */
  stopUltrasonicSound(): void {
    if (this.oscillator && this.isPlaying) {
      try {
        this.oscillator.stop();
        this.oscillator.disconnect();
        this.oscillator = null;
        this.isPlaying = false;
        logger.debug('Audio: Son ultrasonique arrêté');
      } catch (error) {
        logger.error('Audio: Erreur arrêt son', error);
      }
    }
  }

  /**
   * Jouer une séquence de sons (pattern)
   */
  async playPattern(pattern: number[]): Promise<void> {
    logger.info('Audio: Lecture pattern', { pattern });

    for (const frequency of pattern) {
      await new Promise<void>((resolve) => {
        this.playUltrasonicSound(frequency, 300);
        setTimeout(resolve, 500); // 300ms son + 200ms pause
      });
    }

    logger.debug('Audio: Pattern terminé');
  }

  /**
   * Générer un pattern aléatoire
   */
  generateRandomPattern(length: number = 3): number[] {
    const frequencies = [18000, 19000, 20000, 21000, 22000];
    const pattern: number[] = [];

    for (let i = 0; i < length; i++) {
      const randomIndex = Math.floor(Math.random() * frequencies.length);
      pattern.push(frequencies[randomIndex]);
    }

    logger.debug('Audio: Pattern généré', { pattern });
    return pattern;
  }

  /**
   * Vérifier si l'AudioContext est supporté
   */
  isSupported(): boolean {
    // @ts-ignore
    return !!(window.AudioContext || window.webkitAudioContext);
  }

  /**
   * Obtenir les capacités audio
   */
  getCapabilities(): {
    supported: boolean;
    maxFrequency: number;
    minFrequency: number;
  } {
    return {
      supported: this.isSupported(),
      maxFrequency: 22000,
      minFrequency: 18000,
    };
  }

  /**
   * Nettoyer les ressources
   */
  cleanup(): void {
    this.stopUltrasonicSound();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
      logger.debug('Audio: Ressources nettoyées');
    }
  }
}

// Export singleton
export default new AudioService();