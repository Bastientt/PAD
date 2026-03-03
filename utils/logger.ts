// utils/logger.ts

import { LOG_CONFIG } from '@/config/websocket.config';

/**
 * Niveaux de log
 */
export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

/**
 * Couleurs pour les logs (console)
 */
const LOG_COLORS = {
  debug: '#9E9E9E', // Gris
  info: '#2196F3', // Bleu
  warn: '#FF9800', // Orange
  error: '#F44336', // Rouge
};

/**
 * Service de logging
 */
class Logger {
  private enabled: boolean = LOG_CONFIG.enabled;
  private level: LogLevel = LogLevel.DEBUG;

  /**
   * Log debug (détails techniques)
   */
  debug(message: string, data?: any): void {
    if (!this.enabled) return;
    this.log(LogLevel.DEBUG, message, data);
  }

  /**
   * Log info (informations générales)
   */
  info(message: string, data?: any): void {
    if (!this.enabled) return;
    this.log(LogLevel.INFO, message, data);
  }

  /**
   * Log warning (avertissements)
   */
  warn(message: string, data?: any): void {
    if (!this.enabled) return;
    this.log(LogLevel.WARN, message, data);
  }

  /**
   * Log error (erreurs)
   */
  error(message: string, error?: any): void {
    // Les erreurs sont toujours affichées
    this.log(LogLevel.ERROR, message, error);
  }

  /**
   * Log principal
   */
  private log(level: LogLevel, message: string, data?: any): void {
    const timestamp = this.getTimestamp();
    const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
    const color = LOG_COLORS[level];

    // Console avec couleur
    if (LOG_CONFIG.useColors && typeof console !== 'undefined') {
      console.log(
        `%c${prefix}%c ${message}`,
        `color: ${color}; font-weight: bold`,
        'color: inherit'
      );

      if (data !== undefined) {
        if (level === LogLevel.ERROR && data instanceof Error) {
          console.error(data);
        } else {
          console.log(data);
        }
      }
    } else {
      // Console simple
      console.log(`${prefix} ${message}`);
      if (data !== undefined) {
        console.log(data);
      }
    }

    // Envoyer au service de monitoring (optionnel)
    if (level === LogLevel.ERROR) {
      this.sendToMonitoring(level, message, data);
    }
  }

  /**
   * Obtenir timestamp formaté
   */
  private getTimestamp(): string {
    if (!LOG_CONFIG.showTimestamp) return '';

    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const seconds = now.getSeconds().toString().padStart(2, '0');
    const ms = now.getMilliseconds().toString().padStart(3, '0');

    return `${hours}:${minutes}:${seconds}.${ms}`;
  }

  /**
   * Envoyer au service de monitoring (Sentry, LogRocket, etc.)
   */
  private sendToMonitoring(level: LogLevel, message: string, data?: any): void {
    // TODO: Intégrer avec Sentry ou autre service
    // Exemple:
    // if (level === LogLevel.ERROR && Sentry) {
    //   Sentry.captureException(data instanceof Error ? data : new Error(message));
    // }
  }

  /**
   * Log de performance
   */
  time(label: string): void {
    if (!this.enabled) return;
    console.time(label);
  }

  timeEnd(label: string): void {
    if (!this.enabled) return;
    console.timeEnd(label);
  }

  /**
   * Log groupé
   */
  group(label: string): void {
    if (!this.enabled) return;
    console.group(label);
  }

  groupEnd(): void {
    if (!this.enabled) return;
    console.groupEnd();
  }

  /**
   * Activer/désactiver les logs
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Définir le niveau de log
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }
}

// Export singleton
export const logger = new Logger();