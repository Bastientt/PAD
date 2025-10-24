// components/result/result-card.tsx

import React from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';
import { AnalysisResult } from '@/types/analysis.types';

interface ResultCardProps {
  result: AnalysisResult;
}

export function ResultCard({ result }: ResultCardProps) {
  const scaleAnim = React.useRef(new Animated.Value(0)).current;
  const opacityAnim = React.useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    // Animation d'entrée
    Animated.parallel([
      Animated.spring(scaleAnim, {
        toValue: 1,
        tension: 50,
        friction: 7,
        useNativeDriver: true,
      }),
      Animated.timing(opacityAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const isSuccess = result.success;
  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <Animated.View
      style={[
        styles.container,
        isSuccess ? styles.successContainer : styles.failureContainer,
        {
          transform: [{ scale: scaleAnim }],
          opacity: opacityAnim,
        },
      ]}
    >
      {/* Icône et titre principal */}
      <View style={styles.header}>
        <View
          style={[
            styles.iconContainer,
            isSuccess ? styles.iconContainerSuccess : styles.iconContainerFailure,
          ]}
        >
          <Text style={styles.icon}>{isSuccess ? '✓' : '✗'}</Text>
        </View>

        <View style={styles.headerText}>
          <Text style={styles.title}>
            {isSuccess ? 'Identité Vérifiée' : 'Vérification Échouée'}
          </Text>
          <Text style={styles.subtitle}>
            {isSuccess
              ? 'Votre identité a été confirmée avec succès'
              : 'La vérification n\'a pas pu être complétée'}
          </Text>
        </View>
      </View>

      {/* Score de confiance principal */}
      <View style={styles.scoreContainer}>
        <Text style={styles.scoreLabel}>Score de Confiance</Text>
        <View style={styles.scoreRow}>
          <Text
            style={[
              styles.scoreValue,
              isSuccess ? styles.scoreValueSuccess : styles.scoreValueFailure,
            ]}
          >
            {confidencePercent}%
          </Text>
          <View style={styles.scoreBadge}>
            <Text style={styles.scoreBadgeText}>
              {getConfidenceLevel(result.confidence)}
            </Text>
          </View>
        </View>
      </View>

      {/* Message détaillé */}
      {result.message && (
        <View style={styles.messageContainer}>
          <Text style={styles.messageText}>{result.message}</Text>
        </View>
      )}

      {/* Indicateurs visuels */}
      <View style={styles.indicators}>
        <Indicator
          label="Visage"
          status={result.details?.face_detected}
          icon="👤"
        />
        <Indicator
          label="Mouvements"
          status={result.details?.movements_correct}
          icon="↔️"
        />
        <Indicator
          label="Vivacité"
          status={(result.details?.liveness_score || 0) > 0.7}
          icon="✨"
        />
      </View>

      {/* Timestamp */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          🕐 {new Date().toLocaleTimeString('fr-FR', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </Text>
        <Text style={styles.footerText}>
          📅 {new Date().toLocaleDateString('fr-FR')}
        </Text>
      </View>
    </Animated.View>
  );
}

// Composant Indicator pour les checks rapides
interface IndicatorProps {
  label: string;
  status?: boolean;
  icon: string;
}

function Indicator({ label, status = false, icon }: IndicatorProps) {
  return (
    <View style={styles.indicator}>
      <View
        style={[
          styles.indicatorIcon,
          status ? styles.indicatorIconSuccess : styles.indicatorIconFailure,
        ]}
      >
        <Text style={styles.indicatorIconText}>{icon}</Text>
      </View>
      <Text style={styles.indicatorLabel}>{label}</Text>
      <View
        style={[
          styles.indicatorStatus,
          status ? styles.indicatorStatusSuccess : styles.indicatorStatusFailure,
        ]}
      >
        <Text style={styles.indicatorStatusText}>
          {status ? '✓' : '✗'}
        </Text>
      </View>
    </View>
  );
}

// Helper function pour le niveau de confiance
function getConfidenceLevel(confidence: number): string {
  if (confidence >= 0.9) return 'Excellent';
  if (confidence >= 0.8) return 'Très Bon';
  if (confidence >= 0.7) return 'Bon';
  if (confidence >= 0.5) return 'Moyen';
  return 'Faible';
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    padding: 24,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 6,
  },
  successContainer: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#4caf50',
  },
  failureContainer: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#ff9800',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
  },
  iconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  iconContainerSuccess: {
    backgroundColor: '#4caf50',
  },
  iconContainerFailure: {
    backgroundColor: '#ff9800',
  },
  icon: {
    fontSize: 32,
    color: '#fff',
    fontWeight: 'bold',
  },
  headerText: {
    flex: 1,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 13,
    color: '#666',
    lineHeight: 18,
  },
  scoreContainer: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  scoreLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
    fontWeight: '600',
  },
  scoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
    letterSpacing: -2,
  },
  scoreValueSuccess: {
    color: '#4caf50',
  },
  scoreValueFailure: {
    color: '#ff9800',
  },
  scoreBadge: {
    backgroundColor: '#e0e0e0',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  scoreBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
  },
  messageContainer: {
    backgroundColor: '#f0f4ff',
    borderLeftWidth: 4,
    borderLeftColor: '#2196f3',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  messageText: {
    fontSize: 13,
    color: '#333',
    lineHeight: 18,
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#f0f0f0',
  },
  indicator: {
    alignItems: 'center',
    flex: 1,
  },
  indicatorIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 6,
  },
  indicatorIconSuccess: {
    backgroundColor: '#e8f5e9',
  },
  indicatorIconFailure: {
    backgroundColor: '#fff3e0',
  },
  indicatorIconText: {
    fontSize: 20,
  },
  indicatorLabel: {
    fontSize: 11,
    color: '#666',
    marginBottom: 4,
  },
  indicatorStatus: {
    width: 20,
    height: 20,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  indicatorStatusSuccess: {
    backgroundColor: '#4caf50',
  },
  indicatorStatusFailure: {
    backgroundColor: '#ff9800',
  },
  indicatorStatusText: {
    fontSize: 12,
    color: '#fff',
    fontWeight: 'bold',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  footerText: {
    fontSize: 11,
    color: '#999',
  },
});