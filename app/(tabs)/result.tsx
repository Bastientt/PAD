// app/(tabs)/result.tsx

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from 'react-native';
import { useRouter } from 'expo-router';
import { ResultCard } from '@/components/result/result-card';
import { ConfidenceMeter } from '@/components/result/confidence-meter';
import { useWebSocketUpload } from '@/hooks/use-websocket-upload';

export default function ResultScreen() {
  const router = useRouter();
  const { analysisResult, error, retry } = useWebSocketUpload();

  const handleNewVerification = () => {
    retry();
    router.push('/upload');
  };

  if (!analysisResult && !error) {
    return (
      <View style={styles.container}>
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>📋</Text>
          <Text style={styles.emptyStateTitle}>Aucun résultat</Text>
          <Text style={styles.emptyStateText}>
            Effectuez une vérification d'identité pour voir les résultats
          </Text>

          {/* Start new verification */}
          <TouchableOpacity
            style={styles.startButton}
            onPress={() => router.push('/upload')}
          >
            <Text style={styles.startButtonText}>
              Commencer la vérification
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorIcon}>❌</Text>
          <Text style={styles.errorTitle}>Erreur</Text>
          <Text style={styles.errorText}>{error}</Text>

          {/* Retry verification */}
          <TouchableOpacity
            style={styles.retryButton}
            onPress={handleNewVerification}
          >
            <Text style={styles.retryButtonText}>🔄 Réessayer</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Résultat de la Vérification</Text>
        <Text style={styles.subtitle}>
          {new Date().toLocaleDateString('fr-FR', {
            day: 'numeric',
            month: 'long',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </Text>
      </View>

      {/* Main result summary */}
      <ResultCard result={analysisResult!} />

      {/* Confidence score section */}
      <View style={styles.confidenceSection}>
        <Text style={styles.sectionTitle}>Score de Confiance</Text>
        <ConfidenceMeter confidence={analysisResult?.confidence || 0} />
      </View>

      {/* Detailed analysis info */}
      {analysisResult?.details && (
        <View style={styles.detailsSection}>
          <Text style={styles.sectionTitle}>Détails de l'Analyse</Text>
          <View style={styles.detailsCard}>
            <DetailItem
              label="Visage détecté"
              value={analysisResult.details.face_detected}
            />
            <DetailItem
              label="Mouvements corrects"
              value={analysisResult.details.movements_correct}
            />
            <DetailItem
              label="Score de vivacité"
              value={`${Math.round(
                (analysisResult.details.liveness_score || 0) * 100
              )}%`}
            />
            {analysisResult.details.ultrasonic_detected !== undefined && (
              <DetailItem
                label="Son ultrasonique détecté"
                value={analysisResult.details.ultrasonic_detected}
              />
            )}
          </View>
        </View>
      )}

      {/* Buttons for new verification or continuation */}
      <View style={styles.actionsSection}>
        <TouchableOpacity
          style={styles.newVerificationButton}
          onPress={handleNewVerification}
        >
          <Text style={styles.newVerificationButtonText}>
            🔄 Nouvelle vérification
          </Text>
        </TouchableOpacity>

        {analysisResult?.success && (
          <TouchableOpacity
            style={styles.continueButton}
            onPress={() => {
              console.log('Continuer...');
            }}
          >
            <Text style={styles.continueButtonText}>➡️ Continuer</Text>
          </TouchableOpacity>
        )}
      </View>
    </ScrollView>
  );
}

// Composant helper to show details
function DetailItem({ label, value }: { label: string; value: boolean | string }) {
  const displayValue = typeof value === 'boolean' 
    ? (value ? '✅ Oui' : '❌ Non')
    : value;

  return (
    <View style={styles.detailItem}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{displayValue}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
    color: '#666',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyStateIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyStateTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 24,
  },
  startButton: {
    backgroundColor: '#2196f3',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 8,
  },
  startButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  errorIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#f44336',
  },
  errorText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 24,
  },
  retryButton: {
    backgroundColor: '#f44336',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  confidenceSection: {
    margin: 16,
    padding: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  detailsSection: {
    margin: 16,
    marginTop: 0,
  },
  detailsCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
  },
  detailItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  actionsSection: {
    padding: 16,
    gap: 12,
  },
  newVerificationButton: {
    backgroundColor: '#2196f3',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  newVerificationButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  continueButton: {
    backgroundColor: '#4caf50',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  continueButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});