// components/upload/upload-status.tsx

import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { AnalysisResult } from '@/types/analysis.types';

interface UploadStatusProps {
  isUploading: boolean;
  analysisResult: AnalysisResult | null;
  error: string | null;
  onRetry: () => void;
}

export function UploadStatus({
  isUploading,
  analysisResult,
  error,
  onRetry,
}: UploadStatusProps) {
  // En cours d'upload
  if (isUploading) {
    return (
      <View style={[styles.container, styles.uploadingContainer]}>
        <Text style={styles.icon}>⏳</Text>
        <Text style={styles.title}>Analyse en cours...</Text>
        <Text style={styles.message}>
          Votre vidéo est en cours d'analyse par notre système de détection.
        </Text>
      </View>
    );
  }

  // Erreur
  if (error) {
    return (
      <View style={[styles.container, styles.errorContainer]}>
        <Text style={styles.icon}>❌</Text>
        <Text style={styles.title}>Erreur</Text>
        <Text style={styles.message}>{error}</Text>
        <TouchableOpacity style={styles.button} onPress={onRetry}>
          <Text style={styles.buttonText}>🔄 Réessayer</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Résultat disponible
  if (analysisResult) {
    const isSuccess = analysisResult.success;

    return (
      <View
        style={[
          styles.container,
          isSuccess ? styles.successContainer : styles.failureContainer,
        ]}
      >
        <Text style={styles.icon}>{isSuccess ? '✅' : '❌'}</Text>
        <Text style={styles.title}>
          {isSuccess ? 'Vérification réussie !' : 'Vérification échouée'}
        </Text>
        <Text style={styles.message}>
          {analysisResult.message || 
            (isSuccess 
              ? `Identité vérifiée avec ${Math.round(analysisResult.confidence * 100)}% de confiance`
              : 'La vérification a échoué. Veuillez réessayer.'
            )
          }
        </Text>

        {/* Score de confiance */}
        <View style={styles.confidenceContainer}>
          <Text style={styles.confidenceLabel}>Score de confiance :</Text>
          <View style={styles.confidenceBar}>
            <View
              style={[
                styles.confidenceFill,
                {
                  width: `${analysisResult.confidence * 100}%`,
                  backgroundColor: isSuccess ? '#4caf50' : '#ff9800',
                },
              ]}
            />
          </View>
          <Text style={styles.confidenceValue}>
            {Math.round(analysisResult.confidence * 100)}%
          </Text>
        </View>

        {!isSuccess && (
          <TouchableOpacity style={styles.button} onPress={onRetry}>
            <Text style={styles.buttonText}>🔄 Réessayer</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  }

  // État initial
  return null;
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 12,
    padding: 20,
    margin: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  uploadingContainer: {
    backgroundColor: '#e3f2fd',
    borderWidth: 2,
    borderColor: '#2196f3',
  },
  successContainer: {
    backgroundColor: '#e8f5e9',
    borderWidth: 2,
    borderColor: '#4caf50',
  },
  failureContainer: {
    backgroundColor: '#fff3e0',
    borderWidth: 2,
    borderColor: '#ff9800',
  },
  errorContainer: {
    backgroundColor: '#ffebee',
    borderWidth: 2,
    borderColor: '#f44336',
  },
  icon: {
    fontSize: 48,
    marginBottom: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 16,
  },
  confidenceContainer: {
    width: '100%',
    marginTop: 12,
  },
  confidenceLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 6,
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 6,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'right',
  },
  button: {
    backgroundColor: '#2196f3',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginTop: 12,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
});