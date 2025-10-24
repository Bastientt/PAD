// components/upload/upload-progress.tsx

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ProgressBar } from '@/components/ui/progress-bar';

interface UploadProgressProps {
  progress: number; // 0-100
}

export function UploadProgress({ progress }: UploadProgressProps) {
  const getStatusText = () => {
    if (progress < 25) return '📤 Préparation de l\'upload...';
    if (progress < 50) return '⬆️ Envoi en cours...';
    if (progress < 75) return '📡 Transmission des données...';
    if (progress < 100) return '✨ Finalisation...';
    return '✅ Upload terminé !';
  };

  const getStatusColor = () => {
    if (progress < 100) return '#2196f3';
    return '#4caf50';
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Upload en cours</Text>
        <Text style={[styles.status, { color: getStatusColor() }]}>
          {getStatusText()}
        </Text>
      </View>

      <ProgressBar
        progress={progress}
        height={12}
        color={getStatusColor()}
        showPercentage={true}
        animated={true}
      />

      <View style={styles.details}>
        <Text style={styles.detailText}>
          {progress < 100 
            ? 'Ne fermez pas l\'application...' 
            : 'Analyse en cours, veuillez patienter...'}
        </Text>
      </View>

      {/* Animation de chargement */}
      {progress < 100 && (
        <View style={styles.loadingDots}>
          <LoadingDot delay={0} />
          <LoadingDot delay={200} />
          <LoadingDot delay={400} />
        </View>
      )}
    </View>
  );
}

// Composant pour les points de chargement animés
function LoadingDot({ delay }: { delay: number }) {
  const [opacity, setOpacity] = React.useState(0.3);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setOpacity((prev) => (prev === 1 ? 0.3 : 1));
    }, 600);

    const timeout = setTimeout(() => {
      // Démarrer avec le délai
    }, delay);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [delay]);

  return <View style={[styles.dot, { opacity }]} />;
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    marginBottom: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  status: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
  },
  details: {
    marginTop: 12,
  },
  detailText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    fontStyle: 'italic',
  },
  loadingDots: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    marginTop: 16,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#2196f3',
  },
});