// app/(tabs)/upload.tsx

import React from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  ScrollView,
  Alert 
} from 'react-native';
import { useWebSocketUpload } from '@/hooks/use-websocket-upload';
import { VideoRecorder } from '@/components/video/video-recorder';
import { DirectionsDisplay } from '@/components/video/directions-display';
import { UploadProgress } from '@/components/upload/upload-progress';
import { UploadStatus } from '@/components/upload/upload-status';

export default function UploadScreen() {
  const {
    isConnected,
    presignedUrl,
    directions,
    isUploading,
    uploadProgress,
    analysisResult,
    error,
    uploadVideo,
    retry,
  } = useWebSocketUpload();

  const [videoUri, setVideoUri] = React.useState<string | null>(null);
  const [isRecording, setIsRecording] = React.useState(false);

  const handleVideoRecorded = (uri: string) => {
    setVideoUri(uri);
    setIsRecording(false);
  };

  const handleUpload = async () => {
    if (!videoUri) {
      Alert.alert('Erreur', 'Veuillez enregistrer une vidéo d\'abord');
      return;
    }

    try {
      await uploadVideo(videoUri);
    } catch (err) {
      Alert.alert('Erreur', 'Échec de l\'upload');
    }
  };

  const handleStartRecording = () => {
    if (!presignedUrl) {
      Alert.alert('Erreur', 'Connexion en cours...');
      return;
    }
    setIsRecording(true);
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header Section */}
      <View style={styles.header}>
        <Text style={styles.title}>Vérification d'Identité</Text>
        <Text style={styles.subtitle}>
          Suivez les instructions pour enregistrer votre vidéo
        </Text>
      </View>

      {/* Connection Status Indicator */}
      <View style={styles.statusContainer}>
        <View
          style={[
            styles.statusDot,
            isConnected && styles.statusDotConnected,
          ]}
        />
        <Text style={styles.statusText}>
          {isConnected ? '✅ Connecté' : '⏳ Connexion en cours...'}
        </Text>
      </View>

      {/* Directions Display Section */}
      {directions && directions.length > 0 && (
        <DirectionsDisplay directions={directions} />
      )}

      {/* Video Recording Area */}
      {isRecording ? (
        <VideoRecorder
          directions={directions || []}
          onVideoRecorded={handleVideoRecorded}
          onCancel={() => setIsRecording(false)}
        />
      ) : (
        <View style={styles.recordButtonContainer}>
          <TouchableOpacity
            style={[
              styles.recordButton,
              !presignedUrl && styles.recordButtonDisabled,
            ]}
            onPress={handleStartRecording}
            disabled={!presignedUrl}
          >
            <Text style={styles.recordButtonText}>
              {videoUri
                ? '🎥 Réenregistrer'
                : "🎥 Commencer l'enregistrement"}
            </Text>
          </TouchableOpacity>

          {videoUri && (
            <Text style={styles.videoRecordedText}>
              ✅ Vidéo enregistrée
            </Text>
          )}
        </View>
      )}

      {/* Upload Progress Indicator */}
      {isUploading && <UploadProgress progress={uploadProgress} />}

      {/* Upload Button Section */}
      {!isUploading && !analysisResult && videoUri && (
        <TouchableOpacity style={styles.uploadButton} onPress={handleUpload}>
          <Text style={styles.uploadButtonText}>📤 Envoyer la vidéo</Text>
        </TouchableOpacity>
      )}

      {/* General Upload Status (success/error) */}
      <UploadStatus
        isUploading={isUploading}
        analysisResult={analysisResult}
        error={error}
        onRetry={retry}
      />

      {/* Error Message Section */}
      {error && (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>❌ {error}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={retry}>
            <Text style={styles.retryButtonText}>🔄 Réessayer</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
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
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    marginTop: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#ff9800',
    marginRight: 8,
  },
  statusDotConnected: {
    backgroundColor: '#4caf50',
  },
  statusText: {
    fontSize: 14,
    color: '#333',
  },
  recordButtonContainer: {
    padding: 20,
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: '#2196f3',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 8,
    minWidth: 200,
    alignItems: 'center',
  },
  recordButtonDisabled: {
    backgroundColor: '#ccc',
  },
  recordButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  videoRecordedText: {
    marginTop: 12,
    fontSize: 14,
    color: '#4caf50',
  },
  uploadButton: {
    backgroundColor: '#4caf50',
    paddingVertical: 16,
    marginHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorContainer: {
    margin: 20,
    padding: 16,
    backgroundColor: '#ffebee',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#f44336',
  },
  errorText: {
    color: '#d32f2f',
    fontSize: 14,
    marginBottom: 12,
  },
  retryButton: {
    backgroundColor: '#f44336',
    paddingVertical: 12,
    borderRadius: 6,
    alignItems: 'center',
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
});