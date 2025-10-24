// components/video/video-recorder.tsx

import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Camera, CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Video, ResizeMode } from 'expo-av';

interface VideoRecorderProps {
  directions: string[];
  onVideoRecorded: (uri: string) => void;
  onCancel: () => void;
}

export function VideoRecorder({
  directions,
  onVideoRecorded,
  onCancel,
}: VideoRecorderProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [currentDirectionIndex, setCurrentDirectionIndex] = useState(0);
  const [recordedUri, setRecordedUri] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  const startRecording = async () => {
    if (!cameraRef.current) return;

    try {
      setIsRecording(true);
      const video = await cameraRef.current.recordAsync({
  maxDuration: 30,
});

if (video && video.uri) {  // ← Vérification ajoutée
  setRecordedUri(video.uri);
  setIsRecording(false);
} else {
  throw new Error('Aucune vidéo enregistrée');
}
      setIsRecording(false);
    } catch (error) {
      console.error('Erreur enregistrement:', error);
      Alert.alert('Erreur', 'Impossible d\'enregistrer la vidéo');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (cameraRef.current && isRecording) {
      cameraRef.current.stopRecording();
    }
  };

  const handleConfirm = () => {
    if (recordedUri) {
      onVideoRecorded(recordedUri);
    }
  };

  const handleRetake = () => {
    setRecordedUri(null);
    setCurrentDirectionIndex(0);
  };

  // Changer de direction toutes les 3 secondes pendant l'enregistrement
  useEffect(() => {
    if (isRecording && currentDirectionIndex < directions.length - 1) {
      const timer = setTimeout(() => {
        setCurrentDirectionIndex((prev) => prev + 1);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [isRecording, currentDirectionIndex, directions.length]);

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text>Chargement de la caméra...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          Permission caméra requise
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Autoriser</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Prévisualisation après enregistrement
  if (recordedUri) {
    return (
      <View style={styles.container}>
        <View style={styles.previewContainer}>
          <Video
            source={{ uri: recordedUri }}
            style={styles.video}
            useNativeControls
            resizeMode={ResizeMode.CONTAIN}  
            isLooping
            />
        </View>

        <View style={styles.actionsContainer}>
          <TouchableOpacity style={styles.retakeButton} onPress={handleRetake}>
            <Text style={styles.retakeButtonText}>🔄 Refaire</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.confirmButton} onPress={handleConfirm}>
            <Text style={styles.confirmButtonText}>✓ Confirmer</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="front"
      >
        {/* Overlay avec instruction courante */}
        <View style={styles.overlay}>
          <View style={styles.instructionContainer}>
            <Text style={styles.instructionText}>
              {getDirectionInstruction(directions[currentDirectionIndex])}
            </Text>
          </View>

          {/* Indicateur de progression */}
          <View style={styles.progressContainer}>
            {directions.map((_, index) => (
              <View
                key={index}
                style={[
                  styles.progressDot,
                  index <= currentDirectionIndex && styles.progressDotActive,
                ]}
              />
            ))}
          </View>
        </View>
      </CameraView>

      {/* Boutons de contrôle */}
      <View style={styles.controls}>
        <TouchableOpacity style={styles.cancelButton} onPress={onCancel}>
          <Text style={styles.cancelButtonText}>Annuler</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.recordButton, isRecording && styles.recordButtonActive]}
          onPress={isRecording ? stopRecording : startRecording}
        >
          <View style={[styles.recordButtonInner, isRecording && styles.recordButtonInnerActive]} />
        </TouchableOpacity>

        <View style={styles.placeholder} />
      </View>
    </View>
  );
}

// Helper pour traduire les directions
function getDirectionInstruction(direction: string): string {
  const instructions: Record<string, string> = {
    left: '👈 Tournez la tête à GAUCHE',
    right: '👉 Tournez la tête à DROITE',
    up: '👆 Levez la tête en HAUT',
    down: '👇 Baissez la tête en BAS',
    center: '👤 Regardez face à la caméra',
  };
  return instructions[direction.toLowerCase()] || direction;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.3)',
    justifyContent: 'space-between',
    paddingVertical: 40,
  },
  instructionContainer: {
    backgroundColor: 'rgba(33, 150, 243, 0.9)',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginHorizontal: 20,
    alignSelf: 'center',
  },
  instructionText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
  },
  progressDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: 'rgba(255,255,255,0.3)',
  },
  progressDotActive: {
    backgroundColor: '#4caf50',
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 24,
    paddingHorizontal: 40,
    backgroundColor: '#000',
  },
  cancelButton: {
    paddingVertical: 12,
    paddingHorizontal: 20,
  },
  cancelButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  recordButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#fff',
  },
  recordButtonActive: {
    borderColor: '#f44336',
  },
  recordButtonInner: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#f44336',
  },
  recordButtonInnerActive: {
    borderRadius: 6,
    width: 30,
    height: 30,
  },
  placeholder: {
    width: 60,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
  },
  video: {
    width: '100%',
    height: 400,
  },
  actionsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 20,
    backgroundColor: '#000',
  },
  retakeButton: {
    backgroundColor: '#ff9800',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 8,
    flex: 1,
    marginRight: 8,
  },
  retakeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  confirmButton: {
    backgroundColor: '#4caf50',
    paddingVertical: 14,
    paddingHorizontal: 32,
    borderRadius: 8,
    flex: 1,
    marginLeft: 8,
  },
  confirmButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#2196f3',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});