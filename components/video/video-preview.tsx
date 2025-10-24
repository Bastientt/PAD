// components/video/video-preview.tsx

import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';

interface VideoPreviewProps {
  uri: string;
  onDelete?: () => void;
  showControls?: boolean;
}

export function VideoPreview({ 
  uri, 
  onDelete,
  showControls = true 
}: VideoPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = React.useRef<Video>(null);

  const handlePlayPause = async () => {
    if (!videoRef.current) return;

    if (isPlaying) {
      await videoRef.current.pauseAsync();
    } else {
      await videoRef.current.playAsync();
    }
    setIsPlaying(!isPlaying);
  };

  const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
    if (status.isLoaded) {
      setIsPlaying(status.isPlaying);
    }
  };

  return (
    <View style={styles.container}>
      <Video
        ref={videoRef}
        source={{ uri }}
        style={styles.video}
        resizeMode={ResizeMode.CONTAIN}
        isLooping
        onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
      />

      {showControls && (
        <View style={styles.controls}>
          <TouchableOpacity
            style={styles.playButton}
            onPress={handlePlayPause}
          >
            <Text style={styles.playButtonText}>
              {isPlaying ? '⏸' : '▶'}
            </Text>
          </TouchableOpacity>

          {onDelete && (
            <TouchableOpacity
              style={styles.deleteButton}
              onPress={onDelete}
            >
              <Text style={styles.deleteButtonText}>🗑️ Supprimer</Text>
            </TouchableOpacity>
          )}
        </View>
      )}

      <View style={styles.info}>
        <Text style={styles.infoText}>
          📹 Vidéo enregistrée
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#000',
    borderRadius: 12,
    overflow: 'hidden',
    marginVertical: 16,
  },
  video: {
    width: '100%',
    height: 300,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    padding: 12,
    backgroundColor: 'rgba(0,0,0,0.8)',
  },
  playButton: {
    backgroundColor: '#2196f3',
    paddingVertical: 8,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  playButtonText: {
    color: '#fff',
    fontSize: 20,
  },
  deleteButton: {
    backgroundColor: '#f44336',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  deleteButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  info: {
    padding: 8,
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
  },
  infoText: {
    color: '#4caf50',
    fontSize: 12,
    textAlign: 'center',
  },
});