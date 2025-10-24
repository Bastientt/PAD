// components/result/confidence-meter.tsx

import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';

interface ConfidenceMeterProps {
  confidence: number; // 0-1
  size?: number;
  showLabel?: boolean;
  animated?: boolean;
}

export function ConfidenceMeter({
  confidence,
  size = 200,
  showLabel = true,
  animated = true,
}: ConfidenceMeterProps) {
  const animatedValue = useRef(new Animated.Value(0)).current;
  const confidencePercent = Math.round(confidence * 100);

  useEffect(() => {
    if (animated) {
      Animated.timing(animatedValue, {
        toValue: confidence,
        duration: 1500,
        useNativeDriver: false,
      }).start();
    } else {
      animatedValue.setValue(confidence);
    }
  }, [confidence, animated]);

  // Couleur basée sur le score
  const getColor = () => {
    if (confidence >= 0.8) return '#4caf50'; // Vert
    if (confidence >= 0.6) return '#8bc34a'; // Vert clair
    if (confidence >= 0.4) return '#ff9800'; // Orange
    return '#f44336'; // Rouge
  };

  // Angle pour le demi-cercle (0 = gauche, 180 = droite)
  const angle = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 180],
  });

  const rotation = angle.interpolate({
    inputRange: [0, 180],
    outputRange: ['0deg', '180deg'],
  });

  const color = getColor();

  return (
    <View style={[styles.container, { width: size, height: size / 2 + 40 }]}>
      {/* Gauge en demi-cercle */}
      <View style={[styles.gaugeContainer, { width: size, height: size / 2 }]}>
        {/* Background arc */}
        <View
          style={[
            styles.arcBackground,
            {
              width: size,
              height: size,
              borderRadius: size / 2,
              borderWidth: size * 0.15,
            },
          ]}
        />

        {/* Animated arc */}
        <Animated.View
          style={[
            styles.arcForeground,
            {
              width: size,
              height: size,
              borderRadius: size / 2,
              borderWidth: size * 0.15,
              borderColor: color,
              transform: [{ rotate: rotation }],
            },
          ]}
        />

        {/* Needle/Pointer */}
        <Animated.View
          style={[
            styles.needle,
            {
              width: size * 0.6,
              height: 3,
              bottom: size * 0.08,
              left: size * 0.2,
              backgroundColor: color,
              transform: [
                { translateX: size * 0.3 },
                { rotate: rotation },
                { translateX: -size * 0.3 },
              ],
            },
          ]}
        >
          <View
            style={[
              styles.needleCircle,
              {
                width: size * 0.08,
                height: size * 0.08,
                borderRadius: size * 0.04,
                backgroundColor: color,
              },
            ]}
          />
        </Animated.View>
      </View>

      {/* Score central */}
      <View style={styles.scoreContainer}>
        <Text style={[styles.score, { fontSize: size * 0.2, color }]}>
          {confidencePercent}%
        </Text>
        {showLabel && (
          <Text style={[styles.label, { fontSize: size * 0.06 }]}>
            {getConfidenceLabel(confidence)}
          </Text>
        )}
      </View>

      {/* Échelle graduée */}
      <View style={[styles.scale, { width: size }]}>
        <View style={styles.scaleItem}>
          <Text style={styles.scaleValue}>0%</Text>
          <Text style={styles.scaleLabel}>Faible</Text>
        </View>
        <View style={styles.scaleItem}>
          <Text style={styles.scaleValue}>50%</Text>
          <Text style={styles.scaleLabel}>Moyen</Text>
        </View>
        <View style={styles.scaleItem}>
          <Text style={styles.scaleValue}>100%</Text>
          <Text style={styles.scaleLabel}>Excellent</Text>
        </View>
      </View>

      {/* Indicateurs de zone */}
      <View style={[styles.zones, { width: size }]}>
        <View style={[styles.zone, { backgroundColor: '#f443361a' }]} />
        <View style={[styles.zone, { backgroundColor: '#ff98001a' }]} />
        <View style={[styles.zone, { backgroundColor: '#8bc34a1a' }]} />
        <View style={[styles.zone, { backgroundColor: '#4caf501a' }]} />
      </View>
    </View>
  );
}

function getConfidenceLabel(confidence: number): string {
  if (confidence >= 0.9) return '🏆 Excellent';
  if (confidence >= 0.8) return '⭐ Très Bon';
  if (confidence >= 0.7) return '✓ Bon';
  if (confidence >= 0.6) return '~ Acceptable';
  if (confidence >= 0.4) return '⚠ Moyen';
  return '✗ Faible';
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  gaugeContainer: {
    position: 'relative',
    overflow: 'hidden',
    alignItems: 'center',
  },
  arcBackground: {
    position: 'absolute',
    top: 0,
    borderColor: '#e0e0e0',
    borderBottomColor: 'transparent',
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
  },
  arcForeground: {
    position: 'absolute',
    top: 0,
    borderBottomColor: 'transparent',
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
  },
  needle: {
    position: 'absolute',
    transformOrigin: 'left center',
    borderRadius: 2,
  },
  needleCircle: {
    position: 'absolute',
    right: -4,
    top: -4,
  },
  scoreContainer: {
    position: 'absolute',
    bottom: 60,
    alignItems: 'center',
  },
  score: {
    fontWeight: 'bold',
    letterSpacing: -2,
  },
  label: {
    color: '#666',
    marginTop: 4,
    fontWeight: '600',
  },
  scale: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },
  scaleItem: {
    alignItems: 'center',
  },
  scaleValue: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#666',
  },
  scaleLabel: {
    fontSize: 9,
    color: '#999',
    marginTop: 2,
  },
  zones: {
    flexDirection: 'row',
    marginTop: 8,
    height: 4,
    borderRadius: 2,
    overflow: 'hidden',
  },
  zone: {
    flex: 1,
  },
});