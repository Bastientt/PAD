// components/video/directions-display.tsx

import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';

interface DirectionsDisplayProps {
  directions: string[];
  currentIndex?: number;
}

export function DirectionsDisplay({ 
  directions, 
  currentIndex = -1 
}: DirectionsDisplayProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Instructions à suivre :</Text>
      
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.directionsContainer}
      >
        {directions.map((direction, index) => {
          const isActive = index === currentIndex;
          const isPast = index < currentIndex;

          return (
            <View
              key={index}
              style={[
                styles.directionCard,
                isActive && styles.directionCardActive,
                isPast && styles.directionCardPast,
              ]}
            >
              <Text style={styles.directionNumber}>
                {index + 1}
              </Text>
              <Text style={[
                styles.directionIcon,
                isActive && styles.directionIconActive,
              ]}>
                {getDirectionIcon(direction)}
              </Text>
              <Text style={[
                styles.directionText,
                isActive && styles.directionTextActive,
              ]}>
                {getDirectionLabel(direction)}
              </Text>
              
              {isPast && (
                <View style={styles.checkmark}>
                  <Text style={styles.checkmarkText}>✓</Text>
                </View>
              )}
            </View>
          );
        })}
      </ScrollView>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          {directions.length} étape{directions.length > 1 ? 's' : ''} • Durée estimée : {directions.length * 3}s
        </Text>
      </View>
    </View>
  );
}

function getDirectionIcon(direction: string): string {
  const icons: Record<string, string> = {
    left: '👈',
    right: '👉',
    up: '👆',
    down: '👇',
    center: '👤',
  };
  return icons[direction.toLowerCase()] || '•';
}

function getDirectionLabel(direction: string): string {
  const labels: Record<string, string> = {
    left: 'Gauche',
    right: 'Droite',
    up: 'Haut',
    down: 'Bas',
    center: 'Centre',
  };
  return labels[direction.toLowerCase()] || direction;
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    margin: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  directionsContainer: {
    flexDirection: 'row',
    gap: 12,
    paddingVertical: 4,
  },
  directionCard: {
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    padding: 12,
    minWidth: 80,
    alignItems: 'center',
    position: 'relative',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  directionCardActive: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196f3',
  },
  directionCardPast: {
    backgroundColor: '#e8f5e9',
    opacity: 0.7,
  },
  directionNumber: {
    position: 'absolute',
    top: 4,
    right: 4,
    fontSize: 10,
    fontWeight: 'bold',
    color: '#999',
    backgroundColor: '#fff',
    borderRadius: 8,
    width: 16,
    height: 16,
    textAlign: 'center',
    lineHeight: 16,
  },
  directionIcon: {
    fontSize: 32,
    marginBottom: 4,
  },
  directionIconActive: {
    fontSize: 36,
  },
  directionText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
  },
  directionTextActive: {
    color: '#2196f3',
    fontWeight: 'bold',
  },
  checkmark: {
    position: 'absolute',
    top: 4,
    left: 4,
    backgroundColor: '#4caf50',
    borderRadius: 10,
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkmarkText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  footer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  footerText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
});