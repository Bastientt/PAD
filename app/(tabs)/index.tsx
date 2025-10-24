// app/(tabs)/index.tsx

import { View, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

export default function HomeScreen() {
  const router = useRouter();

  return (
    <ThemedView style={styles.container}>
      {/* Header section */}
      <View style={styles.header}>
        <ThemedText type="title" style={styles.title}>
          🔐 Vérification d'Identité
        </ThemedText>
        <ThemedText style={styles.subtitle}>
          Système de détection biométrique sécurisé
        </ThemedText>
      </View>

      {/* Main content */}
      <View style={styles.content}>
        {/* Centered icon */}
        <View style={styles.iconContainer}>
          <ThemedText style={styles.icon}>📱</ThemedText>
        </View>

        {/* Short description */}
        <ThemedText style={styles.description}>
          Créez votre profil numérique en quelques étapes simples
        </ThemedText>

        {/* Key features list */}
        <View style={styles.featuresContainer}>
          <FeatureItem icon="✓" text="Reconnaissance faciale avancée" />
          <FeatureItem icon="✓" text="Détection anti-spoofing" />
          <FeatureItem icon="✓" text="Chiffrement bout-en-bout" />
          <FeatureItem icon="✓" text="Conforme RGPD" />
        </View>
      </View>

      {/* Footer with start button */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => router.push('/upload')}
          activeOpacity={0.8}
        >
          <ThemedText style={styles.buttonText}>
            🚀 Commencer la vérification
          </ThemedText>
        </TouchableOpacity>

        <ThemedText style={styles.footerText}>
          Processus sécurisé • Environ 2 minutes
        </ThemedText>
      </View>
    </ThemedView>
  );
}

// Renders feature items
function FeatureItem({ icon, text }: { icon: string; text: string }) {
  return (
    <View style={styles.featureItem}>
      <ThemedText style={styles.featureIcon}>{icon}</ThemedText>
      <ThemedText style={styles.featureText}>{text}</ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
  },
  header: {
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 32,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    opacity: 0.7,
    textAlign: 'center',
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  iconContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: 'rgba(33, 150, 243, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 24,
  },
  icon: {
    fontSize: 64,
  },
  description: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 32,
    lineHeight: 24,
  },
  featuresContainer: {
    width: '100%',
    gap: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  featureIcon: {
    fontSize: 20,
    color: '#4CAF50',
  },
  featureText: {
    fontSize: 15,
    flex: 1,
  },
  footer: {
    paddingHorizontal: 24,
    paddingBottom: 40,
    gap: 16,
  },
  button: {
    backgroundColor: '#2196F3',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#2196F3',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  footerText: {
    fontSize: 12,
    textAlign: 'center',
    opacity: 0.6,
  },
});