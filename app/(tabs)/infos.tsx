import {
  View,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Linking,
  Animated,
  useWindowDimensions,
} from 'react-native';
import { useEffect, useRef } from 'react';
import { LinearGradient } from 'expo-linear-gradient';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';

export default function InfosScreen() {
  const fade = useRef(new Animated.Value(0)).current;
  const move = useRef(new Animated.Value(40)).current;
  const { width } = useWindowDimensions();
  const small = width < 370;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      }),
      Animated.timing(move, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <ThemedView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        {/* HEADER */}
        <LinearGradient
          colors={['#0f172a', '#1e293b', '#4338ca']}
          style={[
            styles.header,
            {
              paddingTop: small ? 90 : 110,
              paddingBottom: small ? 50 : 70,
            },
          ]}
        >
          <ThemedText
            style={[
              styles.logo,
              {
                fontSize: small ? 30 : 36,
                letterSpacing: small ? 6 : 8,
              },
            ]}
          >
            PAD
          </ThemedText>

          <ThemedText
            style={[
              styles.tagline,
              { fontSize: small ? 12 : 14 },
            ]}
          >
            A Next-Gen Biometric Identity
          </ThemedText>
        </LinearGradient>

        <Animated.View
          style={{
            opacity: fade,
            transform: [{ translateY: move }],
          }}
        >
          {/* ABOUT */}
          <GlassCard icon="shield.lefthalf.filled" title="About PAD" small={small}>
            <ThemedText
              style={[
                styles.text,
                { fontSize: small ? 14 : 15 },
              ]}
            >
              PAD is a secure real-time biometric identity verification
              platform built with React Native and Expo.
              {'\n\n'}
              It integrates facial recognition, liveness detection and
              encrypted communication for seamless and compliant verification.
            </ThemedText>
          </GlassCard>

          {/* TEAM */}
          <GlassCard icon="person.3.sequence.fill" title="Team" small={small}>
            <ModernMember name="Ayman Chergui" small={small} />
            <ModernMember name="Bastien Schneider" small={small} />
            <ModernMember name="Mathieu Fraixanet" small={small} />
          </GlassCard>

          {/* OWNERS */}
          <LinearGradient
            colors={['#1e1b4b', '#312e81']}
            style={[
              styles.ownerCard,
              {
                marginHorizontal: small ? 16 : 22,
                padding: small ? 22 : 28,
              },
            ]}
          >
            <View style={styles.cardHeader}>
              <View style={styles.ownerIconWrapper}>
                <IconSymbol
                  name="crown.fill"
                  size={16}
                  color="#facc15"
                />
              </View>
              <ThemedText style={styles.cardTitle}>
                Owners
              </ThemedText>
            </View>

            <ModernMember name="Jessica Giacobi" highlight small={small} />
            <ModernMember name="Florian SANANES" highlight small={small} />
          </LinearGradient>

          {/* CTA */}
          <TouchableOpacity
            activeOpacity={0.9}
            onPress={() =>
              Linking.openURL('https://github.com/Bastientt/PAD')
            }
            style={[
              styles.buttonWrapper,
              { width: small ? '85%' : undefined },
            ]}
          >
            <LinearGradient
              colors={['#6366f1', '#8b5cf6']}
              style={[
                styles.button,
                {
                  paddingVertical: small ? 14 : 18,
                  paddingHorizontal: small ? 30 : 50,
                },
              ]}
            >
              <IconSymbol
                name="chevron.left.forwardslash.chevron.right"
                size={18}
                color="#fff"
              />
              <ThemedText style={styles.buttonText}>
                GitHub Repository
              </ThemedText>
            </LinearGradient>
          </TouchableOpacity>
        </Animated.View>

        <ThemedText style={styles.footer}>
          PAD • Version 1.0
        </ThemedText>
      </ScrollView>
    </ThemedView>
  );
}

/* ---------- Components ---------- */

function GlassCard({ icon, title, children, small }: any) {
  return (
    <View
      style={[
        styles.card,
        {
          marginHorizontal: small ? 16 : 22,
          padding: small ? 20 : 26,
        },
      ]}
    >
      <View style={styles.cardHeader}>
        <View style={styles.iconWrapper}>
          <IconSymbol name={icon} size={16} color="#818cf8" />
        </View>
        <ThemedText style={styles.cardTitle}>{title}</ThemedText>
      </View>
      {children}
    </View>
  );
}

function ModernMember({
  name,
  highlight = false,
  small,
}: {
  name: string;
  highlight?: boolean;
  small: boolean;
}) {
  return (
    <View style={styles.memberRow}>
      <LinearGradient
        colors={
          highlight
            ? ['#facc15', '#eab308']
            : ['#6366f1', '#4338ca']
        }
        style={[
          styles.modernDot,
          {
            width: small ? 8 : 10,
            height: small ? 8 : 10,
          },
        ]}
      />
      <ThemedText
        style={[
          styles.memberText,
          { fontSize: small ? 14 : 16 },
          highlight && { fontWeight: '700', color: '#facc15' },
        ]}
      >
        {name}
      </ThemedText>
    </View>
  );
}

/* ---------- Styles ---------- */

const styles = StyleSheet.create({
  container: { flex: 1 },

  scroll: { paddingBottom: 100 },

  header: {
    alignItems: 'center',
    borderBottomLeftRadius: 50,
    borderBottomRightRadius: 50,
  },

  logo: {
    fontWeight: '800',
    color: '#fff',
  },

  tagline: {
    marginTop: 10,
    color: '#c7d2fe',
  },

  card: {
    marginTop: 24,
    borderRadius: 24,
    backgroundColor: 'rgba(255,255,255,0.06)',
    shadowColor: '#000',
  },

  ownerCard: {
    marginTop: 28,
    borderRadius: 26,
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 35,
    elevation: 20,
  },

  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 18,
  },

  cardTitle: {
    marginLeft: 12,
    fontSize: 18,
    fontWeight: '700',
  },

  iconWrapper: {
    backgroundColor: 'rgba(99,102,241,0.2)',
    padding: 6,
    borderRadius: 20,
  },

  ownerIconWrapper: {
    backgroundColor: 'rgba(250,204,21,0.2)',
    padding: 6,
    borderRadius: 20,
    marginRight: 8,
  },

  text: {
    lineHeight: 22,
    opacity: 0.9,
  },

  memberRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },

  modernDot: {
    borderRadius: 10,
    marginRight: 12,
  },

  memberText: {},

  buttonWrapper: {
    marginTop: 40,
    alignSelf: 'center',
    borderRadius: 50,
    overflow: 'hidden',
  },

  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },

  buttonText: {
    marginLeft: 12,
    color: '#fff',
    fontWeight: '700',
    fontSize: 15,
  },

  footer: {
    marginTop: 60,
    textAlign: 'center',
    opacity: 0.4,
    fontSize: 12,
  },
});