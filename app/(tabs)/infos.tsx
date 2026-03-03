import { 
  View, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity, 
  Linking 
} from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { IconSymbol } from '@/components/ui/icon-symbol';

export default function InfosScreen() {
  return (
    <ThemedView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >

        {/* Title */}
        <ThemedText type="title" style={styles.title}>
          À propos de PAD
        </ThemedText>

        {/* Intro */}
        <ThemedText style={styles.paragraph}>
          PAD est un système de{' '}
          <ThemedText style={styles.highlight}>
            vérification d’identité biométrique
          </ThemedText>{' '}
          conçue pour garantir{' '}
          <ThemedText style={styles.highlight}>sécurité</ThemedText>,{' '}
          <ThemedText style={styles.highlight}>rapidité</ThemedText> et{' '}
          <ThemedText style={styles.highlight}>fiabilité</ThemedText>.
          {'\n\n'}
          Elle fonctionne en capturant votre visage via la caméra, en appliquant une{' '}
          <ThemedText style={styles.highlight}>
            analyse anti-spoofing avancée
          </ThemedText>
          , puis en{' '}
          <ThemedText style={styles.highlight}>chiffrant les données</ThemedText>{' '}
          avant traitement afin de fournir un{' '}
          <ThemedText style={styles.highlight}>résultat instantané</ThemedText>.
        </ThemedText>

        {/* Security */}
        <Section title="🔐 Sécurité & confidentialité">
          <ThemedText style={styles.paragraph}>
            Toutes les données sont chiffrées avant transmission.
          </ThemedText>

          <ThemedText style={styles.paragraph}>
            PAD respecte strictement les normes RGPD et les meilleures pratiques de sécurité.
          </ThemedText>
        </Section>

        {/* Team */}
        <Section title="👨‍💻 Équipe PAD">
          <Member name="Ayman Chergui" />
          <Member name="Bastien Schneider" />
          <Member name="Mathieu Fraixanet" />
        </Section>

        {/* GitHub */}
        <Section title="💻 Projet GitHub">
          <TouchableOpacity
            style={styles.githubRow}
            activeOpacity={0.7}
            onPress={() =>
              Linking.openURL('https://github.com/Bastientt/PAD')
            }
          >
            <IconSymbol
              name="chevron.left.forwardslash.chevron.right"
              size={22}
              color="#4dabff"
            />
            <ThemedText style={styles.githubText}>
              Voir le projet sur GitHub
            </ThemedText>
          </TouchableOpacity>
        </Section>

        {/* Footer */}
        <ThemedText style={styles.footer}>
          PAD • Version 1.0
        </ThemedText>

      </ScrollView>
    </ThemedView>
  );
}

/* ───────── Components ───────── */

function Section({ title, children }: any) {
  return (
    <View style={styles.section}>
      <ThemedText style={styles.sectionTitle}>{title}</ThemedText>
      {children}
    </View>
  );
}

function Member({ name }: { name: string }) {
  return (
    <View style={styles.memberRow}>
      <View style={styles.memberDot} />
      <ThemedText style={styles.memberText}>{name}</ThemedText>
    </View>
  );
}

/* ───────── Styles ───────── */

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },

  scroll: {
    padding: 24,
    paddingBottom: 60,
    alignItems: 'center',
  },

  title: {
    textAlign: 'center',
    marginBottom: 22,
  },

  paragraph: {
    fontSize: 14,
    lineHeight: 22,
    opacity: 0.75,
    marginBottom: 14,
    textAlign: 'center',
    maxWidth: 420,
  },

  section: {
    marginTop: 28,
    alignItems: 'center',
    width: '100%',
  },

  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 14,
    textAlign: 'center',
    letterSpacing: 0.5,
  },

  memberRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },

  memberDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4dabff',
    marginRight: 10,

    shadowColor: '#4dabff',
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },

  memberText: {
    fontSize: 14,
    opacity: 0.9,
    textAlign: 'center',
  },

  githubRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
  },

  githubText: {
    marginLeft: 12,
    fontSize: 14,
    color: '#4dabff',
    fontWeight: '600',
    textAlign: 'center',
  },

  footer: {
    marginTop: 50,
    textAlign: 'center',
    opacity: 0.5,
    fontSize: 12,
  },

  highlight: {
    color: '#4dabff',
    fontWeight: '700',
  },
});
