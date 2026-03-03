import { View, StyleSheet, Dimensions, Image } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { IconSymbol } from '@/components/ui/icon-symbol';

const screenWidth = Dimensions.get('window').width;

const isSmallMobile = screenWidth < 380;
const isWeb = screenWidth >= 600;

export default function HomeScreen() {
  const router = useRouter();

  return (
    <ThemedView style={styles.container}>

      {/* Main content */}
      <View style={styles.content}>

        {/* Hero image */}
        <View style={styles.iconContainer}>
          <Image
            source={require('@/assets/images/home-logo.png')}
            style={styles.heroImage}
            resizeMode="contain"
          />
        </View>

        {/* Header */}
        <View style={styles.header}>
          <ThemedText type="title" style={styles.title}>
            PAD
          </ThemedText>

          <ThemedText style={styles.subtitle}>
            Secure biometric detection system
          </ThemedText>
        </View>

        {/* Features */}
        <View style={styles.featuresGrid}>

          <Card style={styles.featureCard}>
            <View style={styles.iconCenter}>
              <IconSymbol name="person.crop.circle.fill" size={28} color="#ffffff" />
            </View>
            <ThemedText style={styles.featureTitle}>Face ID</ThemedText>
            <ThemedText style={styles.featureDesc}>Facial recognition</ThemedText>
          </Card>

          <Card style={styles.featureCard}>
            <View style={styles.iconCenter}>
              <IconSymbol name="shield.fill" size={28} color="#ffffff" />
            </View>
            <ThemedText style={styles.featureTitle}>Anti-spoof</ThemedText>
            <ThemedText style={styles.featureDesc}>Fraud protection</ThemedText>
          </Card>

          <Card style={styles.featureCard}>
            <View style={styles.iconCenter}>
              <IconSymbol name="lock.fill" size={28} color="#ffffff" />
            </View>
            <ThemedText style={styles.featureTitle}>Encryption</ThemedText>
            <ThemedText style={styles.featureDesc}>Total security</ThemedText>
          </Card>

          <Card style={styles.featureCard}>
            <View style={styles.iconCenter}>
              <IconSymbol name="doc.text.fill" size={28} color="#ffffff" />
            </View>
            <ThemedText style={styles.featureTitle}>GDPR</ThemedText>
            <ThemedText style={styles.featureDesc}>Protected data</ThemedText>
          </Card>

        </View>

      </View>

      <View style={styles.descriptionWrapper}>

        <View style={styles.descriptionRow}>
          <IconSymbol 
            name="checkmark.circle.fill" 
            size={20} 
            color="#4dabff" 
          />

          <ThemedText style={styles.description}>
            Verify your digital profile{'\n'}
            <ThemedText style={styles.highlight}>
              in just a few simple steps
            </ThemedText>
          </ThemedText>
        </View>

      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Button
          title="🚀 Start verification"
          size="medium"
          onPress={() => router.push('/upload')}
        />

        <ThemedText style={styles.footerText}>
          Secure process • Approximately 2 minutes
        </ThemedText>
      </View>

    </ThemedView>
  );
}
const styles = StyleSheet.create({

  container: {
    flex: 1,
    paddingTop: isWeb ? 60 : 40,
  },

  header: {
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingBottom: 18,
    
  },

  title: {
    fontSize: isWeb ? 28 : 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 6,
  },

  subtitle: {
    fontSize: isWeb ? 14 : 13,
    opacity: 0.7,
    textAlign: 'center',
  },

  content: {
    flex: 1,
    paddingHorizontal: 20,
    alignItems: 'center',
  },

  iconContainer: {
    width: isWeb ? 120 : 100,
    height: isWeb ? 120 : 100,
    borderRadius: isWeb ? 60 : 50,
    backgroundColor: 'rgba(33,150,243,0.15)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 22,

    shadowColor: '#2196F3',
    shadowOpacity: 0.5,
    shadowRadius: 14,
  },

  heroImage: {
    width: '70%',
    height: '70%',
  },

  description: {
    fontSize: isWeb ? 16 : 14,
    textAlign: 'center',
    marginBottom: 26,
    lineHeight: 22,
  },

  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: isWeb ? 'space-between' : 'center',
    gap: isWeb ? 18 : 14,
    marginTop: 10,
  },

  featureCard: {
    width: isWeb ? 160 : isSmallMobile ? '100%' : '46%',
    alignItems: 'center',
  },

  iconCenter: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },

  featureTitle: {
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 4,
    textAlign: 'center',
  },

  featureDesc: {
    fontSize: 11,
    opacity: 0.7,
    textAlign: 'center',
  },

  footer: {
    paddingHorizontal: 24,
    paddingBottom: 36,
    gap: 14,
  },

  footerText: {
    fontSize: 11,
    textAlign: 'center',
    opacity: 0.6,
  },

descriptionWrapper: {
  width: '100%',
  maxWidth: 420,
  alignSelf: 'center',

  marginTop: isWeb ? 400 : 30,

  alignItems: 'center',
},

highlight: {
  color: '#4dabff',
  fontWeight: '500',
},

descriptionRow: {
  flexDirection: 'row',
  alignItems: 'flex-start',
  justifyContent: 'center',
  gap: 2, 
},

});
