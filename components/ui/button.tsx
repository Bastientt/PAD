import React, { useRef } from 'react';
import {
  Pressable,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
  Animated,
  Platform,
  View,
} from 'react-native';

interface ButtonProps {
  title: string;
  onPress: () => void;
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export function Button({
  title,
  onPress,
  size = 'medium',
  disabled = false,
  loading = false,
  style,
  textStyle,
}: ButtonProps) {

  const scale = useRef(new Animated.Value(1)).current;
  const elevation = useRef(new Animated.Value(12)).current;
  const bgAnim = useRef(new Animated.Value(0)).current;
  const hoverShadowOpacity = useRef(new Animated.Value(0)).current;

  const onHoverIn = () => {
    Animated.parallel([
      Animated.timing(scale, {
        toValue: 1.05,
        duration: 220,
        useNativeDriver: true,
      }),
      Animated.timing(elevation, {
        toValue: 26,
        duration: 220,
        useNativeDriver: false,
      }),
      Animated.timing(bgAnim, {
        toValue: 1,
        duration: 220,
        useNativeDriver: false,
      }),
      Animated.timing(hoverShadowOpacity, {
        toValue: 1,
        duration: 220,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const onHoverOut = () => {
    Animated.parallel([
      Animated.timing(scale, {
        toValue: 1,
        duration: 260,
        useNativeDriver: true,
      }),
      Animated.timing(elevation, {
        toValue: 12,
        duration: 260,
        useNativeDriver: false,
      }),
      Animated.timing(bgAnim, {
        toValue: 0,
        duration: 260,
        useNativeDriver: false,
      }),
      Animated.timing(hoverShadowOpacity, {
        toValue: 0,
        duration: 260,
        useNativeDriver: false,
      }),
    ]).start();
  };

  const backgroundColor = bgAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['#ff0000', '#b04747'],
  });

  return (
    <Pressable
      disabled={disabled || loading}
      onPress={onPress}
      onHoverIn={Platform.OS === 'web' ? onHoverIn : undefined}
      onHoverOut={Platform.OS === 'web' ? onHoverOut : undefined}
      style={{ alignItems: 'center' }}
    >
      <View style={styles.shadowWrapper}>

        {/* BLACK SHADOW ON HOVER */}
        <Animated.View
          pointerEvents="none"
          style={[
            styles.blackShadow,
            { opacity: hoverShadowOpacity },
          ]}
        />

        {/* MAIN BUTTON */}
        <Animated.View
          style={[
            styles.button,
            styles[`button_${size}`],
            {
              transform: [{ scale }],
              elevation,
              backgroundColor,
            },
            (disabled || loading) && styles.buttonDisabled,
            style,
          ]}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text
              style={[
                styles.text,
                styles[`text_${size}`],
                textStyle,
              ]}
            >
              {title}
            </Text>
          )}
        </Animated.View>

      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({

  shadowWrapper: {
    alignItems: 'center',
  },

  /* BLACK HOVER SHADOW */
  blackShadow: {
    position: 'absolute',
    width: '100%',
    maxWidth: 360,
    height: '100%',
    borderRadius: 22,

    shadowColor: '#000',
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.45,
    shadowRadius: 26,

    elevation: 26,
  },

  /* MAIN BUTTON */

  button: {
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',

    width: '100%',
    maxWidth: 360,

    /* RED BASE SHADOW */
    shadowColor: '#ff0000',
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.35,
    shadowRadius: 18,

    elevation: 12,
  },

  button_small: {
    paddingVertical: 10,
    paddingHorizontal: 20,
  },

  button_medium: {
    paddingVertical: 15,
    paddingHorizontal: 56,
  },

  button_large: {
    paddingVertical: 18,
    paddingHorizontal: 36,
  },

  buttonDisabled: {
    backgroundColor: '#333',
    opacity: 0.5,
    elevation: 0,
  },

  text: {
    fontWeight: '700',
    letterSpacing: 0.7,
    color: '#ffffff',
  },

  text_small: { fontSize: 13 },
  text_medium: { fontSize: 15 },
  text_large: { fontSize: 17 },
});
