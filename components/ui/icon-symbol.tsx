// Fallback for using MaterialIcons on Android and web.

import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import { SymbolWeight, SymbolViewProps } from 'expo-symbols';
import { ComponentProps } from 'react';
import { OpaqueColorValue, type StyleProp, type TextStyle } from 'react-native';

type IconMapping = Record<SymbolViewProps['name'], ComponentProps<typeof MaterialIcons>['name']>;
type IconSymbolName = keyof typeof MAPPING;

/**
 * Add your SF Symbols to Material Icons mappings here.
 * - see Material Icons in the [Icons Directory](https://icons.expo.fyi).
 * - see SF Symbols in the [SF Symbols](https://developer.apple.com/sf-symbols/) app.
 */
const MAPPING = {
  // Home
  'house.fill': 'home-filled',

  // Send / navigation
  'paperplane.fill': 'send',

  // Code / system
  'chevron.left.forwardslash.chevron.right': 'terminal',

  // Arrow
  'chevron.right': 'arrow-forward-ios',

  // Camera / upload
  'camera.fill': 'photo-camera',

  // Success / result
  'checkmark.circle.fill': 'check-circle',

  // Analytics
  'chart.bar.fill': 'bar-chart',

  // Cloud upload
  'arrow.up.circle.fill': 'cloud-upload',

  // Security
  'person.crop.circle.fill': 'account-circle',
  'shield.fill': 'shield',
  'lock.fill': 'lock',
  'doc.text.fill': 'description',

  // Infos tab (NEW)
  'info.circle.fill': 'info-outline',

  // About
  'shield.lefthalf.filled': 'security',

  // Team
  'person.3.sequence.fill': 'groups',

  // Member
  'person.fill': 'person',

  // Owners
  'crown.fill': 'workspace-premium',

} as IconMapping;

/**
 * An icon component that uses native SF Symbols on iOS, and Material Icons on Android and web.
 * This ensures a consistent look across platforms, and optimal resource usage.
 * Icon `name`s are based on SF Symbols and require manual mapping to Material Icons.
 */

export function IconSymbol({
  name,
  size = 24,
  color,
  style,
}: {
  name: IconSymbolName;
  size?: number;
  color: string | OpaqueColorValue;
  style?: StyleProp<TextStyle>;
  weight?: SymbolWeight;
}) {
  return <MaterialIcons color={color} size={size} name={MAPPING[name]} style={style} />;
}
