# PAD – Mobile Identity Verification Application

## Overview

**PAD (Mobile Identity Verification Application)** is a mobile application developed with **React Native** and **Expo**.  
It provides a secure, real-time identity verification process through facial recognition, video capture, and liveness detection.  
The goal is to offer a smooth and reliable user experience for biometric verification while ensuring data security and GDPR compliance.

---

## Key Features

- **Identity Verification Flow**  
  Step-by-step guided video capture and automated upload to the backend for analysis.

- **Real-Time Communication**  
  WebSocket-based communication for instant feedback during recording and upload.

- **Result Dashboard**  
  Displays the verification outcome with confidence levels and detailed metrics.

- **Modern and Reusable UI**  
  Clean React Native interface with reusable themed components and responsive layouts.

- **Security and Compliance**  
  End-to-end encrypted data transmission and GDPR-compliant user data management.

---

## Technology Stack

| Layer | Technology |
|:------|:------------|
| Framework | React Native (Expo) |
| Language | TypeScript |
| Navigation | Expo Router |
| Video Processing | Expo Camera, Expo AV |
| Real-Time Communication | WebSocket hooks (`useWebSocketUpload`) |
| UI Components | Custom modular React Native components |

---

## Project Structure

```
ui-frontend/
├── app/
│   ├── (tabs)/
│   │   ├── index.tsx        → Home screen
│   │   ├── upload.tsx       → Video recording and upload interface
│   │   └── result.tsx       → Verification results screen
│   └── _layout.tsx          → Application layout and navigation
├── components/
│   ├── video/               → Video recorder and preview components
│   ├── upload/              → Upload progress and status modules
│   ├── result/              → Result and confidence meter components
│   └── ui/                  → Generic UI elements (buttons, cards, etc.)
├── hooks/                   → Custom hooks (e.g., WebSocket management)
├── assets/                  → Icons, images, and static media
└── constants/               → Themes and shared configurations
```

---

## Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bastientt/PAD.git
   cd PAD
   git checkout ui-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npx expo start
   ```

4. **Run the application**
   - Scan the QR code displayed in the terminal using **Expo Go** (Android/iOS), or  
   - Press `a` to launch the Android emulator.

---

## Usage

1. **Home Screen** – Introduction and navigation entry point.  
2. **Upload Screen** – Record a video following the on-screen instructions.  
3. **Result Screen** – View the verification result and confidence metrics.

---

## Contributors

- Ayman Chergui
- Badr Moussaoui
- Bastien SCHNEIDER
- Florian SANANES
- Ranya AMARA

---

## License