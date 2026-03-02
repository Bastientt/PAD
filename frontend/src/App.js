import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView, StatusBar, Platform } from 'react-native';

const API = "http://localhost:8080";

const COLORS = {
  bg: 'transparent',
  glassCard: 'rgba(40, 40, 55, 0.7)',
  primary: '#8A2BE2', 
  secondary: '#00CED1', 
  success: '#32CD32',
  error: '#FF4500',
  textAccent: '#FFFFFF',
  textDim: '#A0A0C0',
  border: 'rgba(255,255,255,0.1)'
};

// Map des icônes pour les directions
const DIRECTION_ICONS = {
  'HAUT': '⬆️',
  'BAS': '⬇️',
  'GAUCHE': '⬅️',
  'DROITE': '➡️'
};

export default function App() {
  const [logs, setLogs] = useState([]);
  const [session, setSession] = useState({ url: null, filename: null, isAuth: false });
  const [challenge, setChallenge] = useState([]); // Nouveau : Stocke la suite de directions
  const [selectedFile, setSelectedFile] = useState(null);
  
  const scrollViewRef = useRef();
  const fileInputRef = useRef(null);

  useEffect(() => scrollViewRef.current?.scrollToEnd({ animated: true }), [logs]);

  const log = (m, type = 'info') => {
    const colors = { info: COLORS.textDim, err: COLORS.error, ia: COLORS.secondary, success: COLORS.success };
    setLogs(prev => [...prev, { id: Date.now(), msg: m, color: colors[type] || COLORS.textAccent }]);
  };

  const doAuth = async () => {
    try {
      await fetch(`${API}/Authentificate`, { credentials: 'include' });
      setSession(p => ({ ...p, isAuth: true }));
      log("🔐 1. Authentifié avec succès", 'success');
    } catch (e) { log("❌ Erreur Auth", 'err'); }
  };

  const doInit = async () => {
    if (!session.isAuth) return log("⚠️ Authentification requise", 'info');
    try {
      const res = await fetch(`${API}/init-sesssion`, { credentials: 'include' });
      if (res.status === 429) return log("🚫 RATE LIMIT : Ban Redis !", 'err');
      
      const data = await res.json();
      const filename = data.url.split('/').pop().split('?')[0];
      
      setSession(p => ({ ...p, url: data.url, filename }));
      setChallenge(data.challenge); // On stocke le challenge reçu du Rust
      
      log(`📂 2. Session prête. Challenge : ${data.challenge.join(' ➔ ')}`, 'success');
    } catch (e) { log(`❌ Erreur Init : ${e.message}`, 'err'); }
  };

  const handleSelectFilePress = () => {
    if (!session.url) return log("⚠️ Initialise une session d'abord", 'info');
    if (Platform.OS === 'web') fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      log(`📹 Vidéo chargée : ${file.name}`, 'secondary');
    }
  };

  const doUpload = async () => {
    if (!session.url || !selectedFile) return log("⚠️ Fichier manquant", 'info');
    try {
      log(`⬆️ 3. Upload de la vidéo vers S3...`, 'info');
      const res = await fetch(session.url, {
        method: 'PUT',
        body: selectedFile,
        headers: { 'Content-Type': selectedFile.type || 'application/octet-stream' }
      });
      if (res.ok) log("✅ Upload réussi. Lancez l'IA.", 'success');
      else throw new Error("Erreur S3");
    } catch (e) { log(`❌ Erreur Upload : ${e.message}`, 'err'); }
  };

  const doIA = () => {
    if (!session.filename) return log("⚠️ Session requise", 'info');
    const ws = new WebSocket(`ws://localhost:8080/ws/${session.filename}`);
    log("🔌 4. Connexion WebSocket...", 'info');

    ws.onopen = () => { 
      log("🤖 WS Ouvert. Vérification du challenge...", 'secondary');
      // Format : UploadDone|HAUT,BAS,GAUCHE
      const msg = `UploadDone|${challenge.join(',')}`;
      ws.send(msg); 
    };

    ws.onmessage = (e) => log(`📥 RÉPONSE IA : ${e.data}`, 'ia');
    ws.onclose = () => log("🔌 WS Déconnecté", 'info');
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      {Platform.OS === 'web' && (
        <input type="file" ref={fileInputRef} style={{display:'none'}} accept="video/*" onChange={handleFileChange} />
      )}

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerSubtitle}>PAD BIOMETRIC POC</Text>
        <Text style={styles.headerTitle}>Liveness Verify</Text>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        
        {/* SECTION CHALLENGE : Affiche la suite à réaliser */}
        {challenge.length > 0 && (
          <View style={[styles.glassCard, { borderColor: COLORS.primary }]}>
            <Text style={[styles.cardLabel, {color: COLORS.primary}]}>CHALLENGE À RÉALISER</Text>
            <View style={styles.challengeRow}>
              {challenge.map((dir, index) => (
                <View key={index} style={styles.dirItem}>
                  <Text style={styles.dirIcon}>{DIRECTION_ICONS[dir]}</Text>
                  <Text style={styles.dirText}>{dir}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        <View style={styles.glassCard}>
          <Text style={styles.cardLabel}>ÉTAPE 1 & 2</Text>
          <View style={styles.row}>
            <GlassBtn title="Auth" icon="🔐" onPress={doAuth} active={!session.isAuth} color={COLORS.primary} style={{flex:1, marginRight:10}} />
            <GlassBtn title="Init" icon="📡" onPress={doInit} active={session.isAuth && !session.url} color={COLORS.primary} style={{flex:1}}/>
          </View>
        </View>

        <View style={[styles.glassCard, { borderColor: COLORS.secondary }]}>
          <Text style={[styles.cardLabel, {color: COLORS.secondary}]}>UPLOAD VIDÉO</Text>
          <View style={styles.fileBox}>
              <Text style={styles.fileName}>{selectedFile ? selectedFile.name : 'Aucun fichier'}</Text>
          </View>
          <View style={styles.row}>
            <GlassBtn title="Choisir" icon="📁" onPress={handleSelectFilePress} active={!!session.url} color={COLORS.secondary} style={{flex:1, marginRight:10}} />
            <GlassBtn title="Upload" icon="⬆️" onPress={doUpload} active={!!selectedFile} color={COLORS.secondary} style={{flex:1}} fill />
          </View>
        </View>

        <View style={styles.glassCard}>
           <Text style={styles.cardLabel}>ANALYSE IA</Text>
           <GlassBtn title="Vérifier la vidéo" icon="🧠" onPress={doIA} active={!!session.filename} color={COLORS.success} big />
        </View>

      </ScrollView>

      {/* Terminal */}
      <View style={styles.terminalContainer}>
        <ScrollView ref={scrollViewRef} style={styles.terminal}>
          {logs.map(l => (
            <Text key={l.id} style={[styles.logText, { color: l.color }]}>
              <Text style={{color:'#444'}}>&gt; </Text>{l.msg}
            </Text>
          ))}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const GlassBtn = ({ title, icon, onPress, active, color, style, big, fill }) => (
  <TouchableOpacity 
    onPress={onPress} 
    disabled={!active}
    style={[styles.glassBtn, { borderColor: active ? color : COLORS.border, backgroundColor: fill && active ? color : COLORS.glassCard, opacity: active ? 1 : 0.4 }, big && {paddingVertical: 20}, style]}
  >
    <Text style={{fontSize: big ? 24 : 18, marginRight: 8}}>{icon}</Text>
    <Text style={{color: fill && active ? '#000' : (active ? color : COLORS.textDim), fontWeight:'bold'}}>{title}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { padding: 25 },
  headerSubtitle: { color: COLORS.primary, fontWeight:'bold', fontSize: 10, letterSpacing: 2 },
  headerTitle: { color: '#fff', fontSize: 28, fontWeight:'900' },
  content: { flex: 1 },
  contentContainer: { paddingHorizontal: 20 },
  glassCard: { backgroundColor: COLORS.glassCard, borderRadius: 20, padding: 15, marginBottom: 15, borderWidth: 1, borderColor: COLORS.border },
  cardLabel: { fontSize: 10, fontWeight: 'bold', marginBottom: 10 },
  row: { flexDirection: 'row' },
  challengeRow: { flexDirection: 'row', justifyContent: 'space-around', paddingVertical: 10 },
  dirItem: { alignItems: 'center' },
  dirIcon: { fontSize: 24, marginBottom: 5 },
  dirText: { color: '#fff', fontSize: 10, fontWeight: 'bold' },
  fileBox: { padding: 10, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: 10, marginBottom: 10 },
  fileName: { color: COLORS.textDim, fontSize: 12, textAlign:'center' },
  glassBtn: { padding: 12, borderRadius: 12, alignItems: 'center', justifyContent: 'center', borderWidth: 1, flexDirection: 'row' },
  terminalContainer: { height: 180, backgroundColor: '#000', padding: 15, borderTopLeftRadius: 25, borderTopRightRadius: 25 },
  terminal: { flex: 1 },
  logText: { fontFamily: 'monospace', fontSize: 11, marginBottom: 4 }
});