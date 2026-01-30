import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, ScrollView, Alert, Animated, Dimensions, Easing, useColorScheme, Appearance } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import { FontAwesome5, MaterialCommunityIcons, Ionicons } from '@expo/vector-icons';

// YOUR RENDER SERVER URL
const SERVER_URL = "https://space-safety-object-detection-1.onrender.com";
const API_URL = `${SERVER_URL}/detect`;

// --- SPLASH SCREEN COMPONENT ---
const SplashScreen = ({ onFinish, isDark }) => {
  const moveAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(1)).current;
  const screenHeight = Dimensions.get('window').height;

  useEffect(() => {
    fetch(SERVER_URL).catch(() => { });

    // Sequence: Launch -> Fade
    Animated.sequence([
      Animated.timing(moveAnim, {
        toValue: 1,
        duration: 2000,
        useNativeDriver: true,
        easing: Easing.out(Easing.cubic),
      }),
      Animated.delay(500),
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 800,
        useNativeDriver: true,
      }),
    ]).start(() => onFinish());
  }, []);

  const rocketTranslateY = moveAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [100, -screenHeight],
  });

  const bgStyle = isDark ? { backgroundColor: '#050b14' } : { backgroundColor: '#FFFFFF' };
  const textTitleStyle = isDark ? { color: '#e0faff' } : { color: '#1A237E' };
  const textSubStyle = isDark ? { color: '#00d2ff' } : { color: '#7986CB' };
  const iconColor = isDark ? "#00d2ff" : "#2962FF";

  return (
    <Animated.View style={[styles.splashContainer, bgStyle, { opacity: fadeAnim }]}>
      <StatusBar style={isDark ? "light" : "dark"} />
      <View style={styles.centerContent}>
        <Animated.View style={{ transform: [{ translateY: rocketTranslateY }] }}>
          {/* APP LOGO IMAGE */}
          <MaterialCommunityIcons
            name="rocket-launch"
            size={120}
            color={iconColor}
            style={{ transform: [{ rotate: '-45deg' }] }}
          />
        </Animated.View>
        <Text style={[styles.splashTitle, textTitleStyle]}>SPACE SAFETY</Text>
        <Text style={[styles.splashSubtitle, textSubStyle]}>AI MONITORING SYSTEM</Text>
        <ActivityIndicator color={iconColor} style={{ marginTop: 20 }} />
      </View>
    </Animated.View>
  );
};

// --- MAIN APP ---
export default function App() {
  const systemScheme = useColorScheme();
  const [manualTheme, setManualTheme] = useState(null); // null = auto, 'light', 'dark'

  // Logic: Use Manual if set, otherwise use System Status
  const isDark = manualTheme ? manualTheme === 'dark' : systemScheme === 'dark';

  const [showSplash, setShowSplash] = useState(true);
  const [serverStatus, setServerStatus] = useState('checking');
  const [image, setImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState([]);

  // COLORS
  const Colors = {
    bg: isDark ? '#050b14' : '#F5F7FA',
    cardBg: isDark ? '#111b2b' : '#FFFFFF',
    textPrimary: isDark ? '#e0faff' : '#0D1B2A',
    textSecondary: isDark ? '#8faab9' : '#5C6B7F',
    accent: isDark ? '#00d2ff' : '#2962FF',
    accentSecondary: isDark ? '#007AA5' : '#1565C0',
    success: '#4CAF50',
    error: '#F44336',
  };

  const toggleTheme = () => {
    // Logic: If currently Dark -> Light. If Light -> Dark.
    setManualTheme(isDark ? 'light' : 'dark');
  };

  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 10000);
    return () => clearInterval(interval);
  }, []);

  const checkServer = async () => {
    try {
      const res = await fetch(SERVER_URL);
      if (res.ok) setServerStatus('online');
      else setServerStatus('offline');
    } catch (e) {
      setServerStatus('offline');
    }
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaType.Images,
      allowsEditing: true,
      quality: 0.8,
    });
    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResultImage(null);
      setDetections([]);
    }
  };

  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Permission required", "Please allow camera access.");
      return;
    }
    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 0.8,
    });
    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResultImage(null);
      setDetections([]);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;
    if (serverStatus === 'offline') {
      Alert.alert("Offline", "AI Server is waking up. Please wait.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: image,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      const data = await response.json();
      if (data.success) {
        setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
        setDetections(data.detections);
      } else {
        Alert.alert("Error", "Analysis failed.");
      }
    } catch (error) {
      Alert.alert("Error", error.message);
    } finally {
      setLoading(false);
    }
  };

  if (showSplash) return <SplashScreen isDark={isDark} onFinish={() => setShowSplash(false)} />;

  return (
    <ScrollView contentContainerStyle={[styles.container, { backgroundColor: Colors.bg }]}>
      <StatusBar style={isDark ? "light" : "dark"} />

      {/* HEADER CARD */}
      <View style={[styles.headerCard, { backgroundColor: Colors.cardBg }]}>
        <View style={styles.headerTop}>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            {/* APP LOGO */}
            <Image
              source={require('./assets/icon.png')}
              style={{ width: 45, height: 45, borderRadius: 10, marginRight: 12 }}
            />
            <View>
              <Text style={[styles.appTitle, { color: Colors.textPrimary }]}>SpaceSafety AI</Text>
              <Text style={[styles.appSubtitle, { color: Colors.textSecondary }]}>Secure Object Detection</Text>
            </View>
          </View>
          <TouchableOpacity onPress={toggleTheme} style={styles.themeToggle}>
            <Ionicons name={isDark ? "sunny" : "moon"} size={24} color={Colors.textPrimary} />
          </TouchableOpacity>
        </View>

        {/* Connection Status Row - Moved below title */}
        <View style={[styles.statusRow, { marginTop: 15 }]}>
          <View style={[styles.statusBadge, { backgroundColor: serverStatus === 'online' ? (isDark ? '#00332a' : '#E8F5E9') : (isDark ? '#330e0e' : '#FFEBEE') }]}>
            <View style={[styles.statusDot, { backgroundColor: serverStatus === 'online' ? Colors.success : Colors.error }]} />
            <Text style={[styles.statusText, { color: serverStatus === 'online' ? Colors.success : Colors.error }]}>
              {serverStatus === 'online' ? 'SYSTEM ACTIVE' : 'OFFLINE'}
            </Text>
          </View>
        </View>

      </View>

      {/* MAIN CONTENT */}
      <View style={styles.content}>

        {/* IMAGE PREVIEW AREA */}
        <View style={[styles.imageCard, { backgroundColor: Colors.cardBg, borderColor: isDark ? '#333' : '#F0F2F5', shadowColor: Colors.accent }]}>
          {resultImage ? (
            <Image source={{ uri: resultImage }} style={styles.previewImage} />
          ) : image ? (
            <Image source={{ uri: image }} style={styles.previewImage} />
          ) : (
            <View style={styles.placeholderContainer}>
              <Ionicons name="image-outline" size={60} color={isDark ? '#334' : '#D1D9E6'} />
              <Text style={[styles.placeholderText, { color: Colors.textSecondary }]}>Upload an image to scan</Text>
            </View>
          )}

          {/* FLOATING ACTION BUTTON */}
          {image && !loading && (
            <TouchableOpacity style={[styles.scanButtonFloating, { backgroundColor: Colors.accent }]} onPress={analyzeImage}>
              <MaterialCommunityIcons name="radar" size={24} color="#FFF" />
              <Text style={styles.scanButtonText}>SCAN NOW</Text>
            </TouchableOpacity>
          )}
          {loading && (
            <View style={[styles.loadingOverlay, { backgroundColor: isDark ? 'rgba(0,0,0,0.8)' : 'rgba(255,255,255,0.9)' }]}>
              <ActivityIndicator size="large" color={Colors.accent} />
              <Text style={[styles.loadingText, { color: Colors.accent }]}>Analyzing...</Text>
            </View>
          )}
        </View>

        {/* CONTROLS */}
        <View style={styles.buttonGrid}>
          <TouchableOpacity style={[styles.actionButton, { backgroundColor: Colors.cardBg }]} onPress={pickImage}>
            <View style={[styles.iconCircle, { backgroundColor: isDark ? 'rgba(0,210,255,0.1)' : '#E3F2FD' }]}>
              <Ionicons name="images" size={24} color={Colors.accent} />
            </View>
            <Text style={[styles.actionText, { color: Colors.textPrimary }]}>Gallery</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.actionButton, { backgroundColor: Colors.cardBg }]} onPress={takePhoto}>
            <View style={[styles.iconCircle, { backgroundColor: isDark ? 'rgba(0,255,157,0.1)' : '#E0F2F1' }]}>
              <Ionicons name="camera" size={24} color={isDark ? '#00ff9d' : '#009688'} />
            </View>
            <Text style={[styles.actionText, { color: Colors.textPrimary }]}>Camera</Text>
          </TouchableOpacity>
        </View>

        {/* RESULTS SECTION */}
        {detections.length > 0 && (
          <View style={[styles.resultsCard, { backgroundColor: Colors.cardBg }]}>
            <Text style={[styles.resultsHeader, { color: Colors.textPrimary }]}>Detection Report</Text>
            {detections.map((item, index) => (
              <View key={index} style={[styles.resultItem, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#F9FAFB' }]}>
                <View style={styles.resultBadge}>
                  <Ionicons name="checkmark-circle" size={16} color={Colors.success} />
                  <Text style={[styles.resultClass, { color: Colors.textPrimary }]}>{item.class}</Text>
                </View>
                <Text style={[styles.resultConf, { color: Colors.textSecondary }]}>{(item.confidence * 100).toFixed(0)}% Conf</Text>
              </View>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  // SPLASH
  splashContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  centerContent: {
    alignItems: 'center',
  },
  splashTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    marginTop: 20,
    letterSpacing: 1,
  },
  splashSubtitle: {
    fontSize: 14,
    marginTop: 5,
    letterSpacing: 2,
  },

  // CONTAINER
  container: {
    flexGrow: 1,
    paddingBottom: 40,
  },

  // HEADER
  headerCard: {
    paddingTop: 60,
    paddingBottom: 20,
    paddingHorizontal: 24,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 5,
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  appTitle: {
    fontSize: 22,
    fontWeight: '800',
  },
  appSubtitle: {
    fontSize: 13,
    marginTop: 2,
  },
  themeToggle: {
    padding: 8,
  },
  statusRow: {
    flexDirection: 'row',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
  },

  // CONTENT
  content: {
    padding: 24,
  },

  // IMAGE CARD
  imageCard: {
    height: 250,
    borderRadius: 24,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 10,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30,
    overflow: 'hidden',
    borderWidth: 1,
  },
  previewImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderContainer: {
    alignItems: 'center',
  },
  placeholderText: {
    marginTop: 10,
    fontSize: 14,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontWeight: 'bold',
    marginTop: 10,
  },

  // BUTTONS
  scanButtonFloating: {
    position: 'absolute',
    bottom: 20,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 30,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  scanButtonText: {
    color: '#FFF',
    fontWeight: 'bold',
    marginLeft: 8,
    fontSize: 14,
  },
  buttonGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  actionButton: {
    flex: 0.48,
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  iconCircle: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
  },
  actionText: {
    fontWeight: '600',
    fontSize: 14,
  },

  // RESULTS
  resultsCard: {
    borderRadius: 20,
    padding: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 3,
  },
  resultsHeader: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#F5F5F5',
    paddingBottom: 10,
  },
  resultItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    padding: 12,
    borderRadius: 10,
  },
  resultBadge: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  resultClass: {
    marginLeft: 8,
    fontSize: 15,
    fontWeight: '500',
  },
  resultConf: {
    fontSize: 13,
    fontWeight: '600',
  },
});
