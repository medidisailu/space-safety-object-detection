import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, ScrollView, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// YOUR RENDER SERVER URL
const API_URL = "https://space-safety-object-detection-1.onrender.com/detect";

export default function App() {
  const [image, setImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detections, setDetections] = useState([]);

  // 1. Pick Image
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResultImage(null); // Reset previous result
      setDetections([]);
    }
  };

  // 2. Take Photo
  const takePhoto = async () => {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    
    if (permissionResult.granted === false) {
      Alert.alert("Permission to access camera is required!");
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResultImage(null);
      setDetections([]);
    }
  };

  // 3. Send to Server
  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    try {
      // Create Form Data
      const formData = new FormData();
      formData.append('file', {
        uri: image,
        name: 'photo.jpg',
        type: 'image/jpeg',
      });

      // Send Request
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (data.success) {
        // Display Result - API returns base64 image
        setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
        setDetections(data.detections);
      } else {
        Alert.alert("Error", "Analysis failed.");
      }
    } catch (error) {
      Alert.alert("Error", "Could not connect to server. " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>Space Safety Object Detection</Text>
      
      {/* Image Display */}
      <View style={styles.imageContainer}>
        {resultImage ? (
          <Image source={{ uri: resultImage }} style={styles.image} />
        ) : image ? (
          <Image source={{ uri: image }} style={styles.image} />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>No Image Selected</Text>
          </View>
        )}
      </View>

      {/* Buttons */}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.button} onPress={pickImage}>
          <Text style={styles.buttonText}>📁 Gallery</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.button, styles.cameraBtn]} onPress={takePhoto}>
          <Text style={styles.buttonText}>📷 Camera</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity 
        style={[styles.analyzeBtn, (!image || loading) && styles.disabledBtn]} 
        onPress={analyzeImage}
        disabled={!image || loading}
      >
        {loading ? (
          <ActivityIndicator color="#000" />
        ) : (
          <Text style={styles.analyzeText}>INITIATE SCAN</Text>
        )}
      </TouchableOpacity>

      {/* Results List */}
      <View style={styles.resultsContainer}>
        <Text style={styles.sectionTitle}>Detected Objects:</Text>
        {detections.length === 0 ? (
           <Text style={{color:'#888', fontStyle:'italic'}}>No scan results yet.</Text>
        ) : (
          detections.map((item, index) => (
            <View key={index} style={styles.resultRow}>
              <Text style={styles.resultName}>• {item.class}</Text>
              <Text style={styles.resultConf}>{(item.confidence * 100).toFixed(0)}%</Text>
            </View>
          ))
        )}
      </View>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#050b14',
    padding: 20,
    alignItems: 'center',
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#00d2ff',
    marginTop: 40,
    marginBottom: 20,
    textAlign: 'center',
    fontFamily: 'System', // Use default font to avoid loading issues
  },
  imageContainer: {
    width: '100%',
    aspectRatio: 16/9,
    borderWidth: 2,
    borderColor: '#00d2ff',
    borderRadius: 10,
    backgroundColor: '#111',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  placeholder: {
    alignItems: 'center',
  },
  placeholderText: {
    color: '#556',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 15,
  },
  button: {
    flex: 1,
    backgroundColor: 'rgba(0, 210, 255, 0.1)',
    borderWidth: 1,
    borderColor: '#00d2ff',
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  cameraBtn: {
    backgroundColor: 'rgba(0, 255, 157, 0.1)',
    borderColor: '#00ff9d',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  analyzeBtn: {
    width: '100%',
    backgroundColor: '#00d2ff',
    padding: 18,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 30,
    shadowColor: '#00d2ff',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  disabledBtn: {
    backgroundColor: '#556',
    shadowOpacity: 0,
  },
  analyzeText: {
    color: '#050b14',
    fontWeight: '900',
    fontSize: 16,
    letterSpacing: 1,
  },
  resultsContainer: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 10,
    padding: 15,
  },
  sectionTitle: {
    color: '#00d2ff',
    fontSize: 18,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#334',
    paddingBottom: 5,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  resultName: {
    color: '#e0faff',
    fontSize: 16,
  },
  resultConf: {
    color: '#00ff9d',
    fontWeight: 'bold',
  },
});
