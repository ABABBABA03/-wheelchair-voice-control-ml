# Voice-Controlled Wheelchair Navigation System (Feature Engineering + ML)

This project simulates a voice-controlled wheelchair navigating through a randomly generated maze. The system integrates real-time audio input with a trained machine learning model to interpret voice commands and convert them into game actions.

Developed for the Feature Engineering course (Engenharia de Atributos), University of Coimbra — 2024.

---

## Project Goal

Enable wheelchair navigation using short voice commands (e.g., “forward”, “left”, “stop”) to demonstrate how sound-based AI interfaces can assist individuals with mobility limitations.

---

## Features

- **Interactive Maze Game**: Built with Pygame; player must reach goal point
- **Real-time Audio Processing**: Captures and segments audio, performs pre-filtering
- **Feature Extraction**:
  - MFCCs (12 coefficients)
  - Zero-Crossing Rate, Spectral Centroid, Bandwidth, Flatness
  - Frequency domain energy distribution
  - DFT-based dominant frequency and energy concentration
- **Machine Learning Pipeline**:
  - Trained with Random Forest
  - Input: engineered features from audio
  - Output: 5 possible commands (FORWARD, BACKWARD, LEFT, RIGHT, STOP)
- Real-time classification with StandardScaler normalization
- Threaded microphone interface for live control

---

## Project Structure

```bash
wheelchair-voice-control-ml/
├── wheelchair_game_v2.py      # Game + microphone interface
├── final_model.joblib         # Trained classifier (binary label prediction)
├── scaler.joblib              # Feature scaler
├── README.md                  # This file
```
---
**Disclaimer**  
This project builds upon a base Pygame framework provided by the course instructor.  
All machine learning integration, feature engineering, audio processing, and real-time interaction were developed independently by the student (Bruno Su Du).
