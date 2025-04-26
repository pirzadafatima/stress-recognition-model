# Stress Recognition From Facial Expressions 

This project introduces an AI-powered stress recognition system using machine learning techniques to analyze facial expressions for real-time stress assessment.  
It offers a more objective and reliable alternative to traditional self-reporting stress detection methods by levaraging facial landmark detection to extract features like furrowed eyebrows, flared nostrils, and clenched jaws which are reliable indicators of stress.  

## Original Dataset

- **Source:** CK+ (Cohn-Kanade Extended) Dataset
- **Size:** 981 images labeled across 7 emotions (anger, contempt, disgust, fear, happy, sadness, surprise)

### Preprocessing Steps
- **Facial feature extraction** using dlib (68 landmarks)
- **Metrics calculation** for:
  - Furrowed Brows Distance
  - Eye to Eyebrow Distance
  - Lip to Nose Distance
  - Nostril Flaring Distance
  - Clenched Jaw Distance
  - Parted Mouth Distance
- **Threshold setting** for stress indicators
- **Stress labeling** based on feature thresholds, converting the ck+ dataset into a stree recognition dataset named stress+
- **Image normalization** (grayscale, resized to 48×48, pixel values between 0-1)

## Workflow

1. Load and preprocess facial images
2. Extract facial landmarks
3. Calculate stress-related metrics
4. Label images as stressed or non-stressed
5. Train machine learning models
6. Deploy for real-time stress detection via webcam

## Methodologies and Approaches 🧠

Various models were tested:
- **CNNs:** Achieved ~95% accuracy
- **Random Forest:** High offline accuracy (~96%) but lower real-time performance
- **Decision Trees, Naive Bayes, Logistic Regression, MLPs:** Moderate to low performance

> Final model: **CNN** with optimized hyperparameters and Adam optimizer.

---

## Comparison with Existing Methods

- Focused on **specific facial features** linked to stress, not general emotions
- Inclusive of **diverse demographics** for better generalization
- Real-world applicability with **real-time webcam** stress detection
- More comprehensive than methods relying only on emotions or self-reports

Exact details, process and results are compiled into a report present in the repository. 



