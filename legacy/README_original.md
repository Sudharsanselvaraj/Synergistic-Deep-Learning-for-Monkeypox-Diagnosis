# Mpox Detector Model

## Overview  
The **Mpox Detector Model** is an advanced diagnostic tool designed to assist in the detection of Monkeypox using two complementary methods:  
1. **Skin Lesion Classification**: Utilizes cutting-edge pretrained deep learning networks to analyze skin lesion images.  
2. **Symptom-Based Detection**: Employs a neural network model for symptom analysis based on user-provided inputs.  

This dual-approach system enhances diagnostic accuracy by combining visual and symptomatic data, providing a reliable, user-friendly solution.

---

## Features  
- **Tri-Net Architecture**: Combines three powerful pretrained networks—EfficientNetB4, Inception-ResNet V2, and DenseNet201—for precise skin lesion classification.  
- **Symptom Analysis**: Uses a convolutional neural network (CNN) to predict Monkeypox based on clinical symptoms such as fever, swollen tonsils, and other associated signs.  
- **User-Friendly Interface**: Accessible via a website for easy prediction using images or symptom inputs.  
- **Interactive Frontend**: Built with Tkinter for offline usage and integrated with Java Swing for enhanced data management.  

---

## Technical Details  

### Skin Lesion Detector  
- **Pretrained Models**: EfficientNetB4, Inception-ResNet V2, DenseNet201.  
- **Input**: Skin lesion images in standard formats.  
- **Output**: Classification into one of six categories: Monkeypox, Chickenpox, HFMD, Measles, Cowpox, or Healthy.  

### Symptom Detector  
- **Neural Network Architecture**:  
  - Input Layer: 12 features (patient symptoms).  
  - Dense Layers: 16 and 32 neurons with dropout regularization.  
  - Output Layer: Sigmoid activation for binary classification.  
- **Optimizer**: SGD with binary cross-entropy loss.  

---

## How It Works  
1. **Upload a Skin Lesion Image** or **Input Symptoms**.  
2. The model processes the input through the respective detection pathway:  
   - Skin lesions are classified using the tri-net architecture.  
   - Symptoms are analyzed using the symptom detector CNN.  
3. Results are displayed on the screen, indicating whether Monkeypox is detected.  

