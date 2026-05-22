Visit Website to see demo working of the model: https://huggingface.co/spaces/aditya-s-yadav/signature-similarity-sys

# 🖊️ Signature Verification System using Siamese Neural Network (SNN)

An AI-powered **Signature Verification System** built using **Siamese Neural Networks (SNN)** to detect forged signatures by comparing the similarity between two signature images. The system automates signature authentication, reducing manual effort and improving verification reliability.

## 🚀 Project Overview

Traditional signature verification methods rely heavily on manual inspection, making them time-consuming and prone to human error. This project leverages **Deep Learning** and **Computer Vision** techniques to build an intelligent system capable of distinguishing between **genuine** and **forged** signatures.

The application compares two signature images, extracts meaningful features using a **Siamese Neural Network**, and computes a **similarity score** using **Euclidean Distance** to determine authenticity.

---

## ✨ Features

- 🔐 **User Authentication System** (Login/Register)
- 🖼️ **Upload Signature Images**
- ✍️ **Canvas-Based Signature Drawing**
- 🧠 **Siamese Neural Network for Signature Matching**
- 📊 **Similarity Score Calculation**
- ⚡ **Real-Time Signature Verification**
- 💾 **SQLite Database Integration**
- 🌐 **Flask-Based Web Application**

---

## 🏗️ Tech Stack

### **Programming & Frameworks**
- Python
- Flask
- PyTorch
- HTML, CSS, JavaScript

### **Libraries Used**
- OpenCV
- Pillow (PIL)
- NumPy
- Flask-SQLAlchemy
- Flask-Bcrypt
- TorchVision

### **Database**
- SQLite

---

## 🧠 Model Architecture

This project uses a **Siamese Neural Network (SNN)** designed for similarity learning.

### Workflow:
1. **Input Signature Pair**
2. **Image Preprocessing**
   - Resize to `100x100`
   - Grayscale Conversion
   - Normalization
3. **Feature Extraction using CNN**
4. **Embedding Generation**
5. **Euclidean Distance Calculation**
6. **Similarity Score Prediction**
7. **Forgery Detection**

If the similarity score crosses a predefined threshold, the signature is classified as **Genuine**; otherwise, it is marked as **Forged**.

---

## 📂 Dataset

The model was trained using the **ICDAR 2011 Signature Dataset**, containing both:

- Genuine Signatures
- Forged Signatures

The dataset was used to train the model for distinguishing authentic signatures from forged ones.

---

## 📈 Results

- Successfully trained a **Siamese Neural Network** for signature comparison.
- Achieved a **consistent decrease in training loss across epochs**.
- Successfully integrated the trained model into a **Flask web application**.
- Real-time verification produced distinguishable similarity scores between genuine and forged signatures.

---

## 📸 Application Modules

### Authentication
- User Registration
- Login System

### Signature Verification
- Upload Signature Image
- Draw Signature via Canvas
- Real-Time Comparison

### Output
- Similarity Score
- Match / No Match Status
- Signature Preview

---

## 🔮 Future Improvements

- Improve accuracy using **larger datasets**
- Implement **Transfer Learning**
- Enhance detection for **skilled forgeries**
- Deploy on **Cloud Platforms**
- Integrate with **Banking & Financial Systems**
- Add **Explainable AI (XAI)** for transparent verification

---

## 📌 Use Cases

- 🏦 Banking Authentication
- 📑 Legal Document Verification
- 🏢 Administrative Verification Systems
- 🔒 Fraud Prevention
- 📱 Secure Remote Authentication

---

## 👨‍💻 Author

**Aditya Yadav**  
BSc Data Science | Aspiring Data Scientist & ML Engineer

---

## ⭐ If you found this project useful, consider giving it a star!
