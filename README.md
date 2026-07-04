# 🩺 Skin Disease Detection Using Hybrid CNN and Grad-CAM

## 📌 Project Overview

This project is a Deep Learning-based Skin Disease Detection system that uses a **Hybrid Convolutional Neural Network (CNN)** to classify skin disease images into multiple categories. The system is trained on a large image dataset and is capable of automatically identifying skin conditions from input images.

To improve transparency and trust in AI predictions, the project also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, which provides visual explanations of model decisions using heatmaps.

## 🎯 Objective

The main objective of this project is to:

* Automatically classify skin diseases using deep learning
* Improve diagnostic assistance using AI
* Provide explainable AI outputs using Grad-CAM
* Build a reliable and efficient image classification system

---

## 🧠 Model Architecture

The model is built using a **Hybrid CNN architecture** consisting of:

* Conv2D (32 filters) + MaxPooling
* Conv2D (64 filters) + MaxPooling
* Conv2D (128 filters) + MaxPooling
* Conv2D (256 filters) + MaxPooling
* Flatten Layer
* Dense Layer (512 neurons, ReLU activation)
* Dropout Layer (0.5)
* Output Layer (Softmax activation)

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Scikit-learn
* Jupyter Notebook

---

## 📂 Dataset

The dataset contains images of different skin disease categories.

**Structure:**

```
SkinDisease/
├── Train/
│   ├── Class_1/
│   ├── Class_2/
│   └── ...
└── Test/
    ├── Class_1/
    ├── Class_2/
    └── ...
```

⚠️ Note: The dataset is not included in this repository due to its large size (1.35GB). You can download it from the provided external link.

---

## 🔗 Dataset Download

👉 Kaggle Link:
https://www.kaggle.com/datasets/pacificrm/skindiseasedataset

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/hussnain144/SkinCare-Detection-Project.git
cd SkinCare-Detection-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Open Jupyter Notebook
2. Run `Skin_Disease_Detection.ipynb`
3. Train the model or load saved model
4. Test with new images

---

## 📊 Model Training

* Image preprocessing using `ImageDataGenerator`
* Data augmentation (rotation, zoom, flipping)
* Training using Adam optimizer
* Loss function: Categorical Crossentropy
* Evaluation using accuracy and validation loss

---

## 📈 Results

The model performance is evaluated using:

* Training Accuracy
* Validation Accuracy
* Loss Curves
* Confusion Matrix
* Classification Report

---

## 🔥 Explainable AI (Grad-CAM)

Grad-CAM is used to generate heatmaps that highlight the important regions of an image that influenced the model’s prediction.

This improves:

* Model transparency
* Trust in predictions
* Visual understanding of AI decisions

---

## 📌 Features

* Skin disease image classification
* Hybrid CNN deep learning model
* Data augmentation for better accuracy
* Model evaluation metrics
* Explainable AI using Grad-CAM
* Saved trained model (.h5 file)

---

## 📁 Project Files

```
SkinCare-Detection-Project/
│
├── Skin_Disease_Detection.ipynb
├── skin_disease_model.h5
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

---

## 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
opencv-python
scikit-learn
jupyter
```

## ⭐ Future Improvements

* Deploy using Streamlit / Flask
* Use Transfer Learning (ResNet, EfficientNet)
* Real-time prediction system
* Mobile application integration
