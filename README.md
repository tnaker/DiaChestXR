# DiaChestXR: AI Portfolio Project – Chest X-ray Diagnosis

DiaChestXR is an end-to-end AI portfolio project for automated chest X-ray analysis.
The system demonstrates practical skills in **deep learning, medical image analysis, and model deployment**, with a strong emphasis on **transfer learning** and **explainable AI**.

This project is designed to reflect real-world AI workflows commonly expected in **AI engineering and research-oriented roles**.

---

## Project Overview

* **Domain:** Medical Imaging (Chest X-ray)
* **Goal:** Assist in screening and diagnosis of thoracic diseases from X-ray images
* **Approach:** Deep Convolutional Neural Networks with pretrained backbones
* **Deployment:** Interactive web application for inference and visualization

---

## Tasks and Models

### Screening Classification (Single-label)

* **Model:** AlexNet (pretrained on ImageNet)
* **Classes:**

  * Covid-19
  * Pneumonia
  * Normal
* **Loss Function:** CrossEntropyLoss
* **Objective:** Fast and reliable initial screening

---

### Detailed Diagnosis (Multi-label)

* **Model:** DenseNet121
* **Output:** 14 thoracic disease labels per image
* **Loss Function:** BCEWithLogitsLoss
* **Objective:** Detect multiple co-existing chest conditions from a single X-ray

---

## Explainable AI (XAI)

To improve model transparency and interpretability, the project integrates **Grad-CAM**:

* Highlights spatial regions that contribute most to model predictions
* Helps visualize suspicious or abnormal areas in chest X-rays
* Demonstrates awareness of explainability requirements in medical AI systems

---

## Technologies Used

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Models:** AlexNet, DenseNet121
* **Data Processing:** Pandas, NumPy, OpenCV, Albumentations
* **Deployment:** Streamlit
* **Visualization:** Grad-CAM, Matplotlib

---

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/tnaker/DiaChestXR.git
cd DiaChestXR
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Skills Demonstrated

* Transfer learning with convolutional neural networks
* Single-label vs multi-label classification
* Medical image preprocessing and augmentation
* Model training, validation, and evaluation
* Explainable AI techniques (Grad-CAM)
* Model inference and deployment with Streamlit
* Clean and reproducible AI project setup

---

## Limitations and Future Improvements

* Intended for educational and research purposes only
* No clinical validation
* Potential dataset bias

Planned extensions:

* AUROC and per-class performance metrics
* External test set evaluation
* Model ensemble techniques
* Docker-based deployment

---

## Author

**Ho Phan The Anh**
Undergraduate Student
Ho Chi Minh City University of Technology (HCMUT)
