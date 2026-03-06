
<img width="2816" height="1536" alt="Gemini_Generated_Image_v9aj78v9aj78v9aj" src="https://github.com/user-attachments/assets/e0e3a5c1-6b0c-480c-b665-740e26f5bdc4" />

# CarotidNet: Multimodal Stroke Risk Predictor

A multimodal machine learning system that predicts stroke risk by fusing carotid ultrasound imaging with clinical patient records. This dual-modality approach addresses the critical limitations of single-source diagnostic models by integrating structural vascular analysis with systemic health indicators.

---

## Architecture

CarotidNet implements a **Y-shaped neural network** built in TensorFlow 2.16.1 with parallel processing streams:

### Visual Branch
- **Base model**: MobileNetV2 with frozen ImageNet weights
- **Input**: Resized TIFF ultrasound images
- **Function**: Extracts spatial features to identify structural vascular anomalies (plaque buildup, vessel wall irregularities, stenosis patterns)

### Clinical Branch
- **Architecture**: Multi-layer perceptron (MLP)
- **Input features**:
  - Age
  - Body mass index (BMI)
  - Hypertension history
  - Average glucose levels
- **Function**: Processes systemic health markers correlated with cerebrovascular risk

### Fusion Layer
- Concatenates encoded representations from both branches
- Dense layer integration for cross-modality feature synthesis
- Sigmoid classification stem outputs binary stroke risk probability

### Performance
- **Validation recall**: 87%
- **Clinical rationale**: High recall is non-negotiable in medical screening contexts. The model prioritizes sensitivity over specificity to minimize false negatives—missing a high-risk patient carries far greater clinical cost than over-referral for additional testing.

---

## Data Engineering

### Synthetic Pairing Strategy

Due to the absence of ethics approval for real paired hospital data, this project employs a **custom synthetic fusion methodology**:

- **Ultrasound imagery**: Sourced from the **Mendeley CUBS dataset**
- **Clinical records**: Sourced from the **Kaggle Stroke Prediction dataset**
- **Pairing logic**:
  - Irregular vessel morphology (metadata-derived) → Mapped to stroke-positive clinical profiles
  - Smooth vessel geometry → Mapped to healthy/low-risk profiles
- **Biological rationale**: Vessel wall irregularity strongly correlates with atherosclerotic burden, a primary stroke risk factor
- **Resulting dataset**: **456 paired samples**

This approach enables multimodal training without access to protected patient information while preserving biologically plausible feature relationships.

---

## Tech Stack

- **Python 3.11**
- **TensorFlow 2.16.1** (Keras 3 model format)
- **Streamlit** (web application framework)
- **OpenCV** (image preprocessing pipeline)
- **Scikit-Learn** (data normalization, preprocessing utilities)

---

## Repository Structure

```
CarotidNet/
│
├── app.py                          # Streamlit web interface
├── final_multimodal_model.h5       # Trained Keras 3 model (87% recall)
├── scaler.pkl                      # Fitted StandardScaler for clinical input normalization
├── requirements.txt                # Python dependencies for cloud deployment
│
├── notebooks/
│   ├── 01_Data_Pairing.ipynb       # Synthetic dataset fusion script
│   └── 02_Multimodal_Training.ipynb # Model training loop and evaluation
│
└── sample_scans/                   # Test ultrasound images
                                    # (Full raw medical dataset excluded due to GitHub storage limits)
```

---

## Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch Application
```bash
streamlit run app.py
```

The web interface allows:
1. Upload of carotid ultrasound images
2. Manual entry of clinical parameters (age, BMI, hypertension status, glucose levels)
3. Real-time stroke risk prediction with probability score

---

## Model Training

Training notebooks are provided in the `notebooks/` directory:

1. **01_Data_Pairing.ipynb**: Executes the synthetic pairing algorithm to generate the unified multimodal dataset
2. **02_Multimodal_Training.ipynb**: Implements the Y-shaped architecture, training loop, and validation metrics

---

## Key Features

- **Dual-modality fusion**: Eliminates blind spots inherent in single-source models
- **Transfer learning**: Leverages MobileNetV2's ImageNet pretraining for efficient feature extraction
- **Clinical interpretability**: Explicit separation of visual and clinical pathways maintains model transparency
- **High recall optimization**: Architecture tuned for medical screening requirements (minimizing false negatives)
- **Deployment-ready**: Streamlit interface and dependency management for cloud hosting (Streamlit Cloud, AWS, GCP)

---

## Limitations

- **Synthetic training data**: Pairing strategy approximates real-world correlations but lacks ground-truth validation
- **Dataset scale**: 456 samples insufficient for clinical-grade generalization
- **Single imaging modality**: Limited to B-mode ultrasound (no Doppler flow analysis)
- **Binary classification**: Does not stratify risk severity levels

---

## Future Work

- Integrate Doppler flow velocity data for hemodynamic analysis
- Expand dataset using federated learning across hospital networks (with appropriate ethics approval)
- Implement multiclass risk stratification (low/moderate/high)
- Deploy attention mechanisms for explainable AI (highlight high-risk plaque regions)

---

## Citation

If you use this work, please cite:
```
CarotidNet: Multimodal Stroke Risk Predictor
https://github.com/asarkar2210/CarotidNet---Multimodal-Stroke-Risk-Predictor
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⚠️ CLINICAL DISCLAIMER

**THIS SOFTWARE IS AN ACADEMIC DEMONSTRATION PROJECT AND IS NOT AN FDA-APPROVED MEDICAL DEVICE.**

- **Not for clinical use**: CarotidNet outputs cannot be used for actual medical diagnosis, treatment decisions, or patient care.
- **Not validated**: The model has not undergone clinical trials, regulatory review, or validation on real-world patient populations.
- **Synthetic data**: Training data was artificially paired using heuristic mapping strategies that do not reflect true clinical correlations.
- **No liability**: The authors assume no responsibility for any harm, injury, or adverse outcomes resulting from the use or misuse of this software.
- **Consult professionals**: All medical decisions must be made by licensed healthcare professionals using validated diagnostic tools and established clinical protocols.

**Use of this software acknowledges acceptance of these terms.**
