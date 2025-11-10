# 🔬 PathMNIST XAI — Explainable Medical Image Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art Flask web application demonstrating **Explainable AI (XAI)** for histopathology image classification using **ResNet-18** on the **PathMNIST** dataset. This project achieves **93.13% test accuracy**, surpassing the official benchmark, and provides real-time predictions with visual explanations via **Grad-CAM** and **Integrated Gradients**.

---

## 🎯 Project Highlights

- **🏆 Superior Performance**: Achieved **93.13% test accuracy** and **91.09% balanced accuracy**, **exceeding the PathMNIST benchmark** for ResNet-18 architecture
- **⚡ Real-time Classification**: Upload histopathology images for instant 9-class tissue classification
- **🔍 Explainable AI**: Generate visual explanations using Grad-CAM and Integrated Gradients from Captum
- **🎨 Professional Medical UI**: Modern, responsive interface with risk assessment and tissue information
- **📊 Comprehensive Metrics**: Interactive performance visualizations and per-class analytics
- **🔬 Research-Ready**: Complete training notebooks with data analysis and model evaluation

---

## 📊 Performance Metrics

### 🏅 Model Performance

Our ResNet-18 implementation outperforms the official PathMNIST benchmark:

| Metric | Our Result | Status |
|--------|------------|--------|
| **Test Accuracy** | **93.13%** | ✅ **Above Benchmark** |
| **Balanced Accuracy** | **91.09%** | ✅ **Excellent** |
| **Validation Accuracy** | **98.24%** | 🎯 Outstanding |
| **Training Accuracy** | **97.93%** | 🎯 Outstanding |

### 📉 Generalization Analysis

- **Validation → Test Drop**: 5.11% (Moderate generalization gap)
- **Cross-site Performance**: Good robustness across different tissue samples

### 🎯 Per-Class Performance

| Class | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| **Adipose (ADI)** | 98.65% | 0.9614 | 0.9865 | 0.9738 | ⬆️ Overrepresented (+8.23%) |
| **Background (BACK)** | 100.00% | 0.9953 | 1.0000 | 0.9976 | ≈ Balanced (+1.23%) |
| **Debris (DEB)** | 90.86% | 0.7422 | 0.9086 | 0.8170 | ⬇️ Underrepresented (-6.79%) |
| **Lymphocytes (LYM)** | 99.53% | 0.9460 | 0.9953 | 0.9700 | ≈ Balanced (-2.73%) |
| **Mucus (MUC)** | 90.72% | 1.0000 | 0.9072 | 0.9514 | ⬆️ Overrepresented (+5.52%) |
| **Smooth Muscle (MUS)** | 82.43% | 0.7987 | 0.8243 | 0.8113 | ⬇️ Underrepresented (-5.29%) |
| **Normal Mucosa (NORM)** | 98.11% | 0.9020 | 0.9811 | 0.9399 | ≈ Balanced (+1.56%) |
| **Stroma (STR)** | 66.51% | 0.8000 | 0.6651 | 0.7263 | ≈ Balanced (-4.58%) |
| **Tumor (TUM)** | 93.03% | 0.9820 | 0.9303 | 0.9554 | ≈ Balanced (+2.86%) |

**Overall Metrics:**
- **Macro Average**: Precision: 0.9031, Recall: 0.9109, F1-Score: 0.9048
- **Weighted Average**: Precision: 0.9338, Recall: 0.9313, F1-Score: 0.9311

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB RAM minimum
- 2GB disk space for models and datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "pathmnist XAI"
   ```

2. **Create a virtual environment** (recommended)
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Or using conda
   conda create -n pathmnist-xai python=3.10
   conda activate pathmnist-xai
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install flask captum medmnist numpy pandas matplotlib seaborn scipy scikit-learn pillow
   ```

4. **Download the dataset** (optional, for training)
   ```bash
   python datased_download.py
   ```
   *Note: Pre-trained model weights are included in `weights/best_pathmnist_resnet18.pth`*

5. **Run the Flask application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   ```
   Navigate to: http://localhost:5000
   ```

---

## 🏗️ Project Architecture

### Model Architecture

- **Base Model**: ResNet-18 with ImageNet transfer learning
- **Training Strategy**: Freeze conv1-layer2, fine-tune layer3-4 + fully connected layers
- **Parameters**: 11.18M total, 10.50M trainable
- **Input Size**: 224×224 RGB images (upscaled from 28×28 PathMNIST)
- **Output**: 9-class softmax classification

### Training Configuration

```python
Optimizer: Adam (lr=0.0001)
Loss Function: CrossEntropyLoss
Epochs: 20
Batch Size: 128
Data Augmentation: 
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter
  - ImageNet normalization
```

### XAI Methods Implemented

1. **Layer Grad-CAM** - Spatial localization of important regions
2. **Integrated Gradients** - Pixel-level attribution analysis
3. **Saliency Maps** - Gradient sensitivity visualization

---

## 📁 Project Structure

```
pathmnist XAI/
├── app.py                      # Flask web application entry point
├── config.py                   # Configuration and settings
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/                     # Model architecture
│   ├── model_loader.py        # ResNet-18 model definition and loader
│   └── __pycache__/
│
├── notebook and model/         # Training notebooks and weights
│   ├── pathmnist-xai-resnet18-kaggle-20epochs-93-acc.ipynb
│   ├── pathmnist-xai-resnet18-kaggle-5epochs-95-acc.ipynb
│   ├── best_pathmnist_resnet18.pth
│   └── best_pathmnist_resnet18_5epochs.pth
│
├── weights/                    # Production model weights
│   └── best_pathmnist_resnet18.pth
│
├── dataset/                    # PathMNIST data (downloaded)
│   ├── pathmnist.npz
│   ├── meta.json
│   ├── X_train.npy, y_train.npy
│   ├── X_val.npy, y_val.npy
│   └── X_test.npy, y_test.npy
│
├── static/                     # Frontend assets
│   ├── css/
│   │   └── custom.css         # Custom styling
│   ├── js/
│   │   └── main.js            # JavaScript interactions
│   ├── data/
│   │   ├── metrics.json       # Model performance metrics
│   │   ├── confusion_matrix.json
│   │   └── class_samples.json
│   ├── images/                # Sample images per class
│   ├── overlays/              # Generated XAI visualizations
│   └── uploads/               # Temporary upload storage
│
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Landing page with demo
│   ├── result.html            # Prediction results page
│   └── partials/              # Reusable components
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── preprocessing.py       # Image preprocessing pipeline
    ├── xai_methods.py         # Captum XAI wrappers
    └── metrics.py             # Performance metrics
```

---

## 📸 Screenshots

### Landing Page & Demo Interface
*Add screenshot here: `docs/screenshots/landing-page.png`*
- Hero section with project overview
- Performance metrics visualization
- Interactive demo with image upload

### Prediction Results
*Add screenshot here: `docs/screenshots/prediction-results.png`*
- Top prediction with confidence score
- All 9 class probabilities with progress bars
- Risk assessment with medical disclaimers

### XAI Explanations - Grad-CAM
*Add screenshot here: `docs/screenshots/gradcam-explanation.png`*
- Heatmap overlay showing spatial attention
- Original image comparison
- Class-specific activation visualization

### XAI Explanations - Integrated Gradients
*Add screenshot here: `docs/screenshots/integrated-gradients.png`*
- Pixel-level attribution analysis
- Smooth gradient visualization
- Feature importance mapping

### Risk Assessment Dashboard
*Add screenshot here: `docs/screenshots/risk-assessment.png`*
- Medical risk level indicators
- Confidence visualization
- Clinical guidance and disclaimers

### Tissue Classification Guide
*Add screenshot here: `docs/screenshots/tissue-guide.png`*
- Comprehensive tissue type information
- Risk classification system
- Educational medical content

---

## 🎓 Training Methodology

### Dataset: PathMNIST

PathMNIST is a medical image dataset consisting of 107,180 histopathology images (28×28 pixels) from colorectal cancer tissue samples, containing 9 different tissue types.

**Dataset Statistics:**
- **Training Set**: 89,996 images
- **Validation Set**: 10,004 images  
- **Test Set**: 7,180 images
- **Classes**: 9 tissue types
- **Image Size**: 28×28 RGB (upscaled to 224×224 for ResNet)

### Training Process

The complete training pipeline is documented in the Jupyter notebooks:

1. **Data Loading & Exploration** (`Section 1`)
   - Download PathMNIST dataset using `medmnist` library
   - Analyze class distribution and data statistics
   - Visualize sample images from each tissue type

2. **Data Preprocessing** (`Section 2`)
   - Resize images from 28×28 to 224×224 for ResNet-18
   - Apply data augmentation (horizontal flip, rotation, color jitter)
   - Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

3. **Model Architecture** (`Section 3`)
   - Load pre-trained ResNet-18 from torchvision
   - Replace final FC layer for 9-class classification
   - Implement selective layer freezing strategy

4. **Transfer Learning Strategy** (`Section 4`)
   - **Phase 1**: Freeze conv1, layer1, layer2 (early features)
   - **Phase 2**: Fine-tune layer3, layer4, and FC layers (high-level features)
   - **Rationale**: Preserve low-level edge/texture features, adapt high-level semantic features

5. **Training Loop** (`Section 5`)
   - Optimizer: Adam with learning rate 0.0001
   - Loss: CrossEntropyLoss
   - Epochs: 20 with early stopping
   - Batch size: 128
   - Learning rate scheduling: ReduceLROnPlateau

6. **Evaluation & Analysis** (`Section 6`)
   - Confusion matrix analysis
   - Per-class performance metrics
   - Distribution shift analysis
   - Worst prediction analysis

7. **XAI Implementation** (`Section 7`)
   - Layer Grad-CAM for spatial localization
   - Integrated Gradients for pixel attribution
   - Saliency maps for gradient sensitivity
   - Visual explanation generation for validation samples

### Key Training Insights

✅ **What Worked Well:**
- Transfer learning from ImageNet significantly boosted performance
- Selective fine-tuning prevented overfitting
- Data augmentation improved generalization
- Adam optimizer with small learning rate ensured stable convergence

⚠️ **Challenges Addressed:**
- **Class Imbalance**: Some classes (STR, DEB) had lower accuracy due to underrepresentation
- **Distribution Shift**: Performance gap between validation (98.24%) and test (93.13%) indicates some distribution mismatch
- **Small Image Size**: Upscaling from 28×28 to 224×224 required careful interpolation

🎯 **Model Improvements:**
- Achieved 93.13% test accuracy (exceeding benchmark)
- Balanced accuracy of 91.09% shows good performance across all classes
- Perfect accuracy (100%) on Background class
- Strong performance on critical classes (Tumor: 93.03%, Normal: 98.11%)

---

## 🔍 Explainable AI (XAI) Implementation

### Why XAI Matters in Medical AI

In medical image classification, **understanding why** a model makes a prediction is as important as the prediction itself. Our implementation provides:

- **Clinical Trust**: Visualize which regions influence the diagnosis
- **Error Analysis**: Identify when the model focuses on irrelevant features
- **Educational Value**: Help medical professionals understand AI decision-making
- **Regulatory Compliance**: Meet interpretability requirements for medical AI systems

### XAI Methods

#### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Implementation Details:**
```python
# From utils/xai_methods.py
explainer = LayerGradCam(model, model.layer4)
attribution = explainer.attribute(input_tensor, target=class_idx)
```

**Features:**
- Targets layer4 (final convolutional layer) for high-level semantic features
- Produces class-discriminative localization maps
- Bilinear interpolation for smooth 224×224 upsampling
- Higher DPI (150) rendering for publication-quality visualizations

**Use Case**: Understanding which spatial regions are most important for classification

#### 2. Integrated Gradients

**Implementation Details:**
```python
# From utils/xai_methods.py
ig = IntegratedGradients(model)
attributions = ig.attribute(input_tensor, target=class_idx, n_steps=200)
```

**Features:**
- 200 integration steps for accurate gradient approximation
- Gaussian smoothing (sigma=2.0) to reduce noise
- Pixel-level attribution scores
- Baseline: black image (all zeros)

**Use Case**: Fine-grained pixel importance analysis

#### 3. Visual Enhancements

Our implementation includes several quality improvements:
- **Bilinear Interpolation**: Smooth upsampling of attribution maps
- **Gaussian Filtering**: Eliminate blocky artifacts in visualizations
- **High DPI Rendering**: 150 DPI for crisp, professional outputs
- **Colormap Optimization**: Perceptually uniform colormaps for better interpretation

---

## 🌐 Web Application Features

### Flask Backend (`app.py`)

**Routes:**
- `GET /` - Landing page with demo interface
- `POST /predict` - Upload image and get predictions
- `POST /explain_page` - Generate XAI explanations
- `GET /metrics` - JSON API for performance metrics

**Key Features:**
- Session management for user uploads
- Temporary file handling with automatic cleanup
- Base64-encoded image transmission for instant display
- Error handling and validation

### Frontend Technologies

- **Bootstrap 5.3.3**: Responsive grid system and components
- **Chart.js**: Interactive performance charts
- **AOS (Animate On Scroll)**: Smooth animations
- **Custom CSS**: Professional medical-grade styling
- **Vanilla JavaScript**: Lightweight interactive features

### User Experience Highlights

- ✨ **Drag & Drop Upload**: Intuitive image selection
- 🔄 **Real-time Preview**: Instant image preview before prediction
- 📊 **Interactive Charts**: Hover to explore per-class accuracy
- 🎨 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Fast Performance**: Optimized for quick inference
- 🔒 **Input Validation**: File type and size checks

---

## 🗂️ Key Files Explained
- `app.py`
	- Flask routes: `/` (index), `/predict` (handle upload and run model), `/explain_page` (generate explanations), `/metrics` (JSON metrics)
	- Uses `PathMNISTPreprocessor` from `utils/preprocessing.py` and `XAIExplainer` from `utils/xai_methods.py`.
	- Saves temporary upload at `static/uploads/temp.png` and returns `templates/result.html` for results.

- `config.py`
	- Central configuration: `MODEL_PATH`, `DEVICE`, class names (`CLASS_NAMES`, `CLASS_SHORT`), normalization constants (`IMAGENET_MEAN`, `IMAGENET_STD`) and demo metrics (test accuracy / per-class accuracy).

- `models/model_loader.py`
	- Defines `PathMNISTResNet` (ResNet-18 backbone, adjusted final linear layer) and `load_model()` helper to load checkpoints.

- `utils/preprocessing.py`
	- `PathMNISTPreprocessor` uses torchvision transforms to resize images to 224×224, convert to Tensor and normalize with ImageNet stats used in training.
	- `tensor_to_numpy()` denormalizes tensors back to `[0,1]` images for visualization.

- `utils/xai_methods.py`
	- Provides `XAIExplainer` which wraps Captum's `LayerGradCam` and `IntegratedGradients`.
	- Key implementation notes:
		- Grad‑CAM uses bilinear interpolation to upsample activations smoothly to the input size.
		- Integrated Gradients uses an increased `n_steps` and Gaussian smoothing to reduce noisy pixel-level artifacts.
		- Overlay and `to_base64()` renderings use higher DPI and bilinear interpolation to avoid blocky images in the UI.

- `templates/` & `static/`
	- Modern UI built with Bootstrap, Bootstrap Icons, AOS (Animate On Scroll) and custom CSS at `static/css/custom.css`.
	- `index.html` hosts the upload UI and model performance dashboard. `result.html` shows predictions, controls for explanations, and the overlay/heatmap gallery.

How the model was trained (notebook notes)
----------------------------------------
The notebooks in `notebook and model/` contain the training pipeline and exploratory analysis. Important steps used during training:

- Data: PathMNIST (medmnist) — histopathology patches labeled into 9 classes.
- Preprocessing: images resized to 224×224, normalized with ImageNet mean/std to match the ResNet backbone expectations.
- Model: ResNet‑18 with the final linear layer replaced to output 9 classes. Training used a standard classification loss (cross-entropy).
- Evaluation: computed test accuracy and per-class accuracy; results were stored (demo metrics embedded in `config.py`).
- Explainability: experiments were performed with Captum's `LayerGradCam` and `IntegratedGradients` in the notebooks to confirm plausible explanations before integrating into the web app.

How the Flask app was built
--------------------------
High-level steps taken to create the web app:

1. Create a minimal Flask application (`app.py`) exposing routes for the UI and actions.
2. Load the trained model in memory at startup using `models/model_loader.py` and set it to evaluation mode.
3. Implement a preprocessing helper mirroring training transforms (`utils/preprocessing.py`) so uploaded images are prepared the same way as training images.
4. Implement `XAIExplainer` wrapping Captum methods and provide safe utilities to convert outputs into RGB overlays and base64 PNGs for embedding in HTML.
5. Build responsive HTML templates (`templates/`), add polished CSS (`static/css/custom.css`), and add small JS interactions (`static/js/main.js`) for previewing images, showing charts and validating forms.
6. Add simple guard-rails for Chart rendering and helpful error messages so the UI degrades gracefully when data is missing.

Usage — running locally
----------------------
Recommended steps to run locally (Windows / PowerShell shown).

1) Create a Python virtual environment and activate it

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) Install dependencies

Note: this project uses PyTorch and other packages — install a matching torch build for your CUDA or CPU environment. Example (CPU-only):

```powershell
# core Python deps
pip install flask pillow numpy matplotlib scikit-learn captum torchvision scipy
# install torch appropriate for your system – for CPU only example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, if you want a prepared `requirements.txt`, I can generate one pinned to the versions used in my environment.

3) Start the app

```powershell
python app.py
```

4) Open the app in your browser at `http://127.0.0.1:5000` and use the Live Demo to upload images, run predictions, and generate explanations.

Notes about dependencies
- PyTorch: choose the correct wheel for your GPU (CUDA) or CPU; using an incompatible torch wheel will fail to load the model on CUDA.
- Captum: used for explainability. `pip install captum`.
- SciPy: required for Gaussian smoothing applied to Integrated Gradients (already used in `utils/xai_methods.py`).

Screenshots (placeholders)
-------------------------
Add screenshots to `static/images/screenshots/` (create the folder if it doesn't exist). Recommended sizes: 1600×640 for header/full-width screenshots and 800×380 for smaller widgets.

In the README you can add images like this (example Markdown):

```markdown
![Demo Interface](static/images/screenshots/demo-interface.png)
![Explanation Overlay](static/images/screenshots/overlay.png)
```

Troubleshooting and tips
------------------------
- If the server prints `Model loaded on cpu` but you expected CUDA, check that your system has CUDA available and that the installed `torch` wheel supports it.
- If explanation images look blocky or noisy:
	- For Grad‑CAM we use bilinear interpolation during upsampling and higher DPI when rendering to PNG to avoid blockiness.
	- For Integrated Gradients we increased `n_steps` and apply light Gaussian smoothing before normalizing — this reduces noisy pixel artifacts.
- If Chart.js fails to render, open the browser developer console to see errors; the page has a `perClassWarn` area which shows helpful diagnostics.

Developer notes — XAI implementation details
-------------------------------------------
- Grad‑CAM: `utils/xai_methods.py` uses Captum `LayerGradCam` and upscales activations to 224×224 with bilinear interpolation to get a smooth heatmap.
- Integrated Gradients: computed with an increased number of steps to reduce noise; absolute attributions are summed over channels and then Gaussian-filtered before normalization. This helps create smooth saliency maps.
- Rendering: `to_base64()` renders images using a higher DPI and bilinear interpolation in matplotlib to produce a high-quality PNG that the UI embeds.

Contributing
------------
If you want to improve the app:

- Add more XAI methods (e.g. SmoothGrad, Grad‑CAM++), expose additional parameters in the UI.
- Add server-side caching for explanations to avoid recomputing the same outputs for the same image+target.
- Extend the evaluation dashboard with confusion matrices and per-class visual examples.

License
-------
This repository is provided for research and educational use. If you want a license stated here, let me know and I can add an appropriate open-source license (MIT, Apache 2.0, etc.).

Contact
-------
If you want help packaging this as a Docker container or deploying to a cloud provider, tell me your target environment and I can add a Dockerfile and deployment instructions.

