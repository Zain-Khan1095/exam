# MNIST CNN Classifier - Data Science Lab Exam

## 📚 Overview

This is a complete implementation of a **Convolutional Neural Network (CNN)** for classifying handwritten digits from the MNIST dataset. The project includes both a command-line training script and an interactive Streamlit web application.

**Total Marks**: 15

## 📋 Tasks

| Task | Description | Marks |
|------|-------------|-------|
| **A** | Data Loading & Exploration | 4 |
| **B** | Preprocessing & Normalization | 2 |
| **C** | CNN Model Building | 3 |
| **D** | Training & Evaluation | 5 |
| **E** | Visualization | 1 |

## 📊 Dataset

- **Name**: MNIST (Modified National Institute of Standards and Technology)
- **Images**: 70,000 total (60,000 training + 10,000 test)
- **Image Size**: 28 × 28 pixels
- **Color**: Grayscale (0-255)
- **Classes**: 10 digits (0-9)

## 🧠 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D (32 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (128) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (10) + Softmax
    ↓
Output (Class probabilities)
```

## 📈 Model Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Batch Size**: 128
- **Epochs**: 10
- **Total Parameters**: ~200,000

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Zain-Khan1095/finalexam.git
cd finalexam
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
finalexam/
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                  # Documentation (this file)
├── mnist_cnn_model.h5         # Trained model (after training)
├── .gitignore                # Git ignore file
└── .github/
    └── workflows/
        └── deploy.yml         # GitHub Actions workflow (optional)
```

## 🎯 Usage

### Streamlit Web App

```bash
streamlit run app.py
```

Navigate using the sidebar to explore:
- **Home**: Overview and task summary
- **Task A**: Data exploration with visualizations
- **Task B**: Preprocessing comparison
- **Task C**: Model architecture details
- **Task D**: Training with metrics and confusion matrix
- **Task E**: Live predictions on test samples

### Command Line Training

```python
python train_and_save.py
```

This trains the model and saves it as `mnist_cnn_model.h5`

## 📊 Expected Performance

- **Test Accuracy**: 98-99%
- **Training Time**: 2-3 minutes (with GPU), 5-10 minutes (CPU)
- **Model Size**: 5.5 MB

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub:
```bash
git add .
git commit -m "Add MNIST CNN app"
git push origin main
```

2. Go to https://share.streamlit.io/
3. Select your GitHub repository
4. Deploy

Your app will be live at: `https://[username]-[repo-name].streamlit.app/`

### Environment Variables (if needed)

Create a `.streamlit/config.toml` file:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = false
```

## 🔧 Troubleshooting

### Problem: "Module not found" error
```bash
pip install -r requirements.txt --upgrade
```

### Problem: Streamlit not found
```bash
pip install streamlit --upgrade
```

### Problem: Model file too large for GitHub
Add to `.gitignore`:
```
*.h5
*.pkl
```

Then push without the model file. Use GitHub LFS for large files if needed.

### Problem: TensorFlow installation fails
```bash
# For CPU
pip install tensorflow-cpu

# For GPU (with CUDA support)
pip install tensorflow[and-cuda]
```

## 📥 Downloading Trained Model from Colab

If training in Google Colab, use:

```python
from google.colab import files

# Save and download model
model.save('mnist_cnn_model.h5')
files.download('mnist_cnn_model.h5')
```

Or save to Google Drive:

```python
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.copy('mnist_cnn_model.h5', '/content/drive/MyDrive/mnist_cnn_model.h5')
```

## 📚 Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CNN Architecture](https://en.wikipedia.org/wiki/Convolutional_neural_network)

## 📝 License

This project is for educational purposes only.

## 👨‍💻 Author

Created for Data Science Lab Exam - Variant 3

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## ❓ Questions?

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: November 2024  
**Version**: 1.0
```

