import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MNIST CNN Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_dataset():
    """Load and preprocess MNIST dataset"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    X_train_reshaped = X_train_norm.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test_norm.reshape(-1, 28, 28, 1)
    
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_train_norm': X_train_norm,
        'X_test_norm': X_test_norm,
        'X_train_reshaped': X_train_reshaped,
        'X_test_reshaped': X_test_reshaped,
        'y_train_cat': y_train_cat,
        'y_test_cat': y_test_cat
    }

@st.cache_resource
def load_trained_model():
    """Load pre-trained model"""
    model_path = 'mnist_cnn_model.h5'
    
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a task:",
    ["🏠 Home", 
     "📊 Task A: Data Exploration", 
     "🔧 Task B: Preprocessing", 
     "🧠 Task C: Model Building", 
     "🚀 Task D: Training & Evaluation", 
     "🎨 Task E: Visualization"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About This App
This application demonstrates a complete CNN pipeline for MNIST digit classification.

**Dataset**: 70,000 handwritten digit images  
**Model**: Convolutional Neural Network  
**Accuracy**: ~98-99%  
**Framework**: TensorFlow/Keras

**Total Marks**: 15
""")

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<div class="title-text">🎯 MNIST CNN Classifier</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the MNIST Digit Recognition System!
    
    This application demonstrates a complete **Convolutional Neural Network (CNN)** pipeline for 
    classifying handwritten digits (0-9) from the MNIST dataset.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Images", "60,000", "📚")
    with col2:
        st.metric("Test Images", "10,000", "🧪")
    with col3:
        st.metric("Classes", "10 digits", "🎯")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📋 Task Overview
    
    | Task | Description | Marks |
    |------|-------------|-------|
    | **A** | Data Loading & Exploration | 4 |
    | **B** | Preprocessing & Normalization | 2 |
    | **C** | CNN Model Building | 3 |
    | **D** | Training & Evaluation | 5 |
    | **E** | Visualization | 1 |
    | **Total** | | **15** |
    
    ### 🚀 Getting Started
    Use the navigation menu on the left to explore each task step by step.
    """)

# ============================================================================
# TASK A: DATA EXPLORATION
# ============================================================================

elif page == "📊 Task A: Data Exploration":
    st.markdown('<div class="section-header">📊 TASK A: DATA LOADING & EXPLORATION (4 Marks)</div>', unsafe_allow_html=True)
    
    st.write("Loading and exploring the MNIST dataset...")
    
    data = load_dataset()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1️⃣ Dataset Shapes")
        shape_table = pd.DataFrame({
            'Data': ['X_train', 'y_train', 'X_test', 'y_test'],
            'Shape': [
                str(data['X_train'].shape),
                str(data['y_train'].shape),
                str(data['X_test'].shape),
                str(data['y_test'].shape)
            ]
        })
        st.table(shape_table)
    
    with col2:
        st.subheader("2️⃣ Pixel Statistics")
        pixel_table = pd.DataFrame({
            'Metric': ['Min Value', 'Max Value', 'Data Type', 'Total Pixels'],
            'Value': [
                f"{data['X_train'].min()}",
                f"{data['X_train'].max()}",
                str(data['X_train'].dtype),
                f"{data['X_train'].size:,}"
            ]
        })
        st.table(pixel_table)
    
    st.subheader("3️⃣ Class Distribution")
    class_dist = pd.Series(data['y_train']).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(class_dist.index, class_dist.values, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Digit Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(class_dist.values):
        ax.text(i, v + 100, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)
    
    st.subheader("4️⃣ Dataset Balance Analysis")
    max_count = class_dist.max()
    min_count = class_dist.min()
    ratio = max_count / min_count
    
    status = "✅ BALANCED" if ratio <= 1.1 else "⚠️ IMBALANCED"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Count", f"{max_count}")
    col2.metric("Min Count", f"{min_count}")
    col3.metric("Imbalance Ratio", f"{ratio:.4f}")
    col4.metric("Status", status)
    
    st.subheader("5️⃣ Sample Images from Dataset")
    n_samples = 10
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 2))
    for i in range(n_samples):
        axes[i].imshow(data['X_train'][i], cmap='gray')
        axes[i].set_title(f"Label: {data['y_train'][i]}", fontsize=10, fontweight='bold')
        axes[i].axis('off')
    st.pyplot(fig)

# ============================================================================
# TASK B: PREPROCESSING
# ============================================================================

elif page == "🔧 Task B: Preprocessing":
    st.markdown('<div class="section-header">🔧 TASK B: PREPROCESSING & PREPARATION (2 Marks)</div>', unsafe_allow_html=True)
    
    data = load_dataset()
    
    st.subheader("1️⃣ Pixel Value Normalization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Normalization:**")
        before_df = pd.DataFrame({
            'Metric': ['Min Value', 'Max Value', 'Data Type', 'Range'],
            'Value': [
                f"{data['X_train'].min()}",
                f"{data['X_train'].max()}",
                str(data['X_train'].dtype),
                "0 - 255"
            ]
        })
        st.table(before_df)
    
    with col2:
        st.write("**After Normalization:**")
        after_df = pd.DataFrame({
            'Metric': ['Min Value', 'Max Value', 'Data Type', 'Range'],
            'Value': [
                f"{data['X_train_norm'].min():.4f}",
                f"{data['X_train_norm'].max():.4f}",
                str(data['X_train_norm'].dtype),
                "0.0 - 1.0"
            ]
        })
        st.table(after_df)
    
    st.subheader("2️⃣ Image Reshaping for CNN")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Shape", str(data['X_train'].shape))
    col2.metric("Reshaped Shape", str(data['X_train_reshaped'].shape))
    col3.metric("CNN Format", "28×28×1")
    
    st.subheader("3️⃣ Label Encoding (One-Hot)")
    st.write(f"**Original Labels**: Shape = {data['y_train'].shape}")
    st.write(f"Sample first 10 labels: {data['y_train'][:10]}")
    
    st.write(f"\n**One-Hot Encoded**: Shape = {data['y_train_cat'].shape}")
    st.write(f"Example: Label 3 → {data['y_train_cat'][3]}")
    
    st.subheader("4️⃣ Visual Comparison")
    fig, axes = plt.subplots(2, 5, figsize=(14, 5))
    
    for i in range(5):
        axes[0, i].imshow(data['X_train'][i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f"Original\nRange: [0-255]", fontsize=9, fontweight='bold')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(data['X_train_norm'][i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f"Normalized\nRange: [0-1]", fontsize=9, fontweight='bold')
        axes[1, i].axis('off')
    
    st.pyplot(fig)

# ============================================================================
# TASK C: MODEL BUILDING
# ============================================================================

elif page == "🧠 Task C: Model Building":
    st.markdown('<div class="section-header">🧠 TASK C: CNN MODEL BUILDING (3 Marks)</div>', unsafe_allow_html=True)
    
    st.subheader("CNN Architecture Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ### Model Layers:
        1. **Input Layer**: 28×28×1 images
        2. **Conv2D**: 32 filters, 3×3 kernel, ReLU
        3. **MaxPooling2D**: 2×2 pool
        4. **Conv2D**: 64 filters, 3×3 kernel, ReLU
        5. **MaxPooling2D**: 2×2 pool
        6. **Flatten**: Convert to 1D vector
        7. **Dense**: 128 neurons, ReLU
        8. **Dropout**: 0.5 (regularization)
        9. **Dense**: 10 neurons, Softmax (output)
        """)
    
    with col2:
        st.write("""
        ### Model Configuration:
        - **Loss Function**: Categorical Crossentropy
        - **Optimizer**: Adam
        - **Metrics**: Accuracy
        - **Batch Size**: 128
        - **Epochs**: 10
        - **Total Parameters**: ~200K
        """)
    
    st.subheader("Layer Summary")
    layer_df = pd.DataFrame({
        'Layer #': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Type': ['Input', 'Conv2D', 'MaxPool', 'Conv2D', 'MaxPool', 'Flatten', 'Dense', 'Dropout', 'Dense'],
        'Filters/Units': ['-', '32', '-', '64', '-', '-', '128', '-', '10'],
        'Kernel': ['-', '3×3', '2×2', '3×3', '2×2', '-', '-', '-', '-'],
        'Activation': ['-', 'ReLU', '-', 'ReLU', '-', '-', 'ReLU', '-', 'Softmax'],
        'Output Shape': ['28×28×1', '26×26×32', '13×13×32', '11×11×64', '5×5×64', '1600', '128', '128', '10']
    })
    st.dataframe(layer_df, use_container_width=True)

# ============================================================================
# TASK D: TRAINING & EVALUATION
# ============================================================================

elif page == "🚀 Task D: Training & Evaluation":
    st.markdown('<div class="section-header">🚀 TASK D: TRAINING & EVALUATION (5 Marks)</div>', unsafe_allow_html=True)
    
    data = load_dataset()
    model = load_trained_model()
    
    if st.button("🎯 Train Model (2-3 minutes with GPU)", key="train_btn"):
        st.info("⏳ Training in progress... Please wait!")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Build model
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        status_text.text("Training model on 60,000 images...")
        
        history = model.fit(
            data['X_train_reshaped'],
            data['y_train_cat'],
            validation_data=(data['X_test_reshaped'], data['y_test_cat']),
            epochs=10,
            batch_size=128,
            verbose=0
        )
        
        # Save model
        model.save('mnist_cnn_model.h5')
        progress_bar.progress(100)
        status_text.text("✅ Training Complete!")
        
        # Display results
        st.subheader("📊 Training Results")
        
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Accuracy", f"{final_train_acc:.4f}", f"{final_train_acc*100:.2f}%")
        col2.metric("Validation Accuracy", f"{final_val_acc:.4f}", f"{final_val_acc*100:.2f}%")
        col3.metric("Model Saved", "✅ Yes")
        
        # Test evaluation
        test_loss, test_accuracy = model.evaluate(data['X_test_reshaped'], data['y_test_cat'], verbose=0)
        st.metric("Test Accuracy", f"{test_accuracy:.4f}", f"{test_accuracy*100:.2f}%")
        
        # Plot training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
        axes[0].set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
        axes[1].set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Confusion Matrix
        y_pred_probs = model.predict(data['X_test_reshaped'], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(data['y_test'], y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10), ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix - MNIST Classification', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("📈 Classification Report")
        report = classification_report(data['y_test'], y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
    
    elif model is not None:
        st.success("✅ Pre-trained model loaded!")
        st.info("Model is ready for predictions in Task E")
    else:
        st.info("👆 Click the button above to train the model")

# ============================================================================
# TASK E: VISUALIZATION
# ============================================================================

elif page == "🎨 Task E: Visualization":
    st.markdown('<div class="section-header">🎨 TASK E: VISUALIZATION (1 Mark)</div>', unsafe_allow_html=True)
    
    data = load_dataset()
    model = load_trained_model()
    
    if model is not None:
        st.subheader("🎲 Random Test Images with Predictions")
        
        n_samples = st.slider("Number of samples to display:", 3, 20, 9)
        
        fig, axes = plt.subplots(3, (n_samples + 2) // 3, figsize=(15, 8))
        axes = axes.flatten()
        
        y_pred_probs = model.predict(data['X_test_reshaped'], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        np.random.seed(42)
        correct_count = 0
        
        for i in range(n_samples):
            idx = np.random.randint(0, len(data['X_test']))
            img = data['X_test'][idx]
            true_label = data['y_test'][idx]
            pred_label = y_pred[idx]
            confidence = np.max(y_pred_probs[idx])
            
            if true_label == pred_label:
                correct_count += 1
            
            axes[i].imshow(img, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(
                f'True: {true_label} | Pred: {pred_label}\nConfidence: {confidence:.2%}',
                color=color, fontweight='bold', fontsize=9
            )
            axes[i].axis('off')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        st.pyplot(fig)
        
        accuracy = correct_count / n_samples
        st.metric("Accuracy on Displayed Samples", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
        
    else:
        st.warning("⚠️ Model not found. Please train the model in Task D first.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>📚 Data Science Lab Exam - Variant 3 | Total Marks: 15</p>
    <p>Built with Streamlit | Powered by TensorFlow & Keras</p>
    <p>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)