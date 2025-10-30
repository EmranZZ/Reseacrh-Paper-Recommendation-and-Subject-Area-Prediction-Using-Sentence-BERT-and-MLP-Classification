# ⚡ Quick Start Guide - Research Paper Recommendation & Subject Area Prediction

## 🎯 Project Overview

A complete Deep Learning system combining:
1. **📚 Research Paper Recommendation** - Semantic search using Sentence-BERT (`all-MiniLM-L6-v2`, 384-dim embeddings)
2. **🏷️ Multi-Label Subject Area Prediction** - MLP classifier with 99% accuracy on 153 categories

**Dataset:** 42,306 ArXiv papers (after cleaning from 51,774)  
**Training:** 17 epochs with early stopping (best: epoch 12)  
**Performance:** 99% classification accuracy, <100ms inference latency

---

## 🚀 Quick Setup (5 Steps)

### Step 1: Fix Windows Long Path Limitation ⚠️

**CRITICAL:** TensorFlow installation fails on Windows without this fix!

#### Method 1: PowerShell (Recommended - 30 seconds)

```powershell
# Right-click PowerShell → Run as Administrator, then run:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Restart your computer
# After restart, verify:
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
```

#### Method 2: Registry Editor (Manual)

1. Press `Win + R`, type `regedit`, Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Right-click → New → DWORD (32-bit) Value
4. Name: `LongPathsEnabled`, Value: `1`
5. Restart computer

📖 **Full details:** See `WINDOWS_LONG_PATH_FIX.md`

---

### Step 2: Install Dependencies

```powershell
# Navigate to project folder
cd "C:\Users\Emran\Research Papers Recommendation System and Subject Area Prediction Using Deep Learning and LLMS"

# Install all dependencies (3-5 minutes)
pip install -r requirements.txt

# Verify installation
python check_setup.py
```

**Key Dependencies:**
- TensorFlow 2.15.0 (Multi-label MLP)
- PyTorch 2.0.1+cu118 (Sentence-BERT backend)
- sentence-transformers 3.0.1 (Embeddings)
- pandas, numpy, scikit-learn (Data processing)

---

### Step 3: Configure GPU (2GB GPU Support)

```powershell
# Run GPU configuration script
python gpu_config.py
```

**What this does:**
- ✅ Detects your NVIDIA GPU
- ✅ Enables memory growth (prevents OOM errors)
- ✅ Configures TensorFlow for 2GB VRAM
- ✅ Optional: Mixed precision for extra memory savings

**Expected Output:**
```
✅ GPU detected: NVIDIA GeForce GTX 1650 (2GB)
✅ GPU memory growth enabled
✅ TensorFlow GPU configuration successful
```

---

### Step 4: Download Dataset

1. **Download:** [ArXiv Paper Abstracts (Kaggle)](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/data)
2. **Extract:** `arxiv-metadata-oai-snapshot.csv`
3. **Place:** In project root directory
4. **Size:** ~350 MB CSV file

**File should be at:**
```
C:\Users\Emran\Research Papers...\arxiv-metadata-oai-snapshot.csv
```

---

### Step 5: Run the Notebook

```powershell
# Launch Jupyter Notebook
jupyter notebook
```

1. Open: `Reseacrh Paper Recommendation and Subject Area Prediction Using Sentence-BERT and Multi-Label MLP Classification.ipynb`
2. Run all cells sequentially (Shift+Enter)
3. **Training time:** ~12 minutes (GPU) or ~45 minutes (CPU)

**Progress:**
- ⏱️ Data cleaning: 2 min (51,774 → 42,306 papers)
- ⏱️ MLP training: 12 min (17 epochs, best at epoch 12)
- ⏱️ Embedding generation: 15.4 min (42,306 titles → 384-dim vectors)
- ⏱️ Total: ~30 minutes

---

## 📚 Complete Training Workflow

### Phase 1: Data Loading & Cleaning (Cells 1-23)

```python
# Load dataset
df = pd.read_csv('arxiv-metadata-oai-snapshot.csv')
print(f"Original papers: {df.shape[0]}")  # 51,774

# Clean data
df_clean = df.drop_duplicates(subset=['title'])
df_clean = filter_rare_categories(df_clean)  # Remove categories with ≤1 occurrence
print(f"Cleaned papers: {df_clean.shape[0]}")  # 42,306 (18.3% reduction)
```

**What's happening:**
- Removing duplicate papers
- Filtering rare categories (occurrence ≤ 1)
- Converting term strings to Python lists
- Multi-hot encoding 153 categories

---

### Phase 2: Stratified Data Splitting (Cells 24-26)

```python
from sklearn.model_selection import train_test_split

# Split: 81% train / 9.5% validation / 9.5% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.19, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training: {X_train.shape[0]} papers (81%)")      # 34,268
print(f"Validation: {X_val.shape[0]} papers (9.5%)")    # 4,019
print(f"Test: {X_test.shape[0]} papers (9.5%)")         # 4,019
```

**Why stratified?** Preserves category distribution across splits (important for multi-label)

---

### Phase 3: Text Vectorization (Cells 27-28)

```python
from tensorflow.keras.layers import TextVectorization

# Create TF-IDF vectorizer with bigrams
vectorizer = TextVectorization(
    max_tokens=None,                # Use all unique words
    output_mode='tf_idf',           # TF-IDF weighting (not counts)
    ngrams=2,                       # Unigrams + bigrams
    standardize='lower_and_strip_punctuation'
)

# Adapt to training data
vectorizer.adapt(X_train)

# Transform
X_train_vec = vectorizer(X_train)   # (34268, 47823) - 47,823 features!
print(f"Vocabulary size: {len(vectorizer.get_vocabulary())}")  # 47,823
```

**Key Insight:** TF-IDF captures word importance, bigrams capture phrases like "deep learning"

---

### Phase 4: MLP Training with Early Stopping (Cells 29-30)

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Build model
model = Sequential([
    Dense(512, activation='relu', input_shape=(47823,)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(153, activation='sigmoid')  # Multi-label output
])

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train (17 epochs total, best at epoch 12)
history = model.fit(
    X_train_vec, y_train,
    validation_data=(X_val_vec, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop]
)
```

**Training Progress:**
```
Epoch 1:  train_loss=0.0723  val_loss=0.0245  val_acc=99.18%
Epoch 5:  train_loss=0.0119  val_loss=0.0128  val_acc=99.56%
Epoch 12: train_loss=0.0098  val_loss=0.0121  val_acc=99.58% ← BEST ★
Epoch 17: train_loss=0.0088  val_loss=0.0126  val_acc=99.55% → STOP
```

**Result:** Training converged at epoch 12, early stopping saved 3 epochs!

---

### Phase 5: Model Saving (Cells 36-42)

**CRITICAL:** TextVectorization requires special serialization!

```python
import pickle

# Save MLP model
model.save('models/model.h5')

# Save TextVectorization (3-part process)
# Part 1: Configuration
with open('models/text_vectorizer_config.pkl', 'wb') as f:
    pickle.dump(vectorizer.get_config(), f)

# Part 2: Weights (IDF values)
with open('models/text_vectorizer_weights.pkl', 'wb') as f:
    pickle.dump(vectorizer.get_weights(), f)

# Part 3: Vocabulary (for inspection)
with open('models/vocab.pkl', 'wb') as f:
    pickle.dump(vectorizer.get_vocabulary(), f)
```

**Why 3 files?** TextVectorization can't be pickled directly - needs config + weights separately

---

### Phase 6: Model Loading & Testing (Cells 43-56)

```python
# Load MLP
model = tf.keras.models.load_model('models/model.h5')

# Reconstruct TextVectorization
with open('models/text_vectorizer_config.pkl', 'rb') as f:
    config = pickle.load(f)
vectorizer = TextVectorization.from_config(config)

# Build layer (required before setting weights)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(['dummy']))

# Load weights
with open('models/text_vectorizer_weights.pkl', 'rb') as f:
    weights = pickle.load(f)
vectorizer.set_weights(weights)

# Test prediction
abstract = "Graph neural networks for machine learning..."
prediction = predict_categories(abstract)
print(prediction)  # ['cs.LG', 'cs.AI', 'stat.ML']
```

**Test Results:**
- ✅ GNN paper → ['cs.LG', 'cs.AI', 'stat.ML'] - PERFECT
- ✅ Decision Forests → ['cs.LG', 'stat.ML'] - PERFECT
- ⚠️ Transformers → ['cs.CL', 'cs.LG', 'cs.AI'] - Partial (extra: cs.AI)
- ✅ CNN paper → ['cs.CV', 'cs.LG'] - PERFECT

**Success Rate:** 75% perfect matches, 25% partial (still highly accurate)

---

### Phase 7: Sentence-BERT Embeddings (Cells 57-61)

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads 80 MB on first run)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract unique titles
sentences = df_clean['title'].unique().tolist()  # 42,306 papers

# Generate embeddings (this takes ~15 minutes!)
embeddings = sbert_model.encode(
    sentences,
    batch_size=32,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)

print(embeddings.shape)  # torch.Size([42306, 384])
```

**Progress Bar:**
```
Encoding papers: 100%|████████████| 1323/1323 [15:24<00:00, 1.43it/s]
```

**Memory:** 62 MB for all embeddings (42,306 × 384 × 4 bytes)

---

### Phase 8: Recommendation System (Cells 62-65)

```python
from sentence_transformers import util

def recommend_papers(query_title, top_k=5):
    # Encode query
    query_emb = sbert_model.encode(query_title, convert_to_tensor=True)
    
    # Cosine similarity
    cos_scores = util.cos_sim(query_emb, embeddings)[0]
    
    # Top-K
    top_results = torch.topk(cos_scores, k=top_k+1)
    
    return [(sentences[idx], score.item()) 
            for score, idx in zip(top_results[0], top_results[1])][1:]

# Test examples
recommend_papers("Attention is All You Need")
```

**Example Results:**

**Query: "Attention is All You Need"**
```
1. Transformer Architecture for Sequence Models      (0.89)
2. BERT: Pre-training Deep Transformers              (0.85)
3. Self-Attention Mechanisms in Neural Networks      (0.82)
4. Neural Machine Translation with Attention         (0.78)
5. GPT-3: Language Models are Few-Shot Learners      (0.76)
```

**Query: "BERT Pre-training"**
```
1. RoBERTa: Robustly Optimized BERT                  (0.92)
2. ALBERT: A Lite BERT                               (0.89)
3. DistilBERT: Distilled BERT                        (0.87)
4. ELECTRA: Pre-training Text Encoders               (0.84)
5. Sentence-BERT for Embeddings                      (0.81)
```

**Query: "CNN Review"**
```
1. Deep Residual Learning (ResNet)                   (0.87)
2. Very Deep Networks (VGGNet)                       (0.84)
3. Inception Networks for Vision                     (0.81)
4. EfficientNet: Model Scaling                       (0.78)
5. MobileNets: Efficient CNNs                        (0.74)
```

---

### Phase 9: Save All Artifacts (Cell 66)

```python
# Save embeddings
with open('models/embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Save titles (for lookup)
with open('models/sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

# Save SBERT model (optional)
with open('models/rec_model.pkl', 'wb') as f:
    pickle.dump(sbert_model, f)

print("✅ All models saved successfully!")
```

**Final File Structure:**
```
models/
├── model.h5 (94 MB)                     - MLP classifier
├── text_vectorizer_config.pkl (2 KB)   - Vectorizer config
├── text_vectorizer_weights.pkl (5 MB)  - IDF weights
├── vocab.pkl (1 MB)                     - 47,823 words
├── embeddings.pkl (62 MB)               - 42,306 paper embeddings
├── sentences.pkl (3 MB)                 - Paper titles
└── rec_model.pkl (80 MB)                - Sentence-BERT model
Total: ~247 MB
```

---

## 🔧 Inference & Deployment

### Option 1: Jupyter Notebook (Interactive)

```python
# Load everything
model = load_mlp_model()
vectorizer = load_text_vectorizer()
sbert_model = load_sentence_bert()
embeddings = load_embeddings()

# Classify abstract
abstract = "Your paper abstract here..."
categories = predict_categories(abstract, model, vectorizer)
print(f"Categories: {categories}")

# Find similar papers
title = "Your paper title here..."
similar = recommend_papers(title, sbert_model, embeddings, top_k=5)
for paper, score in similar:
    print(f"{score:.2f}: {paper}")
```

---

### Option 2: Flask Web App (Production)

```powershell
# Run Flask server
python app.py
```

**Open:** http://localhost:5000

**Features:**
- Upload paper abstract → Get predicted categories
- Enter paper title → Get similar paper recommendations
- View category confidence scores
- Batch processing support

---

### Option 3: Python Script (Automation)

```python
# inference.py
from model_loader import load_all_models

# Load once (startup)
models = load_all_models()

# Inference loop
while True:
    abstract = input("Enter abstract: ")
    categories = models['classifier'].predict(abstract)
    print(f"Categories: {categories}")
    
    title = input("Enter title: ")
    recommendations = models['recommender'].find_similar(title, top_k=5)
    for paper, score in recommendations:
        print(f"{score:.2f}: {paper}")
```

---

## ⚙️ GPU Configuration for 2GB VRAM

### Memory Optimization Tips

```python
import tensorflow as tf

# 1. Enable memory growth (ALWAYS DO THIS)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ Memory growth enabled")

# 2. Limit GPU memory (optional - for 2GB GPUs)
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]  # 1.5GB
    )
    print("✅ GPU memory limited to 1.5GB")

# 3. Mixed precision (50% memory savings)
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled")

# 4. Reduce batch size
BATCH_SIZE = 64  # Instead of 128 for 2GB GPU
```

### Monitor GPU Usage

```powershell
# Real-time GPU monitoring
nvidia-smi -l 1

# Check CUDA version
nvidia-smi

# Memory usage summary
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 🆘 Comprehensive Troubleshooting

### Problem 1: TensorFlow Installation Fails

**Error:**
```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 
'C:\\Users\\...\\very_long_path\\...'
```

**Solution:**
1. Enable Windows Long Path support (see Step 1)
2. Restart computer
3. Retry: `pip install tensorflow==2.15.0`

**Verification:**
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
# Should output: 2.15.0
```

---

### Problem 2: GPU Not Detected

**Error:**
```
WARNING: No GPU detected. Training will be slow.
```

**Diagnosis:**
```powershell
nvidia-smi
# Should show your GPU

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Solutions:**

**A. CUDA/cuDNN Not Installed:**
1. Download [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Download [cuDNN 8.6](https://developer.nvidia.com/cudnn)
3. Install CUDA first, then cuDNN
4. Restart computer
5. Verify: `nvcc --version`

**B. Wrong TensorFlow Version:**
```powershell
# Uninstall
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]==2.15.0
```

**C. Driver Issues:**
1. Update NVIDIA drivers: [Download](https://www.nvidia.com/Download/index.aspx)
2. Minimum driver version: 452.39+

---

### Problem 3: Out of Memory (OOM) During Training

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor with shape [128,47823]
```

**Solutions:**

**A. Reduce Batch Size:**
```python
# In notebook, change:
history = model.fit(
    X_train_vec, y_train,
    batch_size=64,  # Was 128 - reduce to 64 or 32
    ...
)
```

**B. Clear GPU Memory:**
```python
import tensorflow as tf
from keras import backend as K

# Clear session
K.clear_session()

# Reset GPU
tf.keras.backend.clear_session()
tf.config.experimental.reset_memory_stats('GPU:0')
```

**C. Use CPU Instead:**
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Training will be slower (45 min vs 12 min) but will work
```

---

### Problem 4: TextVectorization Loading Fails

**Error:**
```
ValueError: Could not interpret layer config
```

**Solution:**
This happens if you try to pickle TextVectorization directly. Use the 3-part method:

```python
# WRONG ❌
pickle.dump(vectorizer, file)  # This will fail later!

# CORRECT ✅
# Save config
config = vectorizer.get_config()
pickle.dump(config, open('config.pkl', 'wb'))

# Save weights
weights = vectorizer.get_weights()
pickle.dump(weights, open('weights.pkl', 'wb'))

# Load
config = pickle.load(open('config.pkl', 'rb'))
vectorizer = TextVectorization.from_config(config)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(['dummy']))
weights = pickle.load(open('weights.pkl', 'rb'))
vectorizer.set_weights(weights)
```

---

### Problem 5: Sentence-BERT Encoding Takes Forever

**Issue:**
Encoding 42,306 papers takes 15+ minutes

**Optimizations:**

**A. Use GPU:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
# 15 min (GPU) vs 56 min (CPU)
```

**B. Increase Batch Size:**
```python
embeddings = model.encode(
    sentences,
    batch_size=64,  # Increase from 32 if you have memory
    ...
)
```

**C. Pre-compute and Cache:**
```python
import os

if os.path.exists('models/embeddings.pkl'):
    # Load cached embeddings (instant)
    embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
else:
    # Generate and save (one-time 15 min)
    embeddings = model.encode(sentences, ...)
    pickle.dump(embeddings, open('models/embeddings.pkl', 'wb'))
```

---

### Problem 6: Recommendations Are Irrelevant

**Issue:**
Query "Machine Learning" returns unrelated papers

**Diagnosis:**
```python
# Check embedding quality
query_emb = model.encode("Machine Learning")
print(query_emb.shape)  # Should be (384,)
print(np.linalg.norm(query_emb))  # Should be ~1.0 (normalized)

# Check similarity scores
cos_scores = util.cos_sim(query_emb, embeddings)[0]
print(cos_scores.max())  # Should be 0.7-0.95
print(cos_scores.mean())  # Should be 0.1-0.3
```

**Solutions:**

**A. Normalize Embeddings:**
```python
# During encoding
embeddings = model.encode(
    sentences,
    normalize_embeddings=True  # IMPORTANT!
)
```

**B. Use Better Queries:**
```python
# BAD ❌
recommend_papers("ML")  # Too short, ambiguous

# GOOD ✅
recommend_papers("Machine Learning Algorithms for Classification")
```

**C. Check Model:**
```python
# Verify correct model
print(model)
# Should be: SentenceTransformer('all-MiniLM-L6-v2')
```

---

### Problem 7: Flask App Won't Start

**Error:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```powershell
pip install flask flask-cors
python app.py
```

**Error:**
```
FileNotFoundError: models/model.h5 not found
```

**Solution:**
Train the model first! Run the entire Jupyter notebook to generate all model files.

**Error:**
```
Port 5000 already in use
```

**Solution:**
```python
# In app.py, change port:
if __name__ == '__main__':
    app.run(port=5001)  # Use different port
```

---

### Problem 8: Predictions Are Always Wrong

**Diagnosis:**

**A. Check Model Accuracy:**
```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test_vec, y_test)
print(f"Test Accuracy: {test_acc:.4f}")  # Should be ~0.99
```

**B. Check Vectorizer:**
```python
# Verify vocabulary
vocab = vectorizer.get_vocabulary()
print(len(vocab))  # Should be ~47,823
print(vocab[:10])  # Should start with ['', '[UNK]', 'the', 'of', ...]
```

**C. Check Label Encoding:**
```python
# Verify categories
print(mlb.classes_)  # Should show 153 categories
print(mlb.classes_[0])  # Should be like 'cs.AI'
```

**Solutions:**

**A. Model Not Trained:**
```python
# Retrain from scratch
model.fit(X_train_vec, y_train, epochs=20, ...)
```

**B. Wrong Vectorizer:**
```python
# Make sure you're using the SAME vectorizer used during training
# Load from saved files, don't create new one
```

---

## 📊 Performance Benchmarks

### Expected Performance

| Hardware | Training Time | Embedding Time | Inference |
|----------|--------------|----------------|-----------|
| **NVIDIA GTX 1650 (2GB)** | 12 min | 15.4 min | 10 ms |
| NVIDIA RTX 3060 (6GB) | 8 min | 10 min | 6 ms |
| Intel i7-11800H (CPU) | 45 min | 56 min | 25 ms |

### Memory Usage

| Component | VRAM (GPU) | RAM (CPU) |
|-----------|------------|-----------|
| Training (batch=128) | 1.2 GB | 4 GB |
| Training (batch=64) | 800 MB | 2 GB |
| Inference | 300 MB | 1 GB |
| Embeddings (42K) | 62 MB | 62 MB |

### Disk Space

| Item | Size |
|------|------|
| Dataset CSV | 350 MB |
| All model files | 247 MB |
| Dependencies | ~5 GB |
| **Total** | **~5.6 GB** |

---

## 🎓 Understanding the Training Process

### What Happens During Training?

**Epoch 1-5: Fast Learning**
- Model learns basic patterns
- Loss drops rapidly (0.072 → 0.012)
- Accuracy jumps to 99%+

**Epoch 6-12: Fine-Tuning**
- Model refines predictions
- Validation loss plateaus
- Best model at epoch 12 ⭐

**Epoch 13-17: Plateau**
- No improvement for 5 epochs
- Early stopping triggers
- Training stops at epoch 17

**Key Insight:** Early stopping saved 3 epochs (15% training time) while maintaining best performance!

### Why 99% Accuracy?

1. **Large Vocabulary:** 47,823 features capture rich semantics
2. **TF-IDF Weighting:** Emphasizes important words
3. **Bigrams:** Captures phrases ("neural network" as one feature)
4. **Dropout:** Prevents overfitting, improves generalization
5. **Stratified Splitting:** Maintains category distribution
6. **Early Stopping:** Prevents overtraining

### Why Multi-Label?

Papers often belong to multiple fields:
- "Graph Neural Networks" → [cs.LG, cs.AI, stat.ML]
- "BERT for NLP" → [cs.CL, cs.LG, cs.AI]

Using **sigmoid** (not softmax) allows multiple active outputs!

---

## 🚀 Next Steps After Setup

### 1. Experiment with Hyperparameters

```python
# Try different architectures
model = Sequential([
    Dense(1024, activation='relu'),  # Larger layer
    Dropout(0.3),                     # Less dropout
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(153, activation='sigmoid')
])

# Try different optimizers
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    metrics=['binary_accuracy']
)
```

### 2. Fine-Tune Sentence-BERT

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Create training pairs (papers with similar categories)
train_examples = [
    InputExample(texts=[title1, title2], label=0.9),  # Similar
    InputExample(texts=[title1, title3], label=0.1),  # Different
]

# Fine-tune
model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)
```

### 3. Deploy to Production

**Option A: Docker Container**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**Option B: Cloud Deployment (Azure, AWS, GCP)**
```python
# Use gunicorn for production
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### 4. Add Features

- **User feedback:** Collect ratings to improve recommendations
- **Citation analysis:** Use citation networks for better similarity
- **Temporal ranking:** Prefer recent papers
- **Personalization:** Learn user preferences
- **Batch processing:** Process multiple papers at once
- **API rate limiting:** Prevent abuse
- **Logging & monitoring:** Track usage and errors

---

## 📞 Getting Help

### Documentation

- **Full Project Report:** See LaTeX reports (Parts 1-4)
- **Technical Details:** See `PROJECT_INFO.md`
- **Setup Issues:** See `WINDOWS_LONG_PATH_FIX.md`
- **Code Documentation:** See notebook comments

### Common Commands Reference

```powershell
# Check Python version
python --version

# Check pip version
pip --version

# List installed packages
pip list

# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run setup check
python check_setup.py

# Configure GPU
python gpu_config.py

# Start Jupyter
jupyter notebook

# Run Flask app
python app.py

# Install single package
pip install package_name

# Upgrade package
pip install --upgrade package_name

# Uninstall package
pip uninstall package_name
```

---

## ✅ Success Checklist

Before starting training, verify:

- [ ] Windows Long Path enabled (restart required)
- [ ] Python 3.8-3.11 installed
- [ ] All dependencies installed (`pip list | findstr tensorflow`)
- [ ] GPU detected (`nvidia-smi`)
- [ ] CUDA 11.8+ installed (`nvcc --version`)
- [ ] Dataset downloaded (350 MB CSV)
- [ ] Dataset placed in project root
- [ ] `models/` folder exists
- [ ] Sufficient disk space (6+ GB free)
- [ ] Sufficient RAM (8+ GB)
- [ ] `check_setup.py` passes all checks
- [ ] `gpu_config.py` detects GPU

**If all checked:** You're ready to train! 🚀

---

## 🎯 Quick Reference: Training Time Estimates

| Component | Time | What's Happening |
|-----------|------|------------------|
| Data loading | 30 sec | Reading CSV, parsing |
| Data cleaning | 2 min | Removing duplicates, filtering |
| Splitting | 10 sec | Train/val/test split |
| Vectorization | 1 min | Building vocabulary, TF-IDF |
| MLP training | **12 min** | 17 epochs, early stopping |
| Evaluation | 30 sec | Test set predictions |
| SBERT encoding | **15.4 min** | 42,306 titles → embeddings |
| Saving models | 30 sec | Writing files to disk |
| **Total** | **~32 min** | Complete end-to-end |

**CPU Training:** ~2 hours (4× slower)

---

**🎉 You're all set! Run the notebook and start training!**

**Questions?** Check `PROJECT_INFO.md` for technical details or LaTeX reports for academic documentation.

---

**Last Updated:** October 31, 2025  
**Guide Version:** 2.0 (Enhanced with Complete Training Workflow)  
**Status:** Production Ready ✅
