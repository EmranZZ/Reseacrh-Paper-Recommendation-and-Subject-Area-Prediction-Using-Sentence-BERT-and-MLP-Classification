# 🛠️ Complete Setup Guide - Research Paper Recommendation & Subject Area Prediction

**Project:** Research Paper Recommendation and Subject Area Prediction Using Sentence-BERT and Multi-Label MLP Classification  
**Version:** 2.0 (Enhanced for Complete Implementation)  
**Last Updated:** January 2025

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Pre-Installation Checklist](#pre-installation-checklist)
3. [Windows Long Path Fix (CRITICAL)](#windows-long-path-fix-critical)
4. [Python Environment Setup](#python-environment-setup)
5. [GPU Setup (CUDA & cuDNN)](#gpu-setup-cuda--cudnn)
6. [Dependency Installation](#dependency-installation)
7. [Dataset Download](#dataset-download)
8. [Environment Verification](#environment-verification)
9. [Training the Models](#training-the-models)
10. [Running the Application](#running-the-application)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Performance Optimization](#performance-optimization)

---

## 🖥️ System Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **OS** | Windows 10 (64-bit) | Windows 10/11 (64-bit) | Linux/Mac also supported |
| **Python** | 3.8 | 3.10 | Version 3.8-3.11 compatible |
| **RAM** | 8 GB | 16 GB | Training uses ~4 GB RAM |
| **Disk Space** | 6 GB free | 10 GB free | Includes dataset + models |
| **GPU** | None (CPU works) | NVIDIA 2GB+ VRAM | GPU is 4× faster |
| **CUDA** | N/A for CPU | 11.8+ | Required for GPU |
| **cuDNN** | N/A for CPU | 8.6+ | Required for GPU |
| **Internet** | Required | Required | For downloading packages |

### Hardware Performance Comparison

| Hardware | Training Time | Embedding Time | Total Time |
|----------|---------------|----------------|------------|
| **NVIDIA GTX 1650 (2GB)** | 12 min | 15 min | ~30 min |
| NVIDIA RTX 3060 (6GB) | 8 min | 10 min | ~20 min |
| NVIDIA RTX 4090 (24GB) | 5 min | 6 min | ~12 min |
| Intel i7-11800H (CPU) | 45 min | 56 min | ~2 hours |
| AMD Ryzen 9 (CPU) | 38 min | 48 min | ~90 min |

**Recommendation:** GPU is highly recommended but not required. Training on CPU is ~4× slower but works fine.

---

## ✅ Pre-Installation Checklist

Before starting, verify you have:

- [ ] Windows 10/11 (64-bit) or Linux/Mac
- [ ] Administrator access (for Long Path fix and CUDA installation)
- [ ] Stable internet connection (will download ~3-5 GB)
- [ ] 6+ GB free disk space
- [ ] 8+ GB RAM
- [ ] (Optional) NVIDIA GPU with 2+ GB VRAM
- [ ] Python 3.8-3.11 installed (check: `python --version`)
- [ ] pip installed (check: `pip --version`)

---

## ⚠️ Windows Long Path Fix (CRITICAL)

**⚠️ WINDOWS USERS: THIS IS MANDATORY!**

TensorFlow installation will **FAIL** without this fix due to Windows' 260-character path limit.

### Why This Is Needed

Windows has a legacy 260-character limit for file paths. TensorFlow's package structure exceeds this limit, causing installation to fail with:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory
```

### Solution 1: PowerShell (Fastest - 30 seconds)

1. **Right-click PowerShell** → "Run as Administrator"
2. **Run this command:**

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

3. **Restart your computer** (required!)
4. **Verify after restart:**

```powershell
Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled"
```

**Expected output:** `LongPathsEnabled: 1`

### Solution 2: Registry Editor (Manual)

1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Right-click in right pane → New → **DWORD (32-bit) Value**
4. Name: `LongPathsEnabled`
5. Double-click it → Set value to `1`
6. Click OK
7. **Restart your computer**

### Solution 3: Group Policy (Windows Pro/Enterprise)

1. Press `Win + R`, type `gpedit.msc`, press Enter
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Double-click "Enable Win32 long paths"
4. Select **Enabled**
5. Click Apply → OK
6. **Restart your computer**

### Verification

After restart, verify Long Path is enabled:

```powershell
python -c "import winreg; key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\FileSystem'); value, _ = winreg.QueryValueEx(key, 'LongPathsEnabled'); print('Long Path Enabled!' if value == 1 else 'Long Path DISABLED')"
```

**⚠️ DO NOT PROCEED until Long Path is enabled and computer is restarted!**

---

## 🐍 Python Environment Setup

### Step 1: Verify Python Installation

```powershell
python --version
```

**Expected:** `Python 3.8.x`, `Python 3.9.x`, `Python 3.10.x`, or `Python 3.11.x`

**If not installed:**
1. Download from [python.org](https://www.python.org/downloads/)
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Verify installation: `python --version`

### Step 2: Verify pip

```powershell
pip --version
```

**If pip is missing:**

```powershell
python -m ensurepip --upgrade
```

### Step 3: Create Virtual Environment (Recommended)

**Why use venv?** Isolates project dependencies, prevents conflicts, easy to delete/recreate.

```powershell
# Navigate to project folder
cd "C:\Users\Emran\Research Papers Recommendation System and Subject Area Prediction Using Deep Learning and LLMS"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# You should see (venv) in your prompt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**To deactivate later:**
```powershell
deactivate
```

### Step 4: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

---

## 🎮 GPU Setup (CUDA & cuDNN)

**⚠️ GPU USERS ONLY - Skip this section if using CPU**

### Prerequisites

1. **NVIDIA GPU** with compute capability 3.5+ (check: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus))
2. **NVIDIA Drivers** version 452.39 or higher
3. **Windows 10/11 (64-bit)** or Linux

### Step 1: Update GPU Drivers

1. Visit [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx)
2. Enter your GPU model (e.g., "GTX 1650")
3. Download and install latest drivers
4. **Restart computer**

### Step 2: Verify Driver Installation

```powershell
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 536.23       Driver Version: 536.23       CUDA Version: 12.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   45C    P8    N/A /  N/A |    256MiB /  2048MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**If nvidia-smi fails:** Drivers not installed correctly. Reinstall drivers.

### Step 3: Install CUDA 11.8

1. **Download CUDA 11.8:**
   - URL: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Select: Windows → x86_64 → 10/11 → exe (local)

2. **Run installer:**
   - Choose **Custom Installation**
   - Select: CUDA Toolkit, CUDA Runtime, CUDA Development
   - Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

3. **Verify installation:**

```powershell
nvcc --version
```

**Expected output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

4. **Verify CUDA path:**

```powershell
echo $env:CUDA_PATH
```

**Expected:** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

**If empty, add to PATH:**
- Search "Environment Variables" → Edit system environment variables
- System variables → Path → Edit → New
- Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
- Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp`
- OK → Restart PowerShell

### Step 4: Install cuDNN 8.6

1. **Create NVIDIA account** (free): https://developer.nvidia.com/cudnn

2. **Download cuDNN 8.6 for CUDA 11.x:**
   - Login → cuDNN Archive
   - Download: cuDNN v8.6.0 for CUDA 11.x (Windows)
   - File: `cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip`

3. **Extract ZIP:**
   - Extract to: `C:\tools\cudnn-8.6.0-cuda11.8`

4. **Copy files to CUDA directory:**

```powershell
# Copy DLL files
copy "C:\tools\cudnn-8.6.0-cuda11.8\bin\*.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\"

# Copy header files
copy "C:\tools\cudnn-8.6.0-cuda11.8\include\*.h" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\"

# Copy lib files
copy "C:\tools\cudnn-8.6.0-cuda11.8\lib\x64\*.lib" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\"
```

5. **Verify cuDNN:**

```powershell
# Check if DLL exists
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll"
```

**Expected:** `True`

### Step 5: Test GPU with TensorFlow (after installing TensorFlow)

```powershell
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

**Expected (after TensorFlow installation):**
```
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 📦 Dependency Installation

### Step 1: Navigate to Project Directory

```powershell
cd "C:\Users\Emran\Research Papers Recommendation System and Subject Area Prediction Using Deep Learning and LLMS"
```

### Step 2: Install All Dependencies

**Option A: Install from requirements.txt (Recommended)**

```powershell
pip install -r requirements.txt
```

**This will install:**
- TensorFlow 2.15.0 (~500 MB)
- PyTorch 2.0.1 (~2 GB with CUDA)
- sentence-transformers 3.0.1 (~50 MB)
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- flask, streamlit
- jupyter, ipykernel
- And all dependencies

**Total download:** ~3-5 GB  
**Installation time:** 10-20 minutes

**Option B: Install Individually (If Option A fails)**

```powershell
# Core frameworks
pip install tensorflow==2.15.0
pip install torch==2.0.1 torchvision==0.15.2
pip install sentence-transformers==3.0.1

# Data processing
pip install pandas==2.1.4 numpy==1.24.3 scikit-learn==1.3.2

# Visualization
pip install matplotlib==3.8.2 seaborn==0.13.0

# Web frameworks
pip install flask==3.0.0 flask-cors==4.0.0 streamlit==1.29.0

# Utilities
pip install jupyter==1.0.0 psutil==5.9.6 tqdm==4.66.1 ipykernel==6.27.1
```

### Step 3: GPU-Specific Installation (GPU users only)

**For TensorFlow GPU:**

```powershell
pip install tensorflow[and-cuda]==2.15.0
```

**For PyTorch with CUDA 11.8:**

```powershell
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

```powershell
python check_setup.py
```

**This will check:**
- ✓ Python version
- ✓ All dependencies installed
- ✓ GPU availability (if applicable)
- ✓ CUDA/cuDNN (if applicable)
- ✓ Disk space
- ✓ Memory
- ✓ Windows Long Path

**Expected output:**
```
✓ ALL CHECKS PASSED!
You're ready to start training!
```

---

## 📊 Dataset Download

### Step 1: Download from Kaggle

1. **Visit:** https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/data
2. **Login/Create Kaggle account** (free)
3. **Download:** `arxiv-metadata-oai-snapshot.csv` (~350 MB)

**Alternative direct download:**
```powershell
# Using Kaggle API (if installed)
kaggle datasets download -d spsayakpaul/arxiv-paper-abstracts
unzip arxiv-paper-abstracts.zip
```

### Step 2: Place Dataset in Project Root

```
C:\Users\Emran\Research Papers...\
├── arxiv-metadata-oai-snapshot.csv  ← Place here
├── app.py
├── check_setup.py
├── requirements.txt
└── models/
```

### Step 3: Verify Dataset

```powershell
# Check file exists
Test-Path arxiv-metadata-oai-snapshot.csv

# Check file size (should be ~350 MB)
(Get-Item arxiv-metadata-oai-snapshot.csv).Length / 1MB
```

**Expected:** `True` and `~350 MB`

---

## ✔️ Environment Verification

### Run Comprehensive Check

```powershell
python check_setup.py
```

**This performs 10 checks:**

1. ✓ Python version compatibility
2. ✓ pip availability
3. ✓ All dependencies installed
4. ✓ GPU configuration
5. ✓ Dataset file exists
6. ✓ Model files (will fail before training - OK!)
7. ✓ Disk space (6+ GB free)
8. ✓ System memory (8+ GB)
9. ✓ TextVectorization serialization test
10. ✓ Windows Long Path enabled

**Example output:**
```
==============================================================================
                    VERIFICATION SUMMARY                    
==============================================================================

Results:
Total checks: 10
Passed: 9
Failed: 1

Check Details:
Python Version....................................... ✓ PASS
pip Availability..................................... ✓ PASS
Required Packages.................................... ✓ PASS
GPU Configuration.................................... ✓ PASS
Dataset File......................................... ✓ PASS
Model Files.......................................... ✗ FAIL  (OK - will be generated)
Disk Space........................................... ✓ PASS
System Memory........................................ ✓ PASS
TextVectorization Test............................... ✓ PASS
Windows Long Path.................................... ✓ PASS

✓ System ready for training!
```

**Note:** "Model Files FAIL" is expected before training. Models will be generated during training.

### GPU-Specific Verification

```powershell
python gpu_config.py
```

**Expected output (if GPU available):**
```
✅ GPU detected: NVIDIA GeForce GTX 1650
✅ GPU memory growth enabled
✅ TensorFlow GPU configuration successful
✅ PyTorch CUDA available

GPU Details:
  Name: NVIDIA GeForce GTX 1650
  VRAM: 2048 MB
  CUDA Version: 11.8
  cuDNN Version: 8.6
  Compute Capability: 7.5
```

---

## 🚀 Training the Models

### Step 1: Launch Jupyter Notebook

```powershell
jupyter notebook
```

**This will:**
- Start Jupyter server
- Open browser automatically
- Show file browser

### Step 2: Open Training Notebook

Click on:
```
Reseacrh Paper Recommendation and Subject Area Prediction Using Sentence-BERT and Multi-Label MLP Classification.ipynb
```

### Step 3: Run All Cells

**Option A: Run all at once**
- Kernel → Restart & Run All

**Option B: Run cell by cell**
- Click first cell → Shift+Enter
- Repeat for each cell (66 cells total)

### Training Progress (GPU - 2GB VRAM)

| Phase | Time | What's Happening |
|-------|------|------------------|
| Data loading | 30 sec | Reading CSV, parsing |
| Data cleaning | 2 min | 51,774 → 42,306 papers |
| Splitting | 10 sec | Train/val/test (81/9.5/9.5) |
| Vectorization | 1 min | Building 47,823-word vocabulary |
| **MLP training** | **12 min** | **17 epochs, early stopping** |
| Evaluation | 30 sec | Test set predictions (99% acc) |
| **SBERT embedding** | **15.4 min** | **42,306 titles → 384-dim** |
| Saving | 30 sec | Writing 247 MB model files |
| **Total** | **~32 min** | **Complete pipeline** |

**CPU Training:** ~2 hours (4× slower)

### Expected Outputs

**After training, you'll have in `models/` folder:**

```
models/
├── model.h5 (94 MB)                     # MLP classifier
├── text_vectorizer_config.pkl (2 KB)   # Vectorizer config
├── text_vectorizer_weights.pkl (5 MB)  # IDF weights
├── vocab.pkl (1 MB)                     # 47,823 words
├── embeddings.pkl (62 MB)               # 42,306 embeddings
├── sentences.pkl (3 MB)                 # Paper titles
└── rec_model.pkl (80 MB)                # Sentence-BERT
Total: ~247 MB
```

### Training Tips

**For 2GB GPU:**
- Batch size already optimized (128)
- Memory growth enabled automatically
- Mixed precision available if needed
- Monitor GPU: `nvidia-smi -l 1`

**If OOM errors:**
1. Reduce batch size: `batch_size=64` or `32`
2. Enable mixed precision (in notebook)
3. Use CPU: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

---

## 🌐 Running the Application

### Option 1: Flask Web App

```powershell
python app.py
```

**Open:** http://localhost:5000

**Features:**
- Upload abstract → Get categories
- Enter title → Get recommendations
- Batch processing
- API endpoints

### Option 2: Streamlit App

```powershell
streamlit run app.py
```

**Open:** http://localhost:8501

**Features:**
- Interactive UI
- Real-time predictions
- Visualization
- File upload

### Option 3: Jupyter Notebook (Interactive)

```powershell
jupyter notebook
```

Open notebook and use prediction cells

---

## 🔧 Troubleshooting Guide

### Issue 1: TensorFlow Won't Install

**Error:**
```
ERROR: Could not install packages due to an OSError
```

**Solution:**
1. Enable Windows Long Path (see above)
2. **Restart computer** (critical!)
3. Retry: `pip install tensorflow==2.15.0`

### Issue 2: GPU Not Detected

**Diagnosis:**
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**If empty list:**
1. Check drivers: `nvidia-smi`
2. Check CUDA: `nvcc --version`
3. Check cuDNN: Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn64_8.dll"
4. Reinstall TensorFlow GPU: `pip install tensorflow[and-cuda]==2.15.0`

### Issue 3: Out of Memory During Training

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

**A. Reduce Batch Size (in notebook):**
```python
history = model.fit(
    X_train_vec, y_train,
    batch_size=64,  # Change from 128
    ...
)
```

**B. Enable Mixed Precision:**
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

**C. Use CPU:**
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Issue 4: Slow Embedding Generation

**Issue:** Sentence-BERT encoding takes >30 minutes

**Solutions:**

**A. Use GPU:**
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

**B. Increase Batch Size (if memory allows):**
```python
embeddings = model.encode(sentences, batch_size=64)  # From 32
```

**C. Cache Results:**
```python
if os.path.exists('models/embeddings.pkl'):
    embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
else:
    embeddings = model.encode(sentences, ...)
    pickle.dump(embeddings, open('models/embeddings.pkl', 'wb'))
```

### Issue 5: Model Files Not Found

**Error:**
```
FileNotFoundError: models/model.h5 not found
```

**Solution:**
Run the entire Jupyter notebook first to generate all model files.

### Issue 6: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```powershell
pip install tensorflow==2.15.0
```

**For all packages:**
```powershell
pip install -r requirements.txt
```

---

## ⚡ Performance Optimization

### GPU Memory Optimization (2GB VRAM)

**1. Enable Memory Growth (gpu_config.py):**
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**2. Limit GPU Memory:**
```python
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]  # 1.5GB
)
```

**3. Mixed Precision (50% memory savings):**
```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### Training Speed Optimization

**1. Use GPU (4× faster):**
- Install CUDA 11.8 + cuDNN 8.6
- Use TensorFlow GPU: `tensorflow[and-cuda]`

**2. Increase Batch Size (if memory allows):**
```python
batch_size=256  # From 128 (requires 4+ GB VRAM)
```

**3. Use SSD for Dataset:**
- Move CSV to SSD (faster I/O)

### Inference Speed Optimization

**1. Pre-compute Embeddings:**
- Generate embeddings once
- Save to `embeddings.pkl`
- Load cached embeddings (instant)

**2. Batch Predictions:**
```python
# Predict multiple papers at once
predictions = model.predict(X_batch)  # Faster than loop
```

**3. Use GPU for Inference:**
```python
with tf.device('/GPU:0'):
    predictions = model.predict(X)
```

---

## 📝 Next Steps

After successful setup:

1. ✅ **Verify everything works:** `python check_setup.py`
2. ✅ **Train models:** Run Jupyter notebook (~30 min)
3. ✅ **Test predictions:** Try example abstracts in notebook
4. ✅ **Test recommendations:** Try example titles
5. ✅ **Run web app:** `python app.py` or `streamlit run app.py`
6. ✅ **Experiment:** Try your own papers!

### Additional Resources

- **Quick Start Guide:** `QUICK_START_GUIDE.md` - Fast track to running
- **Project Info:** `PROJECT_INFO.md` - Technical specifications
- **Windows Long Path Fix:** `WINDOWS_LONG_PATH_FIX.md` - Detailed fix guide
- **LaTeX Reports:** `Project_Report_Part*.txt` - Academic documentation

---

## 🆘 Getting Help

**If you encounter issues:**

1. **Check documentation:**
   - QUICK_START_GUIDE.md (troubleshooting section)
   - PROJECT_INFO.md (technical details)
   - This file (SETUP_GUIDE.md)

2. **Run diagnostics:**
   ```powershell
   python check_setup.py
   ```

3. **Check specific components:**
   ```powershell
   python gpu_config.py  # GPU issues
   python -c "import tensorflow as tf; print(tf.__version__)"  # TensorFlow
   python -c "import torch; print(torch.cuda.is_available())"  # PyTorch CUDA
   ```

4. **Common solutions:**
   - Restart computer after Long Path fix
   - Restart PowerShell after PATH changes
   - Reinstall package: `pip uninstall package; pip install package==version`
   - Clear pip cache: `pip cache purge`

---

**✨ Setup complete! You're ready to train the models!**

**Last Updated:** October 29, 2025 
**Guide Version:** 2.0 (Complete Setup Instructions)  
**Status:** Production Ready ✅
