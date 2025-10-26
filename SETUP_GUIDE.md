# Setup Guide for Research Papers Recommendation System

## Prerequisites
- Python 3.9 or 3.10 (TensorFlow 2.15.0 supports these versions)
- 2GB GPU (NVIDIA with CUDA support)
- CUDA 11.8 and cuDNN 8.6 (for TensorFlow 2.15.0 GPU support)

## Setup Steps

### 1. Install CUDA and cuDNN (for GPU support)
- Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Download cuDNN 8.6: https://developer.nvidia.com/cudnn
- Install both and add to PATH

### 2. Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Train the Model (Run Notebook)
You MUST run the Jupyter notebook first to generate the missing model files:
- `embeddings.pkl`
- `rec_model.pkl`
- `model.h5`

Open the notebook and run all cells to train and save the models.

### 5. Run the Streamlit App
```powershell
streamlit run app.py
```

## GPU Memory Optimization (for 2GB GPU)

Your 2GB GPU is limited, so you need to:

1. **Enable GPU memory growth** (see gpu_config.py)
2. **Reduce batch size** in training
3. **Use mixed precision training**
4. **Monitor GPU usage**

## Troubleshooting

### GPU Not Detected
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

### Out of Memory Errors
- Reduce batch size in notebook
- Enable memory growth (gpu_config.py)
- Use CPU if GPU fails

### Missing Model Files
- Run the entire notebook first to generate all pickle and .h5 files
- Check models/ folder for all required files
