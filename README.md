# Research Paper Recommendation and Subject Area Prediction Using Sentence-BERT and Multi-Label MLP Classification

## 🎯 Overview
This deep learning project implements two powerful functionalities:
1. **Research Paper Recommendation System** - Using Sentence-BERT embeddings and cosine similarity
2. **Subject Area Prediction** - Using Multi-Label MLP Classification for multi-category prediction

The system provides intelligent, tailored research paper recommendations based on semantic similarity and accurately predicts multiple subject areas for academic papers.

## 🚀 Key Features

### 📚 Research Paper Recommendation System
- **Model:** Sentence-BERT (`all-MiniLM-L6-v2`)
- **Technique:** Sentence embeddings with cosine similarity
- **Purpose:** Recommends similar research papers based on title semantics
- **Capabilities:** 
  - Generates 384-dimensional semantic embeddings
  - Fast similarity search across large paper databases
  - Top-K paper recommendations

### 🧠 Subject Area Prediction (Multi-Label Classification)
- **Model:** Multi-Layer Perceptron (MLP) with Dropout
- **Architecture:** 
  - Input Layer: TextVectorization
  - Hidden Layer 1: 512 neurons (ReLU + Dropout 0.5)
  - Hidden Layer 2: 256 neurons (ReLU + Dropout 0.5)
  - Output Layer: Sigmoid activation (multi-label)
- **Type:** Multi-label classification (papers can belong to multiple categories)
- **Loss Function:** Binary Crossentropy
- **Accuracy:** 99% on test dataset


## 💻 Technologies Used

- **Deep Learning Frameworks:**
  - TensorFlow 2.x / Keras
  - PyTorch (via Sentence Transformers)
  
- **NLP Libraries:**
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - Hugging Face Transformers
  
- **Data Processing:**
  - Pandas, NumPy
  - Scikit-learn
  
- **Web Framework:**
  - Flask (for deployment)

## 📊 Dataset

- **Source:** [ArXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/data)
- **Content:** Research paper titles, abstracts, and subject categories
- **Size:** Large-scale dataset with multiple categories per paper

## 🛠️ How to Use

### Prerequisites
```bash
pip install tensorflow sentence-transformers pandas numpy scikit-learn flask
```

### Training the Models

1. **Prepare Dataset:**
   - Download the ArXiv dataset
   - Place in project directory

2. **Train Models:**
   - Open the Jupyter notebook: `Research Paper recommendation and subject area prediction using sentence transformer.ipynb`
   - Run all cells sequentially to:
     - Train the Multi-Label MLP classifier
     - Generate Sentence-BERT embeddings
     - Save all models to `models/` folder

3. **Run the Flask App:**
```bash
python app.py
```
Then navigate to: `http://localhost:5000`

## 📈 Results

### Research Paper Recommendation
- ✅ Successfully recommends top-K similar papers
- ✅ Uses semantic understanding (not just keyword matching)
- ✅ Fast inference with pre-computed embeddings

### Subject Area Prediction
- ✅ **99% accuracy** on test dataset
- ✅ Multi-label prediction (multiple categories per paper)
- ✅ Handles rare categories effectively

## 📁 Project Structure

```
Research Papers Recommendation System/
├── models/                          # Trained models and embeddings
│   ├── model.h5                    # MLP classifier
│   ├── embeddings.pkl              # Sentence-BERT embeddings
│   ├── sentences.pkl               # Paper titles
│   ├── rec_model.pkl               # Sentence-BERT model
│   ├── vocab.pkl                   # Vocabulary
│   └── text_vectorizer_*.pkl       # Text vectorizer files
├── app.py                          # Flask web application
├── gpu_config.py                   # GPU configuration
├── check_setup.py                  # Setup verification script
├── requirements.txt                # Python dependencies
├── Research Paper recommendation...ipynb  # Training notebook
└── README.md                       # This file
```

## 🎓 Models & Techniques

### 1. Sentence-BERT (all-MiniLM-L6-v2)
- Pre-trained on 1+ billion sentence pairs
- Generates fixed-size embeddings (384 dimensions)
- Optimized for semantic similarity tasks
- Size: 80 MB (lightweight)

### 2. Multi-Label MLP Classifier
- Custom neural network architecture
- Dropout regularization to prevent overfitting
- Binary crossentropy loss for multi-label classification
- Early stopping for optimal training

<!-- ## 📝 License
This project is licensed under **MOHAMMAD EMRAN AHMED** -->

## 🙏 Acknowledgments
- Sentence-BERT paper: [Reimers & Gurevych (2019)](https://arxiv.org/abs/1908.10084)
- ArXiv dataset from Kaggle
- Hugging Face for pre-trained models
- TensorFlow/Keras team

## 👨‍💻 Author
**MOHAMMAD EMRAN AHMED, SHEIKH ASHRAFUL**

---

**⭐ If you find this project helpful, please consider giving it a star!**

