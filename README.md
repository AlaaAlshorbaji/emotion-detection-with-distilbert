# 😃 Emotion Detection using DistilBERT

This project implements an emotion detection system using the pretrained **DistilBERT** model from Hugging Face. The goal is to classify textual inputs into emotion categories such as joy, anger, sadness, etc., through fine-tuning on a labeled dataset.

---

## 🎯 Objective

The objective of this project is to build a deep learning pipeline that can:
- Load and explore an emotion-labeled text dataset.
- Encode textual data using a Transformer tokenizer.
- Train a DistilBERT model to predict emotional labels.
- Evaluate performance using classification accuracy.
- Generate predictions on unseen data.

---

## 🛠️ Tools and Libraries Used

The project uses the following tools and libraries:
- **Transformers** from Hugging Face – for model loading and tokenization.
- **Datasets** from Hugging Face – to handle and format the data.
- **Evaluate** – for model performance metrics (accuracy).
- **PyTorch** – as the backend deep learning framework.
- **Pandas** and **NumPy** – for data manipulation.
- **Matplotlib** – for visualizing label distribution.
- **Google Colab / OS** – for notebook and path operations.

---

## 📂 Dataset Overview

The dataset consists of two columns:
- `text`: contains the sentence or phrase to be classified.
- `label`: contains the emotion category of the text (e.g., "joy", "anger").

Each unique label was converted into an integer ID using a label-to-index dictionary (`label2id`). The model was trained to classify these IDs.

---

## 🔍 Project Workflow

### 1. Data Loading and Exploration
- The CSV file was loaded into a DataFrame.
- A bar plot was created to visualize the distribution of emotion labels to check class balance.

### 2. Dataset Conversion
- The Pandas DataFrame was converted into a Hugging Face `Dataset` object to be compatible with the `Trainer` API.

### 3. Label Encoding
- Unique emotion categories were mapped to numerical IDs.
- Two dictionaries were created: `label2id` and `id2label`.

### 4. Tokenization
- Text data was tokenized using the `DistilBERT` tokenizer.
- Padding and truncation were applied to ensure uniform input lengths.

### 5. Model Setup
- The model used is `DistilBERT for Sequence Classification`.
- The number of output labels was dynamically set based on the dataset.
- Label mappings were passed to the model to retain interpretability.

### 6. Training Configuration
- The training was performed using Hugging Face's `Trainer` API.
- The model was trained for **4 epochs**.
- Batch size was set to **16** for both training and evaluation.
- Learning rate was set to **2e-5**.
- Evaluation was done at the end of each epoch.

### 7. Evaluation
- The primary metric used for evaluation was **accuracy**, computed using the `evaluate` library.
- Model performance was assessed after each epoch to track learning progress.

### 8. Prediction and Decoding
- After training, predictions were made on the test set.
- The predicted label IDs were decoded back into emotion strings using the `id2label` dictionary.

---

## 📈 Results Summary

- The model learned to classify emotion texts with good accuracy.
- Final outputs were displayed with both the predicted and actual emotions for comparison.

---

## 🚀 How to Run This Project

To run this notebook:
1. Open it in Google Colab or Jupyter Notebook.
2. Install the necessary libraries using pip:
   - `transformers`
   - `datasets`
   - `evaluate`
3. Upload or load your dataset and execute the notebook sequentially.

---

## 📌 Possible Improvements

- Add precision, recall, and F1-score as additional evaluation metrics.
- Train on larger datasets with more emotion classes.
- Use cross-validation or early stopping for better generalization.
- Deploy the model using Gradio or Streamlit for a live demo interface.

---

## 👨‍💻 Author

**Alaa Shorbaji**  
Artificial Intelligence Instructor   
Deep Learning & NLP Researcher  


---

## 📜 License

This project is provided for academic and educational purposes. Please provide attribution if re-used.
