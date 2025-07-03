#  Face Matching (Multi-Class Recognition with Distorted Inputs)

This repository contains code and pretrained models for **Task B** of **ComSys Hackathon 5** — verifying identity from distorted face images using a **Siamese embedding model**.

---

## 📁 Project Structure ``` 
- ├── distances_output.xlsx # Distances and results from threshold evaluation
- ├── layers_weights.txt # Extracted layer weights from model (text format)
- ├── requirements.txt # All the required dependencies 
- ├── taskb_siamese.h5 # Full Siamese model (architecture + weights)
- ├── taskb_siamese_embedding.h5 # Embedding model (used for evaluation)
- ├── taskb-get_threshold_with_random_pairs.ipynb # Threshold calculation notebook
- ├── task-b-trainer.ipynb # Model training notebook
- ├── test.py # Final test script for submission ✅ ``` 

---

## 🧠 Model Summary

- Architecture: **Siamese Neural Network** with CNN-based embedding
- Input: **Grayscale images**, resized to **100 × 100**
- Final feature vector: output from the embedding head
- Matching criterion: **Euclidean distance** < threshold

📌 **Threshold** used for evaluation: `82`

---

## 🎯 Evaluation Metrics (on train data)

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `0.8212`  |
| Precision  | `1.0000`  |
| Recall     | `0.8212`  |
| F1-Score   | `0.9018`  |

## 🎯 Evaluation Metrics (on validation)

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `0.7792`  |
| Precision  | `1.0000`  |
| Recall     | `0.7792`  |
| F1-Score   | `0.8759`  |

> _Automatically computed by `test.py` on sampled dataset for saving time

---

## 🧪 Running the Test Script

### 🧾 Folder Structure (expected input):

- test/
- ├── person1/
- │ ├── clean1.jpg
- │ ├── clean2.jpg
- │ └── distortion/
- │ └── distorted1.jpg
- ├── person2/
- │ ├── clean1.jpg
- │ └── distortion/
- │ └── distorted1.jpg
...

### 🚀 Run the Script:

```bash
python test.py "/data_path"
This will:

Generate both matching and non-matching image pairs

Compute: Accuracy, Precision, Recall, F1

Display results in the terminal

💾 Pretrained Model Weights
taskb_siamese.h5 – Full Siamese network

taskb_siamese_embedding.h5 – Embedding head only (used for computing distances)

layers_weights.txt – All learned layer weights (extracted for inspection)

📦 Requirements

To install all required Python dependencies, run:

    pip install -r requirements.txt

The requirements.txt includes:

- tensorflow          → for model inference
- opencv-python       → for image loading and preprocessing
- numpy               → for numerical operations
- scikit-learn        → for evaluation metrics (accuracy, precision, etc.)
- tqdm                → for progress bars

🧠 How Threshold Was Found
Using taskb-get_threshold_with_random_pairs.ipynb:

Random matching and non-matching pairs were generated

Distances were computed using the embedding model

Optimal threshold was selected using the best F1 score and also testing all thresholds between 50-100

Final threshold selected: 82

🤝 Contributors
Prakshay Saini

B.Tech CSE, IIIT Guwahati

prakshay.saini23b@iiitg.ac.in

Rishab Jain

B.Tech CSE, IIIT Guwahati

rishab.jain23b@iiitg.ac.in

