#  Face Matching (Multi-Class Recognition with Distorted Inputs)

This repository contains code and pretrained models for **Task B** of **ComSys Hackathon 5** — verifying identity from distorted face images using a **Siamese embedding model**.

## ▶️ Running the Test Script
To evaluate the model on a custom test dataset, use:

python test.py

⚙️ Instructions:

- Open the test.py file.

- Update the data path inside the script to point to your test folder.

- Make sure the folder structure matches the training/validation data format:

- The script will:

1.Load the pretrained embedding model

2.Generate all possible image pairs (both matching and non-matching)

3.Calculate the following evaluation metrics:
  
   ✅ Accuracy

   ✅ Precision

   ✅ Recall

   ✅ F1 Score
   
Evaluation results are printed to the console and optionally saved to .csv  file.

## 🧠 Model Architecture

This project uses a **Siamese Neural Network** to learn visual similarity between images. The architecture is composed of:

---

### 🔹 1. Embedding Model

A lightweight CNN that maps input images to 64-dimensional embeddings:

```python
def build_embedding_model(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64)  # Final embedding vector
    ])
    return model
```

### 🔹 2. Siamese Model

This model takes two inputs, passes them through the same embedding model, then compares them using L1 (absolute difference) followed by a sigmoid layer to predict similarity.

```python
def build_siamese_model(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    base_model = build_embedding_model(input_shape)

    emb_a = base_model(input_a)
    emb_b = base_model(input_b)

    L1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb_a, emb_b])
    output = Dense(1, activation="sigmoid")(L1_distance)

    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model, base_model
```





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

## 🚀 Run the Script

To evaluate the trained model on a custom dataset:

```bash
python test.py
```

> 📌 **Important:** Open `test.py` and change the data path to your custom test folder (same structure as training/validation).  
> The script will:
> - Generate both matching and non-matching image pairs  
> - Compute **Accuracy, Precision, Recall, F1-score**  
> - Display results in the terminal  

---

## 💾 Pretrained Model Weights

These are included in the repository:

- `taskb_siamese.h5` — Full Siamese network  
- `taskb_siamese_embedding.h5` — Embedding head only (used to compute image distances)  
- `layers_weights.txt` — All learned layer weights (extracted for inspection)

---

## 📦 Requirements

To install all required Python dependencies, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

- `tensorflow` → for model inference  
- `opencv-python` → for image loading and preprocessing  
- `numpy` → for numerical operations  
- `scikit-learn` → for evaluation metrics  
- `tqdm` → for progress bars

---

## 🧠 How the Threshold Was Selected

The notebook `taskb-get_threshold_with_random_pairs.ipynb` was used to:

- Generate random matching and non-matching image pairs  
- Compute distances using the pretrained embedding model  
- Evaluate thresholds from 50 to 100  
- Select the **optimal threshold** based on **F1 score**

✅ **Final threshold selected: 82**

---

## 🤝 Contributors

**Prakshay Saini**  
B.Tech CSE, IIIT Guwahati  
prakshay.saini23b@iiitg.ac.in

**Rishab Jain**  
B.Tech CSE, IIIT Guwahati  
rishab.jain23b@iiitg.ac.in
