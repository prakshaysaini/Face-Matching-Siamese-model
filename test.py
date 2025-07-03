import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

IMAGE_SIZE = (100, 100)
THRESHOLD = 64.33  # from validation

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = img.reshape((100, 100, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_embedding(model, img_path):
    img = preprocess_image(img_path)
    return model.predict(img, verbose=0)[0]

def build_img_to_id_map(data_path):
    img_to_id = {}
    for person in os.listdir(data_path):
        person_path = os.path.join(data_path, person)
        if not os.path.isdir(person_path): continue

        # Regular images
        for f in os.listdir(person_path):
            img_path = os.path.join(person_path, f)
            if f.lower().endswith(('.jpg', '.png')) and f != "distortion":
                img_to_id[img_path] = person

        # Distortion folder
        distortion_path = os.path.join(person_path, "distortion")
        if os.path.exists(distortion_path):
            for f in os.listdir(distortion_path):
                img_path = os.path.join(distortion_path, f)
                if f.lower().endswith(('.jpg', '.png')):
                    img_to_id[img_path] = person
    return img_to_id

def generate_all_pairs(img_to_id):
    id_to_imgs = {}
    for img_path, identity in img_to_id.items():
        if identity not in id_to_imgs:
            id_to_imgs[identity] = []
        id_to_imgs[identity].append(img_path)

    identities = list(id_to_imgs.keys())
    matching_pairs = []
    different_pairs = []

    # Matching pairs: all combinations within same identity
    for identity in id_to_imgs:
        imgs = id_to_imgs[identity]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                matching_pairs.append((imgs[i], imgs[j], 1))

    # Different pairs: one pair per unique identity combination
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id1, id2 = identities[i], identities[j]
            img1 = id_to_imgs[id1][0]
            img2 = id_to_imgs[id2][0]
            different_pairs.append((img1, img2, 0))

    return matching_pairs + different_pairs

def evaluate(model, pairs):
    y_true = []
    y_pred = []

    for img1, img2, label in tqdm(pairs, desc="Evaluating"):
        emb1 = get_embedding(model, img1)
        emb2 = get_embedding(model, img2)
        dist = np.linalg.norm(emb1 - emb2)
        pred = 1 if dist < THRESHOLD else 0
        y_true.append(label)
        y_pred.append(pred)
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n===== Evaluation Metrics =====")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall   : {rec:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")

def main():
    test_data_path=r"E:\comsys_2025\Comys_Hackathon5\Task_B\train"
    model_path = r"taskb_siamese_embedding.h5"
    print("📦 Loading model...")
    model = load_model(model_path)

    print("📂 Loading test images...")
    img_to_id = build_img_to_id_map(test_data_path)

    print("🔁 Generating pairs...")
    pairs = generate_all_pairs(img_to_id)

    evaluate(model, pairs)

if __name__ == "__main__":
    main()
