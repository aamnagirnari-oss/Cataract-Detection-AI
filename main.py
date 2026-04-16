import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PATHS
# -------------------------------
DATA_PATH = "ODIR-5K/data.xlsx"
IMAGE_DIR = "ODIR-5K/preprocessed_images/"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_excel(DATA_PATH)

df["Cataract"] = (
    df["Left-Diagnostic Keywords"].str.contains("cataract", case=False, na=False) |
    df["Right-Diagnostic Keywords"].str.contains("cataract", case=False, na=False)
).astype(int)

df["image_path"] = df["Left-Fundus"].apply(lambda x: os.path.join(IMAGE_DIR, x))

# Metadata (example)
df["Age"] = df["Patient Age"]
df["Gender"] = df["Patient Sex"].map({"Male": 1, "Female": 0})

# -------------------------------
# PREPROCESS IMAGES
# -------------------------------
IMG_SIZE = 224

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

images = np.array([load_image(p) for p in df["image_path"]])
labels = df["Cataract"].values

# -------------------------------
# PREPROCESS METADATA
# -------------------------------
meta = df[["Age", "Gender"]].fillna(0).values
scaler = StandardScaler()
meta = scaler.fit_transform(meta)

# -------------------------------
# MODEL (MULTI-MODAL)
# -------------------------------
def build_model():
    # Image branch
    image_input = Input(shape=(224,224,3))
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=image_input)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Metadata branch
    meta_input = Input(shape=(2,))
    m = Dense(16, activation="relu")(meta_input)

    # Fusion
    combined = Concatenate()([x, m])
    z = Dense(32, activation="relu")(combined)
    output = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[image_input, meta_input], outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# -------------------------------
# STRATIFIED K-FOLD TRAINING
# -------------------------------
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(images, labels)):
    print(f"\nTraining Fold {fold+1}")

    X_train, X_val = images[train_idx], images[val_idx]
    M_train, M_val = meta[train_idx], meta[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    model = build_model()

    model.fit(
        [X_train, M_train], y_train,
        validation_data=([X_val, M_val], y_val),
        epochs=5,
        batch_size=8
    )

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("cataract_multimodal_model.h5")

print("Training complete!")