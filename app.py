import os, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE  = (224, 224)

CLASS_LABELS_STAGE1    = ['allergy', 'infection']
CLASS_LABELS_ALLERGY   = ['Atopic Dermatitis (AD)', 'Contact Dermatitis (CD)', 'Eczema (EC)', 'Seborrheic Dermatitis (SD)']
CLASS_LABELS_INFECTION = ['Scabies (SC)', 'Tinea Corporis (TC)']

DISPLAY_STAGE1 = {'allergy': 'Skin Allergy', 'infection': 'Skin Infection'}
DISPLAY_STAGE2 = {
    'Atopic Dermatitis (AD)':     'Atopic Dermatitis',
    'Contact Dermatitis (CD)':    'Contact Dermatitis',
    'Eczema (EC)':                'Eczema',
    'Seborrheic Dermatitis (SD)': 'Seborrheic Dermatitis',
    'Scabies (SC)':               'Scabies',
    'Tinea Corporis (TC)':        'Tinea Corporis',
}

print(f"TensorFlow: {tf.__version__}")

def build_model(num_classes):
    """Rebuild the exact same architecture used in training."""
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    base.trainable = True
    inputs = keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def load_from_weights(npz_path, num_classes, name):
    print(f"  Loading {name}...")
    model = build_model(num_classes)
    data  = np.load(npz_path, allow_pickle=True)
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    model.set_weights(weights)
    print(f"  OK: {name}  input={model.input_shape}  output={model.output_shape}")
    return model

load_error = None
stage1_model = allergy_model = infection_model = None

print("\nLoading models from weights...")
try:
    stage1_model    = load_from_weights(os.path.join(MODEL_DIR, 'stage1_weights.npz'),    1, 'stage1')
    allergy_model   = load_from_weights(os.path.join(MODEL_DIR, 'allergy_weights.npz'),   4, 'allergy')
    infection_model = load_from_weights(os.path.join(MODEL_DIR, 'infection_weights.npz'), 2, 'infection')
    print("\nAll models loaded successfully.")
except Exception as e:
    load_error = str(e)
    print(f"\nERROR: {load_error}")


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img.resize(IMG_SIZE), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_class(model, tensor, labels):
    preds = model.predict(tensor, verbose=0)
    if preds.shape[-1] == 1:
        conf = float(preds[0][0])
        idx  = 1 if conf > 0.5 else 0
        conf = conf if idx == 1 else 1 - conf
    else:
        idx  = int(np.argmax(preds[0]))
        conf = float(preds[0][idx])
    return labels[idx], round(conf * 100, 2)


@app.route("/")
def index():
    return send_from_directory(MODEL_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if stage1_model is None:
        return jsonify({"error": load_error}), 500
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    try:
        tensor = preprocess(request.files["image"].read())
        s1_raw, s1_conf = predict_class(stage1_model, tensor, CLASS_LABELS_STAGE1)
        if s1_raw == 'allergy':
            s2_raw, s2_conf = predict_class(allergy_model,   tensor, CLASS_LABELS_ALLERGY)
        else:
            s2_raw, s2_conf = predict_class(infection_model, tensor, CLASS_LABELS_INFECTION)
        return jsonify({
            "stage1": {"label": DISPLAY_STAGE1.get(s1_raw, s1_raw), "confidence": s1_conf},
            "stage2": {"label": DISPLAY_STAGE2.get(s2_raw, s2_raw), "confidence": s2_conf},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    return jsonify({
        "models_loaded": stage1_model is not None,
        "load_error": load_error,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
