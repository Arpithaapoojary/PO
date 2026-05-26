import os, io, base64, sqlite3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from flask import render_template
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from datetime import datetime
import tensorflow as tf
import cv2
import matplotlib.cm as cm


from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import tempfile
import base64

app = Flask(__name__)
app.secret_key = 'dermscan_secret_2024'
CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(MODEL_DIR, 'dermscan.db')
IMG_DIR   = os.path.join(MODEL_DIR, 'uploads')
os.makedirs(IMG_DIR, exist_ok=True)
GRADCAM_DIR = os.path.join(IMG_DIR, 'gradcam')
os.makedirs(GRADCAM_DIR, exist_ok=True)
# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        email      TEXT UNIQUE NOT NULL,
        password   TEXT NOT NULL,
        role       TEXT NOT NULL,  -- 'patient' or 'doctor'
        age        INTEGER,
        gender     TEXT,
        phone      TEXT,
        specialization TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id   INTEGER NOT NULL,
        image_path   TEXT,
        stage1_label TEXT,
        stage1_conf  REAL,
        stage2_label TEXT,
        stage2_conf  REAL,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS doctor_notes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id     INTEGER NOT NULL,
        prediction_id INTEGER NOT NULL,
        note          TEXT,
        reviewed_at   TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doctor_id)     REFERENCES users(id),
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )''')

    conn.commit()
    conn.close()
    print("Database ready.")

init_db()

# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
IMG_SIZE = (224, 224)
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

def build_model(num_classes):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
    base.trainable = True
    inputs = keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x) if num_classes == 1 else layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

def load_from_weights(npz_path, num_classes, name):
    print(f"  Loading {name}...")
    model = build_model(num_classes)
    data  = np.load(npz_path, allow_pickle=True)
    model.set_weights([data[f'arr_{i}'] for i in range(len(data.files))])
    print(f"  OK: {name}")
    return model

load_error = None
stage1_model = allergy_model = infection_model = None
print("\nLoading models...")
try:
    stage1_model    = load_from_weights(os.path.join(MODEL_DIR, 'stage1_weights.npz'),    1, 'stage1')
    allergy_model   = load_from_weights(os.path.join(MODEL_DIR, 'allergy_weights.npz'),   4, 'allergy')
    infection_model = load_from_weights(os.path.join(MODEL_DIR, 'infection_weights.npz'), 2, 'infection')
    print("All models loaded.")
    stage1_model.summary()
except Exception as e:
    load_error = str(e)
    print(f"ERROR: {load_error}")





def generate_gradcam(model, img_array):

    # Get MobileNetV2 base model
    base_model = model.layers[1]

    # Create model for Grad-CAM
    grad_model = tf.keras.models.Model(
        [
            base_model.inputs
        ],
        [
            base_model.get_layer("Conv_1").output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap /= (
        np.max(heatmap) + 1e-8
    )

    return heatmap

    # Find MobileNetV2 model automatically
    base_model = None

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            base_model = layer

            break

    if base_model is None:
        raise Exception("Base model not found")

    # Find last Conv2D layer automatically
    last_conv_layer = None

    for layer in reversed(base_model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):

            last_conv_layer = layer

            break

    if last_conv_layer is None:
        raise Exception("No Conv2D layer found")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()

    base_model = model.get_layer("mobilenetv2_1.00_224")

    last_conv_layer = base_model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        pooled_grads * conv_outputs,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()
# ─────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    d = request.get_json()
    name     = d.get('name', '').strip()
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')
    role     = d.get('role', 'patient')
    age      = d.get('age')
    gender   = d.get('gender', '')
    phone    = d.get('phone', '')
    spec     = d.get('specialization', '')

    if not name or not email or not password:
        return jsonify({'error': 'Name, email and password are required.'}), 400
    if role not in ('patient', 'doctor'):
        return jsonify({'error': 'Invalid role.'}), 400

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO users (name,email,password,role,age,gender,phone,specialization) VALUES (?,?,?,?,?,?,?,?)',
            (name, email, hashed, role, age, gender, phone, spec)
        )
        conn.commit()
        conn.close()
        return jsonify({'message': 'Registered successfully. Please login.'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already registered.'}), 400


@app.route('/api/login', methods=['POST'])
def login():
    d        = request.get_json()
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    conn.close()

    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    session['user_id']   = user['id']
    session['user_name'] = user['name']
    session['user_role'] = user['role']

    return jsonify({
        'message': 'Login successful.',
        'user': {
            'id':   user['id'],
            'name': user['name'],
            'role': user['role'],
            'email': user['email'],
        }
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out.'})


@app.route('/api/me', methods=['GET'])
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401
    return jsonify({
        'id':   session['user_id'],
        'name': session['user_name'],
        'role': session['user_role'],
    })

# ─────────────────────────────────────────
# PREDICT (requires login)
# ─────────────────────────────────────────

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
    all_conf = [round(float(p)*100,1) for p in (preds[0] if preds.shape[-1]>1 else [1-float(preds[0][0]), float(preds[0][0])])]
    return labels[idx], round(conf*100, 2), all_conf


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first.'}), 401
    if stage1_model is None:
        return jsonify({'error': load_error}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    try:
        img_bytes = request.files['image'].read()
        tensor    = preprocess(img_bytes)

        s1_raw, s1_conf, s1_all = predict_class(stage1_model, tensor, CLASS_LABELS_STAGE1)
        if s1_raw == 'allergy':
            s2_raw, s2_conf, s2_all = predict_class(allergy_model,   tensor, CLASS_LABELS_ALLERGY)
            s2_labels = CLASS_LABELS_ALLERGY
        else:
            s2_raw, s2_conf, s2_all = predict_class(infection_model, tensor, CLASS_LABELS_INFECTION)
            s2_labels = CLASS_LABELS_INFECTION

        # Save image
                # ─────────────────────────────
        # GENERATE GRAD-CAM
        # ─────────────────────────────

        heatmap = generate_gradcam(
            stage1_model,
            tensor
        )

        # Convert original image
        original_img = Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB")

        original_img = original_img.resize((224, 224))

        original_np = np.array(original_img)

        # Resize heatmap
        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        # Convert heatmap colors
        heatmap = np.uint8(255 * heatmap)

        heatmap = cm.jet(heatmap)[:, :, :3]

        heatmap = np.uint8(heatmap * 255)

        # Overlay heatmap
        superimposed_img = cv2.addWeighted(
            original_np,
            0.6,
            heatmap,
            0.4,
            0
        )

        # Save Grad-CAM image
        gradcam_name = f"gradcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

        gradcam_path = os.path.join(
            GRADCAM_DIR,
            gradcam_name
        )

        Image.fromarray(
            superimposed_img
        ).save(gradcam_path)

        fname = f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        fpath = os.path.join(IMG_DIR, fname)
        Image.open(io.BytesIO(img_bytes)).convert('RGB').save(fpath)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.execute(
            'INSERT INTO predictions (patient_id,image_path,stage1_label,stage1_conf,stage2_label,stage2_conf) VALUES (?,?,?,?,?,?)',
            (session['user_id'], fname, s1_raw, s1_conf, s2_raw, s2_conf)
        )
        pred_id = cur.lastrowid
        conn.commit()
        conn.close()
        return jsonify({

            'prediction_id': pred_id,

    'stage1': {
        'label': DISPLAY_STAGE1.get(s1_raw, s1_raw),
        'raw': s1_raw,
        'confidence': s1_conf,
        'all_conf': s1_all,
        'all_labels': [
            DISPLAY_STAGE1.get(l,l)
            for l in CLASS_LABELS_STAGE1
        ]
    },

    'stage2': {
        'label': DISPLAY_STAGE2.get(s2_raw, s2_raw),
        'raw': s2_raw,
        'confidence': s2_conf,
        'all_conf': s2_all,
        'all_labels': [
            DISPLAY_STAGE2.get(l,l)
            for l in s2_labels
        ]
    },

    'gradcam_image':
        f'/uploads/gradcam/{gradcam_name}'

})
    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({
        'error': str(e)
        }), 500
@app.route('/api/report', methods=['POST'])
def generate_report():

    if 'user_id' not in session:
        return jsonify({'error': 'Please login first.'}), 401

    try:

        data = request.get_json()

        patient = data.get('patient', {})
        stage1 = data.get('stage1', {})
        stage2 = data.get('stage2', {})

        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')

        doc = SimpleDocTemplate(
            temp_pdf.name,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=30
        )

        styles = getSampleStyleSheet()

        story = []

        # TITLE
        story.append(
            Paragraph(
                "<font size=22><b>DermScan AI Medical Report</b></font>",
                styles['Title']
            )
        )

        story.append(Spacer(1, 20))

        # PATIENT DETAILS
        story.append(
            Paragraph(
                "<b>Patient Information</b>",
                styles['Heading2']
            )
        )

        patient_html = f'''
        <br/>
        <b>Name:</b> {patient.get("name", "N/A")}<br/>
        <b>Age:</b> {patient.get("age", "N/A")}<br/>
        <b>Gender:</b> {patient.get("gender", "N/A")}<br/>
        <b>Affected Area:</b> {patient.get("area", "N/A")}<br/>
        '''

        story.append(
            Paragraph(patient_html, styles['BodyText'])
        )

        story.append(Spacer(1, 18))

        # RESULTS
        story.append(
            Paragraph(
                "<b>AI Prediction Results</b>",
                styles['Heading2']
            )
        )

        result_html = f'''
        <br/>
        <b>Stage 1 Category:</b> {stage1.get("raw", "N/A")}<br/>
        <b>Stage 1 Confidence:</b> {stage1.get("confidence", "N/A")}%<br/><br/>

        <b>Detected Disease:</b> {stage2.get("raw", "N/A")}<br/>
        <b>Disease Confidence:</b> {stage2.get("confidence", "N/A")}%<br/>
        '''

        story.append(
            Paragraph(result_html, styles['BodyText'])
        )

        story.append(Spacer(1, 20))

        # DISCLAIMER
        disclaimer = """
        <font size=10>
        <b>Disclaimer:</b><br/>
        This report is generated using an AI-based skin disease detection system
        for educational and research purposes only.
        It should not be considered as a final medical diagnosis.
        Please consult a qualified dermatologist for professional evaluation.
        </font>
        """

        story.append(
            Paragraph(disclaimer, styles['BodyText'])
        )

        story.append(Spacer(1, 20))

        # FOOTER
        story.append(
            Paragraph(
                "<font size=10><i>Generated by DermScan AI</i></font>",
                styles['Italic']
            )
        )

        # BUILD PDF
        doc.build(story)

        # CONVERT TO BASE64
        with open(temp_pdf.name, 'rb') as f:
            pdf_data = f.read()

        encoded = base64.b64encode(pdf_data).decode('utf-8')

        return jsonify({
            'pdf': encoded
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        }), 500
# ─────────────────────────────────────────
# HISTORY (patient sees own, doctor sees all)
# ─────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in.'}), 401

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if session['user_role'] == 'patient':
        rows = conn.execute(
            'SELECT * FROM predictions WHERE patient_id=? ORDER BY created_at DESC',
            (session['user_id'],)
        ).fetchall()
    else:
        rows = conn.execute('''
            SELECT p.*, u.name as patient_name
            FROM predictions p
            JOIN users u ON u.id = p.patient_id
            ORDER BY p.created_at DESC
        ''').fetchall()

    conn.close()
    return jsonify([dict(r) for r in rows])


# ─────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login')
def login_page():
    return render_template('login.html')
@app.route('/uploads/gradcam/<path:filename>')
def gradcam_file(filename):
    return send_from_directory(
        GRADCAM_DIR,
        filename
    )
@app.route('/uploads/<path:filename>')

def uploaded_file(filename):
    return send_from_directory(IMG_DIR, filename)

@app.route('/debug')
def debug():
    return jsonify({'models_loaded': stage1_model is not None, 'load_error': load_error})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
