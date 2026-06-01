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

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)
from reportlab.lib.pagesizes import letter

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
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
        review_status TEXT DEFAULT 'Pending',
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES users(id)
    )''')
    # ADD GRADCAM COLUMN IF MISSING

    try:

        c.execute(
            "ALTER TABLE predictions ADD COLUMN gradcam_path TEXT"
        )

    except:

        pass
    c.execute('''CREATE TABLE IF NOT EXISTS doctor_notes (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id     INTEGER NOT NULL,
        prediction_id INTEGER NOT NULL,
        note          TEXT,
        reviewed_at   TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doctor_id)     REFERENCES users(id),
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )''')

    try:
        c.execute("ALTER TABLE doctor_notes ADD COLUMN medication TEXT")
        c.execute("ALTER TABLE doctor_notes ADD COLUMN dosage TEXT")
        c.execute("ALTER TABLE doctor_notes ADD COLUMN duration TEXT")
    except:
        pass

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

    base_model = model.layers[1]

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
        axis=(0,1,2)
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




    base_model = model.layers[1]

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
        axis=(0,1,2)
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

    d = request.get_json()

    email = d.get('email', '').strip().lower()

    password = d.get('password', '')

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    user = conn.execute(
        'SELECT * FROM users WHERE email=?',
        (email,)
    ).fetchone()

    conn.close()

    if not user or not bcrypt.check_password_hash(
        user['password'],
        password
    ):

        return jsonify({
            'error': 'Invalid email or password.'
        }), 401

    session['user_id'] = user['id']

    session['user_name'] = user['name']

    session['user_role'] = user['role']

    # ROLE BASED REDIRECT

    if user['role'] == 'doctor':

        redirect_url = '/doctor'

    else:

        redirect_url = '/dashboard'

    return jsonify({

        'message': 'Login successful.',

        'redirect': redirect_url,

        'user': {

            'id': user['id'],

            'name': user['name'],

            'role': user['role'],

            'email': user['email']
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
        return jsonify({
            'error': 'Please login first.'
        }), 401

    if stage1_model is None:
        return jsonify({
            'error': load_error
        }), 500

    if 'image' not in request.files:
        return jsonify({
            'error': 'No image uploaded.'
        }), 400

    try:

        img_bytes = request.files['image'].read()

        tensor = preprocess(img_bytes)

        # STAGE 1

        s1_raw, s1_conf, s1_all = predict_class(
            stage1_model,
            tensor,
            CLASS_LABELS_STAGE1
        )

        # STAGE 2

        if s1_raw == 'allergy':

            s2_raw, s2_conf, s2_all = predict_class(
                allergy_model,
                tensor,
                CLASS_LABELS_ALLERGY
            )

            s2_labels = CLASS_LABELS_ALLERGY

        else:

            s2_raw, s2_conf, s2_all = predict_class(
                infection_model,
                tensor,
                CLASS_LABELS_INFECTION
            )

            s2_labels = CLASS_LABELS_INFECTION

        # ─────────────────────────────
        # GENERATE GRAD-CAM
        # ─────────────────────────────

        heatmap = generate_gradcam(
            stage1_model,
            tensor
        )

        original_img = Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB")

        original_img = original_img.resize((224,224))

        original_np = np.array(original_img)

        heatmap = cv2.resize(
            heatmap,
            (224,224)
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cm.jet(heatmap)[:,:,:3]

        heatmap = np.uint8(heatmap * 255)

        superimposed_img = cv2.addWeighted(
            original_np,
            0.6,
            heatmap,
            0.4,
            0
        )

        gradcam_name = (
            f"gradcam_"
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        )

        gradcam_path = os.path.join(
            GRADCAM_DIR,
            gradcam_name
        )

        Image.fromarray(
            superimposed_img
        ).save(gradcam_path)

        # SAVE ORIGINAL IMAGE

        fname = (
            f"{session['user_id']}_"
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        )

        fpath = os.path.join(
            IMG_DIR,
            fname
        )

        Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB").save(fpath)

        # SAVE DATABASE

        conn = sqlite3.connect(DB_PATH)

        cur = conn.execute(
            '''
            INSERT INTO predictions
            (
                patient_id,
                image_path,
                gradcam_path,
                stage1_label,
                stage1_conf,
                stage2_label,
                stage2_conf
            )
            VALUES (?,?,?,?,?,?,?)
            ''',
            (
                session['user_id'],
                fname,
                gradcam_name,
                s1_raw,
                s1_conf,
                s2_raw,
                s2_conf
            )
        )

        pred_id = cur.lastrowid

        conn.commit()

        conn.close()

        return jsonify({

            'prediction_id': pred_id,

            'stage1': {

                'label': DISPLAY_STAGE1.get(
                    s1_raw,
                    s1_raw
                ),

                'raw': s1_raw,

                'confidence': s1_conf,

                'all_conf': s1_all,

                'all_labels': [
                    DISPLAY_STAGE1.get(l,l)
                    for l in CLASS_LABELS_STAGE1
                ]
            },

            'stage2': {

                'label': DISPLAY_STAGE2.get(
                    s2_raw,
                    s2_raw
                ),

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
@app.route('/api/report', methods=['POST'])
def generate_report():

    if 'user_id' not in session:

        return jsonify({
            'error': 'Please login first.'
        }), 401

    try:

        data = request.get_json()

        patient = data.get('patient', {})

        stage1 = data.get('stage1', {})

        stage2 = data.get('stage2', {})

        original_image = data.get('original_image')

        gradcam_image = data.get('gradcam_image')

        doctor_note = data.get('doctor_note', '')
        
        medication = data.get('medication', '')
        dosage = data.get('dosage', '')
        duration = data.get('duration', '')

        review_status = data.get(
            'review_status',
            'Pending'
        )

        temp_pdf = tempfile.NamedTemporaryFile(
            delete=False,
            suffix='.pdf'
        )

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
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#0F766E'),
            spaceAfter=30
        )
        story.append(Paragraph("<b>DermScan Clinical Report</b>", title_style))

        # PATIENT DETAILS
        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#334155'),
            spaceAfter=10,
            spaceBefore=15
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#475569'),
            leading=16
        )

        story.append(Paragraph("<b>Patient Information</b>", section_heading))
        patient_data = [
            [Paragraph("<b>Name:</b>", body_style), Paragraph(patient.get("name", "N/A"), body_style),
             Paragraph("<b>Age:</b>", body_style), Paragraph(str(patient.get("age", "N/A")), body_style)],
            [Paragraph("<b>Gender:</b>", body_style), Paragraph(patient.get("gender", "N/A"), body_style),
             Paragraph("<b>Affected Area:</b>", body_style), Paragraph(patient.get("area", "N/A"), body_style)]
        ]
        patient_table = Table(patient_data, colWidths=[60, 150, 60, 150])
        patient_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))

        # IMAGES (Side by Side)
        story.append(Paragraph("<b>Clinical Images</b>", section_heading))
        img_data = [[None, None], [Paragraph("<b>Original Image</b>", body_style), Paragraph("<b>AI Attention Heatmap</b>", body_style)]]
        
        orig_img_obj = None
        grad_img_obj = None

        if original_image:
            try:
                original_bytes = base64.b64decode(original_image.split(',')[1])
                original_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                original_temp.write(original_bytes)
                original_temp.close()
                orig_img_obj = RLImage(original_temp.name, width=200, height=200)
            except:
                pass
                
        if gradcam_image:
            try:
                gradcam_bytes = base64.b64decode(gradcam_image.split(',')[1])
                gradcam_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                gradcam_temp.write(gradcam_bytes)
                gradcam_temp.close()
                grad_img_obj = RLImage(gradcam_temp.name, width=200, height=200)
            except:
                pass

        if orig_img_obj or grad_img_obj:
            img_data[0][0] = orig_img_obj if orig_img_obj else ""
            img_data[0][1] = grad_img_obj if grad_img_obj else ""
            img_table = Table(img_data, colWidths=[250, 250])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,1), (-1,1), 10),
            ]))
            story.append(img_table)
        
        story.append(Spacer(1, 20))

        # AI RESULTS
        story.append(Paragraph("<b>AI Prediction Results</b>", section_heading))
        ai_data = [
            [Paragraph("<b>Stage 1 Category:</b>", body_style), Paragraph(str(stage1.get("raw", "N/A")), body_style)],
            [Paragraph("<b>Stage 1 Confidence:</b>", body_style), Paragraph(f"{stage1.get('confidence', 'N/A')}%", body_style)],
            [Paragraph("<b>Detected Disease:</b>", body_style), Paragraph(str(stage2.get("raw", "N/A")), body_style)],
            [Paragraph("<b>Disease Confidence:</b>", body_style), Paragraph(f"{stage2.get('confidence', 'N/A')}%", body_style)]
        ]
        ai_table = Table(ai_data, colWidths=[150, 250])
        ai_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(ai_table)
        story.append(Spacer(1, 20))

        # DOCTOR REVIEW
        story.append(Paragraph("<b>Clinical Review</b>", section_heading))
        review_text = doctor_note if doctor_note else "Awaiting doctor review."
        rev_data = [
            [Paragraph("<b>Review Status:</b>", body_style), Paragraph(str(review_status), body_style)],
            [Paragraph("<b>Doctor Notes:</b>", body_style), Paragraph(review_text, body_style)]
        ]
        
        if medication or dosage or duration:
            rx_text = ""
            if medication: rx_text += f"<b>Medication:</b> {medication}<br/>"
            if dosage: rx_text += f"<b>Dosage:</b> {dosage}<br/>"
            if duration: rx_text += f"<b>Duration:</b> {duration}"
            rev_data.append([Paragraph("<b>E-Prescription:</b>", body_style), Paragraph(rx_text, body_style)])
        rev_table = Table(rev_data, colWidths=[120, 300])
        rev_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(rev_table)
        story.append(Spacer(1, 30))

        # DISCLAIMER & FOOTER
        disclaimer = """
        <font size=9 color="#64748B">
        <b>Disclaimer:</b><br/>
        This report is generated using an AI-assisted skin disease classification system.
        The generated predictions should not be treated as a final medical diagnosis.
        Please consult a qualified dermatologist for professional clinical evaluation.
        </font>
        """
        story.append(Paragraph(disclaimer, styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Paragraph("<font size=9 color='#94A3B8'><i>Generated by DermScan</i></font>", styles['Normal']))

        # BUILD PDF

        doc.build(story)

        # READ PDF

        with open(temp_pdf.name, 'rb') as f:

            pdf_data = f.read()

        encoded = base64.b64encode(
            pdf_data
        ).decode('utf-8')

        return jsonify({
            'pdf': encoded
        })

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({
            'error': str(e)
        }), 500
# ─────────────────────────────────────────
# HISTORY (patient sees own, doctor sees all)
# ─────────────────────────────────────────
@app.route('/api/add-note', methods=['POST'])
def add_note():

    if 'user_id' not in session:
        return jsonify({
            'error': 'Please login first.'
        }), 401

    if session['user_role'] != 'doctor':
        return jsonify({
            'error': 'Only doctors can add notes.'
        }), 403

    data = request.get_json()

    prediction_id = data.get('prediction_id')
    note = data.get('note', '').strip()
    status = data.get('status', 'Approved')
    medication = data.get('medication', '').strip()
    dosage = data.get('dosage', '').strip()
    duration = data.get('duration', '').strip()

    if not prediction_id or not note:
        return jsonify({
            'error': 'Missing fields.'
        }), 400

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        '''
        INSERT INTO doctor_notes
        (doctor_id, prediction_id, note, medication, dosage, duration)
        VALUES (?, ?, ?, ?, ?, ?)
        ''',
        (
            session['user_id'],
            prediction_id,
            note,
            medication,
            dosage,
            duration
        )
    )

    conn.execute(
        '''
        UPDATE predictions
        SET review_status=?
        WHERE id=?
        ''',
        (
            status,
            prediction_id
        )
    )

    conn.commit()
    conn.close()

    return jsonify({
        'message': 'Doctor review saved.'
    })


@app.route('/api/history', methods=['GET'])
def history():

    if 'user_id' not in session:

        return jsonify({
            'error': 'Not logged in.'
        }), 401

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    # PATIENT HISTORY

    if session['user_role'] == 'patient':

        rows = conn.execute('''

            SELECT
                p.*,
                dn.note as doctor_note,
                dn.medication,
                dn.dosage,
                dn.duration

            FROM predictions p

            LEFT JOIN doctor_notes dn
            ON dn.prediction_id = p.id

            WHERE p.patient_id=?

            ORDER BY p.created_at DESC

        ''',
        (session['user_id'],)
        ).fetchall()

    # DOCTOR HISTORY

    else:

        rows = conn.execute('''

            SELECT
                p.*,
                u.name as patient_name,
                dn.note as doctor_note,
                dn.medication,
                dn.dosage,
                dn.duration

            FROM predictions p

            JOIN users u
            ON u.id = p.patient_id

            LEFT JOIN doctor_notes dn
            ON dn.prediction_id = p.id

            ORDER BY p.created_at DESC

        ''').fetchall()

    conn.close()

    return jsonify([
        dict(r)
        for r in rows
    ])


# ─────────────────────────────────────────
# STATIC FILES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:

        return render_template('login.html')

    if session['user_role'] != 'patient':

        return render_template(
            'doctor_dashboard.html'
        )

    return render_template(
        'dashboard.html'
    )
@app.route('/doctor')
def doctor_dashboard():

    if 'user_id' not in session:

        return render_template('login.html')

    if session['user_role'] != 'doctor':

        return render_template(
            'dashboard.html'
        )

    return render_template(
        'doctor_dashboard.html'
    )

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
