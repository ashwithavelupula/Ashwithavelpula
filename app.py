#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline Multipurpose Chatbot Web Application
Features:
- Object Detection using MobileNet SSD (OpenCV DNN)
- Face Detection and Registration using Haar Cascades
- MySQL Database for user logs, history, and analytics
- Modular design for easy model swapping
- Offline-first capability with optional Firebase layer
"""

import os
import re
import json
import base64
import cv2
import logging
import secrets
import threading
import time
import uuid
import numpy as np
from collections import defaultdict
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, Response,
    send_from_directory, session as flask_session
)
import pymysql
import pymysql.cursors
from llm_engine import (
    engine as llm, safe_calc, convert_units,
    password_generate, text_analyze, summarize, doc_qa,
    get_llm_params, set_llm_params,
)

# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)

# Thread-safety locks (Flask threaded dev server + CPU model backends)
NET_LOCK = threading.Lock()
LLM_LOCK = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# --- local .env file (written by db_setup.py / setup_and_run.bat) ---
def _load_dotenv(path):
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except FileNotFoundError:
        pass


_load_dotenv(os.path.join(BASE_DIR, '.env'))

from firebase_sync import (
    push_qa as rtdb_push, push_voice_qa as rtdb_push_voice,
    push_text as rtdb_push_text,
    push_emotion as rtdb_push_emotion,
    rtdb_status, set_enabled as rtdb_set_enabled,
)

# MySQL Configuration
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DB', 'chatbot_db'),
    'charset': 'utf8mb4',
    'autocommit': True,
    'connect_timeout': 3,
    'read_timeout': 5,
}

# Object Detection Config
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', 0.3))
PHONE_CONFIDENCE_THRESHOLD = float(os.getenv('PHONE_CONFIDENCE_THRESHOLD', 0.08))
PHONE_NMS_THRESHOLD = float(os.getenv('PHONE_NMS_THRESHOLD', 0.45))

# Face Detection Config
FACE_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_CONFIDENCE_THRESHOLD', 0.5))

# Load class names (fall back to built-in 20-class list if file is missing)
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.txt')
try:
    with open(CLASSES_PATH, 'r') as f:
        CLASSES = [line.strip() for line in f.readlines() if line.strip()]
    if not CLASSES:
        raise ValueError('empty classes file')
except Exception as e:
    print(f"[WARNING] Could not load classes.txt ({e}); using built-in 20-class list")
    CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
               'sofa', 'train', 'tvmonitor']

# Load face cascade
FACE_CASCADE_PATH = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load MobileNet SSD model
PROTOTXT_PATH = os.path.join(MODEL_DIR, 'MobileNet-SSD_deploy.prototxt')
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'mobilenet_iter_73000.caffemodel')

# Try to load the model; if weights/prototxt are missing, run without it
try:
    if os.path.exists(PROTOTXT_PATH) and os.path.exists(WEIGHTS_PATH):
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)
        print("[INFO] MobileNet SSD model loaded successfully")
    else:
        raise FileNotFoundError(
            f"missing prototxt={os.path.exists(PROTOTXT_PATH)} "
            f"weights={os.path.exists(WEIGHTS_PATH)}")
except Exception as e:
    print(f"[WARNING] Object detection model unavailable, running without it: {e}")
    net = None

PHONE_MODEL_PATH = os.path.join(MODEL_DIR, 'object_detection_yolox_2022nov.onnx')
try:
    if not os.path.isfile(PHONE_MODEL_PATH):
        raise FileNotFoundError(PHONE_MODEL_PATH)
    phone_net = cv2.dnn.readNetFromONNX(PHONE_MODEL_PATH)
    phone_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    phone_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] COCO mobile phone detector loaded successfully")
except Exception as e:
    logging.warning("Mobile phone detection model unavailable: %r", e)
    phone_net = None

# ============================================================
# DATABASE HELPER CLASS
# ============================================================

class Database:
    """Handles all MySQL database operations (with graceful offline fallback)"""

    def __init__(self, config=MYSQL_CONFIG):
        self.config = config
        self.connection = None
        self.available = False
        self._lock = threading.Lock()
        self.connect()

    def connect(self):
        try:
            # First ensure the database exists (connect without db selected)
            cfg_no_db = {k: v for k, v in self.config.items() if k != 'database'}
            tmp = pymysql.connect(**cfg_no_db)
            try:
                with tmp.cursor() as cur:
                    cur.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.config['database']}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                    )
                tmp.commit()
            finally:
                tmp.close()
            self.connection = pymysql.connect(**self.config)
            self.available = True
            print("[INFO] Database connected successfully")
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            print("[INFO] Running in OFFLINE mode (no MySQL). Detection still works; history/analytics are kept in memory.")
            self.connection = None
            self.available = False

    def is_connected(self):
        if not self.connection:
            return False
        try:
            self.connection.ping(reconnect=False)
            return True
        except Exception:
            return False

    def execute(self, query, params=None):
        if not self.connection:
            return None
        with self._lock:
            try:
                with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
                    cursor.execute(query, params)
                    if cursor.description:
                        result = cursor.fetchall()
                        return result
                    self.connection.commit()
                    return cursor.rowcount
            except Exception as e:
                print(f"[ERROR] Query execution failed: {e}")
                try:
                    self.connection.ping(reconnect=True)
                except Exception:
                    pass
                return None
    
    def insert_one(self, query, params=None):
        return self.execute(query, params)
    
    def get_one(self, query, params=None):
        result = self.execute(query, params)
        return result[0] if result else None
    
    def fetch_all(self, query, params=None):
        result = self.execute(query, params)
        return result if result else []


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def init_db(db):
    """Initialize database tables"""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) NOT NULL UNIQUE,
            email VARCHAR(120) UNIQUE,
            password_hash VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP NULL,
            is_active BOOLEAN DEFAULT TRUE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            user_message TEXT,
            bot_response TEXT,
            detected_objects JSON,
            face_detected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS face_registry (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            face_encoding LONGTEXT,
            face_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            action_type VARCHAR(50),
            action_data JSON,
            ip_address VARCHAR(45),
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            level VARCHAR(20) DEFAULT 'INFO',
            message TEXT,
            module VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    ]
    
    for query in queries:
        db.execute(query)


def decode_upload(file, max_dim=1600):
    """Decode an uploaded image file; downscale so max side <= max_dim. None if invalid."""
    try:
        data = file.read()
    except Exception:
        return None
    if not data:
        return None
    try:
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    return frame


class _BytesFile:
    """Minimal file-like wrapper so raw bytes reuse decode_upload."""
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


def decode_upload_bytes(data, max_dim=1600):
    return decode_upload(_BytesFile(data), max_dim=max_dim)


def encode_jpg(frame, quality=90):
    """Encode a frame to base64 JPEG. Returns None on failure."""
    try:
        ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception:
        return None


def detect_objects(frame, net, classes, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Detect objects in a frame using MobileNet SSD
    Returns: list of dicts with label, confidence, box
    """
    if net is None:
        return []
    if frame is None or frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Object detection requires a non-empty color image")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    if not 0 <= NMS_THRESHOLD <= 1:
        raise ValueError("NMS threshold must be between 0 and 1")
    h, w = frame.shape[:2]

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(
        frame,
        1.0 / 127.5,
        (300, 300),
        127.5,
        swapRB=False,
        crop=False
    )

    with NET_LOCK:
        net.setInput(blob)
        detections = net.forward()

    if (detections is None or detections.ndim != 4 or detections.shape[0] < 1
            or detections.shape[1] != 1 or detections.shape[3] < 7):
        raise RuntimeError("Object detection model returned an invalid result")

    results = []
    for detection in detections[0, 0]:
        confidence = float(detection[2])
        class_value = float(detection[1])
        if (not np.isfinite(confidence) or confidence < confidence_threshold
                or not np.isfinite(class_value) or not class_value.is_integer()):
            continue
        class_id = int(class_value)
        if class_id < 0 or class_id >= len(classes) or classes[class_id].lower() == 'background':
            continue

        coordinates = np.asarray(detection[3:7], dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            continue
        start_x, start_y, end_x, end_y = coordinates * np.array([w, h, w, h])
        start_x = max(0, min(w - 1, round(start_x)))
        start_y = max(0, min(h - 1, round(start_y)))
        end_x = max(0, min(w - 1, round(end_x)))
        end_y = max(0, min(h - 1, round(end_y)))
        if end_x <= start_x or end_y <= start_y:
            continue

        results.append({
            'label': classes[class_id],
            'confidence': confidence,
            'box': {
                'startX': start_x,
                'startY': start_y,
                'endX': end_x,
                'endY': end_y
            }
        })

    kept = []
    for candidate in sorted(results, key=lambda item: item['confidence'], reverse=True):
        candidate_box = candidate['box']
        candidate_area = ((candidate_box['endX'] - candidate_box['startX'])
                          * (candidate_box['endY'] - candidate_box['startY']))
        overlaps_same_class = False
        for selected in kept:
            if selected['label'] != candidate['label']:
                continue
            selected_box = selected['box']
            intersection_width = max(
                0, min(candidate_box['endX'], selected_box['endX'])
                - max(candidate_box['startX'], selected_box['startX']))
            intersection_height = max(
                0, min(candidate_box['endY'], selected_box['endY'])
                - max(candidate_box['startY'], selected_box['startY']))
            intersection = intersection_width * intersection_height
            selected_area = ((selected_box['endX'] - selected_box['startX'])
                             * (selected_box['endY'] - selected_box['startY']))
            union = candidate_area + selected_area - intersection
            if union > 0 and intersection / union > NMS_THRESHOLD:
                overlaps_same_class = True
                break
        if not overlaps_same_class:
            kept.append(candidate)
    return kept


PHONE_INPUT_SIZE = 640
COCO_CELL_PHONE_CLASS_ID = 67
_phone_grids = []
_phone_expanded_strides = []
for _stride in (8, 16, 32):
    _grid_size = PHONE_INPUT_SIZE // _stride
    _x_grid, _y_grid = np.meshgrid(np.arange(_grid_size), np.arange(_grid_size))
    _grid = np.stack((_x_grid, _y_grid), axis=2).reshape(1, -1, 2)
    _phone_grids.append(_grid)
    _phone_expanded_strides.append(np.full((*_grid.shape[:2], 1), _stride))
PHONE_GRIDS = np.concatenate(_phone_grids, axis=1)
PHONE_EXPANDED_STRIDES = np.concatenate(_phone_expanded_strides, axis=1)


def detect_mobile_objects(frame, detector=None, confidence_threshold=PHONE_CONFIDENCE_THRESHOLD):
    """Detect COCO cell-phone objects with the local YOLOX model."""
    detector = detector if detector is not None else phone_net
    if detector is None:
        return []
    if frame is None or frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Mobile detection requires a non-empty color image")
    if not 0 < confidence_threshold <= 1:
        raise ValueError("Phone confidence threshold must be between 0 and 1")
    if not 0 < PHONE_NMS_THRESHOLD <= 1:
        raise ValueError("Phone NMS threshold must be between 0 and 1")

    height, width = frame.shape[:2]
    scale = min(PHONE_INPUT_SIZE / height, PHONE_INPUT_SIZE / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded = np.full((PHONE_INPUT_SIZE, PHONE_INPUT_SIZE, 3), 114.0, dtype=np.float32)
    padded[:resized_height, :resized_width] = resized
    input_blob = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]

    with NET_LOCK:
        detector.setInput(input_blob)
        output = detector.forward()

    if (output is None or output.ndim != 3 or output.shape[0] != 1
            or output.shape[1] != PHONE_GRIDS.shape[1] or output.shape[2] <= 5 + COCO_CELL_PHONE_CLASS_ID):
        raise RuntimeError("Mobile phone model returned an invalid result")

    predictions = output[0].astype(np.float32, copy=True)
    phone_scores = predictions[:, 4] * predictions[:, 5 + COCO_CELL_PHONE_CLASS_ID]
    candidate_indices = np.flatnonzero(
        np.isfinite(phone_scores) & (phone_scores >= confidence_threshold)
    )
    if not candidate_indices.size:
        return []

    selected = predictions[candidate_indices]
    decoded_centers = (selected[:, :2] + PHONE_GRIDS[0, candidate_indices]) * PHONE_EXPANDED_STRIDES[0, candidate_indices]
    decoded_sizes = np.exp(np.clip(selected[:, 2:4], -10, 10)) * PHONE_EXPANDED_STRIDES[0, candidate_indices]
    x1 = (decoded_centers[:, 0] - decoded_sizes[:, 0] / 2) / scale
    y1 = (decoded_centers[:, 1] - decoded_sizes[:, 1] / 2) / scale
    x2 = (decoded_centers[:, 0] + decoded_sizes[:, 0] / 2) / scale
    y2 = (decoded_centers[:, 1] + decoded_sizes[:, 1] / 2) / scale

    boxes = []
    valid_scores = []
    valid_coordinates = []
    for index, score in enumerate(phone_scores[candidate_indices]):
        start_x = max(0, min(width - 1, round(float(x1[index]))))
        start_y = max(0, min(height - 1, round(float(y1[index]))))
        end_x = max(0, min(width - 1, round(float(x2[index]))))
        end_y = max(0, min(height - 1, round(float(y2[index]))))
        if end_x <= start_x or end_y <= start_y:
            continue
        boxes.append([start_x, start_y, end_x - start_x, end_y - start_y])
        valid_scores.append(float(score))
        valid_coordinates.append((start_x, start_y, end_x, end_y))

    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes, valid_scores, float(confidence_threshold), float(PHONE_NMS_THRESHOLD)
    )
    if indices is None or len(indices) == 0:
        return []

    results = []
    for index in np.asarray(indices).reshape(-1):
        start_x, start_y, end_x, end_y = valid_coordinates[int(index)]
        results.append({
            'label': 'cell phone',
            'confidence': valid_scores[int(index)],
            'box': {
                'startX': start_x,
                'startY': start_y,
                'endX': end_x,
                'endY': end_y
            }
        })
    return results


def detect_faces(gray_frame, face_cascade, confidence_threshold=FACE_CONFIDENCE_THRESHOLD):
    """
    Detect faces in a grayscale frame using Haar Cascade
    Returns: list of dicts with bbox and confidence
    """
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    for (x, y, w, h) in faces:
        area = w * h
        confidence = min(1.0, area / (200 * 200) * 0.8 + 0.2)
        
        if confidence >= confidence_threshold:
            results.append({
                'bbox': {
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h)
                },
                'confidence': float(confidence)
            })
    
    return results


def _is_person_label(label):
    norm = str(label).lower().strip()
    return norm in {'person', 'people', 'human', 'humans'} or norm.startswith('person')


def _normalize_object_label(label):
    norm = str(label).lower().strip()
    aliases = {
        'cell phone': 'Mobile',
        'cellphone': 'Mobile',
        'mobile phone': 'Mobile',
        'mobile': 'Mobile',
        'phone': 'Mobile',
        'tvmonitor': 'Monitor',
        'monitor': 'Monitor',
        'pottedplant': 'Plant',
        'diningtable': 'Table',
        'motorbike': 'Motorbike',
        'aeroplane': 'Airplane',
    }
    return aliases.get(norm, label)


def _bbox_coverage(box, face_box):
    left = max(box['startX'], face_box['x'])
    top = max(box['startY'], face_box['y'])
    right = min(box['endX'], face_box['x'] + face_box['w'])
    bottom = min(box['endY'], face_box['y'] + face_box['h'])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = max(0, box['endX'] - box['startX']) * max(0, box['endY'] - box['startY'])
    return intersection / area if area else 0.0


def _filter_mobile_monitor_duplicates(object_dets, mobile_dets):
    mobile_boxes = [detection['box'] for detection in mobile_dets]
    return [
        detection for detection in object_dets
        if _normalize_object_label(detection.get('label', '')).lower() not in {'monitor', 'tv'}
        or not any(
            _bbox_coverage(detection['box'], {
                'x': mobile_box['startX'],
                'y': mobile_box['startY'],
                'w': mobile_box['endX'] - mobile_box['startX'],
                'h': mobile_box['endY'] - mobile_box['startY']
            }) >= 0.8
            for mobile_box in mobile_boxes
        )
    ]


def _filter_face_as_object(object_dets, face_boxes):
    filtered = []
    for det in object_dets:
        label = det.get('label', '')
        if (not _is_person_label(label)
                and any(_bbox_coverage(det['box'], face['bbox']) >= 0.7 for face in face_boxes)):
            continue
        det = dict(det)
        det['label'] = _normalize_object_label(label)
        filtered.append(det)
    return filtered


def detect_face_emotion(frame, face_bbox):
    """Estimate a coarse facial emotion from the face ROI using Haar features."""
    try:
        h, w = frame.shape[:2]
        x = max(0, int(face_bbox['x']))
        y = max(0, int(face_bbox['y']))
        fw = max(1, int(face_bbox['w']))
        fh = max(1, int(face_bbox['h']))
        face_roi = frame[max(0, y):min(h, y + fh), max(0, x):min(w, x + fw)]
        if face_roi.size == 0:
            return {'emotion': 'Idle', 'confidence': 0.4}

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        cascade_dir = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascades')
        eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))
        smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_smile.xml'))
        mouth_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_mcs_mouth.xml'))

        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(15, 15))
        smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=18, minSize=(20, 20))
        mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=12, minSize=(12, 12)) if mouth_cascade.empty() is False else ()

        face_area = float(fw * fh)
        smile_area = sum((m[2] * m[3]) for m in smiles) if len(smiles) else 0.0
        mouth_area = sum((m[2] * m[3]) for m in mouths) if len(mouths) else 0.0

        if len(smiles) > 0 and len(eyes) >= 1 and (smile_area / max(face_area, 1.0)) > 0.02:
            return {'emotion': 'Happy', 'confidence': 0.92}
        if len(mouths) > 0 and (mouth_area / max(face_area, 1.0)) > 0.03 and len(eyes) >= 1:
            return {'emotion': 'Sad', 'confidence': 0.76}
        if len(eyes) >= 2 and (smile_area / max(face_area, 1.0)) < 0.015:
            return {'emotion': 'Idle', 'confidence': 0.70}
        if len(eyes) >= 1 and len(smiles) == 0 and (mouth_area / max(face_area, 1.0)) > 0.02:
            return {'emotion': 'Angry', 'confidence': 0.70}
        if len(eyes) == 0 and len(smiles) == 0:
            return {'emotion': 'Idle', 'confidence': 0.55}
        return {'emotion': 'Idle', 'confidence': 0.60}
    except Exception:
        return {'emotion': 'Idle', 'confidence': 0.5}


def encode_face(face_roi):
    """
    Simple face encoding: downscale ROI to 32x32 grayscale, flatten/normalize.
    In a production system, use deep learning face embeddings (e.g., FaceNet)
    """
    gray = face_roi
    try:
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = face_roi
    # Resize to small standard size (32x32 grayscale = 1024 values)
    face_resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    # Flatten and normalize
    encoding = (face_resized.flatten() / 255.0).tolist()
    return encoding


def log_analytics(db, user_id, action_type, action_data=None, ip_address=None, user_agent=None):
    """Log analytics event (safe no-op when DB is unavailable)"""
    try:
        if not db.is_connected():
            return
        query = """
        INSERT INTO analytics (user_id, action_type, action_data, ip_address, user_agent)
        VALUES (%s, %s, %s, %s, %s)
        """
        params = (user_id, action_type, json.dumps(action_data) if action_data else None,
                  ip_address, user_agent)
        db.insert_one(query, params)
    except Exception as e:
        print(f"[WARN] analytics log skipped: {e}")


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
_secret_key = os.getenv('SECRET_KEY')
if _secret_key:
    app.secret_key = _secret_key
else:
    app.secret_key = secrets.token_hex(32)
    print("[WARNING] SECRET_KEY env not set - using a random per-boot key; sessions will not persist across restarts")

# Initialize database
db = Database(MYSQL_CONFIG)
init_db(db)

# Ensure models are loaded at startup
print(f"[INFO] Classes loaded: {len(CLASSES)}")
print(f"[INFO] Face cascade loaded: {FACE_CASCADE_PATH}")
print(f"[INFO] Detection models ready")


@app.route('/api/status')
def status():
    """Check system status"""
    user_id = flask_session.get('user_id')
    try:
        db_connected = db.is_connected()
    except Exception:
        db_connected = False
    try:
        face_ok = face_cascade is not None and not face_cascade.empty()
    except Exception:
        face_ok = False

    return jsonify({
        'loggedIn': user_id is not None,
        'user': {'username': flask_session.get('username', 'Guest')} if user_id else None,
        'dbConnected': db_connected,
        'models': {
            'object_net': net is not None or phone_net is not None,
            'mobile_phone_detector': phone_net is not None,
            'face_cascade': face_ok
        }
    })


@app.route('/api/live-camera', methods=['POST'])
def live_camera_analysis():
    """Run a lightweight real-time object and face analysis on a single camera frame."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400
    if net is None and phone_net is None:
        return jsonify({
            'success': False,
            'message': 'Object detection models are unavailable. Check the model files in the models folder.'
        }), 503

    try:
        frame = decode_upload(file, max_dim=1200)
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, face_cascade)
        object_dets = detect_objects(frame, net, CLASSES) if net is not None else []
        mobile_dets = detect_mobile_objects(frame, phone_net) if phone_net is not None else []
        object_dets = _filter_mobile_monitor_duplicates(object_dets, mobile_dets)
        object_dets.extend(mobile_dets)
        object_dets = _filter_face_as_object(object_dets, faces)

        include_emotions = request.form.get('include_emotions', 'true').lower() not in ('0', 'false', 'off')
        face_results = []
        dominant_emotion = 'No face detected' if not faces else 'Emotion analysis disabled'
        best_score = 0.0
        for face in faces:
            entry = {
                'bbox': face['bbox'],
                'confidence': face['confidence']
            }
            if include_emotions:
                emotion = detect_face_emotion(frame, face['bbox'])
                entry.update({
                    'emotion': emotion['emotion'],
                    'emotion_confidence': emotion['confidence']
                })
                if emotion['confidence'] > best_score:
                    dominant_emotion = emotion['emotion']
                    best_score = emotion['confidence']
            face_results.append(entry)

        response = {
            'success': True,
            'face_count': len(face_results),
            'object_count': len(object_dets),
            'mobile_phone_detector': phone_net is not None,
            'emotion': dominant_emotion,
            'dominant_emotion': dominant_emotion,
            'emotion_confidence': round(best_score, 2),
            'faces': face_results,
            'face_emotions': [
                {
                    'emotion': item['emotion'],
                    'confidence': round(item['emotion_confidence'], 2)
                }
                for item in face_results
                if 'emotion' in item
            ],
            'objects': [{
                'label': d['label'],
                'confidence': float(d['confidence']),
                'box': d['box']
            } for d in object_dets],
            'object_names': [
                f"{d['label']} ({d['confidence']:.0%})" for d in object_dets
            ]
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error('Live camera detection failed: %r', e)
        return jsonify({'success': False, 'message': 'Live detection failed'}), 500


@app.route('/')
def index():
    """Main chatbot interface"""
    user_id = flask_session.get('user_id')
    return render_template('index.html', user_id=user_id, classes=CLASSES)


@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint (works offline when MySQL is unavailable)"""
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()

    if not username:
        return jsonify({'success': False, 'message': 'Username required'}), 400

    # Offline fallback when DB is unavailable
    if not db.is_connected():
        flask_session['user_id'] = 'guest-' + uuid.uuid4().hex
        flask_session['username'] = username
        return jsonify({'success': True, 'message': 'Logged in (offline mode, no MySQL)', 'user': username, 'offline': True})

    # For offline demo, accept any non-empty password
    # In production, verify hashed password
    query = "SELECT id, username FROM users WHERE username = %s AND is_active = TRUE"
    user = db.get_one(query, (username,))
    
    if user:
        flask_session['user_id'] = user['id']
        flask_session['username'] = user['username']
        log_analytics(db, user['id'], 'login', ip_address=request.remote_addr)
        return jsonify({'success': True, 'message': 'Login successful', 'user': user['username']})
    else:
        # Create user if not exists (demo mode)
        insert_query = """
        INSERT INTO users (username, is_active) VALUES (%s, TRUE)
        ON DUPLICATE KEY UPDATE is_active = TRUE, last_login = CURRENT_TIMESTAMP
        """
        db.execute(insert_query, (username,))
        user = db.get_one(query, (username,))
        if not user:
            flask_session['user_id'] = 'guest-' + uuid.uuid4().hex
            flask_session['username'] = username
            return jsonify({'success': True, 'message': 'Logged in (offline mode)', 'user': username, 'offline': True})
        flask_session['user_id'] = user['id']
        flask_session['username'] = user['username']
        log_analytics(db, user['id'], 'login', ip_address=request.remote_addr)
        return jsonify({'success': True, 'message': 'Account created/loaded', 'user': user['username']})


@app.route('/api/logout')
def logout():
    """User logout"""
    user_id = flask_session.get('user_id')
    if user_id:
        log_analytics(db, user_id, 'logout', ip_address=request.remote_addr)
    flask_session.clear()
    return jsonify({'success': True, 'message': 'Logged out'})


@app.route('/api/detect-objects', methods=['POST'])
def detect_objects_api():
    """Object detection endpoint"""
    if net is None:
        return jsonify({'success': False, 'message': 'object model missing'}), 503
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400

    try:
        # Read image
        frame = decode_upload(file)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400

        # Detect objects
        results = detect_objects(frame, net, CLASSES)

        # Draw bounding boxes on the frame for visualization
        for det in results:
            box = det['box']
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.rectangle(frame, (box['startX'], box['startY']), (box['endX'], box['endY']), (0, 255, 0), 2)
            cv2.putText(frame, label, (box['startX'], box['startY'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode result image
        result_img_base64 = encode_jpg(frame)

        return jsonify({
            'success': True,
            'detections': results,
            'result_image': result_img_base64,
            'total_objects': len(results)
        })

    except Exception as e:
        app.logger.error("Object detection failed: %r", e)
        return jsonify({'success': False, 'message': 'Detection failed'}), 500


@app.route('/api/detect-faces', methods=['POST'])
def detect_faces_api():
    """Face detection endpoint"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400
    
    try:
        # Read image
        frame = decode_upload(file)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        results = detect_faces(gray, face_cascade)

        # Draw rectangles on the frame
        for face in results:
            bbox = face['bbox']
            cv2.rectangle(frame, (bbox['x'], bbox['y']),
                         (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                         (255, 0, 0), 2)
            cv2.putText(frame, f"Face: {face['confidence']:.2f}",
                       (bbox['x'], bbox['y'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode result image
        result_img_base64 = encode_jpg(frame)

        # Log face detection
        user_id = flask_session.get('user_id')
        if user_id and results:
            log_analytics(db, user_id, 'face_detected',
                         action_data={'faces_detected': len(results)},
                         ip_address=request.remote_addr)

        return jsonify({
            'success': True,
            'detections': results,
            'result_image': result_img_base64,
            'total_faces': len(results)
        })

    except Exception as e:
        app.logger.error("Face detection failed: %r", e)
        return jsonify({'success': False, 'message': 'Detection failed'}), 500


@app.route('/api/register-face', methods=['POST'])
def register_face():
    """Register a face for future recognition"""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400
    
    try:
        # Read image
        frame = decode_upload(file)

        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        
        # Detect faces in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, face_cascade)
        
        if not faces:
            return jsonify({'success': False, 'message': 'No face detected in image'}), 400
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f['bbox']['w'] * f['bbox']['h'])
        bbox = largest_face['bbox']
        
        # Extract face ROI
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        face_roi = frame[y:y+h, x:x+w]
        
        # Encode face
        face_encoding = encode_face(face_roi)
        
        # Get face name from request (works with both JSON and form-data)
        payload = request.get_json(silent=True) or {}
        face_name = str(
            payload.get('name')
            or request.form.get('name')
            or request.args.get('name')
            or f'user_{user_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        ).strip()[:64]
        if not face_name:
            return jsonify({'success': False, 'message': 'Face name required'}), 400

        # Store in face registry
        query = """
        INSERT INTO face_registry (user_id, face_encoding, face_name)
        VALUES (%s, %s, %s)
        """
        encoding_json = json.dumps(face_encoding)
        if db.is_connected():
            db.insert_one(query, (user_id, encoding_json, face_name))
        
        # Log analytics
        log_analytics(db, user_id, 'face_registered',
                     action_data={'face_name': face_name, 'faces_detected': 1})
        
        return jsonify({
            'success': True,
            'message': f'Face registered as "{face_name}"',
            'face_name': face_name,
            'encoding_length': len(face_encoding)
        })
        
    except Exception as e:
        app.logger.error("Face registration failed: %r", e)
        return jsonify({'success': False, 'message': 'Face registration failed'}), 500


@app.route('/api/faces')
def list_faces():
    """List registered faces for the logged-in user"""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401

    if not db.is_connected():
        return jsonify({'success': True, 'faces': [], 'offline': True})

    query = """
    SELECT id, face_name, created_at FROM face_registry
    WHERE user_id = %s
    ORDER BY created_at DESC
    """
    faces = db.fetch_all(query, (user_id,))

    return jsonify({
        'success': True,
        'faces': faces
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat message with optional object/face detection"""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    
    data = request.get_json(silent=True) or {}
    if not data and request.form:
        data = dict(request.form)
    user_message = (data.get('message') or '').strip()
    
    if not user_message:
        return jsonify({'success': False, 'message': 'Message required'}), 400
    if len(user_message) > 4000:
        return jsonify({'success': False, 'message': 'Message too long (max 4000 chars)'}), 400

    # Single decode; draw object boxes AND face boxes on the same frame
    detected_objects = []
    result_image_base64 = None
    face_detected = False
    frame_received = False
    img_bytes = None

    if 'image' in request.files:
        _f = request.files['image']
        if _f.filename != '':
            try:
                img_bytes = _f.read()
            except Exception as e:
                app.logger.warning("Chat image read failed: %r", e)
                img_bytes = None

    if img_bytes:
        try:
            frame = decode_upload_bytes(img_bytes)
            if frame is not None:
                frame_received = True
                if net is None:
                    app.logger.warning("Chat object detection skipped: object model missing")
                else:
                    detected_objects = detect_objects(frame, net, CLASSES)
                    for det in detected_objects:
                        box = det['box']
                        label = f"{det['label']}: {det['confidence']:.2f}"
                        cv2.rectangle(frame, (box['startX'], box['startY']),
                                     (box['endX'], box['endY']), (0, 255, 0), 2)
                        cv2.putText(frame, label, (box['startX'], box['startY'] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces(gray, face_cascade)
                face_detected = len(faces) > 0
                for face in faces:
                    bbox = face['bbox']
                    cv2.rectangle(frame, (bbox['x'], bbox['y']),
                                 (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                                 (255, 0, 0), 2)
                result_image_base64 = encode_jpg(frame)
        except Exception as e:
            app.logger.error("Chat vision failed: %r", e)
    
    # Generate bot response with the LLM (context-aware)
    context = None
    if frame_received:
        parts = [
            'An image frame was provided, but this chat model is text-only. '
            'Answer only from the available detector results and do not infer visual details.'
        ]
        if detected_objects:
            parts.append('Supported objects detected: ' + ', '.join(
                f"{d['label']} ({d['confidence']:.0%})" for d in detected_objects))
        else:
            parts.append('No supported objects were detected.')
        if face_detected:
            parts.append('At least one face was detected.')
        else:
            parts.append('No face was detected.')
        context = ' '.join(parts)
    with LLM_LOCK:
        bot_response = llm.chat(user_message, context=context)
    
    # Save to chat history (skipped for offline guests / no DB)
    save_history(user_id, user_message, bot_response,
                 detected_objects=detected_objects, face_detected=face_detected)
    rtdb_push(user_message, bot_response)
    
    # Log analytics
    log_analytics(db, user_id, 'chat_message',
                 action_data={'message_length': len(user_message),
                            'objects_detected': len(detected_objects),
                            'face_detected': face_detected})
    
    return jsonify({
        'success': True,
        'response': bot_response,
        'detected_objects': detected_objects,
        'face_detected': face_detected,
        'result_image': result_image_base64
    })


@app.route('/api/history')
def get_history():
    """Get chat history for user"""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401

    if not db.is_connected():
        return jsonify({'success': True, 'history': [], 'offline': True})

    query = """
    SELECT * FROM chat_history 
    WHERE user_id = %s 
    ORDER BY created_at DESC 
    LIMIT 50
    """
    history = db.fetch_all(query, (user_id,))
    
    return jsonify({
        'success': True,
        'history': history
    })


@app.route('/api/analytics')
def get_analytics():
    """Get system analytics (admin endpoint)"""
    if flask_session.get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    if not db.is_connected():
        return jsonify({'success': True, 'analytics': [], 'offline': True})

    query = """
    SELECT action_type, COUNT(*) as count, DATE(created_at) as date
    FROM analytics 
    WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    GROUP BY action_type, DATE(created_at)
    ORDER BY created_at DESC
    LIMIT 100
    """
    analytics = db.fetch_all(query)
    
    return jsonify({
        'success': True,
        'analytics': analytics
    })


@app.route('/api/users')
def list_users():
    """List all users (admin endpoint)"""
    if flask_session.get('username') != 'admin':
        return jsonify({'success': False, 'message': 'Admin only'}), 403
    if not db.is_connected():
        return jsonify({'success': True, 'users': [], 'offline': True})
    query = "SELECT id, username, email, created_at, last_login FROM users WHERE is_active = TRUE"
    users = db.fetch_all(query)
    
    return jsonify({
        'success': True,
        'users': users
    })


# ============================================================
# LLM + SMART TOOLS + IMAGE TOOLS
# ============================================================

def _is_guest(uid):
    return isinstance(uid, str) and uid.startswith('guest-')


def save_history(user_id, user_message, bot_response, detected_objects=None, face_detected=False):
    """Persist one chat turn; skipped for offline guest ids or when DB is down."""
    try:
        if _is_guest(user_id) or not db.is_connected():
            return
        objects_json = json.dumps(detected_objects) if detected_objects else None
        db.insert_one(
            "INSERT INTO chat_history (user_id, user_message, bot_response, detected_objects, face_detected)"
            " VALUES (%s,%s,%s,%s,%s)",
            (user_id, user_message, bot_response, objects_json, face_detected),
        )
    except Exception as e:
        print(f"[WARN] history save skipped: {e}")


def _llm_overrides(data):
    """Validated per-request temperature/max_tokens (None = use global)."""
    temperature = data.get('temperature')
    max_tokens = data.get('max_tokens')
    try:
        temperature = None if temperature is None else float(temperature)
        if temperature is not None and not 0.0 <= temperature <= 1.5:
            temperature = None
    except (ValueError, TypeError):
        temperature = None
    try:
        max_tokens = None if max_tokens is None else int(max_tokens)
        if max_tokens is not None and not 32 <= max_tokens <= 512:
            max_tokens = None
    except (ValueError, TypeError):
        max_tokens = None
    return temperature, max_tokens


@app.route('/api/llm-info')
def llm_info():
    """Which LLM backend is active and which GGUF files are available."""
    info = llm.info()
    try:
        info['dbConnected'] = db.is_connected()
    except Exception:
        info['dbConnected'] = False
    return jsonify({'success': True, **info})


@app.route('/api/llm', methods=['POST'])
def llm_chat():
    """Direct chat with the real offline LLM."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'success': False, 'message': 'Message required'}), 400
    if len(message) > 4000:
        return jsonify({'success': False, 'message': 'Message too long (max 4000 chars)'}), 400
    temperature, max_tokens = _llm_overrides(data)
    with LLM_LOCK:
        reply = llm.chat(message, temperature=temperature, max_tokens=max_tokens)
    save_history(user_id, message, reply)
    log_analytics(db, user_id, 'llm_chat', action_data={'message_length': len(message)})
    if data.get('mode') == 'voice':
        rtdb_push_voice(message, reply)
    else:
        rtdb_push(message, reply)
    return jsonify({'success': True, 'response': reply, 'backend': llm.info()['backend']})


@app.route('/api/llm-stream', methods=['POST'])
def llm_stream():
    """Stream an LLM reply token-by-token (Server-Sent Events over POST)."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'success': False, 'message': 'Message required'}), 400
    if len(message) > 4000:
        return jsonify({'success': False, 'message': 'Message too long (max 4000 chars)'}), 400
    temperature, max_tokens = _llm_overrides(data)
    if not LLM_LOCK.acquire(blocking=False):
        return jsonify({'success': False, 'message': 'Model busy, please retry', 'retry': True}), 429

    def gen():
        full = ''
        last_emit = time.time()
        try:
            for chunk in llm.stream(message, temperature=temperature, max_tokens=max_tokens):
                now = time.time()
                if now - last_emit >= 15:
                    yield ": ping\n\n"
                    last_emit = now
                if chunk:
                    full = chunk  # snapshots: latest replaces previous
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                    last_emit = time.time()
        except Exception as e:
            app.logger.error("LLM stream failed: %r", e)
            yield f"data: {json.dumps({'error': 'Generation failed'})}\n\n"
        finally:
            try:
                LLM_LOCK.release()
            except RuntimeError:
                pass
        reply = (full or '').strip()
        save_history(user_id, message, reply)
        if reply:
            rtdb_push(message, reply)
        log_analytics(db, user_id, 'llm_chat', action_data={'message_length': len(message)})
        yield f"data: {json.dumps({'done': True, 'backend': llm.info()['backend']})}\n\n"

    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/llm-params', methods=['GET', 'POST'])
def llm_params():
    """Read or update runtime LLM params (temperature 0-1.5, max_tokens 32-512)."""
    if request.method == 'GET':
        return jsonify({'success': True, **get_llm_params()})
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    params, error = set_llm_params(data.get('temperature'), data.get('max_tokens'))
    if error:
        return jsonify({'success': False, 'message': error}), 400
    log_analytics(db, user_id, 'llm_params', action_data=params)
    return jsonify({'success': True, **params})


@app.route('/api/doc-qa', methods=['POST'])
def doc_qa_api():
    """Extractive Q&A over an uploaded/pasted document (fully offline)."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    doc_text = data.get('doc_text', '')
    question = (data.get('question') or '').strip()
    if 'file' in request.files and not doc_text:
        f = request.files['file']
        try:
            doc_text = f.read().decode('utf-8', errors='replace')
        except Exception:
            return jsonify({'success': False, 'message': 'Could not read file (use .txt/.md)'}), 400
    if not doc_text.strip():
        return jsonify({'success': False, 'message': 'Document text required'}), 400
    if not question:
        return jsonify({'success': False, 'message': 'Question required'}), 400
    try:
        result = doc_qa(doc_text, question)
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    log_analytics(db, user_id, 'doc_qa',
                  action_data={'doc_chars': len(doc_text), 'truncated': result['truncated']})
    return jsonify({'success': True, **result})


NOTES_DIR = os.path.join(BASE_DIR, 'data')
NOTES_LOCK = threading.Lock()
SCHEDULE_LOCK = threading.Lock()


def _notes_path(username):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', username or 'guest')[:64]
    return os.path.join(NOTES_DIR, f'notes_{safe}.json')


def _load_notes(username):
    path = _notes_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        return notes if isinstance(notes, list) else []
    except Exception:
        return []


def _save_notes(username, notes):
    os.makedirs(NOTES_DIR, exist_ok=True)
    tmp = _notes_path(username) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False)
    os.replace(tmp, _notes_path(username))


def _schedule_path(username):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', username or 'guest')[:64]
    return os.path.join(NOTES_DIR, f'schedule_{safe}.json')


def _load_schedule(username):
    path = _schedule_path(username)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        return entries if isinstance(entries, list) else []
    except (OSError, json.JSONDecodeError) as e:
        app.logger.error("Could not load schedule for %s: %r", username, e)
        raise


def _save_schedule(username, entries):
    os.makedirs(NOTES_DIR, exist_ok=True)
    path = _schedule_path(username)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False)
    os.replace(tmp, path)


@app.route('/api/schedule', methods=['GET', 'PUT'])
def daily_schedule():
    """Read or replace the signed-in user's daily work timetable."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    username = flask_session.get('username', 'guest')

    if request.method == 'GET':
        try:
            with SCHEDULE_LOCK:
                entries = _load_schedule(username)
        except OSError:
            return jsonify({'success': False, 'message': 'Could not load the saved timetable'}), 500
        return jsonify({'success': True, 'entries': entries})

    data = request.get_json(silent=True) or {}
    entries = data.get('entries')
    if not isinstance(entries, list):
        return jsonify({'success': False, 'message': 'Timetable entries must be a list'}), 400
    if len(entries) > 50:
        return jsonify({'success': False, 'message': 'A timetable can contain at most 50 entries'}), 400

    clean_entries = []
    seen_ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return jsonify({'success': False, 'message': 'Each timetable entry must be an object'}), 400
        title = str(entry.get('title', '')).strip()
        start = str(entry.get('start', ''))
        end = str(entry.get('end', ''))
        entry_id = str(entry.get('id', '')).strip() or uuid.uuid4().hex
        if not title or len(title) > 100:
            return jsonify({'success': False, 'message': 'Activity names must be 1–100 characters'}), 400
        if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', start):
            return jsonify({'success': False, 'message': 'Start times must use 24-hour HH:MM format'}), 400
        if not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', end):
            return jsonify({'success': False, 'message': 'End times must use 24-hour HH:MM format'}), 400
        if end <= start:
            return jsonify({'success': False, 'message': 'End time must be later than start time'}), 400
        if len(entry_id) > 64 or entry_id in seen_ids:
            return jsonify({'success': False, 'message': 'Timetable entry IDs must be unique and at most 64 characters'}), 400
        seen_ids.add(entry_id)
        clean_entries.append({
            'id': entry_id,
            'title': title,
            'start': start,
            'end': end,
            'enabled': entry.get('enabled') is not False,
        })

    try:
        with SCHEDULE_LOCK:
            _save_schedule(username, clean_entries)
    except OSError as e:
        app.logger.error("Could not save schedule for %s: %r", username, e)
        return jsonify({'success': False, 'message': 'Could not save the timetable'}), 500
    return jsonify({'success': True, 'entries': clean_entries})


@app.route('/api/notes', methods=['GET', 'POST'])
def notes_collection():
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    username = flask_session.get('username', 'guest')
    if request.method == 'GET':
        q = (request.args.get('q') or '').lower()
        with NOTES_LOCK:
            notes = _load_notes(username)
        if q:
            notes = [n for n in notes if q in (n.get('title', '') + ' ' + n.get('body', '')).lower()]
        notes = sorted(notes, key=lambda n: n.get('updated', ''), reverse=True)[:200]
        return jsonify({'success': True, 'notes': notes})
    data = request.get_json(silent=True) or {}
    title = str(data.get('title', ''))[:120]
    body = str(data.get('body', ''))
    if len(body) > 20 * 1024:
        return jsonify({'success': False, 'message': 'Note too large (max 20 KB)'}), 400
    if not title.strip() and not body.strip():
        return jsonify({'success': False, 'message': 'Empty note'}), 400
    note = {'id': uuid.uuid4().hex, 'title': title.strip() or 'Untitled',
            'body': body, 'updated': datetime.now().isoformat(timespec='seconds')}
    with NOTES_LOCK:
        notes = _load_notes(username)
        notes.append(note)
        _save_notes(username, notes)
    log_analytics(db, user_id, 'note_created')
    return jsonify({'success': True, 'note': note})


@app.route('/api/notes/<note_id>', methods=['PUT', 'DELETE'])
def notes_item(note_id):
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    username = flask_session.get('username', 'guest')
    with NOTES_LOCK:
        notes = _load_notes(username)
        idx = next((i for i, n in enumerate(notes) if n.get('id') == note_id), None)
        if idx is None:
            return jsonify({'success': False, 'message': 'Note not found'}), 404
        if request.method == 'DELETE':
            del notes[idx]
            _save_notes(username, notes)
            return jsonify({'success': True})
        data = request.get_json(silent=True) or {}
        if 'title' in data:
            notes[idx]['title'] = str(data['title'])[:120]
        if 'body' in data:
            body = str(data['body'])
            if len(body) > 20 * 1024:
                return jsonify({'success': False, 'message': 'Note too large (max 20 KB)'}), 400
            notes[idx]['body'] = body
        notes[idx]['updated'] = datetime.now().isoformat(timespec='seconds')
        _save_notes(username, notes)
        return jsonify({'success': True, 'note': notes[idx]})


@app.route('/api/detect-batch', methods=['POST'])
def detect_batch():
    """Detect objects in up to 6 images; returns per-file results + combined counts."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    files = request.files.getlist('images')
    if not files:
        return jsonify({'success': False, 'message': 'No images provided'}), 400
    if len(files) > 6:
        return jsonify({'success': False, 'message': 'Max 6 images per batch'}), 400
    if net is None:
        return jsonify({'success': False, 'message': 'object model missing'}), 503
    results = []
    combined = {}
    for f in files:
        name = f.filename or 'image'
        try:
            frame = decode_upload(f, max_dim=1280)
            if frame is None:
                results.append({'filename': name, 'error': 'Invalid image'})
                continue
            # NOTE: no NET_LOCK here — detect_objects() locks internally
            # (Lock is not reentrant; locking twice deadlocks).
            dets = detect_objects(frame, net, CLASSES)
            for d in dets:
                combined[d['label']] = combined.get(d['label'], 0) + 1
            thumb = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
            results.append({'filename': name, 'detections': dets,
                            'total': len(dets), 'thumbnail': encode_jpg(thumb, quality=70)})
        except Exception as e:
            app.logger.error("Batch item failed: %r", e)
            results.append({'filename': name, 'error': 'Processing failed'})
    log_analytics(db, user_id, 'detect_batch',
                  action_data={'files': len(files), 'objects': sum(combined.values())})
    return jsonify({'success': True, 'results': results, 'combined_counts': combined})


@app.route('/api/rtdb-status')
def rtdb_status_api():
    """Firebase RTDB sync status for the LCD/hardware layer."""
    return jsonify({'success': True, **rtdb_status()})


@app.route('/api/rtdb-toggle', methods=['POST'])
def rtdb_toggle():
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    on = data.get('enabled', True)
    return jsonify({'success': True, 'enabled': rtdb_set_enabled(on)})


@app.route('/api/rtdb-push', methods=['POST'])
def rtdb_push_api():
    """Manually re-push text to the LCD node (e.g. resend a missed answer)."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'success': False, 'message': 'Text required'}), 400
    rtdb_push_text(text[:500])
    return jsonify({'success': True, 'message': 'Sent to display'})


@app.route('/api/face-emotion', methods=['POST'])
def face_emotion_sync_api():
    """Queue a privacy-minimized live emotion snapshot for Firebase RTDB."""
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or type(data.get('enabled')) is not bool:
        return jsonify({'success': False, 'message': 'Invalid emotion snapshot'}), 400
    faces = data.get('faces')
    if not isinstance(faces, list) or len(faces) > 20:
        return jsonify({'success': False, 'message': 'Invalid face emotion list'}), 400

    allowed_emotions = {
        'Angry', 'Disgusted', 'Fearful', 'Happy',
        'Neutral', 'Sad', 'Surprised',
    }
    snapshot_faces = []
    for face in faces:
        if not isinstance(face, dict):
            return jsonify({'success': False, 'message': 'Invalid face emotion entry'}), 400
        emotion = face.get('emotion')
        confidence = face.get('confidence')
        if (not isinstance(emotion, str)
                or emotion not in allowed_emotions
                or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1):
            return jsonify({'success': False, 'message': 'Invalid face emotion entry'}), 400
        snapshot_faces.append({
            'emotion': emotion,
            'confidence': round(float(confidence), 3),
        })

    dominant = max(snapshot_faces, key=lambda face: face['confidence'], default=None)
    person_count = len(snapshot_faces)
    if not data['enabled']:
        person_status = 'detection_disabled'
        summary = 'Face/emotion detection disabled'
    elif person_count:
        person_status = 'detected'
        emotions = list(dict.fromkeys(face['emotion'] for face in snapshot_faces))
        if person_count == 1:
            summary = f'Person detected; face emotion: {emotions[0]}'
        else:
            summary = (
                f'{person_count} people detected; face emotions: '
                f'{", ".join(emotions)}')
    else:
        person_status = 'not_detected'
        summary = 'No person detected'

    snapshot = {
        'ts': int(time.time()),
        'enabled': data['enabled'],
        'face_count': person_count,
        'person_detected': data['enabled'] and person_count > 0,
        'person_count': person_count if data['enabled'] else 0,
        'person_status': person_status,
        'summary': summary,
        'active': data['enabled'] and person_count > 0,
        'status': 'disabled' if not data['enabled'] else (
            'detected' if snapshot_faces else 'no_face'),
        'dominant_emotion': dominant['emotion'] if dominant else None,
        'emotion_confidence': dominant['confidence'] if dominant else 0,
        'faces': snapshot_faces,
    }
    if not rtdb_push_emotion(snapshot):
        return jsonify({'success': False, 'message': 'Firebase sync is disabled'}), 503
    return jsonify({'success': True, 'message': 'Emotion snapshot queued'})


@app.route('/api/tool/calc', methods=['POST'])
def tool_calc():
    data = request.get_json(silent=True) or {}
    expr = str(data.get('expression', ''))
    try:
        return jsonify({'success': True, 'result': safe_calc(expr)})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/tool/convert', methods=['POST'])
def tool_convert():
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get('value', 0))
        result = convert_units(value, str(data.get('from', '')), str(data.get('to', '')))
        return jsonify({'success': True, 'result': round(result, 6)})
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/tool/password')
def tool_password():
    length = request.args.get('length', '16')
    symbols = request.args.get('symbols', '1') == '1'
    try:
        return jsonify({'success': True, 'password': password_generate(length, symbols)})
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/api/tool/text', methods=['POST'])
def tool_text():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'success': False, 'message': 'Text required'}), 400
    return jsonify({'success': True, 'analysis': text_analyze(text)})


@app.route('/api/tool/summarize', methods=['POST'])
def tool_summarize():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'success': False, 'message': 'Text required'}), 400
    try:
        n = max(1, min(int(data.get('sentences', 3)), 10))
    except (ValueError, TypeError):
        n = 3
    return jsonify({'success': True, 'summary': summarize(text, n)})


@app.route('/api/image-tool', methods=['POST'])
def image_tool():
    """Offline image utilities: grayscale, edges, resize, thumbnail."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No image selected'}), 400
    operation = (request.form.get('operation') or 'grayscale').lower()
    try:
        frame = decode_upload(file)
        if frame is None:
            return jsonify({'success': False, 'message': 'Invalid image format'}), 400
        if operation == 'grayscale':
            out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif operation == 'edges':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out = cv2.Canny(gray, 100, 200)
        elif operation == 'resize':
            try:
                width = max(16, min(int(request.form.get('width', 640)), 4000))
            except (ValueError, TypeError):
                width = 640
            h, w = frame.shape[:2]
            out = cv2.resize(frame, (width, int(h * width / w)))
        elif operation == 'thumbnail':
            out = cv2.resize(frame, (256, 256))
        else:
            return jsonify({'success': False, 'message': f'Unknown operation: {operation}'}), 400
        img_b64 = encode_jpg(out)
        user_id = flask_session.get('user_id')
        if user_id:
            log_analytics(db, user_id, 'image_tool',
                          action_data={'operation': operation}, ip_address=request.remote_addr)
        return jsonify({'success': True, 'operation': operation, 'result_image': img_b64})
    except Exception as e:
        app.logger.error("image tool failed: %r", e)
        return jsonify({'success': False, 'message': 'Image processing failed'}), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'message': 'Upload too large (max 8 MB)'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'success': False, 'message': 'Method not allowed'}), 405


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Offline Multipurpose Chatbot Web Application")
    print("=" * 60)
    print(f"[MODE] Offline detection active")
    print(f"[MODEL] MobileNet SSD + Haar Cascade face detection")
    print(f"[DATABASE] MySQL configured at {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
    print(f"[CLASSES] {len(CLASSES)} classes loaded")
    print("=" * 60)
    print("Available endpoints:")
    print("  GET  /           - Chat interface")
    print("  POST /api/login  - User login/registration")
    print("  POST /api/detect-objects - Object detection")
    print("  POST /api/detect-faces - Face detection")
    print("  POST /api/register-face - Register face")
    print("  POST /api/chat   - Chat with detection")
    print("  GET  /api/history - Chat history")
    print("  GET  /api/analytics - Analytics (admin)")
    print("=" * 60)
    
    # Start Flask application
    app.run(
        host=os.getenv('FLASK_HOST', '127.0.0.1'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=False,
        use_reloader=False,
        threaded=True
    )