#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fetch the Apache-2.0 OpenCV Zoo YOLOX model used for mobile phone detection."""

import hashlib
import os
import sys
import urllib.error
import urllib.request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'object_detection_yolox_2022nov.onnx')
TEMP_PATH = MODEL_PATH + '.download'
MODEL_URL = (
    'https://media.githubusercontent.com/media/opencv/opencv_zoo/'
    '47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox/'
    'object_detection_yolox_2022nov.onnx'
)
EXPECTED_SIZE = 35_858_002
EXPECTED_SHA256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) == EXPECTED_SIZE and _sha256(MODEL_PATH) == EXPECTED_SHA256:
            print('Mobile phone detector is present and verified.')
            return
        raise ValueError(f'Existing model failed integrity verification: {MODEL_PATH}')

    print('Downloading the 35 MB offline mobile phone detector...')
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response, open(TEMP_PATH, 'wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError('Downloaded mobile detector failed integrity verification')
        os.replace(TEMP_PATH, MODEL_PATH)
        print('Mobile phone detector downloaded and verified.')
    finally:
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)


if __name__ == '__main__':
    try:
        ensure_model()
    except (OSError, urllib.error.URLError, ValueError) as error:
        print(f'ERROR: Could not prepare the mobile phone detector: {error}', file=sys.stderr)
        sys.exit(1)
