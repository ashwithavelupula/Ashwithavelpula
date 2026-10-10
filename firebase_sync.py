#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Firebase RTDB sync for hardware pickup (e.g. LCD on ESP32/Arduino).

What it does
------------
Every chat answer is pushed as a SMALL flat node that any beginner can read:

    nexus_live: {
        "seq": 42,                 # increments each push; hardware polls for change
        "q": "short question",     # max ~64 chars
        "a": "short answer",       # max ~160 chars
        "line1": "16-char LCD",    # pre-wrapped LCD lines (see LCD_COLS/LCD_ROWS)
        "line2": "16-char LCD",
        "ts": 1728000000,          # unix time of push
        "ttl": 20,                 # seconds the node stays alive
        "expires_at": 1728000020,  # ts + ttl; hardware can ignore stale data
        "status": "new"
    }

The node is DELETED automatically after `ttl` seconds, so a missing node
means "nothing new to display". All pushes run in a background thread and
never break chat if Firebase is unreachable.

Live camera face/person snapshots are written separately to `nexus_emotion`,
including `person_detected`, `person_count`, `dominant_emotion`, and a
human-readable `summary`, and the latest `nexus_text` sent to the LCD.
Distinct detected emotion states are also appended to `nexus_emotion_history`,
so later snapshots do not replace earlier ones. Voice turns are also appended
there as `type: "voice"` records and tagged in the temporary LCD payload.

Config (env vars)
-----------------
FIREBASE_URL         RTDB root URL (default: the srdoor project URL)
FIREBASE_CREDENTIALS service-account JSON path (Admin SDK mode)
FIREBASE_NODE        node name (default: nexus_live)
FIREBASE_EMOTION_NODE node name for live face emotion (default: nexus_emotion)
FIREBASE_EMOTION_HISTORY_NODE node name for emotion history (default: nexus_emotion_history)
FIREBASE_TTL_SEC     seconds the node lives (default: 20)
FIREBASE_ENABLED     1/0 master switch (default: 1)
LCD_COLS / LCD_ROWS  display size for pre-wrapped lines (default 16x2)
"""

import os
import re
import json
import time
import threading
import urllib.request
import urllib.error

FIREBASE_URL = os.getenv('FIREBASE_URL', 'https://robo---project-default-rtdb.firebaseio.com').rstrip('/')
FIREBASE_CREDENTIALS = os.getenv(
    'FIREBASE_CREDENTIALS',
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'robo---project-firebase-adminsdk-fbsvc-d3a2e20524.json'))
FIREBASE_NODE = os.getenv('FIREBASE_NODE', 'nexus_live').strip('/') or 'nexus_live'
FIREBASE_EMOTION_NODE = os.getenv(
    'FIREBASE_EMOTION_NODE', 'nexus_emotion').strip('/') or 'nexus_emotion'
FIREBASE_EMOTION_HISTORY_NODE = os.getenv(
    'FIREBASE_EMOTION_HISTORY_NODE', 'nexus_emotion_history').strip('/') or 'nexus_emotion_history'
FIREBASE_HISTORY_NODE = os.getenv('FIREBASE_HISTORY_NODE', 'nexus_history').strip('/') or 'nexus_history'
FIREBASE_TTL = max(5, int(os.getenv('FIREBASE_TTL_SEC', '20')))
LCD_COLS = max(8, int(os.getenv('LCD_COLS', '16')))
LCD_ROWS = min(8, max(1, int(os.getenv('LCD_ROWS', '2'))))
Q_MAX = 64
A_MAX = 160

_state_lock = threading.Lock()
_seq = 0
_timer = None
_enabled = os.getenv('FIREBASE_ENABLED', '1') == '1'
_last_push = None
_last_error = None
_last_archive_push = None
_last_archive_error = None
_last_emotion_push = None
_last_emotion_error = None
_last_emotion_history_push = None
_last_emotion_history_error = None
_last_emotion_history_record_push = None
_last_emotion_history_record_error = None
_last_voice_history_push = None
_last_voice_history_error = None
_last_emotion_history_signature = None
_pending_emotion_history_signatures = set()
_last_nexus_text = ''


def _strip_md(text):
    text = re.sub(r'```[\s\S]*?```', ' ', text or '')
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'[*_#>|]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _shorten(text, limit):
    text = _strip_md(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(' ', 1)[0] or text[:limit]
    return cut.rstrip() + '…'


def _lcd_lines(answer):
    """Word-wrap the short answer into LCD_COLSxLCD_ROWS lines."""
    words, lines, cur = _shorten(answer, A_MAX).split(' '), [], ''
    for w in words:
        while len(w) > LCD_COLS:  # hard-split overlong words
            if cur:
                lines.append(cur)
                cur = ''
            lines.append(w[:LCD_COLS])
            w = w[LCD_COLS:]
        trial = (cur + ' ' + w).strip()
        if len(trial) <= LCD_COLS:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) >= LCD_ROWS:
            break
    if cur and len(lines) < LCD_ROWS:
        lines.append(cur)
    lines = lines[:LCD_ROWS]
    while len(lines) < LCD_ROWS:
        lines.append('')
    return lines


def _rest_write(payload, node=FIREBASE_NODE, append=False):
    """REST fallback: PUT a live value or POST a permanent history record."""
    url = f'{FIREBASE_URL}/{node}.json'
    secret = os.getenv('FIREBASE_DB_SECRET', '')
    if secret:
        url += f'?auth={secret}'
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'},
                                 method='POST' if append else 'PUT')
    with urllib.request.urlopen(req, timeout=10) as r:
        if r.status not in (200, 204):
            raise RuntimeError(f'RTDB write http {r.status}')


def _rest_delete(node=FIREBASE_NODE):
    url = f'{FIREBASE_URL}/{node}.json'
    secret = os.getenv('FIREBASE_DB_SECRET', '')
    if secret:
        url += f'?auth={secret}'
    req = urllib.request.Request(url, method='DELETE')
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


_admin_refs = {}


def _admin_reference(node):
    """Admin SDK path (service account bypasses DB rules)."""
    if node not in _admin_refs:
        import firebase_admin
        from firebase_admin import credentials, db
        if not os.path.exists(FIREBASE_CREDENTIALS):
            raise RuntimeError('service-account JSON not found')
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS)
            firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
        _admin_refs[node] = db.reference(node)
    return _admin_refs[node]


def _admin_write(payload, node=FIREBASE_NODE, append=False):
    ref = _admin_reference(node)
    if append:
        ref.push(payload)
    else:
        ref.set(payload)


def _admin_delete():
    try:
        if FIREBASE_NODE in _admin_refs:
            _admin_refs[FIREBASE_NODE].delete()
        else:
            _rest_delete()
    except Exception:
        _rest_delete()


def _push(payload, node=FIREBASE_NODE, append=False):
    """Try Admin SDK, fall back to REST. Returns (ok, mode_or_error)."""
    try:
        _admin_write(payload, node=node, append=append)
        return True, 'admin-sdk'
    except Exception as e1:
        try:
            _rest_write(payload, node=node, append=append)
            return True, f'rest (admin failed: {e1})'
        except Exception as e2:
            return False, f'admin: {e1} | rest: {e2}'


def _expire(expected_seq):
    with _state_lock:
        global _timer
        if expected_seq != _seq:
            return
        _timer = None
    try:
        _admin_delete()
    except Exception:
        pass


def _queue_push(question, answer, archive, mode=None, emotion_history_type=None):
    """Queue a live LCD update and, for chat replies, an archive record."""
    global _seq, _timer, _last_push, _last_error
    global _last_archive_push, _last_archive_error
    global _last_emotion_history_record_push, _last_emotion_history_record_error
    global _last_voice_history_push, _last_voice_history_error
    global _last_nexus_text
    if not _enabled:
        return
    question = (question or '').strip()
    answer = (answer or '').strip()
    if not answer:
        return
    lcd_question = _shorten(question, Q_MAX)
    lcd_answer = _shorten(answer, A_MAX)
    lines = _lcd_lines(f'VOICE: {lcd_answer}' if mode == 'voice' else lcd_answer)
    now = int(time.time())
    with _state_lock:
        _seq += 1
        seq = _seq
        if _timer is not None:
            _timer.cancel()
        payload = {'seq': seq, 'q': lcd_question, 'a': lcd_answer,
                   'ts': now, 'ttl': FIREBASE_TTL, 'expires_at': now + FIREBASE_TTL,
                   'status': 'new'}
        if mode:
            payload['mode'] = mode
        _last_nexus_text = lcd_answer
        for i, ln in enumerate(lines, 1):
            payload[f'line{i}'] = ln
        archive_payload = {'q': question, 'a': answer, 'ts': now}
        if mode:
            archive_payload['mode'] = mode
        if not archive:
            archive_payload = None
        emotion_history_payload = None
        if emotion_history_type:
            emotion_history_payload = {
                'type': emotion_history_type,
                'source': 'nexus_history',
                'q': question,
                'a': answer,
                'ts': now,
            }
            if mode:
                emotion_history_payload['mode'] = mode
        _timer = threading.Timer(FIREBASE_TTL, _expire, args=(seq,))
        _timer.daemon = True
        _timer.start()

    def _bg():
        global _last_push, _last_error, _last_archive_push, _last_archive_error
        global _last_emotion_history_record_push, _last_emotion_history_record_error
        global _last_voice_history_push, _last_voice_history_error
        ok, info = _push(payload)
        with _state_lock:
            if ok:
                _last_push = {'seq': seq, 'ts': now, 'mode': info}
                _last_error = None
            else:
                _last_error = info
        if archive_payload is not None:
            archived, archive_info = _push(
                archive_payload, node=FIREBASE_HISTORY_NODE, append=True)
            with _state_lock:
                if archived:
                    _last_archive_push = {'ts': now, 'mode': archive_info}
                    _last_archive_error = None
                else:
                    _last_archive_error = archive_info
        if emotion_history_payload is not None:
            archived, archive_info = _push(
                emotion_history_payload, node=FIREBASE_EMOTION_HISTORY_NODE, append=True)
            with _state_lock:
                if archived:
                    _last_emotion_history_record_push = {'ts': now, 'mode': archive_info}
                    _last_emotion_history_record_error = None
                    if mode == 'voice':
                        _last_voice_history_push = _last_emotion_history_record_push
                        _last_voice_history_error = None
                else:
                    _last_emotion_history_record_error = archive_info
                    if mode == 'voice':
                        _last_voice_history_error = archive_info

    t = threading.Thread(target=_bg, daemon=True)
    t.start()


def push_qa(question, answer):
    """Store a chat turn in emotion history and publish a temporary LCD update."""
    _queue_push(
        question, answer, archive=False, emotion_history_type='nexus_history')


def push_voice_qa(question, answer):
    """Queue a tagged voice record in both histories and the temporary LCD node."""
    _queue_push(
        question, answer, archive=True, mode='voice', emotion_history_type='voice')


def push_text(text):
    """Queue a temporary LCD update without adding a chat-history record."""
    _queue_push('', text, archive=False)


def push_emotion(snapshot):
    """Queue the latest emotion state and archive each distinct state."""
    global _last_emotion_push, _last_emotion_error
    global _last_emotion_history_push, _last_emotion_history_error
    global _last_emotion_history_signature
    if not _enabled:
        return False

    snapshot = dict(snapshot)
    emotions = sorted(face['emotion'] for face in snapshot.get('faces', []))
    signature = json.dumps({
        'enabled': snapshot.get('enabled'),
        'person_count': snapshot.get('person_count'),
        'emotions': emotions,
    }, sort_keys=True)
    with _state_lock:
        snapshot['nexus_text'] = _last_nexus_text
        archive_state = (
            signature != _last_emotion_history_signature
            and signature not in _pending_emotion_history_signatures
        )
        if archive_state:
            _pending_emotion_history_signatures.add(signature)

    def _bg():
        global _last_emotion_push, _last_emotion_error
        global _last_emotion_history_push, _last_emotion_history_error
        global _last_emotion_history_signature
        ok, info = _push(snapshot, node=FIREBASE_EMOTION_NODE)
        with _state_lock:
            if ok:
                _last_emotion_push = {'ts': snapshot['ts'], 'mode': info}
                _last_emotion_error = None
            else:
                _last_emotion_error = info
        if archive_state:
            archived, archive_info = _push(
                snapshot, node=FIREBASE_EMOTION_HISTORY_NODE, append=True)
            with _state_lock:
                _pending_emotion_history_signatures.discard(signature)
                if archived:
                    _last_emotion_history_signature = signature
                    _last_emotion_history_push = {
                        'ts': snapshot['ts'], 'mode': archive_info}
                    _last_emotion_history_error = None
                else:
                    _last_emotion_history_error = archive_info

    threading.Thread(target=_bg, daemon=True).start()
    return True


def set_enabled(on):
    global _enabled
    _enabled = bool(on)
    return _enabled


def rtdb_status():
    with _state_lock:
        return {'enabled': _enabled, 'url': FIREBASE_URL, 'node': FIREBASE_NODE,
                'history_node': FIREBASE_HISTORY_NODE,
                'ttl': FIREBASE_TTL, 'lcd': f'{LCD_COLS}x{LCD_ROWS}',
                'backend': 'admin-sdk' if os.path.exists(FIREBASE_CREDENTIALS) else 'rest',
                'last_push': _last_push, 'last_error': _last_error,
                'last_archive_push': _last_archive_push,
                'last_archive_error': _last_archive_error,
                'emotion_node': FIREBASE_EMOTION_NODE,
                'last_emotion_push': _last_emotion_push,
                'last_emotion_error': _last_emotion_error,
                'emotion_history_node': FIREBASE_EMOTION_HISTORY_NODE,
                'last_emotion_history_push': _last_emotion_history_push,
                'last_emotion_history_error': _last_emotion_history_error,
                'last_emotion_history_record_push': _last_emotion_history_record_push,
                'last_emotion_history_record_error': _last_emotion_history_record_error,
                'last_voice_history_push': _last_voice_history_push,
                'last_voice_history_error': _last_voice_history_error}
