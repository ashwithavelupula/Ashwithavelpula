#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline LLM engine with pluggable backends.

Backend selection (env LLM_BACKEND = auto|ollama|gpt4all|llama_cpp|local):
  1. ollama     : talks to a local Ollama server (http://127.0.0.1:11434)
       using the first installed model. Zero setup here if Ollama runs.
  2. gpt4all    : loads a GGUF model from models/ via the gpt4all package
       (prebuilt wheels, no compiler).
  3. llama_cpp  : loads a GGUF model from models/ via llama-cpp-python.
     - Set LLM_MODEL_FILE to pick a specific file, else the first *.gguf
       (sorted by size, smallest first) is used. Drop any GGUF in models/
       to swap models without code changes.
  4. local      : built-in offline brain (intents, math, knowledge base,
       extractive summarizer). Always available, zero dependencies.

All chat turns are also written to MySQL (via the caller) for history/analytics.
"""

import os
import re
import glob
import math
import random
import secrets
import string
import threading
import ast
from datetime import datetime

STOP_SEQUENCES = ['<|user|>', 'User:', '###']
PROMPT_MAX_CHARS = 1500


def truncate_prompt(text, limit=PROMPT_MAX_CHARS):
    """Truncate text to `limit` chars at a word boundary."""
    if text is None:
        return ''
    s = str(text)
    if len(s) <= limit:
        return s
    cut = s[:limit]
    idx = cut.rfind(' ')
    if idx > 0:
        return cut[:idx]
    return cut


def format_answer_bullets(text):
    """Render conversational answers as concise, individually readable points."""
    if not isinstance(text, str):
        return text

    output = []
    in_code_block = False
    for line in text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            if output and output[-1] != '':
                output.append('')
            continue
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            output.append(stripped)
            continue
        if in_code_block or re.match(r'^(?:[-*+]\s+|\d+[.)]\s+)', stripped):
            output.append(stripped)
            continue
        if stripped.startswith('# '):
            output.append(stripped)
            continue

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', stripped)
        output.extend(f'- {sentence.strip()}' for sentence in sentences if sentence.strip())

    while output and output[-1] == '':
        output.pop()
    return '\n'.join(output)


def clamp_output(text):
    """Clamp output to end at the last sentence boundary ('.!?').

    Appends ' …' when trailing text was removed. Short texts without
    any sentence boundary are returned unchanged.
    """
    if not text:
        return text
    s = text.rstrip()
    if not s:
        return s
    if s[-1] in '.!?':
        return s
    idx = max(s.rfind('.'), s.rfind('!'), s.rfind('?'))
    if idx == -1:
        return s
    return s[:idx + 1].rstrip() + ' …'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LLM_BACKEND = os.getenv('LLM_BACKEND', 'auto').lower()
LLM_MODEL_FILE = os.getenv('LLM_MODEL_FILE', '')
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '256'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
PROMPT_MAX_CHARS = int(os.getenv('PROMPT_MAX_CHARS', '1500'))


def get_llm_params():
    return {'temperature': LLM_TEMPERATURE, 'max_tokens': LLM_MAX_TOKENS}


def set_llm_params(temperature=None, max_tokens=None):
    """Validate + apply runtime LLM params. Returns (params, error)."""
    global LLM_TEMPERATURE, LLM_MAX_TOKENS
    if temperature is not None:
        try:
            temperature = float(temperature)
        except (ValueError, TypeError):
            return None, 'temperature must be a number'
        if not 0.0 <= temperature <= 1.5:
            return None, 'temperature must be 0.0-1.5'
        LLM_TEMPERATURE = temperature
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (ValueError, TypeError):
            return None, 'max_tokens must be an integer'
        if not 32 <= max_tokens <= 512:
            return None, 'max_tokens must be 32-512'
        LLM_MAX_TOKENS = max_tokens
    return get_llm_params(), None


def find_gguf_models():
    """Return sorted list of *.gguf files in models/ (smallest first)."""
    files = glob.glob(os.path.join(MODEL_DIR, '*.gguf'))
    try:
        files.sort(key=lambda p: os.path.getsize(p))
    except OSError:
        files.sort()
    return files


def _rt(temperature=None, max_tokens=None):
    """Resolve per-request overrides against validated globals."""
    try:
        t = LLM_TEMPERATURE if temperature is None else float(temperature)
    except (ValueError, TypeError):
        t = LLM_TEMPERATURE
    try:
        n = LLM_MAX_TOKENS if max_tokens is None else int(max_tokens)
    except (ValueError, TypeError):
        n = LLM_MAX_TOKENS
    t = min(1.5, max(0.0, t))
    n = min(512, max(32, n))
    return t, n


class OllamaBackend:
    """Real offline LLM via a local Ollama server (http://127.0.0.1:11434)."""

    def __init__(self, model):
        import json as _json
        import urllib.request as _url
        self.model = model
        self._json = _json
        self._url = _url

    def generate(self, prompt, system=None, temperature=None, max_tokens=None):
        temp, ntok = _rt(temperature, max_tokens)
        payload = self._json.dumps({
            'model': self.model,
            'prompt': ((system + '\n\n' if system else '') + truncate_prompt(prompt)),
            'stream': False,
            'options': {'temperature': temp, 'num_predict': ntok,
                        'repeat_penalty': 1.1, 'stop': STOP_SEQUENCES},
        }).encode()
        req = self._url.Request('http://127.0.0.1:11434/api/generate', data=payload,
                                headers={'Content-Type': 'application/json'})
        with self._url.urlopen(req, timeout=120) as r:
            return clamp_output(self._json.load(r)['response'].strip())


def find_ollama_model():
    """Return the first installed Ollama model name, or None if Ollama is absent."""
    try:
        import json as _json
        import urllib.request as _url
        with _url.urlopen('http://127.0.0.1:11434/api/tags', timeout=3) as r:
            tags = _json.load(r).get('models', [])
        return tags[0]['name'] if tags else None
    except Exception:
        return None


ASSISTANT_SYSTEM = (
    'You are Nexus, a friendly offline assistant. Keep replies short and easy to scan. '
    'Present answers as concise Markdown bullet points, with one key point per bullet. '
    'Avoid prose paragraphs; use a short heading only when it helps.'
)


class GPT4AllBackend:
    """Real offline LLM via the gpt4all package (prebuilt wheels, no compiler)."""

    def __init__(self, model_path):
        from gpt4all import GPT4All
        self.llm = GPT4All(model_path)
        self.model_path = model_path

    def _prompt(self, prompt, system):
        sys = system or ASSISTANT_SYSTEM
        return f"<|system|>\n{sys}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"

    ECHO_RE = re.compile(
        r'^\s*as an ai language model\b|special tokens?|system instructions?',
        re.IGNORECASE)
    SIGNOFF_RE = re.compile(
        r'(\n\s*(best regards|sincerely|kind regards)[^\n]*(\n\s*\[[^\n\]]+\])?|\n\s*\[[^\n\]]{0,40}\])\s*$',
        re.IGNORECASE)

    @classmethod
    def _clean(cls, text, clamp=True):
        if '<|assistant|>' in text:
            text = text.rsplit('<|assistant|>', 1)[-1]
        text = re.split(r'<\|user\|>', text, flags=re.IGNORECASE)[0]
        text = re.split(r'\n\s*user\s*:', text, flags=re.IGNORECASE)[0]
        text = re.sub(r'</?\s*s\s*>', '', text)
        text = re.sub(r'<\|.*?\|>', '', text)
        text = text.split('###')[0]
        text = cls.SIGNOFF_RE.sub('', text)
        text = text.replace('[Your Name]', '').replace('[your name]', '')
        kept = [s for s in re.split(r'(?<=[.!?])\s+', text)
                if s.strip() and not cls.ECHO_RE.search(s)]
        text = ' '.join(kept) if kept else text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        if clamp:
            text = clamp_output(text)
        return text

    def generate(self, prompt, system=None, temperature=None, max_tokens=None):
        prompt = truncate_prompt(prompt)
        temp, ntok = _rt(temperature, max_tokens)
        try:
            out = self.llm.generate(
                self._prompt(prompt, system),
                max_tokens=ntok, temp=temp,
                top_p=0.9, top_k=40, repeat_penalty=1.1,
                stop=STOP_SEQUENCES,
            )
        except TypeError:
            out = self.llm.generate(
                self._prompt(prompt, system),
                max_tokens=ntok, temp=temp,
                top_p=0.9, top_k=40, repeat_penalty=1.1,
            )
        if isinstance(out, str):
            return self._clean(out)
        return self._clean(''.join(out))

    def stream(self, prompt, system=None, temperature=None, max_tokens=None):
        """Yield cumulative cleaned snapshots as generation progresses.

        Snapshots (not deltas) because the runtime's streamed token pieces do
        not preserve leading whitespace; cleaning the whole prefix every time
        keeps spacing exact. The UI replaces the bubble text per snapshot.
        """
        prompt = truncate_prompt(prompt)
        temp, ntok = _rt(temperature, max_tokens)
        try:
            try:
                gen = self.llm.generate(
                    self._prompt(prompt, system),
                    max_tokens=ntok, temp=temp,
                    top_p=0.9, top_k=40, repeat_penalty=1.1, streaming=True,
                    stop=STOP_SEQUENCES,
                )
            except TypeError:
                gen = self.llm.generate(
                    self._prompt(prompt, system),
                    max_tokens=ntok, temp=temp,
                    top_p=0.9, top_k=40, repeat_penalty=1.1, streaming=True,
                )
            if isinstance(gen, str):
                yield self._clean(gen)
                return
            raw = ''
            last = ''
            for tok in gen:
                raw += tok
                if len(raw) - len(last) >= 8 or any(c in tok for c in '.!?\n'):
                    snap = self._clean(raw, clamp=False)
                    if snap and snap != last:
                        last = snap
                        yield snap
            final = self._clean(raw)
            if final and final != last:
                yield final
        except TypeError:
            yield self.generate(prompt, system=system)


class LlamaCppBackend:
    """Real offline LLM via llama-cpp-python."""

    def __init__(self, model_path):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=int(os.getenv('LLM_CTX', '2048')),
            n_threads=int(os.getenv('LLM_THREADS', str(os.cpu_count() or 4))),
            verbose=False,
        )
        self.model_path = model_path

    def generate(self, prompt, system=None, temperature=None, max_tokens=None):
        prompt = truncate_prompt(prompt)
        temp, ntok = _rt(temperature, max_tokens)
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=ntok,
            temperature=temp,
            stop=STOP_SEQUENCES,
            repeat_penalty=1.1,
        )
        return clamp_output(out['choices'][0]['message']['content'].strip())


class LocalBrain:
    """Built-in offline assistant: intents + knowledge + tools."""

    KNOWLEDGE = {
        'object detection': (
            "I use an offline MobileNet SSD model that recognises 20 everyday classes: "
            "person, car, bus, bicycle, motorbike, aeroplane, boat, train, bird, cat, dog, "
            "horse, sheep, cow, bottle, chair, sofa, dining table, potted plant and TV/monitor. "
            "Use the Object button and upload a photo."
        ),
        'face': (
            "Face detection runs fully offline with Haar cascades. You can detect faces in a "
            "photo with the Faces button, and store a face under a name with Register Face "
            "for later recognition."
        ),
        'voice': (
            "Voice mode uses your browser microphone: press the mic button, speak, and I will "
            "transcribe and answer out loud. Grant microphone permission when asked."
        ),
        'offline': (
            "Everything core runs offline on your machine: chat, object detection, face "
            "detection, image tools and smart tools. Only an optional Firebase layer would "
            "need the internet."
        ),
        'mysql': (
            "Chat history, registered faces, logs and analytics are stored in MySQL tables: "
            "users, chat_history, face_registry, analytics and system_logs."
        ),
        'model': (
            "The LLM backend auto-loads any .gguf file placed in the models folder (needs the "
            "llama-cpp runtime), or connects to Ollama automatically if it is running with a "
            "model installed. To swap models, drop in another GGUF and restart, or install "
            "Ollama and pull any model."
        ),
    }

    def try_direct(self, prompt):
        """Answer simple intents deterministically; return None to use the model."""
        msg = (prompt or '').strip()
        low = msg.lower()
        if not low:
            return "Please type a message or use one of the feature buttons."
        low_ns = re.sub(r'[^\w\s]', '', low).strip()
        if re.match(r'(hi+|hello|hey|yo|namaste|good\s*(morning|afternoon|evening))\b', low):
            return ("Hi there! I'm Nexus, your offline assistant. Ask me anything, or try the "
                    "buttons above: Objects, Faces, Image Tools, Smart Tools.")
        if low_ns in ('who are you', 'what are you', 'your name', 'about yourself',
                      'who r u', 'what r you'):
            return ("I'm Nexus, an offline assistant running right here on your machine. I can chat, "
                    "detect objects and faces in photos, edit images, calculate, convert units and more.")
        if re.search(r'\bthank(s| you)\b', low):
            return "You're welcome! Anything else I can help with?"
        if re.match(r'(bye|goodbye|good ?night|see you|take care)\b', low):
            return "Goodbye! I'll be here whenever you need me."
        if low_ns in ('ok', 'okay', 'k', 'got it', 'gotit', 'fine', 'cool', 'nice'):
            return "Got it! Let me know if you need anything else."
        if any(k in low for k in ('help', 'what can you do', 'features', 'options')):
            return (
                "I can: 1) Chat and answer questions, 2) Detect objects in photos, "
                "3) Detect faces, 4) Register faces by name, 5) Edit images "
                "(grayscale, resize, edges), 6) Calculate, convert units, generate "
                "passwords, analyse text, 7) Talk via voice mode. Pick a button above."
            )
        if 'time' in low and any(w in low for w in ('what', 'current', 'now', 'time is')):
            return "Current time is " + datetime.now().strftime('%I:%M %p on %A, %d %B %Y') + "."
        if 'date' in low and any(w in low for w in ('what', 'today', 'current')):
            return "Today is " + datetime.now().strftime('%A, %d %B %Y') + "."
        m = re.search(r'(?:calc(?:ulate)?\s*)?([0-9\.\s\+\-\*\/\(\)%\^=]+)$', low)
        if m:
            raw = m.group(1)
            # Only auto-calculate on explicit intent: 'calc' keyword or '='.
            # Bare '10/10' or '12-30' is left for the model (avoids hijacking chat).
            if ('calc' in low) or ('=' in raw):
                expr = raw.replace('=', '').strip()
                if expr and re.search(r'\d', expr):
                    try:
                        return f"Result: {safe_calc(expr)}"
                    except ValueError:
                        pass
        for key, answer in self.KNOWLEDGE.items():
            if key in low:
                return answer
        return None

    def generate(self, prompt, system=None, context=None):
        direct = self.try_direct(prompt or '')
        if direct is not None and not context:
            return direct
        msg = (prompt or '').strip()
        if context:
            return context + "\n\n" + self._fallback(msg)
        return self._fallback(msg)

    def _fallback(self, msg):
        words = len(msg.split())
        return (
            f"You said ({words} words): \"{msg[:200]}\". I am running in built-in offline mode "
            "right now. For deeper reasoning, drop any GGUF model file into the models folder "
            "and restart the server — I will switch to it automatically. Meanwhile try the "
            "feature buttons: Objects, Faces, Image Tools or Smart Tools."
        )


# ---------------------------------------------------------------- tools

def safe_calc(expr):
    """Evaluate a basic arithmetic expression safely."""
    expr = str(expr).strip().replace('^', '**').replace(' ', '')
    if len(expr) > 200 or len(expr) == 0:
        raise ValueError('Not a math expression')
    if not re.fullmatch(r'[0-9\.\+\-\*\/\(\)%]+', expr) or not re.search(r'\d', expr):
        raise ValueError('Not a math expression')
    tree = ast.parse(expr, mode='eval')
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Num, ast.Constant, ast.Add,
        ast.Sub, ast.Mult, ast.Div,
        ast.Mod, ast.Pow, ast.USub,
        ast.UAdd, ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed + (ast.Expr, ast.Module)):
            raise ValueError('Not a math expression')
    # Power guard: reject huge exponents that would hang the CPU (e.g. 9**9**9)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp = node.right
            if not (isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float))
                    and abs(exp.value) <= 30):
                raise ValueError('Exponent too large')
    result = eval(compile(tree, '<calc>', 'eval'), {'__builtins__': {}}, {})  # noqa: S307 - sandboxed above
    if isinstance(result, (int, float)) and abs(result) == float('inf'):
        raise ValueError('Result out of range')
    if isinstance(result, int) and result.bit_length() > 4096:
        raise ValueError('Result out of range')
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return result


LENGTH = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.0, 'in': 0.0254, 'ft': 0.3048,
          'yd': 0.9144, 'mi': 1609.344}
WEIGHT = {'mg': 0.001, 'g': 1.0, 'kg': 1000.0, 'oz': 28.349523125, 'lb': 453.59237}
TEMP = {'c', 'f', 'k'}


def convert_units(value, frm, to):
    frm, to = frm.lower(), to.lower()
    if frm in LENGTH and to in LENGTH:
        return value * LENGTH[frm] / LENGTH[to]
    if frm in WEIGHT and to in WEIGHT:
        return value * WEIGHT[frm] / WEIGHT[to]
    if frm in TEMP and to in TEMP:
        c = {'c': value, 'f': (value - 32) * 5 / 9, 'k': value - 273.15}[frm]
        return {'c': c, 'f': c * 9 / 5 + 32, 'k': c + 273.15}[to]
    raise ValueError(f'Cannot convert {frm} to {to}')


def password_generate(length=16, symbols=True):
    alphabet = string.ascii_letters + string.digits + ('!@#$%^&*-_+' if symbols else '')
    return ''.join(secrets.choice(alphabet) for _ in range(max(4, min(int(length), 128))))


def text_analyze(text):
    words = re.findall(r"[A-Za-z0-9']+", text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    chars = len(text)
    freq = {}
    for w in words:
        w = w.lower()
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        'characters': chars,
        'words': len(words),
        'sentences': len(sentences),
        'avg_word_length': round(sum(len(w) for w in words) / len(words), 2) if words else 0,
        'top_words': [{'word': w, 'count': c} for w, c in top],
        'reading_time_sec': round(len(words) / 200 * 60, 1),
    }


def doc_qa(doc_text, question, top_n=3):
    """Extractive Q&A over an offline document: returns best matching sentences.

    Returns dict(answer, quotes[{sentence, score}], truncated).
    """
    MAX_DOC = 200 * 1024
    truncated = len(doc_text) > MAX_DOC
    if truncated:
        doc_text = doc_text[:MAX_DOC]
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', doc_text.strip()) if s.strip()]
    if not sentences:
        raise ValueError('Document is empty')
    qwords = [w for w in re.findall(r'[a-z]+', question.lower()) if len(w) >= 3]
    if not qwords:
        raise ValueError('Question has no searchable words')
    scored = []
    n = len(sentences)
    for i, s in enumerate(sentences):
        words = set(re.findall(r'[a-z]+', s.lower()))
        overlap = sum(1 for q in qwords if q in words)
        if overlap:
            bonus = 0.3 if i < 3 else 0.0  # intro sentences often define terms
            scored.append((overlap + bonus, i, s))
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:max(1, min(int(top_n), 5))]
    if not top:
        return {'answer': summarize(doc_text, 3),
                'quotes': [],
                'truncated': truncated,
                'notice': 'No direct match — here is a summary instead.'}
    quotes = [{'sentence': s, 'score': round(sc, 2)} for sc, _, s in top]
    idx = sorted(i for _, i, _ in top)
    answer = ' '.join(sentences[i] for i in idx)
    return {'answer': answer, 'quotes': quotes, 'truncated': truncated}


def summarize(text, max_sentences=3):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return text.strip()
    freq = {}
    for w in re.findall(r'[a-z]+', text.lower()):
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    scored = sorted(
        enumerate(sentences),
        key=lambda p: sum(freq.get(w, 0) for w in re.findall(r'[a-z]+', p[1].lower())),
        reverse=True,
    )
    keep = sorted(i for i, _ in scored[:max_sentences])
    return ' '.join(sentences[i] for i in keep)


# ---------------------------------------------------------------- engine

class Engine:
    def __init__(self):
        self.backend_name = 'local'
        self.backend = LocalBrain()
        self.model_path = None
        self._lock = threading.Lock()
        # 1) Ollama server with an installed model = real LLM, zero setup here
        if LLM_BACKEND in ('auto', 'ollama'):
            model = find_ollama_model()
            if model:
                try:
                    self.backend = OllamaBackend(model)
                    self.backend_name = 'ollama:' + model
                    print(f"[INFO] LLM backend: ollama ({model})")
                    return
                except Exception as e:
                    print(f"[WARN] ollama backend failed ({e})")
        # 2) Local GGUF via gpt4all (prebuilt) or llama-cpp-python
        if LLM_BACKEND in ('auto', 'gpt4all', 'llama_cpp'):
            candidates = [LLM_MODEL_FILE] if LLM_MODEL_FILE else find_gguf_models()
            candidates = [c for c in candidates if c and os.path.exists(c)
                          and os.path.getsize(c) > 50 * 1024 * 1024]
            if candidates and LLM_BACKEND in ('auto', 'gpt4all'):
                try:
                    self.backend = GPT4AllBackend(candidates[0])
                    self.backend_name = 'gpt4all'
                    self.model_path = candidates[0]
                    print(f"[INFO] LLM backend: gpt4all ({os.path.basename(candidates[0])})")
                    return
                except Exception as e:
                    print(f"[WARN] gpt4all load failed ({e})")
            if candidates and LLM_BACKEND in ('auto', 'llama_cpp'):
                try:
                    self.backend = LlamaCppBackend(candidates[0])
                    self.backend_name = 'llama_cpp'
                    self.model_path = candidates[0]
                    print(f"[INFO] LLM backend: llama_cpp ({os.path.basename(candidates[0])})")
                except Exception as e:
                    print(f"[WARN] llama_cpp load failed ({e}); using local brain")
        print(f"[INFO] LLM backend active: {self.backend_name}")

    def chat(self, prompt, system=None, context=None, temperature=None, max_tokens=None):
        try:
            if isinstance(self.backend, LocalBrain):
                reply = self.backend.generate(prompt, system=system, context=context)
                return format_answer_bullets(reply)
            if not context:
                direct = LocalBrain().try_direct(prompt or '')
                if direct is not None:
                    return format_answer_bullets(direct)
            enriched = truncate_prompt(prompt if not context else f"{context}\n\nUser: {prompt}")
            with self._lock:
                reply = self.backend.generate(enriched, system=system or ASSISTANT_SYSTEM,
                                              temperature=temperature, max_tokens=max_tokens)
            reply = clamp_output(reply.strip()) if isinstance(reply, str) else reply
            return format_answer_bullets(reply)
        except Exception as e:
            print(f"[ERROR] LLM generate failed: {e}")
            return format_answer_bullets(LocalBrain().generate(prompt, context=context))

    def stream(self, prompt, system=None, context=None, temperature=None, max_tokens=None):
        """Yield reply chunks; falls back to a single chunk for non-streaming backends."""
        if hasattr(self.backend, 'stream'):
            try:
                if not context:
                    direct = LocalBrain().try_direct(prompt or '')
                    if direct is not None:
                        yield format_answer_bullets(direct)
                        return
                enriched = truncate_prompt(prompt if not context else f"{context}\n\nUser: {prompt}")
                self._lock.acquire()
                try:
                    for chunk in self.backend.stream(enriched, system=system or ASSISTANT_SYSTEM,
                                                     temperature=temperature, max_tokens=max_tokens):
                        yield format_answer_bullets(chunk)
                finally:
                    self._lock.release()
                return
            except Exception as e:
                print(f"[ERROR] LLM stream failed: {e}")
        yield self.chat(prompt, system=system, context=context,
                        temperature=temperature, max_tokens=max_tokens)

    def info(self):
        return {
            'backend': self.backend_name,
            'model': os.path.basename(self.model_path) if self.model_path else None,
            'models_available': [os.path.basename(p) for p in find_gguf_models()],
        }


engine = Engine()
