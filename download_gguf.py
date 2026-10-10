"""Resumable GGUF download with retries. Run: python download_gguf.py"""
import os
import time
import urllib.request

URL = ('https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/'
       'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
OUT = os.path.join('models', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')
EXPECTED = 668788096

for attempt in range(1, 30):
    have = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    if have >= EXPECTED:
        print(f'DONE {have} bytes', flush=True)
        break
    req = urllib.request.Request(URL, headers={'Range': f'bytes={have}-'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(OUT, 'ab') as f:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        print(f'progress: {os.path.getsize(OUT)}/{EXPECTED}', flush=True)
    except Exception as e:
        print(f'attempt {attempt}: {e} (have {os.path.getsize(OUT) if os.path.exists(OUT) else 0})',
              flush=True)
        time.sleep(5)
else:
    print('FAILED after retries', flush=True)
