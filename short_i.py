"""Versions of Jax's (Heart) short i sound for the family to pick from (parent's sound check, 2026-10-12: "Meet i sounds
like it's saying e"). Writes samples/short-i/<n>.m4a and samples/short-i/index.html, and prints the sounds read."""
import html, os, subprocess, tempfile
import numpy as np, soundfile as sf
from kokoro import KPipeline

SPEED = 0.8  # the speed the app's recordings use (rerecord.py)
WAYS = [('ɪ', 'Now'), ('ˈɪ', 'Stressed'), ('ˈɪh', 'With a breath after'), ('ˈɪʔ', 'Cut off short'), ('ˈɪɪ', 'Held a little longer'), ('ˈɪˈɪ', 'Said twice, ih ih')]
LINE = 'It says [ih](/{p}/), like the start of insect. Say it with me: [ih](/{p}/).'
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
os.makedirs('samples/short-i', exist_ok=True)
tmp = os.path.join(tempfile.gettempdir(), 'i.wav')
rows = []
for n, (p, label) in enumerate(WAYS, 1):
    said, chunks = [], []
    for _, ps, a in pipe(LINE.format(p=p), voice='af_heart', speed=SPEED): said.append(ps); chunks.append(np.asarray(a))
    sf.write(tmp, np.concatenate(chunks), 24000)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '48k', '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', f'samples/short-i/{n}.m4a'], check=True)
    print('said', n, p, '->', ' | '.join(said), flush=True)
    rows.append(f'<tr><th>{n}. {html.escape(label)}</th><td><audio controls preload="none" src="{n}.m4a"></audio></td></tr>')
open('samples/short-i/index.html', 'w').write('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Short i</title><style>body{font:16px system-ui;margin:16px}td,th{border:1px solid #ddd;padding:6px;text-align:left}</style><h1>Jax\'s short i</h1><p>Which one says ih, like the start of insect?</p><table>' + ''.join(rows) + '</table>')
