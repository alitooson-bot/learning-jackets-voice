"""Voice samples for the family to compare (speeds and voices), plus the sounds the voice reads for each sentence.
Writes samples/<voice>-<speed>-<n>.m4a and samples/index.html (GitHub Pages: /learning-jackets-voice/samples/)."""
import html, os, subprocess, tempfile
import numpy as np, soundfile as sf
from kokoro import KPipeline

SENTENCES = [
    'Now pick your goal for today. Learn a new letter sound. Finish today’s path. Or be a helper at home.',
    'This is s. Its sound is sss. Listen: snake. snake starts with that sound.',
    'snake',
    'Snake.',
    'Snake starts with sss. Moon starts with mmm.',
    'Great job! Tap the big green button to keep going.',
]
VOICES = [('af_heart', 1.0, 'Now: Heart, normal speed'), ('af_heart', 0.9, 'Heart, a little slower'), ('af_heart', 0.85, 'Heart, slower'),
          ('af_heart', 0.8, 'Heart, slowest'), ('af_bella', 0.85, 'Bella, slower'), ('af_nicole', 0.85, 'Nicole (soft), slower'),
          ('am_michael', 0.85, 'Michael (man), slower'), ('bf_emma', 0.85, 'Emma (British), slower')]
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
brit = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M', device='cpu')
os.makedirs('samples', exist_ok=True)
tmp = os.path.join(tempfile.gettempdir(), 'sample.wav')
rows = []
for voice, speed, label in VOICES:
    cells = []
    for n, text in enumerate(SENTENCES):
        p = brit if voice.startswith('b') else pipe
        said, chunks = [], []
        for _, ps, a in p(text, voice=voice, speed=speed): said.append(ps); chunks.append(np.asarray(a))
        name = f'{voice}-{speed}-{n}.m4a'
        sf.write(tmp, np.concatenate(chunks), 24000)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '48k', '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', os.path.join('samples', name)], check=True)
        print('said', voice, speed, repr(text), '->', ' | '.join(said), flush=True)
        cells.append(f'<td><audio controls preload="none" src="{name}"></audio></td>')
    rows.append(f'<tr><th>{html.escape(label)}</th>{"".join(cells)}</tr>')
head = ''.join(f'<th>{html.escape(t)}</th>' for t in SENTENCES)
open('samples/index.html', 'w').write(f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jax voice samples</title><style>body{{font:16px system-ui;margin:16px;background:#fffdf5;color:#2b2d42}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px;vertical-align:top;text-align:left}}th{{font-size:14px;max-width:220px}}audio{{width:150px}}.wrap{{overflow-x:auto}}</style>
<h1>Jax voice samples</h1><p>Each row is one voice and speed; each column is a sentence from Noah’s lessons. The first row is the voice the app uses now. Pick the row that sounds clearest for a 4-year-old.</p>
<div class="wrap"><table><tr><th>Voice</th>{head}</tr>{"".join(rows)}</table></div>''')
