"""How the voice reads stretched letter sounds (sss, fff, shhh...), and the same sounds written as phonemes.
Prints both, and records each to samples/sounds/ with a page to listen (samples/sounds/index.html)."""
import html, os, subprocess, tempfile
import numpy as np, soundfile as sf
from kokoro import KPipeline
SOUNDS = [('sss', 'sss'), ('fff', 'fff'), ('rrr', 'ɹɹɹ'), ('zzz', 'zzz'), ('mmm', 'mmm'), ('nnn', 'nnn'), ('vvv', 'vvv'), ('thhh', 'θθθ'),
          ('shh', 'ʃʃ'), ('shhh', 'ʃʃʃ'), ('lll', 'lll'), ('eee', 'iii'), ('ooo', 'uuu'), ('aaa', 'æææ'), ('iii', 'ɪɪɪ')]
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
os.makedirs('samples/sounds', exist_ok=True); tmp = os.path.join(tempfile.gettempdir(), 's.wav'); rows = []
for word, ph in SOUNDS:
    cells = []
    for tag, text in (('now', f'Its sound is {word}. Listen: {word}.'), ('fixed', f'Its sound is [{word}](/{ph}/). Listen: [{word}](/{ph}/).')):
        said, chunks = [], []
        for _, ps, a in pipe(text, voice='af_heart', speed=0.9): said.append(ps); chunks.append(np.asarray(a))
        name = f'{word}-{tag}.m4a'; sf.write(tmp, np.concatenate(chunks), 24000)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '48k', '-ac', '1', '-f', 'mp4', f'samples/sounds/{name}'], check=True)
        print('said', tag, repr(text), '->', ' | '.join(said), flush=True)
        cells.append(f'<td><audio controls preload="none" src="{name}"></audio></td>')
    rows.append(f'<tr><th>{html.escape(word)}</th>{"".join(cells)}</tr>')
open('samples/sounds/index.html', 'w').write('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Letter sounds</title><style>body{font:16px system-ui;margin:16px}td,th{border:1px solid #ddd;padding:6px}</style><h1>Letter sounds: now and fixed</h1><table><tr><th>Sound</th><th>Now</th><th>Fixed</th></tr>' + ''.join(rows) + '</table>')
