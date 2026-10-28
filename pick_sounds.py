"""Versions of Jax's (Heart) short i and n sounds for the family to pick from (Ali, 2026-10-28: the recorded "ih" and
stretched "nnn" in Sound it out still sounded wrong). Each version is said alone and in "Which letters say ...?".
Writes samples/pick/<sound>-<n>-{alone,line}.m4a, samples/pick/sounds.json, and prints the sounds read."""
import json, os, subprocess, tempfile
import numpy as np, soundfile as sf
from kokoro import KPipeline

SPEED = 0.8  # the speed the app's recordings use (rerecord.py)
WAYS = {
    'i': [('ˈɪh', 'Now: ih with a breath after'), ('ɪ', 'Plain ih'), ('ˈɪ', 'Plain ih, stressed'), ('ˈɪʔ', 'ih, cut off short'), ('ˈɪˈɪ', 'ih ih, said twice')],
    'n': [('nnn', 'Now: long nnn'), ('n', 'Short n'), ('nn', 'n, a little longer'), ('nə', 'nuh'), ('ˈɛn', 'The letter name, en')],
}
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
os.makedirs('samples/pick', exist_ok=True)
tmp = os.path.join(tempfile.gettempdir(), 'p.wav'); out = []
def record(text, name):
    said, chunks = [], []
    for _, ps, a in pipe(text, voice='af_heart', speed=SPEED): said.append(ps); chunks.append(np.asarray(a))
    sf.write(tmp, np.concatenate(chunks), 24000)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'libmp3lame', '-b:a', '64k', '-ac', '1', f'samples/pick/{name}.mp3'], check=True)
    return ' | '.join(said)
for sound, ways in WAYS.items():
    for n, (p, label) in enumerate(ways, 1):
        word = 'ih' if sound == 'i' else 'nnn'
        alone = record(f'[{word}](/{p}/)', f'{sound}-{n}-alone')
        line = record(f'Which letters say [{word}](/{p}/)?', f'{sound}-{n}-line')
        print('said', sound, n, p, '->', alone, '/', line, flush=True)
        out.append({'sound': sound, 'n': n, 'phonemes': p, 'label': label, 'alone': alone, 'line': line})
json.dump(out, open('samples/pick/sounds.json', 'w'), ensure_ascii=False, indent=1)
