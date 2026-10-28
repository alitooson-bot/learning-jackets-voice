"""Sound it out for "chin", before and after (Ali, 2026-10-28: "it gets the i sound wrong and the n is too many n's").
Before: lines with no recording were read by Piper (en_US-hfc_female-medium, the app's on-device voice), which says
"ih" as "eye" and "nnn" as "en, en, en". After: the same lines recorded ahead of time with Jax's voice, the sounds written
as phonemes (redo.json). Writes samples/chin/<n>-before.wav, <n>-after.m4a and index.html, and prints what each voice says."""
import html, json, os, subprocess, tempfile, wave
import numpy as np, soundfile as sf
from pathlib import Path
from kokoro import KPipeline
from piper import PiperVoice
from piper.download_voices import download_voice

SPEED = 0.8  # the speed the app's recordings use (rerecord.py)
LINES = [("Let's sound it out!", "Let's sound it out!"), ('Listen: chin.', 'Listen: chin.'), ('chuh', 'chuh'), ('ih', '[ih](/ˈɪh/)'),
         ('nnn', '[nnn](/nnn/)'), ('Which letters say ih?', 'Which letters say [ih](/ˈɪh/)?'), ('Yes! ih.', 'Yes! [ih](/ˈɪh/).'),
         ('Which letters say nnn?', 'Which letters say [nnn](/nnn/)?'), ('Yes! nnn.', 'Yes! [nnn](/nnn/).'),
         ('That one says eh.', 'That one says [eh](/ɛ/).'), ('We want ih. ih.', 'We want [ih](/ˈɪh/). [ih](/ˈɪh/).'),
         ('chin! chin is spelled c, h, i, n.', 'chin! chin is spelled c, h, i, n.')]
os.makedirs('samples/chin', exist_ok=True)
VOICES = os.path.join(tempfile.gettempdir(), 'piper'); os.makedirs(VOICES, exist_ok=True)
download_voice('en_US-hfc_female-medium', Path(VOICES))
piper = PiperVoice.load(os.path.join(VOICES, 'en_US-hfc_female-medium.onnx'))
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
tmp = os.path.join(tempfile.gettempdir(), 'c.wav'); rows = []; report = []
for n, (text, say) in enumerate(LINES, 1):
    with wave.open(f'samples/chin/{n}-before.wav', 'wb') as w: piper.synthesize_wav(text, w)
    heard_before = ' '.join(''.join(s) for s in piper.phonemize(text))
    said, chunks = [], []
    for _, ps, a in pipe(say, voice='af_heart', speed=SPEED): said.append(ps); chunks.append(np.asarray(a))
    sf.write(tmp, np.concatenate(chunks), 24000)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '48k', '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', f'samples/chin/{n}-after.m4a'], check=True)
    heard_after = ' | '.join(said)
    print(n, repr(text), 'before:', heard_before, 'after:', heard_after, flush=True)
    report.append({'n': n, 'text': text, 'before': heard_before, 'after': heard_after})
    rows.append(f'<tr><th>{html.escape(text)}</th><td><audio controls preload="none" src="{n}-before.wav"></audio><br><small>{html.escape(heard_before)}</small></td>'
                f'<td><audio controls preload="none" src="{n}-after.m4a"></audio><br><small>{html.escape(heard_after)}</small></td></tr>')
json.dump(report, open('samples/chin/sounds.json', 'w'), ensure_ascii=False, indent=1)
open('samples/chin/index.html', 'w').write('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Chin, sounded out</title><style>body{font:16px system-ui;margin:16px}td,th{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}</style><h1>Sound it out: chin</h1><table><tr><th>Line</th><th>Before (Piper)</th><th>After (recorded)</th></tr>' + ''.join(rows) + '</table>')
