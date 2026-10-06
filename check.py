"""Prints the sounds the voice reads for each sentence in check.json (no audio), so we can see letter names vs sounds."""
import json
from kokoro import KPipeline
try:
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', model=False)
except Exception:
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
for text in json.load(open('check.json')):
    ps = ' | '.join(p for _, p, *_ in pipe(text) if p)
    print('said', repr(text), '->', ps, flush=True)
