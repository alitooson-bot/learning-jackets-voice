"""Records the sentences listed in redo.json again, replacing the old files in heart/<2 hex>/<id>.m4a.
Used when the words to SAY for a sentence change (for example two-way words like "live" and "read", which are
written as [word](/sounds/) so the voice says the right one). Shard k of n takes every n-th sentence."""
import json, os, subprocess, sys, tempfile, time
import numpy as np, soundfile as sf
from kokoro import KPipeline

SPEED = 0.8  # the family chose 'Heart, slowest' on the samples page (2026-10-06)

shard, shards = int(sys.argv[1]), int(sys.argv[2])
# Optional 3rd argument: the list to record again (default redo.json; phrases.json re-records every sentence).
mine = json.load(open(sys.argv[3] if len(sys.argv) > 3 else 'redo.json'))[shard::shards]
quiet = len(sys.argv) > 4  # a whole-list run prints only failures
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
tmp = os.path.join(tempfile.gettempdir(), f'r{shard}.wav')
made = failed = 0; t0 = time.time()
for it in mine:
    d = os.path.join('heart', it['id'][:2]); f = os.path.join(d, it['id'] + '.m4a')
    try:
        said = []; chunks = []
        for _, ps, a in pipe(it['say'], voice='af_heart', speed=SPEED): said.append(ps); chunks.append(np.asarray(a))
        if not chunks: raise RuntimeError('no audio')
        sf.write(tmp, np.concatenate(chunks), 24000)
        os.makedirs(d, exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '32k', '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', f + '.part'], check=True)
        os.replace(f + '.part', f); made += 1
        if not quiet: print('said', it['id'], ' '.join(said), flush=True)
        elif made % 200 == 0: print(f'shard {shard}: {made} made, {(time.time()-t0)/60:.1f} min', flush=True)
    except Exception as e:
        failed += 1; print('failed', it['id'], repr(it['say']), e, flush=True)
print(f'shard {shard} done: {made} made, {failed} failed in {(time.time()-t0)/60:.1f} min')
sys.exit(1 if failed else 0)
