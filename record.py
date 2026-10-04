"""Records a shard of phrases.json with the free Kokoro voice 'af_heart' (Apache 2.0) into heart/<2 hex>/<id>.m4a.
Sentences are in priority order; shard k of n takes the k-th contiguous slice. Skips files that already exist."""
import json, os, subprocess, sys, tempfile, time
import numpy as np, soundfile as sf
from kokoro import KPipeline

shard, shards = int(sys.argv[1]), int(sys.argv[2])
items = json.load(open('phrases.json'))
per = -(-len(items) // shards)
mine = items[shard * per:(shard + 1) * per]
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device='cpu')
tmp = os.path.join(tempfile.gettempdir(), f'k{shard}.wav')
made = skipped = failed = 0; t0 = time.time()
for it in mine:
    d = os.path.join('heart', it['id'][:2]); f = os.path.join(d, it['id'] + '.m4a')
    if os.path.exists(f): skipped += 1; continue
    try:
        chunks = [np.asarray(a) for _, _, a in pipe(it['say'], voice='af_heart', speed=1.0)]
        if not chunks: raise RuntimeError('no audio')
        sf.write(tmp, np.concatenate(chunks), 24000)
        os.makedirs(d, exist_ok=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:a', 'aac', '-b:a', '32k', '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', f + '.part'], check=True)
        os.replace(f + '.part', f); made += 1
    except Exception as e:
        failed += 1; print('failed', it['id'], repr(it['say']), e, flush=True)
    if (made + failed) % 50 == 0 and made: print(f'shard {shard}: {made} made, {skipped} skipped, {failed} failed, {(time.time()-t0)/60:.1f} min', flush=True)
print(f'shard {shard} done: {made} made, {skipped} skipped, {failed} failed in {(time.time()-t0)/60:.1f} min')
