"""TEMPORARY resumable range downloader for the pinned smoke checkpoint only."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from src.models.smoke_qwen import MODEL_ID, REVISION, CACHE


def main():
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true', timeout=30))
    entry = next(x for x in info['siblings'] if x['rfilename'].endswith('.safetensors'))
    size = entry['size']
    if size > 5_000_000_000:
        raise RuntimeError('Download exceeds user confirmation threshold')
    expected = entry['lfs']['sha256']
    target = CACHE / 'model' / entry['rfilename']
    partial = target.with_suffix('.partial')
    progress = target.with_suffix('.progress.json')
    chunk_size = 16 * 1024 * 1024
    done = set(json.loads(progress.read_text()) if progress.exists() and partial.exists() else [])
    if not partial.exists():
        with partial.open('wb') as f:
            f.truncate(size)
    starts = [start for start in range(0, size, chunk_size) if start not in done]

    def fetch(start):
        end = min(start + chunk_size, size) - 1
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{entry["rfilename"]}?download=true&smoke_chunk={start}'
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError('Server did not honor requested byte range')
                    raw = response.read(end-start+2)
                if len(raw) != end-start+1:
                    raise ValueError('Incomplete chunk')
                return start, raw
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)

    print(f'Range download: {size} bytes; {len(done)} chunks already complete', flush=True)
    with partial.open('r+b') as f, concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch, start) for start in starts]):
            start, raw = future.result()
            f.seek(start)
            f.write(raw)
            f.flush()
            done.add(start)
            progress.write_text(json.dumps(sorted(done)), encoding='utf-8')
            if len(done) % 8 == 0 or len(done) == (size+chunk_size-1)//chunk_size:
                print(f'{min(len(done)*chunk_size,size)/1e9:.2f}/{size/1e9:.2f} GB downloaded', flush=True)
    sha = hashlib.sha256()
    with partial.open('rb') as f:
        for raw in iter(lambda: f.read(8*1024*1024), b''):
            sha.update(raw)
    if sha.hexdigest() != expected:
        raise ValueError('Checkpoint checksum mismatch')
    partial.replace(target)
    (target.parent / 'smoke_revision.txt').write_text(REVISION, encoding='utf-8')
    print(f'Checkpoint SHA-256 verified: {expected}', flush=True)


if __name__ == '__main__':
    main()
