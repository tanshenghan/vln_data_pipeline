"""Resumable complete scene mirror at a pinned Hugging Face revision.

Every repository file is enumerated (including pagination), downloaded or copied
from an existing local cache, then verified against its HF LFS/git object hash.
A scene is marked complete only after all files verify. Errors remain visible and
are retried; never rewrite the shared source cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import quote


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def matches(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    h = hashlib.sha256() if 'lfs' in row else hashlib.sha1()
    if 'lfs' not in row: h.update(f"blob {row['size']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest() == row.get('lfs', {}).get('oid', row['oid'])


def tree(scene, revision, metadata):
    cached = metadata / (scene + '.json')
    if cached.exists():
        data = json.loads(cached.read_text())
        if data['revision'] == revision: return data['files']
    url = f'https://huggingface.co/api/datasets/Eyz/VLNVerse_scene/tree/{revision}/{scene}?recursive=true&limit=1000'
    rows = []
    page = metadata / (scene + '.page')
    headers = metadata / (scene + '.headers')
    while url:
        subprocess.run(['curl', '-fsSL', '--retry', '4', '--connect-timeout', '15', '--max-time', '120',
                        '-D', str(headers), url, '-o', str(page)], check=True, stdout=subprocess.DEVNULL)
        rows.extend(json.loads(page.read_text()))
        links = re.findall(r'^link:\s*(.*)$', headers.read_text(), re.M | re.I)
        next_url = re.search(r'<([^>]+)>;\s*rel="next"', links[-1]) if links else None
        url = next_url.group(1) if next_url else None
        if url and not url.startswith('https://huggingface.co/api/datasets/'):
            raise ValueError('unexpected pagination host')
    files = [r for r in rows if r['type'] == 'file']
    if not files or len({r['path'] for r in files}) != len(files): raise ValueError('invalid scene inventory')
    atomic_json(cached, dict(revision=revision, files=files))
    return files


def sync(args):
    args.root.mkdir(parents=True, exist_ok=True); args.state.mkdir(parents=True, exist_ok=True)
    lock = (args.state/'sync.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = json.loads(args.repo_info.read_text())
    revision = info['sha']
    # The info API silently truncates siblings for large repositories. Enumerate
    # the pinned root tree and every paginated subtree instead.
    root_tree = args.state/'root-tree.json'
    subprocess.run(['curl','-fsSL','--retry','4','--connect-timeout','15','--max-time','120',
        f'https://huggingface.co/api/datasets/Eyz/VLNVerse_scene/tree/{revision}?limit=1000', '-o',str(root_tree)],check=True)
    roots = json.loads(root_tree.read_text())
    scenes = sorted(r['path'] for r in roots if r['type']=='directory')
    if any(not re.fullmatch(r'kujiale_[0-9]+(?:_fix)?',name) for name in scenes):
        raise ValueError('unexpected repository directory')
    extra_rows = [r for r in roots if r['type']=='file']
    metadata = args.state/'manifests'; metadata.mkdir(exist_ok=True)
    status = dict(repo='Eyz/VLNVerse_scene', revision=revision, root=str(args.root),
        status='inventory', total_scenes=len(scenes), total_files=None, verified_files=0,
        verified_bytes=0, total_bytes=None, completed_scenes=0, scenes={}, errors={})
    mutex = threading.Lock()
    def save():
        status['updated_at'] = time.time(); atomic_json(args.state/'status.json', status)
    save()
    inventory = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(tree, scene, revision, metadata):scene for scene in scenes}
        for future in as_completed(jobs):
            scene = jobs[future]; inventory[scene] = future.result()
            status['scenes'][scene] = dict(status='pending', files=len(inventory[scene]), verified_files=0)
            save()
    status['total_files'] = sum(len(rows) for rows in inventory.values())+len(extra_rows)
    status.update(status='downloading', total_bytes=sum(r['size'] for rows in inventory.values() for r in rows)+sum(r['size'] for r in extra_rows))
    save()
    verified = set()
    def fetch(row, scene=None):
        path = (args.root/row['path']).resolve()
        if not path.is_relative_to(args.root.resolve()): raise ValueError('unsafe file path')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.partial')
        if not matches(path,row) and matches(partial,row):
            partial.replace(path)
        if not matches(path,row):
            reused = False
            for base in args.reuse:
                source = base/row['path']
                if matches(source,row):
                    shutil.copyfile(source,partial); reused=True; break
            if not reused:
                url = f'https://huggingface.co/datasets/Eyz/VLNVerse_scene/resolve/{revision}/'+quote(row['path'],safe='/')
                result = subprocess.run(['curl','-fsSL','--retry','3','--retry-all-errors','--connect-timeout','15',
                    '--max-time','900','--speed-time','60','--speed-limit','1024','-C','-',url,'-o',str(partial)],capture_output=True,text=True)
                if result.returncode == 33:
                    # Some CDN responses do not support range requests.
                    partial.unlink(missing_ok=True)
                    result = subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','15','--max-time','900',url,'-o',str(partial)],capture_output=True,text=True)
                if result.returncode: raise RuntimeError(result.stderr[-400:])
            if not matches(partial,row):
                partial.unlink(missing_ok=True)
                raise ValueError('hash mismatch: '+row['path'])
            partial.replace(path)
        with mutex:
            if row['path'] not in verified:
                verified.add(row['path']);status['verified_files']+=1;status['verified_bytes']+=row['size']
                if scene:status['scenes'][scene]['verified_files']+=1
            status['errors'].pop(row['path'],None)
            if status['verified_files']%50==0:save()
    for row in extra_rows: fetch(row)
    # Prioritize the acceptance scene, then process all scenes, with bounded I/O.
    pending = sorted(scenes,key=lambda s:(s!='kujiale_0254',s))
    def process_scene(scene):
        marker=args.root/scene/'platform_asset_manifest.json'
        if marker.exists() and json.loads(marker.read_text()).get('revision') != revision:
            marker.rename(marker.with_name('platform_asset_manifest.previous.json'))
        with mutex:
            status['current_scene']=scene;status['scenes'][scene]['status']='downloading';save()
        failed=False
        with ThreadPoolExecutor(max_workers=max(1,args.workers//4)) as pool:
            jobs={pool.submit(fetch,r,scene):r for r in inventory[scene] if r['path'] not in verified}
            for future in as_completed(jobs):
                row=jobs[future]
                try:future.result()
                except Exception as exc:
                    failed=True
                    with mutex:status['errors'][row['path']]=str(exc);save()
        if not failed:
            atomic_json(args.root/scene/'platform_asset_manifest.json',dict(repo=status['repo'],revision=revision,complete=True,files=inventory[scene]))
        with mutex:
            if not failed:
                status['scenes'][scene]['status']='complete';status['completed_scenes']+=1
            else:status['scenes'][scene]['status']='retry_pending'
            save()
            print(json.dumps({k:status[k] for k in ['current_scene','completed_scenes','total_scenes','verified_files','total_files','verified_bytes']}),flush=True)
        return not failed
    while pending:
        remaining=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs={pool.submit(process_scene,scene):scene for scene in pending}
            for future in as_completed(jobs):
                if not future.result():remaining.append(jobs[future])
        pending=remaining
        if pending:time.sleep(30)
    status['status']='complete';save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--repo-info',type=Path,required=True)
    parser.add_argument('--reuse',type=Path,action='append',default=[])
    parser.add_argument('--workers',type=int,default=24)
    sync(parser.parse_args())
