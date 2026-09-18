"""Download one VLNVerse scene from an explicit HF file manifest; verify each file."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


def fetch_scene(tree, root, scene_id, revision='main', workers=12):
    rows=json.loads(Path(tree).read_text())
    files=[r for r in rows if r['type']=='file' and r['path'].startswith(scene_id+'/')]
    if not files:raise ValueError('empty scene manifest')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    def matches(path,row):
        if not path.exists() or path.stat().st_size!=row['size']:return False
        digest=hashlib.sha256() if 'lfs' in row else hashlib.sha1()
        if 'lfs' not in row:digest.update(f"blob {row['size']}\0".encode())
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        return digest.hexdigest()==row.get('lfs',{}).get('oid',row['oid'])
    def fetch(row):
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('unsafe scene file path')
        if matches(path,row):return
        path.parent.mkdir(parents=True,exist_ok=True)
        partial=path.with_name(path.name+'.partial')
        url='https://huggingface.co/datasets/Eyz/VLNVerse_scene/resolve/'+quote(revision,safe='')+'/'+quote(row['path'],safe='/')
        subprocess.run(['curl','-fsSL','--retry','2','--connect-timeout','8','--max-time','180',url,'-o',str(partial)],check=True)
        if not matches(partial,row):raise ValueError('scene file integrity mismatch: '+row['path'])
        partial.replace(path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(fetch,r) for r in files]
        for i,future in enumerate(as_completed(jobs),1):
            future.result()
            if i%50==0 or i==len(files):print(json.dumps({'downloaded_files':i,'total_files':len(files)}),flush=True)
    manifest={'repo':'Eyz/VLNVerse_scene','revision':revision,'files':files,'complete':True}
    (root/scene_id/'platform_asset_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--revision',default='main')
    args=parser.parse_args()
    fetch_scene(args.tree,args.root,args.scene,args.revision)
