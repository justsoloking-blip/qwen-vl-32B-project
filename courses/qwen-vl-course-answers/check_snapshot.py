from pathlib import Path
import argparse,hashlib,json
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--sha256',action='store_true');a=ap.parse_args()
meta=json.loads((root/'common/model_metadata/repository.json').read_text());rows=[]
for item in meta['siblings']:
 name=item['rfilename']
 if name in ['README.md','.gitattributes']:continue
 f=a.directory/name;assert f.is_file(),f'Missing {name}';assert f.stat().st_size==item['size'],f'Wrong size: {name}'
 row={'file':name,'size_checked':True}
 if a.sha256 and 'lfs' in item:
  h=hashlib.sha256()
  with f.open('rb') as src:
   for block in iter(lambda:src.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==item['lfs']['sha256'],f'Hash mismatch: {name}';row['sha256_checked']=True
 rows.append(row)
print(json.dumps({'passed':True,'revision':meta['sha'],'scope':'Files and sizes; LFS hashes only if --sha256','files':rows},indent=2))
