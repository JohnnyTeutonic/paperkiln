"""Standard jobs_vm preflight: source, spec and uploaded data only; no training launch."""
import hashlib
import json
from pathlib import Path
import sys
import os
import tarfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify(binding,spec_path,upload,repo):
    if sha(spec_path)!=binding['spec_sha256']:raise ValueError('DPO spec differs')
    for name,value in binding['upload_sha256'].items():
        if Path(name).name!=name or sha(Path(upload)/name)!=value:raise ValueError('DPO upload differs')
    archive=Path(upload)/'training_source.tar'
    if 'training_source.tar' not in binding['upload_sha256']:raise ValueError('exact training source archive missing')
    # Verify every archived tracked file against the actual checkout, not only selected consumers.
    with tarfile.open(archive,'r:') as tar:
        for entry in tar:
            name=Path(entry.name)
            if name.is_absolute() or '..' in name.parts:raise ValueError('source archive path escapes checkout')
            target=Path(repo)/name
            if entry.isfile():
                h=hashlib.sha256()
                with tar.extractfile(entry) as handle:
                    for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
                if sha(target)!=h.hexdigest():raise ValueError('actual checkout differs from archived source')
            elif entry.issym():
                if not target.is_symlink() or os.readlink(target)!=entry.linkname:raise ValueError('source link differs')
            elif not entry.isdir():raise ValueError('unsupported source archive entry')
    if (Path(upload)/'ember_src.tar.gz').exists():raise ValueError('unbound Ember override would break common builder identity')
    for name,value in binding['source_files_sha256'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or sha(Path(repo)/p)!=value:raise ValueError('DPO source differs')
    if 'vocab.gguf' in binding['upload_sha256']:
        for alias in ['pksft.gguf','vocab_llama3_bpe.gguf']:
            target=Path(upload)/alias
            if target.exists():
                if sha(target)!=binding['upload_sha256']['vocab.gguf']:raise ValueError('existing model alias differs')
            else:target.symlink_to('vocab.gguf')
    return {'kind':'DPO_VM_PREFLIGHT_PASSED','training_launched':False,'source_commit':binding['source_commit'],'spec_sha256':binding['spec_sha256']}

def main():
    binding,spec,out=map(Path,sys.argv[1:]);record=verify(json.loads(binding.read_text()),spec,binding.parent,Path.cwd())
    out.mkdir(parents=True,exist_ok=True);(out/'preflight.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
