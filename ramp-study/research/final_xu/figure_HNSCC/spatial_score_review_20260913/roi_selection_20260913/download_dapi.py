"""Read exact uncompressed DAPI planes from official BioImage Archive TIFFs."""
from pathlib import Path
import hashlib
import json
import ssl
import struct
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.request import Request,urlopen
import certifi

HERE=Path(__file__).resolve().parent
BASE='https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/638/S-BIAD1638/Files/'
CTX=ssl.create_default_context(cafile=certifi.where())

def get_range(url,start,end):
    for trial in range(3):
        try:
            req=Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'HNSCC-scientific-figure/1.0'})
            with urlopen(req,timeout=45,context=CTX) as r:
                cr=r.headers.get('Content-Range','')
                if r.status!=206 or not cr.startswith(f'bytes {start}-{end}/'):
                    raise IOError(f'Expected checked 206 byte range, got {r.status} {cr}')
                b=r.read(end-start+2)
            assert len(b)==end-start+1
            return b
        except Exception:
            if trial==2:raise
            time.sleep(1+trial)

def parse_tiff(h):
    endian={'MM':'>','II':'<'}[h[:2].decode()]
    assert struct.unpack(endian+'H',h[2:4])[0]==42
    offset=struct.unpack(endian+'I',h[4:8])[0]
    n=struct.unpack(endian+'H',h[offset:offset+2])[0]
    tags={}
    for i in range(n):
        p=offset+2+i*12
        tag,kind,count=struct.unpack(endian+'HHI',h[p:p+8]);value=h[p+8:p+12]
        if kind not in [3,4]:continue
        width={3:2,4:4}[kind]
        if count*width>4:
            pos=struct.unpack(endian+'I',value)[0];value=h[pos:pos+count*width]
        else:value=value[:count*width]
        if len(value)==count*width:tags[tag]=struct.unpack(endian+str(count)+{3:'H',4:'I'}[kind],value)
    assert tags[256]==(1920,) and tags[257]==(1440,) and tags[258]==(16,)
    # TIFF defaults to uncompressed, one sample when these optional tags are absent.
    assert tags.get(259,(1,))==(1,) and tags.get(277,(1,))==(1,)
    assert len(tags[273])==1 and tags[279]==(5529600,)
    return dict(endian=endian,width=1920,height=1440,bits=16,compression=1,strip_start=tags[273][0],strip_bytes=tags[279][0],tags={str(k):list(v) for k,v in tags.items()})

def download(record):
    roi=record['roi'];out=HERE/'images'/roi;out.mkdir(parents=True,exist_ok=True)
    url=BASE+record['official_image_path']
    plane=out/'DAPI.raw';manifest=out/'DOWNLOAD.json'
    if plane.exists() and manifest.exists():
        m=json.loads(manifest.read_text())
        assert hashlib.sha256(plane.read_bytes()).hexdigest()==m['sha256']
        print(roi,'verified cached plane',flush=True);return
    header=get_range(url,0,8191);(out/'tiff_header_8192.bin').write_bytes(header)
    spec=parse_tiff(header);start=spec['strip_start'];total=spec['strip_bytes']
    print(roi,'verified TIFF header',spec['width'],spec['height'],spec['endian'],start,total,flush=True)
    pieces=32;step=(total+pieces-1)//pieces;parts={}
    jobs=[(i,start+i*step,min(start+(i+1)*step,start+total)-1) for i in range(pieces) if i*step<total]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(get_range,url,a,b):i for i,a,b in jobs}
        for future in as_completed(futures):
            i=futures[future];parts[i]=future.result()
            if len(parts)%8==0:print(roi,f'{len(parts)}/{len(jobs)} checked ranges',flush=True)
    payload=b''.join(parts[i] for i in range(len(jobs)));assert len(payload)==total
    plane.write_bytes(payload)
    manifest.write_text(json.dumps(dict(source=url,roi=roi,plane_index=0,channel='DAPI',http_checked_ranges=True,**spec,
            sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)),indent=2)+'\n')
    print(roi,'DONE',flush=True)

if __name__=='__main__':
    records=json.loads((HERE/'RANKING_AUDIT.json').read_text())['top']
    seen=set()
    for r in records.values():
        if r['roi'] not in seen:download(r);seen.add(r['roi'])
