from pathlib import Path
import json,time
C=Path(__file__).resolve().parents[1]
s=json.loads((C/'STATUS.json').read_text());print('Overall:',s['status'],'elapsed_min:',round((time.time()-s['started_unix'])/60,1))
for name,j in s['jobs'].items():
 state=j.get('status');elapsed=round((j.get('ended_unix',time.time())-j.get('started_unix',time.time()))/60,1)
 print(name,state,'minutes',elapsed)
for key in ['harmony','fastmnn','cycombine','cytonorm']:
 p=C/'logs'/f'{key}.log'
 if p.exists():print(key,'tail:',p.read_text(errors='replace')[-180:].replace('\n',' | '))
for i in range(3):
 p=C/'maxfuse_full_panel/logs'/f'maxfuse_hub{i}.log'
 if p.exists():print('maxfuse',i,'tail:',p.read_text(errors='replace')[-170:].replace('\n',' | '))
