import json
from pathlib import Path
import time
import urllib.request
import urllib.error

def call(path, data=None):
    req = urllib.request.Request('http://127.0.0.1:8000'+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)

s=call('/sessions',{'scenario':'baseline','seed':42})
s=call(f"/sessions/{s['id']}/edits",{'revision':s['revision'],'kind':'compute','value':69.6})
job=call(f"/sessions/{s['id']}/missions",{'revision':s['revision']})
deadline=time.monotonic()+120
while time.monotonic()<deadline:
    result=call('/missions/'+job['id'])
    if result['status'] not in ('queued','running'): break
    time.sleep(.5)
assert result['status']=='complete',result
assert len(result['outages'])==59
assert any(p['source']=='WAM cap proposal' and p['imagined'] for p in result['proposals'])
assert all(len(p['cases'])==60 for p in result['proposals'])
p=result['best'] or result['proposals'][-1]
branch=call(f"/sessions/{s['id']}/missions/{job['id']}/branch",{'revision':s['revision'],'proposal':p['id']})
assert branch['id']!=s['id'] and branch['compute']['requested']==69.6
assert branch['compute']['bus']==p['bus'] and branch['compute']['cap']==p['cap_mw']
parent=call(f"/sessions/{s['id']}/timeline")['frames'][-1]
assert parent['revision']==s['revision'] and parent['compute']==s['compute']
edited=call(f"/sessions/{s['id']}/edits",{'revision':s['revision'],'kind':'rating','target':2,'value':.7})
try:
    call(f"/sessions/{s['id']}/missions/{job['id']}/branch",{'revision':edited['revision'],'proposal':p['id']})
    raise AssertionError('Stale study accepted')
except urllib.error.HTTPError as e: assert e.code==409
(Path(__file__).resolve().parents[1] / 'artifacts/local-smoke/mission-api-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':'passed','outages':len(result['outages']),'proposals':len(result['proposals']),'solves':result['solves'],'duration_s':result['duration_s'],'checks':['WAM provenance','full active-branch coverage','isolated proposal branch','stale study rejected']},indent=2))
