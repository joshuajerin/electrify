"""Exercise the real HTTP workflow, including checkpoint-backed studies."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--wam",action="store_true")
    args.add_argument("--modal",action="store_true")
    args.add_argument("--export")
    args.add_argument("--restudy",action="store_true")
    args.add_argument("--requested-mw",type=float,default=60.)
    opts = args.parse_args()
    base = "http://127.0.0.1:8000/v2"
    def call(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        with urlopen(Request(base+path,payload,{"Content-Type":"application/json"}),timeout=140) as r:
            return json.load(r)

    status = call("/runs/current")
    if opts.wam:
        assert status["available"],status
    s = call("/sessions",{"seed":42,"requested_mw":opts.requested_mw})
    url = f"/sessions/{s['id']}"
    def run(revision):
        job = call(url+"/studies",{"revision":revision})
        end = time.monotonic()+150
        while time.monotonic()<end:
            report = call(f"/studies/{job['id']}")
            if report["status"] in ("complete","error","cancelled"):
                break
            time.sleep(.7)
        assert report["status"]=="complete",report.get("progress")
        return job,report

    job,report = run(s["revision"])
    if opts.modal:
        assert report["inference_backend"]=="Modal" and report.get("gpu")
    if opts.wam:
        learned = next(p for p in report["proposals"] if p["source"]=="WAM allocation")
        assert len(learned["imagined"])==12 and len(learned["trajectory"])==36
    split = next((p for p in report["proposals"] if len(p["allocation"])>1),report["proposals"][0])
    branched = call(url+f"/studies/{job['id']}/branch",{"revision":s["revision"],"proposal":split["id"]})
    assert branched["id"]!=s["id"] and len(branched["campuses"])==len(split["allocation"])
    updated = call(f"/sessions/{branched['id']}/step",{"revision":branched["revision"],"mode":"learned" if opts.wam else "heuristic"})
    assert updated["tick"]==branched["tick"]+1
    frames = call(f"/sessions/{branched['id']}/timeline")["frames"]
    assert len(frames)==2
    edit = call(url+"/edits",{"revision":s["revision"],"kind":"rating","target":11,"value":.85})
    assert edit["revision"]==s["revision"]+1
    try:
        call(url+f"/studies/{job['id']}/branch",{"revision":edit["revision"],"proposal":split["id"]})
    except HTTPError as exc:
        assert exc.code==409
    else:
        raise AssertionError("Stale study accepted")
    edited_report = None
    if opts.restudy:
        _,edited_report = run(edit["revision"])
        assert edited_report["revision"]==edit["revision"] and edited_report["id"]!=report["id"]
        if opts.modal:
            assert edited_report.get("gpu")
    if opts.export:
        Path(opts.export).write_text(json.dumps(dict(original=report,edited=edited_report),indent=2))
    print({"wam":opts.wam,"inference_backend":report["inference_backend"],
           "proposals":len(report["proposals"]),"solves":report["solves"],"branch_campuses":len(branched["campuses"]),
           "advanced_tick":updated["tick"],"stale_rejected":True})


if __name__=="__main__":
    main()
