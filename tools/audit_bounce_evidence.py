import argparse,csv
from pathlib import Path
from linecaller.bounce_v2.evidence_integrity import BounceEvidenceIntegrity,EvidenceSample

def val(r,names):
    for n in names:
        if n in r and r[n] not in ("",None): return r[n]
def fl(r,n): 
    v=val(r,n); return None if v is None else float(v)
def it(r,n):
    v=val(r,n); return None if v is None else int(float(v))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--track-csv",required=True); p.add_argument("--bounce-csv",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--window",type=int,default=5); p.add_argument("--max-inference-gap",type=int,default=6)
    p.add_argument("--min-observed-confidence",type=float,default=.10); p.add_argument("--min-inferred-confidence",type=float,default=.10)
    a=p.parse_args(); samples=[]
    with open(a.track_csv,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fr=it(r,["frame","frame_number"])
            if fr is None: continue
            samples.append(EvidenceSample(fr,fl(r,["raw_x","det_x","yolo_x"]),fl(r,["raw_y","det_y","yolo_y"]),fl(r,["tracked_x","x"]),fl(r,["tracked_y","y"]),fl(r,["confidence","conf"]) or 0.,str(val(r,["source"]) or ""),it(r,["missed_frames"]) or 0))
    eng=BounceEvidenceIntegrity(a.window,a.max_inference_gap,a.min_observed_confidence,a.min_inferred_confidence)
    rows=[]; accepted=[]
    with open(a.bounce_csv,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fr=it(r,["frame","frame_number"])
            if fr is None: continue
            z=eng.evaluate(fr,samples); q=dict(r)
            q.update(evidence_accepted=z.accepted,evidence_class=z.evidence.value,evidence_contact_frame=z.contact_frame,evidence_contact_x=z.contact_x,evidence_contact_y=z.contact_y,evidence_confidence=round(z.confidence,4),before_observed_frame=z.before_observed_frame,after_observed_frame=z.after_observed_frame,evidence_reason=z.reason)
            rows.append(q)
            if z.accepted:
                d=dict(q); d["frame"]=z.contact_frame; d["x"]=z.contact_x; d["y"]=z.contact_y; accepted.append(d)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    audit=out/"bounce_evidence_audit.csv"; verified=out/"evidence_verified_bounce_events.csv"
    if rows:
        with audit.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if accepted:
        with verified.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(accepted[0])); w.writeheader(); w.writerows(accepted)
    else: verified.write_text("",encoding="utf-8")
    print(f"candidates={len(rows)}"); print(f"accepted={len(accepted)}")
    print(f"observed={sum(r['evidence_class']=='OBSERVED_CONTACT' for r in rows)}")
    print(f"inferred={sum(r['evidence_class']=='INFERRED_CONTACT' for r in rows)}")
    print(f"unverified={sum(r['evidence_class']=='UNVERIFIED_CONTACT' for r in rows)}")
    print(f"audit={audit}"); print(f"verified={verified}")
    for r in rows: print(f"frame={r.get('frame')} class={r['evidence_class']} accepted={r['evidence_accepted']} contact_frame={r['evidence_contact_frame']} confidence={r['evidence_confidence']} before={r['before_observed_frame']} after={r['after_observed_frame']} reason={r['evidence_reason']}")
if __name__=="__main__": main()
