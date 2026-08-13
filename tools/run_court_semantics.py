from __future__ import annotations
import argparse, csv
from pathlib import Path
from linecaller.court_semantics import CourtSemanticModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--court-events", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--line-tolerance-in", type=float, default=2.0)
    a=p.parse_args()

    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    model=CourtSemanticModel(line_tolerance_in=a.line_tolerance_in)
    rows=[]

    with open(a.court_events, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r=model.classify(float(row["court_x_ft"]), float(row["court_y_ft"]))
            rows.append({
                **row,
                "semantic_zone": r.zone.value,
                "semantic_nearest_line": r.nearest_line.value,
                "semantic_signed_distance_ft": round(r.signed_distance_ft,4),
                "semantic_distance_in": round(r.absolute_distance_ft*12.0,3),
                "semantic_on_line": r.on_line,
                "semantic_inside_outer_court": r.inside_outer_court,
            })

    output_csv=out/"semantic_events.csv"
    if rows:
        with output_csv.open("w",newline="",encoding="utf-8") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)

    print(f"events={len(rows)}")
    print(f"output={output_csv}")
    for row in rows:
        print(f"frame={row['frame']} zone={row['semantic_zone']} "
              f"line={row['semantic_nearest_line']} "
              f"distance={row['semantic_distance_in']}in "
              f"on_line={row['semantic_on_line']}")

if __name__=="__main__":
    main()
