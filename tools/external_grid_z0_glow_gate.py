from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from linecaller.dcf.external_grid_pillow_watcher import BallColorProfile, ExternalGridCalibration
from linecaller.dcf.external_grid_z0_game import ExternalGridZ0Game
from linecaller.dcf.z0_floor_glow_renderer import render_soft_floor_glow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--grid-calibration', required=True)
    ap.add_argument('--active-side', choices=['FAR', 'NEAR'], required=True)
    ap.add_argument('--depth-bu', type=int, default=4)
    ap.add_argument('--start-frame', type=int, required=True)
    ap.add_argument('--end-frame', type=int, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    cal = ExternalGridCalibration.load(args.grid_calibration)
    profile = BallColorProfile.from_template(args.template)
    game = ExternalGridZ0Game(cal, profile, active_side=args.active_side, depth_bu=args.depth_bu)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot create output: {out}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    print('\n' + '=' * 72)
    print(' CP-0035.9.4.2 RC8.4 - EXTERNAL GRID Z0 AFTER-FACT GLOW')
    print('=' * 72)
    print(f'ACTIVE SIDE....................... {args.active_side}')
    print(f'ACTIVE DEPTH...................... {args.depth_bu} BU')
    print(f'ACTIVE CELLS...................... {len(game.cells)}')
    print('ORDER............................. FRAME -> CELL LOOP -> PILLOW')
    print('PILLOW REFERENCE.................. CALIBRATED BACKGROUND')
    print('CANDIDATES VISIBLE................ MAX 1')
    print('PATH / HISTORY.................... NO')
    print('TRACKER / XYZ..................... NO')
    print('DOWN.............................. NO')
    print('AFTER THE FACT.................... ABOVE ONLY')
    print('CONFIRMED Z0...................... CANDIDATE CELL')
    print('GLOW.............................. SOFT FLOOR ONLY\n')

    frame_no = args.start_frame
    candidates = []
    confirmations = []

    while frame_no <= args.end_frame:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay = bgr.copy()

        conf = game.check_above_after_fact(frame_no, rgb)
        if conf is not None:
            confirmations.append(conf)
            overlay = render_soft_floor_glow(overlay, conf.cell.polygon)
            cv2.putText(overlay, f'CONFIRMED Z0 cell={conf.cell.cell_id} candidate={conf.candidate_frame} UP={conf.confirm_frame}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        elif game.pending is None:
            c = game.scan_for_one_candidate(frame_no, rgb)
            if c is not None:
                candidates.append(c)
                poly = np.round(np.asarray(c.cell.polygon, dtype=np.float32)).astype(np.int32)
                cv2.polylines(overlay, [poly], True, (0,255,255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (int(round(c.center_px[0])), int(round(c.center_px[1]))), 3, (255,255,255), -1, cv2.LINE_AA)
                cv2.putText(overlay, f'ONE Z0 CANDIDATE cell={c.cell.cell_id} grid=({c.cell.gx},{c.cell.gy})', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(overlay, 'NO Z0 CANDIDATE', (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        else:
            p = game.pending
            cv2.putText(overlay, f'WAITING ABOVE AFTER FACT for cell={p.cell.cell_id}', (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(overlay, 'ONE BALL / ONE Z0 CELL | NO PATH | NO DOWN | ABOVE -> GLOW', (12, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255,255,255), 1, cv2.LINE_AA)
        writer.write(overlay)
        frame_no += 1

    cap.release()
    writer.release()

    print(f'FRAMES SCANNED.................... {max(0, frame_no - args.start_frame)}')
    print(f'Z0 CANDIDATES..................... {len(candidates)}')
    print(f'Z0 CONFIRMATIONS.................. {len(confirmations)}')
    if candidates:
        print('\nCANDIDATES:')
        for c in candidates[:20]:
            print(f'  frame={c.frame} cell={c.cell.cell_id} grid=({c.cell.gx},{c.cell.gy}) floor={c.cell.floor_position_bu} expected_scale={c.cell.expected_scale_px:.2f}px observed_scale={c.scale_px:.2f}px')
    if confirmations:
        print('\nCONFIRMED Z0 / GLOW:')
        for c in confirmations[:20]:
            print(f'  Z0 frame={c.candidate_frame} -> UP frame={c.confirm_frame} cell={c.cell.cell_id} grid=({c.cell.gx},{c.cell.gy})')
    print(f'\nOUTPUT............................ {out}')
    print('PRODUCTION GRID................... INVISIBLE')
    print('GLOW COLOR........................ NEUTRAL ENGINEERING ONLY')
    print('COMMIT / TAG...................... NO')
    print('RC8_4_EXTERNAL_GRID_Z0_GLOW_GATE=COMPLETE')

if __name__ == '__main__':
    main()
