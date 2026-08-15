import json
from pathlib import Path
import pytest
from linecaller.live_bridge.officiating_events import OfficiatingTimeline
def E(decision='OUT',frame=29):return {'decision':decision,'confidence':.92,'explanation':['trusted'],'bounce_frame':frame,'image':{'x':885.5,'y':485.1},'nearest_line':'FAR_BASELINE','signed_distance_m':-.0368,'total_uncertainty_m':.01,'bounce_confidence':.8,'calibration_valid':True}
def W(tmp_path,items):p=tmp_path/'events.jsonl';p.write_text('\n'.join(json.dumps(x) for x in items)+'\n',encoding='utf-8');return p
def test_load(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E()])).events[0].decision=='OUT'
def test_xy(tmp_path):e=OfficiatingTimeline.load(W(tmp_path,[E()])).events[0];assert e.image_x==885.5 and e.image_y==485.1
def test_geom(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E()])).events[0].signed_distance_m<0
def test_review(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E('REVIEW')])).events[0].decision=='REVIEW'
def test_bad(tmp_path):
 with pytest.raises(ValueError):OfficiatingTimeline.load(W(tmp_path,[E('MAYBE')]))
def test_hold(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E(frame=100)])).event_for_frame(120,45)
def test_expire(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E(frame=100)])).event_for_frame(146,45) is None
def test_latest(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E(frame=10),E('IN',20)])).event_for_frame(22).decision=='IN'
def test_json(tmp_path):assert OfficiatingTimeline.load(W(tmp_path,[E()])).to_dict()['events'][0]['explanation']==['trusted']
def test_react():s=Path('ui/src/LiveCourtVideo.tsx').read_text(encoding='utf-8');assert 'syncedDecision' in s and 'bounceMarker' in s and 'onTimeUpdate' in s
