from pathlib import Path
import pytest
from linecaller.live_bridge.video import VideoSourceRegistry,content_type_for,parse_range_header
def test_missing(tmp_path):
 with pytest.raises(FileNotFoundError):VideoSourceRegistry().bind("c",tmp_path/"x.mp4")
def test_bind(tmp_path):
 p=tmp_path/"x.mp4";p.write_bytes(b"x");r=VideoSourceRegistry();r.bind("c",p);assert r.get("c")==p.resolve()
def test_describe(tmp_path):
 p=tmp_path/"x.mp4";p.write_bytes(b"x");r=VideoSourceRegistry();r.bind("c",p);assert r.describe("c")["available"]
def test_empty():assert not VideoSourceRegistry().describe("c")["available"]
def test_type():assert content_type_for("x.mp4")=="video/mp4"
def test_open():assert parse_range_header("bytes=10-",100)==(10,99)
def test_bound():assert parse_range_header("bytes=10-19",100)==(10,19)
def test_suffix():assert parse_range_header("bytes=-10",100)==(90,99)
def test_invalid():assert parse_range_header("bytes=200-300",100) is None
def test_component_exists():assert Path("ui/src/LiveCourtVideo.tsx").exists()
