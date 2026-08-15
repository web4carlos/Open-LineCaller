from pathlib import Path
R=Path(__file__).resolve().parents[1]
def rd(p):return (R/p).read_text(encoding="utf-8")
def test_player_out_only(): assert 'x==="OUT"' in rd("ui/src/presentationMode.ts")
def test_ref_all(): 
 s=rd("ui/src/presentationMode.ts"); assert all(x in s for x in ['"IN"','"OUT"','"REVIEW"'])
def test_voice(): assert 'SpeechSynthesisUtterance("OUT")' in rd("ui/src/useOutVoice.ts")
def test_dwell(): assert "2800" in rd("ui/src/RoleAwareCallOverlay.tsx")
def test_default_player(): assert '"PLAYER"' in rd("ui/src/presentationMode.ts")
def test_player_overlay(): assert "playerOutOverlay" in rd("ui/src/RoleAwareCallOverlay.tsx")
def test_ref_evidence(): assert "nearest_line" in rd("ui/src/RoleAwareCallOverlay.tsx")
def test_ref_distance(): assert "signed_distance_in" in rd("ui/src/RoleAwareCallOverlay.tsx")
def test_switcher_css(): assert ".modeSwitcher" in rd("ui/src/styles.cp00344b.css")
def test_video_wrapper_css(): assert ".videoRoleWrap" in rd("ui/src/styles.cp00344b.css")
