from linecaller.open_play import OpenPlayPlayer, OpenPlayRotationEngine

def players(n,start=1):
    return [OpenPlayPlayer(str(i),f"P{i}") for i in range(start,start+n)]

def test_four_players_fill_court():
    e=OpenPlayRotationEngine()
    for p in players(4): e.add_player(p)
    r=e.fill_court()
    assert r.action=="FOUR_ON"
    assert len(r.on_court)==4

def test_three_players_cannot_form_doubles_game():
    e=OpenPlayRotationEngine()
    for p in players(3): e.add_player(p)
    assert e.fill_court().action=="WAITING"

def test_eight_players_rotate_four_on_four_off():
    e=OpenPlayRotationEngine()
    for p in players(8): e.add_player(p)
    e.fill_court()
    r=e.complete_game()
    assert r.action=="FOUR_OFF_FOUR_ON"
    assert [p.name for p in r.on_court]==["P5","P6","P7","P8"]

def test_departing_four_go_to_back_of_queue():
    e=OpenPlayRotationEngine()
    for p in players(8): e.add_player(p)
    e.fill_court(); e.complete_game()
    assert [p.name for p in e.waiting]==["P1","P2","P3","P4"]

def test_twelve_players_preserve_fifo():
    e=OpenPlayRotationEngine()
    for p in players(12): e.add_player(p)
    e.fill_court(); e.complete_game()
    assert [p.name for p in e.on_court]==["P5","P6","P7","P8"]
    assert [p.name for p in e.waiting]==["P9","P10","P11","P12","P1","P2","P3","P4"]

def test_second_rotation_advances_next_four():
    e=OpenPlayRotationEngine()
    for p in players(12): e.add_player(p)
    e.fill_court(); e.complete_game(); e.complete_game()
    assert [p.name for p in e.on_court]==["P9","P10","P11","P12"]

def test_duplicate_player_is_rejected():
    e=OpenPlayRotationEngine()
    p=players(1)[0]
    e.add_player(p)
    assert e.add_player(p).reason=="PLAYER_ALREADY_PRESENT"

def test_no_rotation_when_less_than_four_waiting():
    e=OpenPlayRotationEngine()
    for p in players(6): e.add_player(p)
    e.fill_court()
    before=e.on_court
    r=e.complete_game()
    assert r.action=="WAITING"
    assert e.on_court==before

def test_rotation_reports_rotated_off_players():
    e=OpenPlayRotationEngine()
    for p in players(8): e.add_player(p)
    e.fill_court()
    r=e.complete_game()
    assert [p.name for p in r.rotated_off]==["P1","P2","P3","P4"]

def test_queue_cycles_fairly():
    e=OpenPlayRotationEngine()
    for p in players(8): e.add_player(p)
    e.fill_court()
    e.complete_game()
    e.complete_game()
    assert [p.name for p in e.on_court]==["P1","P2","P3","P4"]
