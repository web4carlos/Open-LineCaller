from linecaller.open_play import OpenPlayPlayer,OpenPlayRotationEngine

def names(xs): return ",".join(p.name for p in xs) or "-"

def show(label,r):
    print(f"{label}: action={r.action} on_court={names(r.on_court)} waiting={names(r.waiting)} reason={r.reason}")

def main():
    e=OpenPlayRotationEngine()
    for i in range(1,13):
        e.add_player(OpenPlayPlayer(str(i),f"P{i}"))

    show("initial_four",e.fill_court())
    show("game1_complete",e.complete_game())
    show("game2_complete",e.complete_game())
    show("game3_complete",e.complete_game())

if __name__=="__main__":
    main()
