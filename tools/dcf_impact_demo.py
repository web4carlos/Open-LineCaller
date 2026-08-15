from linecaller.dcf import CellIndex, DynamicCourtField, ImpactEngine

def show(engine, cell):
    event=engine.process_contact(cell)
    print(
        f"cell={cell} region={event.region.value} "
        f"out={event.is_out} floor_on={event.illuminate_floor} "
        f"sound={event.audio_call or 'SILENT'}"
    )

def main():
    field=DynamicCourtField()
    impact=ImpactEngine(field)

    print("=== DCF IMPACT DEMO ===")
    show(impact, CellIndex(field.x0+10, field.y0+30, 0))
    show(impact, CellIndex(field.x0, field.y0+30, 0))
    show(impact, CellIndex(field.x1+1, field.y0+30, 0))

if __name__=="__main__":
    main()
