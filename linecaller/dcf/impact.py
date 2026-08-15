from .models import CellIndex, FieldRegion, ImpactEvent
from .engine import DynamicCourtField

OUT_REGIONS = {
    FieldRegion.OUT_LEFT,
    FieldRegion.OUT_RIGHT,
    FieldRegion.OUT_NEAR,
    FieldRegion.OUT_FAR,
    FieldRegion.OUT_CORNER,
}

class ImpactEngine:
    """
    Converts a confirmed z=0 DCF contact into an officiating impact event.

    Player Mode rule:
      INSIDE / BOUNDARY -> silent
      EXTERNAL          -> illuminate floor tile + audio "OUT"
    """
    def __init__(self, field: DynamicCourtField):
        self.field=field
        self.last_impact=None

    def process_contact(self, cell: CellIndex):
        region=self.field.contact_region(cell)
        if region is None:
            return None

        is_out=region in OUT_REGIONS
        event=ImpactEvent(
            cell=cell,
            region=region,
            is_out=is_out,
            illuminate_floor=is_out,
            audio_call="OUT" if is_out else None,
        )
        self.last_impact=event
        return event
