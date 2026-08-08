# CP-0014C User Guide

## First run

Open Dataset Studio and press Propose.

If the YOLO model is available, the suggested box comes from YOLO.
If the model cannot load, Dataset Studio stays usable and temporal proposals
can still be used.

## Generic baseline limitations

The default model recognizes the generic COCO `sports ball` class.
A pickleball is small and often blurred, so false negatives are expected.

Do not judge final Open-LineCaller accuracy from this baseline.

The important milestone is that real AI inference now flows through the same
ProposalEngine interface used by the annotation system.

## Custom weights

Set:

    $env:OPEN_LINECALLER_YOLO_WEIGHTS = "C:\models\pickleball.pt"

before opening Dataset Studio.
