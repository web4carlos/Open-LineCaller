# CP-0012 User Guide

## Create dataset

    python -m tools.dataset_init --root .\datasets\pickleball_v1

## Add clips

For now, copy videos into:

    datasets\pickleball_v1\clips\

and register them in manifest.json.

The graphical Dataset Studio planned next will automate this.

## Validate dataset

    python -m tools.dataset_validate --root .\datasets\pickleball_v1

## Split dataset

    python -m tools.dataset_split --root .\datasets\pickleball_v1 --seed 42

## Export YOLO

    python -m tools.dataset_export_yolo --root .\datasets\pickleball_v1 --output .\datasets\exports\pickleball_v1_yolo
