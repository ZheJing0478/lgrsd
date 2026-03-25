# LG-RSD-AttnFPN for SAR Ship Detection

Code release for the SAR ship detection project built on a local Ultralytics YOLOv8 fork with two project-specific components:

- `AttnFPN`: lightweight channel recalibration on fused pyramid features
- `LG-RSD`: training-only local-global region self-distillation

The main detector used in this repository is `YOLOv8n + LG-RSD + AttnFPN`. LG-RSD is enabled only during training and adds no inference-time branch.

## What Is Included

- Custom model definitions in `custom_models/`
- Project-specific extensions in `ultralytics_ext/`
- Training, evaluation, export, and verification scripts in `tools/`
- The local Ultralytics fork in `method/yolov8/`
- Packaging and documentation files for a code-only open-source release

## What Is Not Included

- Raw datasets
- Prepared dataset outputs under `datasets/`
- Training runs under `runs/`
- Model weights, checkpoints, and binary exports
- Internal notes or machine-specific artifacts

## Quick Start

1. Create and activate a Python environment.
2. Install PyTorch separately for your CUDA or CPU setup.
3. Install project dependencies:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -e method/yolov8
   ```

4. Prepare raw datasets as described in [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).
5. Generate YOLO-format datasets:

   ```bash
   python tools/prepare_datasets.py --dataset all --overwrite
   ```

6. Run training:

   ```bash
   bash RUN_ALL.sh --dataset hrsid --methods final --epochs 100 --imgsz 1024 --batch 8 --model_scale n
   ```

If you prefer not to activate an environment first, you can set `CONDA_ENV`:

```bash
CONDA_ENV=myenv bash tools/train_all.sh --dataset ssdd --methods baseline
```

## Documentation

- Dataset preparation: [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)
- Reproduction commands: [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)
- Packaging the clean release: [`docs/PACKAGING.md`](docs/PACKAGING.md)

## Repository Layout

- `custom_models/`: custom YAML model specs
- `method/yolov8/`: local Ultralytics fork used by this project
- `tools/`: training, evaluation, export, and verification scripts
- `ultralytics_ext/`: AttnFPN and LG-RSD implementation

## Packaging a Clean Open-Source Bundle

Run:

```bash
bash tools/package_open_source.sh
```

This creates a code-only release under `dist/` and excludes datasets, runs, weights, local paths, and user-specific information.

## License

This repository includes and extends Ultralytics code. The packaged release copies the AGPL-3.0 license from the local Ultralytics fork into the release bundle.

