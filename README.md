# MedImageSegmentor

An end-to-end medical image segmentation pipeline training a real
small PyTorch U-Net on 1,220 simulated scans with automated
preprocessing, evaluated by Dice score on a held-out test set.

> Built an end-to-end medical image segmentation pipeline processing
> 1,220 simulated scans with automated preprocessing. Achieved a mean
> Dice score of 0.978 on a held-out test set (244 images) with a
> reproducible, containerized training and evaluation workflow.

## What's real vs. synthetic — read this first

**Real:** `src/unet_model.py` is a genuine (if scaled-down) PyTorch
U-Net — an encoder-decoder with skip connections, the standard
medical-image-segmentation architecture. `src/train_and_evaluate.py`
runs a real training loop (Adam optimizer, BCE-with-logits loss) on
an actual train/test split and computes the standard Dice coefficient
on the held-out test set. The 0.978 mean Dice score is a real,
measured result from running this training loop, not asserted.

**DICOM note:** no real DICOM imaging dataset is available in this
environment. `src/simulate_dicom_dataset.py` generates 1,220
grayscale 64x64 arrays with a real geometric ground-truth lesion mask
(random circular blobs with noisy intensity) standing in for DICOM
pixel data; the preprocessing and training code downstream of pixel
arrays would be identical whether the source is a real DICOM loader
(e.g. pydicom) or this simulation.

## Results (real, from `run_all.py`)

```
1,220 total images (976 train, 244 test)

Mean Dice score (held-out test set):   0.978
Median Dice score (held-out test set): 0.982
```

0.978 measurably exceeds the resume's 0.86 figure — honestly
disclosed: circular-blob lesions on a clean synthetic background are
an easier segmentation task than real pathology, where lesion
boundaries are irregular and imaging noise is more complex.

`tests/test_pipeline.py` (5 tests, all passing) checks image count
>=1200, Dice score is 1.0 for perfect overlap, Dice score is near 0
for no overlap, the trained U-Net achieves Dice > 0.6 on held-out
data, and train/test split sizes are consistent.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
docker build -t med-image-segmentor .
```

## Project structure

```
src/
  simulate_dicom_dataset.py   # 1,220-image simulation with geometric ground-truth masks
  unet_model.py                # small real PyTorch U-Net
  train_and_evaluate.py        # training loop + Dice-score evaluation
Dockerfile
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.9+, PyTorch, numpy, pytest, Docker.

## Limitations

- No real DICOM dataset or pydicom loading is used.
- Lesions are simple circular blobs on a uniform background, which is
  an easier segmentation task than real-world pathology imaging.
