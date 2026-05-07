# Change Detection on EO-SAR Imagery

## Description
Binary pixel-level change detection using paired EO and SAR imagery.

## Requirements
- Python 3.10+
- torch
- segmentation-models-pytorch
- albumentations
- numpy
- Pillow
- scikit-learn
- pyyaml

## Environment Setup
pip install torch segmentation-models-pytorch albumentations numpy Pillow scikit-learn pyyaml

## Training
python train.py

## Evaluation
python eval.py --data_path /path/to/test --weights /path/to/best_model.pth

## Model Weights
Download from Google Drive: https://drive.google.com/file/d/1YR9du4oqcXO0CF-OVl3OPfZN3KOalV4l/view?usp=drivesdk

## Results
### Validation Split
| Metric | Score |
|--------|-------|
| IoU | 0.3769 |
| Precision | 0.4631 |
| Recall | 0.6696 |
| F1 Score | 0.5475 |

### Test Split
| Metric | Score |
|--------|-------|
| IoU | 0.0275 |
| Precision | 0.0410 |
| Recall | 0.0776 |
| F1 Score | 0.0536 |
