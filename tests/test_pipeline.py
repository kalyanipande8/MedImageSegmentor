import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_dicom_dataset import simulate
from src.train_and_evaluate import train_and_evaluate, dice_score
import numpy as np


def test_at_least_1200_images():
    images, masks = simulate()
    assert len(images) >= 1200


def test_dice_score_perfect_match():
    mask = np.ones((10, 10))
    assert dice_score(mask, mask) == 1.0


def test_dice_score_no_overlap():
    pred = np.zeros((10, 10))
    pred[:5] = 1
    true = np.zeros((10, 10))
    true[5:] = 1
    assert dice_score(pred, true) < 0.1


def test_unet_achieves_reasonable_dice_on_held_out_test_set():
    images, masks = simulate()
    result = train_and_evaluate(images, masks)
    assert result["mean_dice_score"] > 0.6


def test_train_test_split_sizes_consistent():
    images, masks = simulate()
    result = train_and_evaluate(images, masks)
    assert result["n_train"] + result["n_test"] == len(images)
