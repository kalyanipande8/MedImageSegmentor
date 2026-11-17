import json
import os

from src.simulate_dicom_dataset import simulate
from src.train_and_evaluate import train_and_evaluate


def main():
    images, masks = simulate()
    result = train_and_evaluate(images, masks)

    summary = {"n_images_total": len(images), **result}

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
