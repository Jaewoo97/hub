import json
import os
from pathlib import Path

import pytest

fixtures_directory = Path(__file__).resolve().parent


@pytest.fixture
def melanoma_model():
    melanoma_directory = os.path.join(fixtures_directory, "melanoma")

    model_key = "921d61d5ceefbc953f3d0cfe7d91dc0a"
    project_secret = None

    images = [
        os.path.join(melanoma_directory, "361.jpg"),
        os.path.join(melanoma_directory, "362.jpg"),
        os.path.join(melanoma_directory, "363.jpg")
    ]

    image_size = (1024, 1024)
    threshold = 0.7

    predictions_filename = ("predictions_windows.json"
                            if os.name == "nt" else "predictions.json")

    with open(os.path.join(melanoma_directory,
                           predictions_filename)) as predictions_file:
        predictions = json.load(predictions_file)

    return (model_key, project_secret, images, image_size, threshold,
            predictions)
