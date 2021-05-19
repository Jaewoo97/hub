"""Predictions for trained TensorFlow models."""
import os
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_image_into_numpy_array(
        path: str, height: int,
        width: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load and resize an image from a file into a NumPy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    :param path: The file path to the image
    :param width: The width the image should have
    :param height: The height the image should have
    :return: uint8 NumPy array with image data, and a tuple containing the
        original image dimensions
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((height, width))
    return np.array(image_resized), (image_shape[0], image_shape[1])


def run_prediction(
    trained_model,
    images: List[str],
    image_size: Tuple[int, int],
    threshold: float,
) -> Any:
    """Run a prediction on a set of images.

    :param trained_model: The TensorFlow model to use for prediction
    :param images: A list of file paths containing the images to predict
    :param image_size: The size the images must be resized to for the model
    :param threshold: Prediction confidence threshold
    :return: The model predictions
    """
    height, width = image_size

    def predict_one_image(image):
        """Run prediction for one image."""
        image_resized, _ = load_image_into_numpy_array(image, height, width)

        input_tensor = tf.convert_to_tensor(image_resized)

        input_tensor = input_tensor[tf.newaxis, ...]

        detections_output = trained_model(input_tensor)

        num_detections = int(detections_output.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections_output.items()
        }

        detections["num_detections"] = num_detections

        indexes = np.where(detections["detection_scores"] > threshold)

        bboxes = detections["detection_boxes"][indexes].tolist()
        classes = (detections["detection_classes"][indexes].astype(
            np.int64).tolist())
        scores = detections["detection_scores"][indexes].tolist()

        return {"bboxes": bboxes, "classes": classes, "scores": scores}

    return [predict_one_image(image) for image in images]


def predictions_equal(actual, expected) -> bool:
    """Determine if two prediction objects are equal.

    :param actual: The actual prediction object
    :param expected: The expected prediction object
    :return: True if actual == expected, False otherwise
    """

    def one_prediction_equal(actual_one, expected_one):
        bboxes_actual = np.array(actual_one["bboxes"])
        classes_actual = np.array(actual_one["classes"])
        scores_actual = np.array(actual_one["scores"])

        bboxes_expected = np.array(expected_one["bboxes"])
        classes_expected = np.array(expected_one["classes"])
        scores_expected = np.array(expected_one["scores"])

        return (np.allclose(bboxes_actual, bboxes_expected)
                and np.allclose(classes_actual, classes_expected)
                and np.allclose(scores_actual, scores_expected))

    return all((one_prediction_equal(actual[i], expected_one)
                for i, expected_one in enumerate(expected)))
