"""Module containing all visualize functions."""
import numpy as np
import cv2
from PIL import Image

from .color_switch import color_switch


def visualize_bbox(
    image_directory,
    detections_output,
    category_index,
    threshold,
):
    """Visualize bounding boxes.

    Prints the bounding boxes indicating the detection,
    along with their confidence scores on the image.
    :param image_directory: The path of the image to perform visualizations on
    :param detections_output: The detections that are being returned by the
        model
    :param category_index: The category index list
    :param threshold: The threshold confidence level required to be visualized
    """
    num_detections = int(detections_output.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections_output.items()
    }
    detections["num_detections"] = num_detections

    # Filter out predictions below threshold
    indexes = np.where(detections["detection_scores"] > float(threshold))

    # Extract predictions
    bboxes = detections["detection_boxes"][indexes]
    classes = detections["detection_classes"][indexes].astype(np.int64)
    scores = detections["detection_scores"][indexes]

    # Draw Predictions
    image_origi = np.array(Image.open(image_directory).convert("RGB"))
    origi_shape = image_origi.shape

    if len(bboxes) != 0:
        for idx, each_bbox in enumerate(bboxes):
            color = color_switch(classes[idx] - 1)

            # Draw bounding box
            image_origi = cv2.rectangle(
                image_origi,
                (
                    int(each_bbox[1] * origi_shape[1]),
                    int(each_bbox[0] * origi_shape[0]),
                ),
                (
                    int(each_bbox[3] * origi_shape[1]),
                    int(each_bbox[2] * origi_shape[0]),
                ),
                color,
                2,
            )

            # Draw label background
            image_origi = cv2.rectangle(
                image_origi,
                (
                    int(each_bbox[1] * origi_shape[1]),
                    int(each_bbox[2] * origi_shape[0]),
                ),
                (
                    int(each_bbox[3] * origi_shape[1]),
                    int(each_bbox[2] * origi_shape[0] + 15),
                ),
                color,
                -1,
            )

            # Insert label class & score
            image_origi = cv2.putText(
                image_origi,
                "Class: {}, Score: {}".format(
                    str(category_index[classes[idx]]["name"]),
                    str(round(scores[idx], 2)),
                ),
                (
                    int(each_bbox[1] * origi_shape[1]),
                    int(each_bbox[2] * origi_shape[0] + 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return image_origi
