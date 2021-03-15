"""A loader for models trained on the Datature platform."""
import enum
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional
import zipfile

import requests
import tensorflow as tf


class ModelType(enum.Enum):

    """A type of machine learning model."""

    TF = "TF"
    """ProtoBuf model usable with TensorFlow"""


def get_default_hub_dir():
    """Get the default directory where downloaded models are saved."""
    return os.path.join(Path.home(), ".dataturehub")


def _get_signed_url_for_model(auth_key: str, deploy_key: str) -> str:
    nexpresso_endpoint = "http://.../api/deploy"

    response = requests.get(nexpresso_endpoint,
                            params={
                                "authKey": auth_key,
                                "deployKey": deploy_key
                            })

    response.raise_for_status()

    response_json = response.json()

    return response_json["signedURL"]["url"]


def _save_model_file(signed_url: str, destination_path: str,
                     progress: bool) -> None:
    """Download a model file and display progress information if requested."""
    if progress:
        sys.stderr.write("Downloading model from Datature Hub...\n")

    with open(destination_path, "wb") as model_file:
        response = requests.get(signed_url, stream=True)

        response.raise_for_status()

        total_length = response.headers.get('content-length')

        if total_length is None:
            model_file.write(response.content)
            return

        total_length_mib = int(total_length) / (1024 * 1024)
        downloaded_so_far_mib = 0
        progress_bar_size = 50
        progress_bar_progress = 0

        for data in response.iter_content(chunk_size=4096):
            if progress:
                downloaded_so_far_mib += len(data) / (1024 * 1024)
                progress_bar_progress = int(progress_bar_size *
                                            downloaded_so_far_mib /
                                            total_length_mib)

                sys.stderr.write(
                    f"\r[{'=' * (progress_bar_progress)}"
                    f"{' ' * (progress_bar_size - progress_bar_progress)}"
                    f"] {downloaded_so_far_mib:.2f} / "
                    f"{total_length_mib:.2f} MiB")
                sys.stderr.flush()

            model_file.write(data)

        sys.stderr.write("\n")
        sys.stderr.flush()


def download_model(auth_key: str,
                   deploy_key: str,
                   destination: Optional[str] = None,
                   model_type: ModelType = ModelType.TF,
                   progress: bool = True) -> str:
    """Download a model, placing it in the ``destination`` directory.

    :param auth_key: Your Datature Hub authentication key
    :param deploy_key: The deploy key of the model to download
    :param destination: The path to the directory where the model will be
        saved, or ``None`` to use the default hub directory.
    :param model_type: The type of the model that should be downloaded
    :param progress: Whether to display progress information as the model
        downloads.
    :return: The directory where the model has been downloaded.
    """
    if destination is None:
        destination = get_default_hub_dir()

    model_folder = os.path.join(destination, deploy_key)

    Path(model_folder).mkdir(parents=True, exist_ok=True)

    try:
        model_signed_url = _get_signed_url_for_model(auth_key, deploy_key)

        if model_type == ModelType.TF:
            model_zip_path = os.path.join(model_folder, "model.zip")

            _save_model_file(model_signed_url, model_zip_path, progress)

            if progress:
                sys.stderr.write("Extracting model...\n")
                sys.stderr.flush()

            with zipfile.ZipFile(model_zip_path, "r") as model_zip_file:
                model_zip_file.extractall(model_folder)

            os.remove(model_zip_path)
        else:
            raise ValueError(f"Invalid model type {model_type}.")
    except Exception as exc:
        shutil.rmtree(model_folder, ignore_errors=True)
        raise exc

    return model_folder


def load_tf_model(auth_key: str,
                  deploy_key: str,
                  hub_dir: Optional[str] = None,
                  force_download: bool = False,
                  progress: bool = True,
                  **kwargs) -> Any:
    """Load a TensorFlow model.

    :param auth_key: Your Datature Hub authentication key
    :param deploy_key: The deploy key for the model to download
    :param hub_dir: The path to the model cache folder, or
        ``None`` to use the default hub directory.
    :param force_download: Whether to download the model from Datature Hub
        even if a copy already exists in the model cache folder.
    :param progress: Whether to display progress information as the model
        downloads.
    :param **kwargs: Additional keyword arguments to pass to the TensorFlow
        model loader.
    :return: The loaded TensorFlow model
    """
    if hub_dir is None:
        hub_dir = get_default_hub_dir()

    model_folder = os.path.join(hub_dir, deploy_key)

    if force_download or not os.path.exists(model_folder):
        download_model(auth_key, deploy_key, hub_dir, ModelType.TF, progress)

    return tf.saved_model.load(
        os.path.join(model_folder, "exported_model", "saved_model"), **kwargs)
