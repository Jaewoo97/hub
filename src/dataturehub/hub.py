"""A loader for models trained on the Datature platform."""
import enum
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional, NamedTuple
import zipfile

import requests
import tensorflow as tf

_config = {"hub_endpoint": "https://api.datature.io/hub"}


class ModelType(enum.Enum):

    """A type of machine learning model."""

    TF = "TF"
    """ProtoBuf model usable with TensorFlow"""


_ModelURLWithHash = NamedTuple("ModelURLWithHash", [('url', str),
                                                    ('checksum', str)])
"""A URL to download a model file along with its SHA256 checksum."""


def get_default_hub_dir():
    """Get the default directory where downloaded models are saved."""
    return os.path.join(Path.home(), ".dataturehub")


def _set_hub_endpoint(endpoint: str) -> None:
    """Set the Datature Hub API endpoint to a different URL."""
    _config["hub_endpoint"] = endpoint


def _get_model_url_and_hash(
        model_key: str, project_secret: Optional[str]) -> _ModelURLWithHash:
    """Get the URL and SHA256 hash of a model file."""
    api_params = {"modelKey": model_key}

    if project_secret is not None:
        api_params["projectSecret"] = project_secret

    response = requests.get(_config["hub_endpoint"], params=api_params)

    response.raise_for_status()

    response_json = response.json()

    if response_json["status"] != "ready":
        raise RuntimeError("Model is not ready to download.")

    if not response_json["projectSecretNeeded"] and project_secret is not None:
        sys.stderr.write(
            "WARNING: Project secret unnecessarily supplied when downloading"
            f"public model {model_key}.")
        sys.stderr.flush()

    return _ModelURLWithHash(response_json["signedUrl"], response_json["hash"])


def _get_sha256_hash_of_file(filepath: str, progress: bool) -> str:
    """Compute the SHA256 checksum of a file."""
    hash_f = hashlib.sha256()
    chunk_size = 1024 * 1024

    with open(filepath, 'rb') as file_to_hash:
        total_mib = os.fstat(file_to_hash.fileno()).st_size / (1024 * 1024)
        read_mib = 0

        while True:
            chunk = file_to_hash.read(chunk_size)

            if not chunk:
                break

            read_mib += len(chunk) / (1024 * 1024)
            if progress:
                sys.stderr.write(
                    f"\rVerifying {read_mib:.2f} / {total_mib:.2f} MiB...")
                sys.stderr.flush()

            hash_f.update(chunk)

    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    return hash_f.hexdigest()


def _save_and_verify_model(url_with_hash: _ModelURLWithHash,
                           destination_path: str, progress: bool) -> None:
    """Download and verify the integrity of a model file."""
    if progress:
        sys.stderr.write("Downloading model from Datature Hub...\n")

    with open(destination_path, "wb") as model_file:
        response = requests.get(url_with_hash.url, stream=True)

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

    file_checksum = _get_sha256_hash_of_file(destination_path, progress)

    if file_checksum != url_with_hash.checksum:
        raise RuntimeError("Checksum of downloaded file "
                           f"({file_checksum}) does not match the expected "
                           f" value ({url_with_hash.checksum})")


def download_model(model_key: str,
                   project_secret: Optional[str] = None,
                   destination: Optional[str] = None,
                   model_type: ModelType = ModelType.TF,
                   progress: bool = True) -> str:
    """Download a model, placing it in the ``destination`` directory.

    :param model_key: The key of the model to download
    :param project_secret: The project secret, or ``None`` if no secret key
        is necessary.
    :param destination: The path to the directory where the model will be
        saved, or ``None`` to use the default hub directory.
    :param model_type: The type of the model that should be downloaded
    :param progress: Whether to display progress information as the model
        downloads.
    :return: The directory where the model has been downloaded.
    """
    if destination is None:
        destination = get_default_hub_dir()

    model_folder = os.path.join(destination, model_key)

    Path(model_folder).mkdir(parents=True, exist_ok=True)

    try:
        url_with_hash = _get_model_url_and_hash(model_key, project_secret)

        if model_type == ModelType.TF:
            model_zip_path = os.path.join(model_folder, "model.zip")

            _save_and_verify_model(url_with_hash, model_zip_path, progress)

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


def load_tf_model(model_key: str,
                  project_secret: Optional[str] = None,
                  hub_dir: Optional[str] = None,
                  force_download: bool = False,
                  progress: bool = True,
                  **kwargs) -> Any:
    """Load a TensorFlow model.

    :param model_key: The key of the model to load
    :param project_secret: The project secret, or ``None`` if no secret key
        is necessary.
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

    model_folder = os.path.join(hub_dir, model_key)

    if force_download or not os.path.exists(model_folder):
        download_model(model_key, project_secret, hub_dir, ModelType.TF,
                       progress)

    return tf.saved_model.load(os.path.join(model_folder, "saved_model"),
                               **kwargs)
