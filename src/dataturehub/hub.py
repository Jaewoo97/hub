"""A loader for models trained on the Datature platform."""
import enum
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional, NamedTuple
import zipfile

from PIL import Image
import numpy as np

import requests
import tensorflow as tf


_config = {
    "hub_endpoint": "https://asia-northeast1-datature-alchemy.cloudfunctions.net/invoke-hub-staging"
}


class ModelType(enum.Enum):

    """A type of machine learning model."""

    TF = "TF"
    """ProtoBuf model usable with TensorFlow"""


_ModelURLWithHash = NamedTuple(
    "ModelURLWithHash", [("url", str), ("checksum", str)]
)
"""A URL to download a model file along with its SHA256 checksum."""


class hub:
    def _set_hub_endpoint(self, endpoint: str) -> None:
        """Set the Datature Hub API endpoint to a different URL."""
        _config["hub_endpoint"] = endpoint

    def _get_model_url_and_hash(
        self, model_key: str, project_secret: Optional[str]
    ) -> _ModelURLWithHash:
        """Get the URL and SHA256 hash of a model file."""
        api_params = {"modelKey": model_key}

        if project_secret is not None:
            api_params["projectSecret"] = self.project_secret

        response = requests.post(_config["hub_endpoint"], json=api_params)

        response.raise_for_status()

        response_json = response.json()

        if response_json["status"] != "ready":
            raise RuntimeError("Model is not ready to download.")

        if (
            not response_json["projectSecretNeeded"]
            and project_secret is not None
        ):
            sys.stderr.write(
                "WARNING: Project secret unnecessarily supplied when downloading"
                f"public model {model_key}."
            )
            sys.stderr.flush()

        return _ModelURLWithHash(
            response_json["signedUrl"], response_json["hash"]
        )

    def _get_sha256_hash_of_file(self, filepath: str, progress: bool) -> str:
        """Compute the SHA256 checksum of a file."""
        hash_f = hashlib.sha256()
        chunk_size = 1024 * 1024

        with open(filepath, "rb") as file_to_hash:
            total_mib = os.fstat(file_to_hash.fileno()).st_size / (1024 * 1024)
            read_mib = 0

            while True:
                chunk = file_to_hash.read(chunk_size)

                if not chunk:
                    break

                read_mib += len(chunk) / (1024 * 1024)
                if progress:
                    sys.stderr.write(
                        f"\rVerifying {read_mib:.2f} / {total_mib:.2f} MiB..."
                    )
                    sys.stderr.flush()

                hash_f.update(chunk)

        if progress:
            sys.stderr.write("\n")
            sys.stderr.flush()

        return hash_f.hexdigest()

    def __init__(
        self,
        model_key: Optional[str] = None,
        project_secret: Optional[str] = None,
    ):
        self.hub_dir = os.path.join(Path.home(), ".dataturehub")
        self.model_key = model_key
        self.project_secret = project_secret
        self.model_url_and_hash = (
            (self._get_model_url_and_hash(self.model_key, self.project_secret))
            if model_key is not None
            else None
        )

    def get_default_hub_dir(self):
        return self.hub_dir

    def _save_and_verify_model(
        self, destination_path: str, progress: bool
    ) -> None:
        """Download and verify the integrity of a model file."""
        if progress:
            sys.stderr.write("Downloading model from Datature Hub...\n")

        with open(destination_path, "wb") as model_file:
            response = requests.get(self.model_url_and_hash.url, stream=True)

            response.raise_for_status()

            total_length = response.headers.get("content-length")

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
                    progress_bar_progress = int(
                        progress_bar_size
                        * downloaded_so_far_mib
                        / total_length_mib
                    )

                    sys.stderr.write(
                        f"\r[{'=' * (progress_bar_progress)}"
                        f"{' ' * (progress_bar_size - progress_bar_progress)}"
                        f"] {downloaded_so_far_mib:.2f} / "
                        f"{total_length_mib:.2f} MiB"
                    )
                    sys.stderr.flush()

                model_file.write(data)

            sys.stderr.write("\n")
            sys.stderr.flush()

        file_checksum = self._get_sha256_hash_of_file(
            destination_path, progress
        )

        if file_checksum != self.model_url_and_hash.checksum:
            raise RuntimeError(
                "Checksum of downloaded file "
                f"({file_checksum}) does not match the expected "
                f" value ({self.model_url_and_hash.checksum})"
            )

    def download_model(
        self,
        destination: Optional[str] = None,
        model_type: ModelType = ModelType.TF,
        progress: bool = True,
    ) -> str:
        """Download a model, placing it in the ``destinaton`` directory.

        :param destination: The path to the directory where the model will be
            saved, or ``None`` to use the default hub directory.
        :param model_type: The type of the model that should be downloaded
        :param progress: Whether to display progress information as the model
            downloads.
        :return: The directory where the model has been downloaded.
        """
        if destination is None:
            destination = self.hub_dir

        model_folder = os.path.join(destination, self.model_key)

        Path(model_folder).mkdir(parents=True, exist_ok=True)

        try:

            if model_type == ModelType.TF:
                model_zip_path = os.path.join(
                    model_folder, self.project_secret
                )

                self._save_and_verify_model(model_zip_path, progress)
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

    def load_tf_model(
        self,
        model_cache_dir: Optional[str] = None,
        force_download: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> Any:
        """Load a TensorFlow model.

        :param model_dir: The path to the model cache folder, or
            ``None`` to use the default hub directory.
        :param force_download: Whether to download the model from Datature Hub
            even if a copy already exists in the model cache folder.
        :param progress: Whether to display progress information as the model
            downloads.
        :param **kwargs: Additional keyword arguments to pass to the TensorFlow
            model loader.
        :return: The loaded TensorFlow model
        """
        if model_cache_dir is None:
            model_cache_dir = self.hub_dir
            model_folder = os.path.join(model_cache_dir, self.model_key)
        else:
            model_folder = model_cache_dir
        if force_download or not os.path.exists(model_folder):
            self.download_model(
                model_cache_dir,
                ModelType.TF,
                progress,
            )

        return tf.saved_model.load(
            os.path.join(model_folder, "saved_model"), **kwargs
        )

    def load_label_map_from_file(
        self,
        label_map_path: str,
    ) -> Any:
        """Load the label map for the Tensorflow model.

        :param label_map_path: The supplied directory to load the label map
        """
        label_map = {}

        with open(label_map_path, "r") as label_file:
            for line in label_file:
                if "id" in line:
                    label_index = int(line.split(":")[-1])
                    label_name = (
                        next(label_file).split(":")[-1].strip().strip("'")
                    )
                    label_map[label_index] = {
                        "id": label_index,
                        "name": label_name,
                    }

        return label_map

    def load_label_map(self) -> Any:
        """Load the label map for the Tensorflow model using the model key."""
        model_folder = os.path.join(self.hub_dir, self.model_key)
        if not os.path.exists(model_folder):
            raise FileNotFoundError(
                "The directory for model key " + model_key + " does not exist."
            )
        label_map_path = os.path.join(model_folder, "label_map.pbtxt")
        return self.load_label_map_from_file(label_map_path=label_map_path)

    def load_image(
        self,
        path: str,
        height: int,
        width: int,
    ) -> Any:
        """Load Image.

        Take in the path of an image, along with
        (height and width) parameters and returns an image tensor.
        :param path: The path of the image
        :param height: The height required by the model
        :param width: The width required by the model
        """
        image = Image.open(path).convert("RGB")
        image = image.resize((height, width))

        return tf.convert_to_tensor(np.array(image))[tf.newaxis, ...]

    def load_image_with_model_dimensions(
        self,
        path: str,
    ) -> Any:
        """Load Image with image settings retrieved from model.

        :param path: The path of the image
        """
        model_folder = os.path.join(self.hub_dir, self.model_key)
        if not os.path.exists(model_folder):
            raise FileNotFoundError(
                "The directory for model key "
                + self.model_key
                + " does not exist."
            )
        pipeline_path = os.path.join(model_folder, "pipeline.config")

        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                "pipeline.config not found in the model key directory. \
                Try re-downloading the model by \
                calling load_tf_model with force_download parameter = True."
            )
        with open(pipeline_path, "r") as opened_file:
            line_list = opened_file.readlines()
            for index, contents in enumerate(line_list):
                if "fixed_shape_resizer" in contents:
                    model_height = int(
                        line_list[index + 1].strip().replace("height: ", "")
                    )
                    model_width = int(
                        line_list[index + 2].strip().replace("width: ", "")
                    )
                    break
        return self.load_image(path, model_height, model_width)
