import os

import dataturehub.hub as hub

import predict


def test_load_tf_model(tmpdir, melanoma_model):

    if os.environ.get("HUB_CUSTOM_ENDPOINT", None) is not None:
        hub._set_hub_endpoint(os.environ["HUB_CUSTOM_ENDPOINT"])

    (
        model_key,
        project_secret,
        images,
        image_size,
        threshold,
        predictions,
    ) = melanoma_model
    hub_model = hub.HubModel(
        model_key=model_key, project_secret=project_secret, hub_dir=str(tmpdir)
    )

    trained_model = hub_model.load_tf_model()

    actual_predictions = predict.run_prediction(
        trained_model, images, image_size, threshold
    )

    assert predict.predictions_equal(actual_predictions, predictions)
