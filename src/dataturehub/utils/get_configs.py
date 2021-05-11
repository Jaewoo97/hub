import tensorflow as tf
from google.protobuf import text_format

from .object_detection.protos import pipeline_pb2


def create_configs_from_pipeline_proto(pipeline_config):
    """Creates a configs dictionary from pipeline_pb2.TrainEvalPipelineConfig.

    :param pipeline_config: pipeline_pb2.TrainEvalPipelineConfig proto object.
    :return: Dictionary of configuration objects. Keys are `model`, `train_config`,
        `train_input_config`, `eval_config`, `eval_input_configs`. Value are
        the corresponding config objects or list of config objects (only for
        eval_input_configs).
    """
    configs = {}
    configs["model"] = pipeline_config.model
    configs["train_config"] = pipeline_config.train_config
    configs["train_input_config"] = pipeline_config.train_input_reader
    configs["eval_config"] = pipeline_config.eval_config
    configs["eval_input_configs"] = pipeline_config.eval_input_reader
    #   Keeps eval_input_config only for backwards compatibility. All clients should
    #   read eval_input_configs instead.
    if configs["eval_input_configs"]:
        configs["eval_input_config"] = configs["eval_input_configs"][0]
    if pipeline_config.HasField("graph_rewriter"):
        configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

    return configs


def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
    """Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.

    :param pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text proto.
    :param config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to override pipeline_config_path.
    :return:Dictionary of configuration objects. Keys are `model`, `train_config`, `train_input_config`, `eval_config`, `eval_input_config`.
    Value are the corresponding config objects.
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    if config_override:
        text_format.Merge(config_override, pipeline_config)
    return create_configs_from_pipeline_proto(pipeline_config)


def get_height_width_from_pipeline_file(pipeline_config_path):
    config_dict = get_configs_from_pipeline_file(pipeline_config_path)
    model_configs = config_dict["model"]
    if hasattr(model_configs, "ssd"):
        neural_network_configs = model_configs.ssd
    elif hasattr(model_configs, "faster_rcnn"):
        neural_network_configs = model_configs.faster_rcnn
    else:
        raise AttributeError(
            "ssd or faster_rcnn not found in the model_configs. perhaps a new model has been added?"
        )
    image_resizer_configs = neural_network_configs.image_resizer
    if hasattr(image_resizer_configs, "keep_aspect_ratio_resizer"):
        height = image_resizer_configs.keep_aspect_ratio_resizer.max_dimension
        width = image_resizer_configs.keep_aspect_ratio_resizer.max_dimension
    elif hasattr(image_resizer_configs, "fixed_shape_resizer"):
        height = image_resizer_configs.fixed_shape_resizer.height
        width = image_resizer_configs.fixed_shape_resizer.width
    else:
        raise AttributeError(
            "attributes 'keep_aspect_ratio_resizer' and 'fixed_shape_resizer' \
                not found in the pipeline config. Perhaps there's a new type?"
        )
    return (height, width)


if __name__ == "__main__":
    get_height_width_from_pipeline_file("pipeline_bbox.config")
