"""This module contains the utility function to obtain the.

height and width form the pipeline config filepath
"""


def dims_from_config(pipeline_config_file):
    """Get height and width from the pipeline config file path.

    :param: pipeline_config_file: pipeline config file path
    :return: tuple of (model_height, model_width)
    """
    with open(pipeline_config_file, "r") as opened_file:
        line_list = opened_file.readlines()
        if "fixed_shape_resizer" in line_list[4]:
            model_height = int(line_list[5].strip().replace("height: ", ""))
            model_width = int(line_list[6].strip().replace("width: ", ""))
        elif "fixed_shape_resizer" in line_list[5]:
            model_height = int(line_list[6].strip().replace("height: ", ""))
            model_width = int(line_list[7].strip().replace("width: ", ""))
        elif "keep_aspect_ratio_resizer" in line_list[4]:
            model_height = int(
                line_list[6].strip().replace("max_dimension: ", "")
            )
            model_width = model_height
    return (model_height, model_width)
