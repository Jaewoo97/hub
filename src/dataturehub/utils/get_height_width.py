def get_height_width_from_pipeline_config(pipeline_config_file):
    with open(pipeline_config_file, "r") as opened_file:
        line_list = opened_file.readlines()
        if "fixed_shape_resizer" in line_list[4]:
            model_height = int(line_list[5].strip().replace("height: ", ""))
            model_width = int(line_list[6].strip().replace("width: ", ""))
        elif "keep_aspect_ratio_resizer" in line_list[4]:
            model_height = int(
                line_list[6].strip().replace("max_dimension: ", "")
            )
            model_width = model_height
    return (model_height, model_width)
