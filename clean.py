import os
import argparse
import shutil
import yaml


def read_yaml(fn: str) -> dict:
    """_summary_
    Args:
        fn (str): yaml file
    Returns:
        _type_: dictionary
    """
    with open(fn, "r") as stream:
        d = yaml.safe_load(stream)
    return d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        required=False,
        nargs="?",
        help='Choose a model type: full_model, k210, tflite or onnx',
    )
    
    args = parser.parse_args()

    # Read yaml
    conf = read_yaml("./config.yml")

    if args.model_type is None:
        # Delete all results (i.e. results folder)
        debug_output_dir = conf['debug_output_dir']
        shutil.rmtree(debug_output_dir)
    else:
        # Verify that selected model type is correct
        if args.model_type not in conf['models'].keys():
            raise ValueError('Wrong model type. Enter full_model, k210, tflite or onnx.')
        debug_output_dir = os.path.join(conf['debug_output_dir'], args.model_type)
        # Delete results of the selected model
        shutil.rmtree(debug_output_dir)