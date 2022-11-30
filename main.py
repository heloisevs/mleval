import yaml
import os
import argparse
import shutil
import subprocess
from inference.full_model.run_full_model import FullModel
from inference.k210.run_k210 import K210
from inference.tflite.run_tflite import TFLite
from utils.format_data import FormatDataK210, FormatDataFullModel, FormatDataTFLite


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

def update_yaml(path_to_yalm:str, conf:dict):
    with open(path_to_yalm, 'w') as f:
        yaml.dump({"nc":conf['nc'],"names":conf['labels']}, f, default_flow_style=None)

def clean(model_type: str) -> None:
    """ Delete the results folder of the selected model type after processing. """
    debug_output_dir = os.path.join(conf['debug_output_dir'], model_type)
    shutil.rmtree(debug_output_dir)

def run_inference(model_type: str, conf: dict) -> dict:
    if model_type == 'full_model':
        pb_model = FullModel(model_type, conf)
        jsons = pb_model.process()
        format_data_pb = FormatDataFullModel(jsons, conf)
        format_data_pb.jsons_to_csv()
    elif model_type == 'k210':
        k210_model = K210(model_type, conf)
        jsons = k210_model.process()
        format_data_k210 = FormatDataK210(jsons, conf)
        format_data_k210.jsons_to_csv()
    elif model_type == 'tflite':
        tflite_model = TFLite(model_type, conf)
        jsons = tflite_model.process()
        format_data_tflite = FormatDataTFLite(jsons, conf)
        format_data_tflite.jsons_to_csv()
    prediction_csv = os.path.join(conf['debug_output_dir'], model_type, model_type + '_debug.csv')
    return prediction_csv



def run_eval(ground_truth_csv:str, prediction_csv:str, conf:dict, model_type:str):
    eval_result_directory =  os.path.join("./results", model_type, "eval")
    # update the data yaml from the mlutils repo
    update_yaml("../mlutils/data/metric/data.yaml", conf)
    command = "python ../mlutils/eval.py --ground_truth_csv " + ground_truth_csv + " --prediction_csv " + prediction_csv + " --data_yaml ../mlutils/data/metric/data.yaml --conf_thresh 0.01 --result_directory " + eval_result_directory
    subprocess.run(command, shell=True, check=True)

def run(model_type: str, conf: dict, ground_truth_csv:str):
    prediction_csv = run_inference(model_type, conf)
    run_eval(ground_truth_csv, prediction_csv, conf, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        required=True,
        nargs="?",
        help='Choose a model type: full_model, k210, tflite or onnx',
    )
    parser.add_argument(
        "-gt",
        "--ground_truth_csv",
        type=str,
        required=True,
        nargs="?",
        help='Ground truth csv file label',
    )

    args = parser.parse_args()
    
    # Reading yaml
    conf = read_yaml('./config.yml')
    num_clases = int(conf["nc"])
    labels = {i: k for i, k in enumerate(conf["labels"])}
    print("labels ", labels)
    
    # Verify that selected model type is correct
    if args.model_type not in conf['models'].keys():
        raise ValueError('Wrong model type. Enter full_model, k210, tflite or onnx.')
    
    # Run
    run(args.model_type, conf, args.ground_truth_csv)
