import os
import subprocess
import ast
from tqdm import tqdm
from pathlib import Path
import argparse


def get_ncc_paths(model_dir):
    ncc_file_path = os.path.join(
        "/home/heloise/workspace/vergesense",
        "k210/model_tools/nncase_release/0.2_beta4/ncc",
    )
    process_inference_file_path = os.path.join(
        "/home/heloise/workspace/vergesense",
        "k210/model_tools/process_inference/ProcessInference",
    )
    weights_file_path = os.path.join(model_dir, "anchor_weights.bin")
    return ncc_file_path, process_inference_file_path, weights_file_path


def k210_predict(model_dir, images_dir, debug_output_dir, subclasses_file):
    return_code_list = []

    (
        ncc_file_path,
        process_inference_file_path,
        weights_file_path,
    ) = get_ncc_paths(model_dir)

    images = os.listdir(images_dir)
    num_classes = sum(1 for line in open(subclasses_file)) + 1
    os.makedirs(debug_output_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_output_dir, 'json'), exist_ok=True)
    for img in tqdm(images):
        base_name, ext = img.split(".")
        img_kmodel_bin_path = os.path.join(debug_output_dir, 'bin', base_name + ".bin")
        debug_json_out_path = os.path.join(debug_output_dir, 'json', base_name + ".json")

        cmd = [
            process_inference_file_path,
            "-f",
            img_kmodel_bin_path,
            "-a",
            weights_file_path,
            "-x",
            str(num_classes),
            "-s",
            subclasses_file,
            "-i",
            "0.1",
            "-c",
            "0.4",
            "-t",
            "json",
            "-m",
            "-o",
            debug_json_out_path,
        ]

        # process = subprocess.run(cmd, capture_output=True, text=True)
        # print("Process stdout ", process)

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        data = stdout.decode("utf-8")
        data = ast.literal_eval(data)
        detections = data["detections"]
        error = stderr.decode("utf-8")
        if process.returncode != 0:
            return_code_list.append(process.returncode)
            print("return code ", process.returncode)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        default="/home/heloise/workspace/vergesense/pretrained/signs_of_life/model_zoo/Triceratops_B/v2/k210",
        nargs="?",
        help="path to k210 model",
    )

    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        default="/home/habib/data/k210/pred/test/jsons",
        nargs="?",
        help="path for storing model predictions in json format",
    )

    parser.add_argument(
        "--bin_images_dir",
        type=str,
        required=True,
        default="/home/habib/data/k210/bin/test/",
        nargs="?",
        help="Path where binary images are stored that will be used to run infrence",
    )

    parser.add_argument(
        "--subclasses_file",
        type=str,
        required=True,
        default="/home/habib/k210_svm/sol.txt",
        nargs="?",
        help="Path to subclass file which contains class indices",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        default="/home/habib/data/k210/pred/test/labels.csv",
        nargs="?",
        help="Path for storing csv file After json file is generated they will be converted to 1 csv file",
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        default=320,
        nargs="?",
        help="Widht of the image",
    )


    parser.add_argument(
        "--height",
        type=int,
        required=True,
        default=240,
        nargs="?",
        help="Height of the image",
    )

    args = parser.parse_args()
    #runing prediction and generating json files
    k210_predict(args.model_dir, args.bin_images_dir, args.json_dir, args.subclasses_file)
    #getting all json files
    jsons = list(Path(args.json_dir).glob('*.json'))
    print("jsons pred ", jsons)