import os
import subprocess
import argparse



def CreateBinairies(
    ncc_file_path, model_dir, path_to_model, images_dir, debug_output_dir
):
    ncc_output_dir = debug_output_dir
    run_ncc = False
    if not os.path.isdir(ncc_output_dir):
        os.makedirs(ncc_output_dir)
        run_ncc = True

    if run_ncc:
        print("Run NCC")
        cmd = [
            ncc_file_path,
            "infer",
            path_to_model,
            ncc_output_dir,
            "--dataset",
            images_dir,
        ]
        out = subprocess.call(cmd)
    return ncc_output_dir


def Set_Ncc_Process_Paths(model_dir):
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


def make_binaries(model_dir, images_dir, debug_output_dir, path_to_model):
    return_code_list = []
    (
        ncc_file_path,
        process_inference_file_path,
        weights_file_path,
    ) = Set_Ncc_Process_Paths(model_dir)
    print('ncc_file_path, process_inference_file_path, weights_file_path ', ncc_file_path, process_inference_file_path, weights_file_path)
    images = os.listdir(images_dir)
    ncc_output_dir = CreateBinairies(
        ncc_file_path, model_dir, path_to_model, images_dir, debug_output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        default="/home/heloise/workspace/vergesense/pretrained/signs_of_life/model_zoo/Triceratops_B/v2/k210",
        nargs="?",
        help="path to k210 model directory",
    )

    parser.add_argument(
        "--path_to_model",
        type=str,
        required=True,
        default="/home/heloise/workspace/vergesense/pretrained/signs_of_life/model_zoo/Triceratops_B/v2/k210/pod.kmodel",
        nargs="?",
        help="path to k210 model",
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        default="/home/habib/data/k210/images/roche",
        nargs="?",
        help="path for storing model predictions in json format",
    )

    parser.add_argument(
        "--output_bin_dir",
        type=str,
        required=True,
        default="/home/habib/data/k210/bin/roche/",
        nargs="?",
        help="Path where generated binary images will be stored",
    )

    args = parser.parse_args()
    make_binaries(args.model_dir, args.images_dir, args.output_bin_dir, args.path_to_model)
