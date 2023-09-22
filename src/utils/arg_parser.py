import os
import argparse


def eval_parse_args() -> argparse.Namespace:
    """ This function parses the arguments passed to the script.

    Returns:
        argparse.Namespace: Namespace containing the arguments.
    """
    
    parser = argparse.ArgumentParser(description="Multimodal Garment Designer argparse.")

    # Diffusion parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    #  destination folder
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions will be written.",
    )
    
    # Accelerator parameters
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    # dataset parameters
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument("--category", type=str, default="", help="category to use")
    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"],
                        help="Test order, should be either paired or unpaired")

    # dataloader parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (per device) for the test dataloader.")
    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="Number of workers for the test dataloader.")

    
    #  input parameters
    parser.add_argument("--mask_type", type=str, default="bounding_box", choices=["keypoints", "bounding_box"])
    parser.add_argument("--no_pose", action="store_true", help="exclude posemap from input")


    # disentagle classifier free guidance parameters
    parser.add_argument("--disentagle", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="text guidance scale, use with disentagle")
    parser.add_argument("--guidance_scale_pose", type=float, default=7.5,
                        help="pose guidance scale, use with disentagle")
    parser.add_argument("--guidance_scale_sketch", type=float, default=7.5,
                        help="sketch guidance scale, use with disentagle")
    
    # sketch conditioninig paramters
    parser.add_argument("--sketch_cond_rate", type=float, default=0.2, help="Sketch conditioning rate")
    parser.add_argument("--start_cond_rate", type=float, default=0.0, help="offset sketch cond rate")

    # miscelaneous parameters
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--save_name", type=str, required=True, help="Folder name of the saved images")

    args = parser.parse_args()

    # if not, set default local rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
