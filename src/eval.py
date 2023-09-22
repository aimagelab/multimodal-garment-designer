import os

#  external libraries
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

# custom imports
from datasets.dresscode import DressCodeDataset
from datasets.vitonhd import VitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from mgd_pipelines.mgd_pipe_disentangled import MGDPipeDisentangled
from utils.arg_parser import eval_parse_args
from utils.image_from_pipe import generate_images_from_mgd_pipe
from utils.set_seeds import set_seed

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def main() -> None:
    args = eval_parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    unet = torch.hub.load(dataset=args.dataset, repo_or_dir='aimagelab/multimodal-garment-designer', source='github',
                          model='mgd', pretrained=True)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Enable memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.category:
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            radius=5,
            sketch_threshold_range=(20, 20),
            tokenizer=tokenizer,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            sketch_threshold_range=(20, 20),
            radius=5,
            tokenizer=tokenizer,
            size=(512, 384),
        )
    else:
        raise NotImplementedError

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers_test,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    unet.eval()
    # Select fast classifier free guidance or disentagle classifier free guidance according to the disentagle parameter in args
    with torch.inference_mode():
        if args.disentagle:
            val_pipe = MGDPipeDisentangled(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet.to(vae.dtype),
                tokenizer=tokenizer,
                scheduler=val_scheduler,
            ).to(device)
        else:
            val_pipe = MGDPipe(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet.to(vae.dtype),
                tokenizer=tokenizer,
                scheduler=val_scheduler,
            ).to(device)

        val_pipe.enable_attention_slicing()
        test_dataloader = accelerator.prepare(test_dataloader)
        generate_images_from_mgd_pipe(
            test_order=args.test_order,
            pipe=val_pipe,
            test_dataloader=test_dataloader,
            save_name=args.save_name,
            dataset=args.dataset,
            output_dir=args.output_dir,
            guidance_scale=args.guidance_scale,
            guidance_scale_pose=args.guidance_scale_pose,
            guidance_scale_sketch=args.guidance_scale_sketch,
            sketch_cond_rate=args.sketch_cond_rate,
            start_cond_rate=args.start_cond_rate,
            no_pose=False,
            disentagle=False,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
