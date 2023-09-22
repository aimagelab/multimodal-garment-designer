import os
from tqdm import tqdm
import torch

import torchvision.transforms as T
from diffusers.pipeline_utils import DiffusionPipeline
from torch.utils.data import DataLoader
from src.utils.image_composition import compose_img, compose_img_dresscode


@torch.inference_mode()
def generate_images_from_mgd_pipe(
    test_order: bool,
    pipe: DiffusionPipeline,
    test_dataloader: DataLoader,
    save_name: str,
    dataset: str,
    output_dir: str,
    guidance_scale: float = 7.5,
    guidance_scale_pose: float = 7.5,
    guidance_scale_sketch: float = 7.5,
    sketch_cond_rate: float = 1.0,
    start_cond_rate: float = 0.0,
    no_pose: bool = False,
    disentagle: bool = False,
    seed: int = 1234,
    ) -> None:
    #This function generates images from the given test dataloader and saves them to the output directory.
    """
    Args:
        test_order: The order of the test dataset.
        pipe: The diffusion pipeline.
        test_dataloader: The test dataloader.
        save_name: The name of the saved images.
        dataset: The name of the dataset.
        output_dir: The output directory.
        guidance_scale: The guidance scale.
        guidance_scale_pose: The guidance scale for the pose.
        guidance_scale_sketch: The guidance scale for the sketch.
        sketch_cond_rate: The sketch condition rate.
        start_cond_rate: The start condition rate.
        no_pose: Whether to use the pose.
        disentagle: Whether to use disentagle.
        seed: The seed.
        
        Returns:
        None
    """ 
    assert(save_name != ""), "save_name must be specified"
    assert(output_dir != ""), "output_dir must be specified"

    path = os.path.join(output_dir, f"{save_name}_{test_order}", "images")

    os.makedirs(path, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(seed)

    for batch in tqdm(test_dataloader):
        model_img = batch["image"]
        mask_img = batch["inpaint_mask"]
        mask_img = mask_img.type(torch.float32)
        prompts = batch["original_captions"]  # prompts is a list of length N, where N=batch size.
        pose_map = batch["pose_map"]
        sketch = batch["im_sketch"]
        ext = ".jpg"

        if disentagle:
            guidance_scale = guidance_scale
            num_samples = 1
            guidance_scale_pose = guidance_scale_pose
            guidance_scale_sketch = guidance_scale_sketch
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                guidance_scale_pose=guidance_scale_pose,
                guidance_scale_sketch=guidance_scale_sketch,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images
        else:
            guidance_scale = 7.5
            num_samples = 1
            generated_images = pipe(
                prompt=prompts,
                image=model_img,
                mask_image=mask_img,
                pose_map=pose_map,
                sketch=sketch,
                height=512,
                width=384,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
                sketch_cond_rate=sketch_cond_rate,
                start_cond_rate=start_cond_rate,
                no_pose=no_pose,
            ).images

            for i in range(len(generated_images)):
                model_i = model_img[i] * 0.5 + 0.5
                if dataset == "vitonhd":
                    final_img = compose_img(model_i, generated_images[i], batch['im_parse'][i])
                else: # dataset == Dresscode
                    face = batch["stitch_label"][i].to(model_img.device)
                    face = T.functional.resize(face, 
                                               size=(512,384), 
                                               interpolation=T.InterpolationMode.BILINEAR, 
                                               antialias = True
                                               )

                    final_img = compose_img_dresscode(
                        gt_img = model_i, 
                        fake_img = T.functional.to_tensor(generated_images[i]).to(model_img.device), 
                        im_head = face
                        )
                
                final_img = T.functional.to_pil_image(final_img)
                final_img.save(
                    os.path.join(path, batch["im_name"][i].replace(".jpg", ext)))
