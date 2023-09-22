dependencies = ['torch', 'diffusers']
import torch
from diffusers import UNet2DConditionModel


# mgd is the name of entrypoint
def mgd(dataset: str, pretrained: bool = True, **kwargs) -> UNet2DConditionModel:
    """ # This docstring shows up in hub.help()
    MGD model
    pretrained (bool): kwargs, load pretrained weights into the model
    """

    config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-inpainting", subfolder="unet")
    config['in_channels'] = 28
    unet = UNet2DConditionModel.from_config(config)

    if pretrained:
        checkpoint = f"https://github.com/aimagelab/multimodal-garment-designer/releases/download/weights/{dataset}.pth"
        unet.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))

    return unet
