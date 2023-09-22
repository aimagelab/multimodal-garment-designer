# File havily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py


import json
import os
import pathlib
import random
import sys
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes
from src.utils.posemap import get_coco_body25_mapping
from src.utils.posemap import kpoint_to_heatmap


class VitonHDDataset(data.Dataset):
    def __init__(
            self,
            dataroot_path: str,
            phase: str,
            tokenizer,
            radius=5,
            caption_folder='captions.json',
            sketch_threshold_range: Tuple[int, int] = (20, 127),
            order: str = 'paired',
            outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_cloth', 'shape', 'pose_map',
                                      'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                      'im_sketch', 'captions', 'original_captions'),
            size: Tuple[int, int] = (512, 384),
    ):

        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_folder = caption_folder
        self.sketch_threshold_range = sketch_threshold_range
        self.category = ('upper_body')
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['c_name', 'im_name', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array',
                            'im_mask', 'inpaint_mask', 'parse_mask_total', 'im_sketch', 'captions',
                            'original_captions', 'category']

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(os.path.join(self.dataroot, self.caption_folder)) as f:
            # self.captions_dict = json.load(f)['items']
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        dataroot = self.dataroot
        if phase == 'train':
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot, f"{phase}_pairs.txt")

        with open(filename, 'r') as f:
            data_len = len(f.readlines())

        with open(filename, 'r') as f:
            for line in f.readlines():
                if phase == 'train':
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == 'paired':
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        """
        For each index return the corresponding sample in the dataset
        :param index: data index
        :type index: int
        :return: dict containing dataset samples
        :rtype: dict
        """
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist or "original_captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            # take a random caption if there are multiple
            if self.phase == 'train':
                random.shuffle(captions)
            captions = ", ".join(captions)

            original_captions = captions

        if "captions" in self.outputlist:
            cond_input = self.tokenizer([captions], max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
            cond_input = cond_input.squeeze(0)
            max_length = cond_input.shape[-1]
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            ).input_ids.squeeze(0)
            captions = cond_input
            captions_uncond = uncond_input

        if "image" in self.outputlist or "im_head" in self.outputlist or "im_cloth" in self.outputlist:
            # Person image
            # image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = Image.open(os.path.join(dataroot, self.phase, 'image', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "im_sketch" in self.outputlist:
            # Person image
            # im_sketch = Image.open(os.path.join(dataroot, 'im_sketch', c_name.replace(".jpg", ".png")))
            if self.order == 'unpaired':
                im_sketch = Image.open(
                    os.path.join(dataroot, self.phase, 'im_sketch_unpaired',
                                 os.path.splitext(im_name)[0] + '_' + c_name.replace(".jpg", ".png")))
            elif self.order == 'paired':
                im_sketch = Image.open(os.path.join(dataroot, self.phase, 'im_sketch', im_name.replace(".jpg", ".png")))
            else:
                raise ValueError(
                    f"Order should be either paired or unpaired"
                )

            im_sketch = im_sketch.resize((self.width, self.height))
            im_sketch = ImageOps.invert(im_sketch)
            # threshold grayscale pil image
            im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
            # im_sketch = im_sketch.convert("RGB")
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
            im_sketch = 1 - im_sketch

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist or "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist or "pose_map" in self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist:
            # Label Map
            # parse_name = im_name.replace('_0.jpg', '_4.png')
            parse_name = im_name.replace('.jpg', '.png')
            im_parse = Image.open(os.path.join(dataroot, self.phase, 'image-parse-v3', parse_name))
            im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
            im_parse_final = transforms.ToTensor()(im_parse) * 255
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 4).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)

            parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 18).astype(np.float32) + \
                                (parse_array == 19).astype(np.float32)

            # parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
            parser_mask_changeable = (parse_array == 0).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

            parse_cloth = (parse_array == 5).astype(np.float32) + \
                          (parse_array == 6).astype(np.float32) + \
                          (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 5).astype(np.float32) + \
                         (parse_array == 6).astype(np.float32) + \
                         (parse_array == 7).astype(np.float32)

            parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                                (parse_array == 12).astype(np.float32)  # the lower body is fixed

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            # dilation
            parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
            shape = self.transform2D(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]

                # rescale keypoints on the base of height and width
                pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
                pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

            pose_mapping = get_coco_body25_mapping()

            point_num = len(pose_mapping)

            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
                point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []

            for idx in range(point_num):
                ux = pose_data[pose_mapping[idx], 0]  # / (192)
                uy = (pose_data[pose_mapping[idx], 1])  # / (256)

                # scale posemap points
                px = ux  # * self.width
                py = uy  # * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2D(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)

            # do in any case because i have only upperbody
            with open(os.path.join(dataroot, self.phase, 'openpose_json', pose_name), 'r') as f:
                data = json.load(f)
                data = data['people'][0]['pose_keypoints_2d']
                data = np.array(data)
                data = data.reshape((-1, 3))[:, :2]

                # rescale keypoints on the base of height and width
                data[:, 0] = data[:, 0] * (self.width / 768)
                data[:, 1] = data[:, 1] * (self.height / 1024)

                shoulder_right = np.multiply(tuple(data[pose_mapping[2]]), 1)
                shoulder_left = np.multiply(tuple(data[pose_mapping[5]]), 1)
                elbow_right = np.multiply(tuple(data[pose_mapping[3]]), 1)
                elbow_left = np.multiply(tuple(data[pose_mapping[6]]), 1)
                wrist_right = np.multiply(tuple(data[pose_mapping[4]]), 1)
                wrist_left = np.multiply(tuple(data[pose_mapping[7]]), 1)

                ARM_LINE_WIDTH = int(90 / 512 * self.height)
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)
                parse_mask += im_arms
                parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            # im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            # here we have to modify the mask and get the bounding box
            bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
            bboxes = bboxes.type(torch.int32)  # xmin, ymin, xmax, ymax format
            xmin = bboxes[0, 0]
            xmax = bboxes[0, 2]
            ymin = bboxes[0, 1]
            ymax = bboxes[0, 3]

            inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
                torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
                torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

            inpaint_mask = inpaint_mask.unsqueeze(0)
            im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
            parse_mask_total = parse_mask_total.numpy()
            parse_mask_total = parse_array * parse_mask_total
            parse_mask_total = torch.from_numpy(parse_mask_total)

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        result['im_parse'] = im_parse_final
        result['hands'] = torch.from_numpy(hands)

        # Output interpretation
        # "c_name" -> filename of inshop cloth
        # "im_name" -> filename of model with cloth
        # "cloth" -> img of inshop cloth
        # "image" -> img of the model with that cloth
        # "im_cloth" -> cut cloth from the model
        # "im_mask" -> black mask of the cloth in the model img
        # "cloth_sketch" -> sketch of the inshop cloth
        # "im_sketch" -> sketch of "im_cloth"
        # inpaint_mask -> bb of the model img where the cloth is
        # ...
        return result

    def __len__(self):
        return len(self.c_names)
