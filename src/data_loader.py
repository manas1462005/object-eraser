import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import json

class COCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file

        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.image_id_to_annotations = self._map_image_id_to_annotations()

        # Image transformations
        self.transform = transform if transform else T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Mask transformations
        self.mask_transform = mask_transform if mask_transform else T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def _map_image_id_to_annotations(self):
        mapping = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(ann)
        return mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        mask = Image.new('L', image.size)
        img_id = image_info['id']
        anns = self.image_id_to_annotations.get(img_id, [])

        for ann in anns:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    draw.polygon(seg, outline=1, fill=1)

        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)

        return image, mask
