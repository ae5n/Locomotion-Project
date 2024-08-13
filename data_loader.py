import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class BaseLocomotionDataset(Dataset):
    def __init__(self, data, image_folder):
        """
        Base dataset class for locomotion tasks.
        """
        self.data = data
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def load_image(self, image_id):
        """
        Load an image given its ID.
        """
        img_path = os.path.join(self.image_folder, f"{image_id}.jpg")
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None
        
        return Image.open(img_path).convert("RGB")

class CLIPLocomotionDataset(BaseLocomotionDataset):
    def __init__(self, data, image_folder, processor, mode='image_and_text'):
        """
        CLIP-specific dataset class supporting image-only, text-only, and image-and-text modes.
        """
        super().__init__(data, image_folder)
        self.processor = processor
        self.mode = mode

    def __getitem__(self, idx):
        """
        Get a sample from the dataset for CLIP model based on the mode.
        """
        entry = self.data[idx]
        image_id = entry['id']
        text = entry['text']
        label = entry['label']

        image = self.load_image(image_id) if self.mode in ['image_only', 'image_and_text'] else None

        inputs = {}
        if self.mode == 'image_only':
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            return inputs['pixel_values'][0], label
        elif self.mode == 'text_only':
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            return inputs['input_ids'][0], inputs['attention_mask'][0], label
        elif self.mode == 'image_and_text':
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            return inputs['pixel_values'][0], inputs['input_ids'][0], inputs['attention_mask'][0], label

def clip_collate_fn(batch, mode='image_and_text'):
    """
    Function to collate data into batches and pad sequences.
    """
    batch = [item for item in batch if item is not None]
    
    if mode == 'image_only':
        pixel_values = torch.stack([item[0] for item in batch])
        labels = [item[1] for item in batch]
        return pixel_values, None, None, labels
    
    elif mode == 'text_only':
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return None, input_ids_padded, attention_masks_padded, labels
    
    elif mode == 'image_and_text':
        pixel_values = torch.stack([item[0] for item in batch])
        input_ids = [item[1] for item in batch]
        attention_masks = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return pixel_values, input_ids_padded, attention_masks_padded, labels
