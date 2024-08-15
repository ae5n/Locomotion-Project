import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class BaseLocomotionDataset(Dataset):
    def __init__(self, data, image_folder):
        """
        Base dataset class for locomotion tasks.
        """
        self.data = data
        self.image_folder = image_folder
        self.label_map = self.create_label_map()

    def create_label_map(self):
        # Create a label map by extracting all unique labels and assigning unique integers
        unique_labels = set(entry['label'] for entry in self.data)
        return {label: idx for idx, label in enumerate(unique_labels)}

    def get_label_map(self):
        # Returns the label map.
        return self.label_map

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
    def __init__(self, data, image_folder, processor, mode='image_text'):
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

        image = self.load_image(image_id) if self.mode in ['image_only', 'image_text'] else None

        inputs = {}
        if self.mode == 'image_only':
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            return inputs['pixel_values'][0], self.label_map[label]
        elif self.mode == 'text_only':
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            return inputs['input_ids'][0], inputs['attention_mask'][0], self.label_map[label]
        elif self.mode == 'image_text':
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
            return inputs['pixel_values'][0], inputs['input_ids'][0], inputs['attention_mask'][0], self.label_map[label]

def clip_collate_fn(batch, mode='image_text'):
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
    
    elif mode == 'image_text':
        pixel_values = torch.stack([item[0] for item in batch])
        input_ids = [item[1] for item in batch]
        attention_masks = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        return pixel_values, input_ids_padded, attention_masks_padded, labels

class ViLTLocomotionDataset(BaseLocomotionDataset):
    def __init__(self, data, image_folder, processor):
        """
        ViLT-specific dataset class.
        """
        super().__init__(data, image_folder)
        self.processor = processor

    def __getitem__(self, idx):
        """
        Get a sample from the dataset for ViLT model.
        """
        entry = self.data[idx]
        image_id = entry['id']
        text = entry['text']
        label = entry['label']

        image = self.load_image(image_id)

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return inputs['pixel_values'][0], inputs['input_ids'][0], inputs['attention_mask'][0], self.label_map[label]

def vilt_collate_fn(batch):
    """
    Function to collate data into batches for ViLT.
    """
    batch = [item for item in batch if item is not None]
    pixel_values = torch.stack([item[0] for item in batch])
    input_ids = [item[1] for item in batch]
    attention_masks = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return pixel_values, input_ids_padded, attention_masks_padded, labels

class ImageBindLocomotionDataset(BaseLocomotionDataset):
    def __init__(self, data, image_folder, audio_folder=None, mode='image_text', device='cpu'):
        super().__init__(data, image_folder)
        self.audio_folder = audio_folder
        self.mode = mode
        self.device = device
        self.imagebind_data = None
        self.ModalityType = None
        self._lazy_import_imagebind()

    def _lazy_import_imagebind(self):
        from imagebind import data as imagebind_data
        from imagebind.models.imagebind_model import ModalityType
        self.imagebind_data = imagebind_data
        self.ModalityType = ModalityType

    def __getitem__(self, idx):
        entry = self.data[idx]
        inputs = {}

        # Handle image data if included in mode
        if 'image' in self.mode:
            image_path = os.path.join(self.image_folder, f"{entry['id']}.jpg")
            image = self.imagebind_data.load_and_transform_vision_data([image_path], self.device)
            inputs[self.ModalityType.VISION] = image[0]  # Assuming method returns a list of tensors

        # Handle text data if included in mode
        if 'text' in self.mode:
            text_input = self.imagebind_data.load_and_transform_text([entry.get('text', '')], self.device)
            inputs[self.ModalityType.TEXT] = text_input[0]  # Assuming method returns a list of tensors

        # Handle audio data if included in mode and audio path is specified
        if 'audio' in self.mode and self.audio_folder:
            audio_path = os.path.join(self.audio_folder, f"{entry['id']}.wav")
            audio = self.imagebind_data.load_and_transform_audio_data([audio_path], self.device)
            inputs[self.ModalityType.AUDIO] = audio[0]  # Assuming method returns a list of tensors

        # Convert string labels to integers using the label map
        label = self.label_map[entry['label']]
        label_tensor = torch.tensor(label, dtype=torch.long)

        return inputs, label_tensor

    def __len__(self):
        return len(self.data)
