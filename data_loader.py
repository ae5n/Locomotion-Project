import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DataLoaderBase:
    def __init__(self, data_path, image_folder, audio_folder=None):
        self.data_path = data_path
        self.image_folder = image_folder
        self.audio_folder = audio_folder
        self.data = self.load_json()

    def load_json(self):
        """Load the JSON file containing the dataset annotations."""
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def get_image_path(self, image_id):
        """Get the path to the image file, assuming all images are in .jpg format."""
        return os.path.join(self.image_folder, f"{image_id}.jpg")

    def load_image(self, image_id):
        """Load the image given its ID."""
        image_path = self.get_image_path(image_id)
        try:
            return Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image {image_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def load_audio(self, audio_id):
        """Load the audio file if needed (functionality to be implemented)."""
        audio_path = self.get_audio_path(audio_id)
        if audio_path:
            pass  # Implement audio loading logic if needed
        return None

class ImageTextDataset(DataLoaderBase, Dataset):
    def __init__(self, json_data, image_folder, processor):
        """Initialize the dataset with JSON data and the image folder."""
        super().__init__(data_path=json_data, image_folder=image_folder)
        self.processor = processor

    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a processed image and its corresponding label."""
        entry = self.data[idx]
        image = self.load_image(entry['id'])
        
        if image is None:
            raise ValueError(f"Failed to load image with ID {entry['id']}.")
        
        text = entry['text']
        
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, entry['label']
