import torch
import torch.nn as nn

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_model, num_classes, mode='image_text'):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.mode = mode
        self.fc_image = nn.Linear(self.clip_model.vision_model.config.hidden_size, num_classes) if mode in ['image_only', 'image_text'] else None
        self.fc_text = nn.Linear(self.clip_model.text_model.config.hidden_size, num_classes) if mode in ['text_only', 'image_text'] else None

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        logits_image, logits_text = None, None

        if self.mode in ['image_only', 'image_text']:
            vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output
            logits_image = self.fc_image(image_embeds)

        if self.mode in ['text_only', 'image_text']:
            text_outputs = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.pooler_output
            logits_text = self.fc_text(text_embeds)

        return logits_image, logits_text


class CustomViLTModel(nn.Module):
    def __init__(self, vilt_model, num_classes):
        super(CustomViLTModel, self).__init__()
        self.vilt_model = vilt_model
        self.fc = nn.Linear(vilt_model.config.hidden_size, num_classes)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        outputs = self.vilt_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits, None

class CustomImageBindModel(nn.Module):
    def __init__(self, num_classes, mode='image_text'):
        super(CustomImageBindModel, self).__init__()
        self.mode = mode
        from imagebind.models import imagebind_model
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        # Adjust the input dimension to 1024 as identified from the embeddings structure
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        embeddings = self.imagebind(inputs)

        if 'vision' in embeddings and 'text' in embeddings:
            # Example: averaging the embeddings from 'vision' and 'text'
            combined_embeddings = (embeddings['vision'] + embeddings['text']) / 2
        elif 'vision' in embeddings:
            combined_embeddings = embeddings['vision']
        elif 'text' in embeddings:
            combined_embeddings = embeddings['text']
        else:
            raise ValueError("Unexpected output structure from ImageBind model")

        logits = self.fc(combined_embeddings)
        return logits