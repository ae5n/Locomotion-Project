import torch
import torch.nn as nn
import json

class CustomFusionModule(nn.Module):
    def __init__(self, method, embedding_dim1, embedding_dim2=None):
        super(CustomFusionModule, self).__init__()
        self.method = method
        if embedding_dim2 is None:
            embedding_dim2 = embedding_dim1

        if self.method == 'concat':
            self.fc = nn.Linear(embedding_dim1 + embedding_dim2, embedding_dim1)
        elif self.method == 'cross_attention':
            self.proj_emb1 = nn.Linear(embedding_dim1, embedding_dim1)
            self.proj_emb2 = nn.Linear(embedding_dim2, embedding_dim1)
            self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim1, num_heads=8)

    def forward(self, emb1, emb2):
        if self.method == 'average':
            return (emb1 + emb2) / 2
        elif self.method == 'concat':
            combined = torch.cat((emb1, emb2), dim=-1)
            return self.fc(combined)
        elif self.method == 'cross_attention':
            emb1 = self.proj_emb1(emb1).unsqueeze(1).permute(1, 0, 2)  # (batch_size, embed_dim) -> (1, batch_size, embed_dim) -> (1, batch_size, embed_dim)
            emb2 = self.proj_emb2(emb2).unsqueeze(1).permute(1, 0, 2)  # (batch_size, embed_dim) -> (1, batch_size, embed_dim) -> (1, batch_size, embed_dim)
            attn_output, _ = self.cross_attention(emb1, emb2, emb2)
            return attn_output.permute(1, 0, 2).squeeze(1)  # Back to (batch_size, embed_dim)

        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_model, num_classes, mode='image_text', fusion_method='average'):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model
        self.mode = mode
        self.fusion_method = fusion_method
        self.fc_image = nn.Linear(self.clip_model.vision_model.config.hidden_size, num_classes) if mode in ['image_only', 'image_text'] else None
        self.fc_text = nn.Linear(self.clip_model.text_model.config.hidden_size, num_classes) if mode in ['text_only', 'image_text'] else None
        self.fusion = CustomFusionModule(fusion_method, self.clip_model.vision_model.config.hidden_size, self.clip_model.text_model.config.hidden_size)

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

        if self.mode == 'image_text':
            fused_output = self.fusion(image_embeds, text_embeds)
            return fused_output

        return logits_image if logits_image is not None else logits_text


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
    def __init__(self, imagebind_model, num_classes, mode='image_text', fusion_method='average'):
        super(CustomImageBindModel, self).__init__()
        self.mode = mode
        self.imagebind = imagebind_model
        self.fusion_method = fusion_method
        self.fusion = CustomFusionModule(fusion_method, 1024, 1024)  # Adjust dimensions for ImageBind
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        embeddings = self.imagebind(inputs)

        if self.mode == 'image_text':
            combined_embeddings = self.fusion(embeddings['vision'], embeddings['text'])
        elif self.mode == 'image_audio':
            combined_embeddings = self.fusion(embeddings['vision'], embeddings['audio'])
        elif 'vision' in embeddings:
            combined_embeddings = embeddings['vision']
        elif 'text' in embeddings:
            combined_embeddings = embeddings['text']
        elif 'audio' in embeddings:
            combined_embeddings = embeddings['audio']
        else:
            raise ValueError("Unexpected output structure from ImageBind model")

        logits = self.fc(combined_embeddings)
        return logits

class CustomGPT4oModel(nn.Module):
    def __init__(self, num_classes, model_name, mode='image_text', strategy='default', client=None):
        super(CustomGPT4oModel, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.model_name = model_name
        self.client = client
        self.valid_labels = None
        self.strategy = strategy 
    def set_valid_labels(self, label_mapping):
        self.valid_labels = list(label_mapping.keys())
    
    def construct_prompt(self, text=None):
        """ Construct the prompt based on the strategy and mode. """
        if self.strategy == 'default':
            if self.mode == 'image_only':
                return (
                    "You are provided with an image containing field-of-view (FOV) frames from smart glasses worn by a user performing a locomotion activity "
                    "in an industrial environment. The 9 frames in the image are sampled in chronological order over a 5-second period, with 2 seconds before and 3 seconds after the command was given, "
                    "providing context for the user's activity. "
                    "Based on the frames, identify the locomotion activity the user is performing."
                )
            elif self.mode == 'text_only':
                return (
                    f"The user is performing a locomotion activity in an industrial environment and has issued the following command: \"{text}\". "
                    "Based on the command, identify the locomotion activity the user is performing."
                )
            elif self.mode == 'image_text':
                return (
                    "You are provided with an image containing field-of-view (FOV) frames from smart glasses worn by a user performing a locomotion activity "
                    "in an industrial environment, along with a spoken command issued by the user. "
                    "The 9 frames in the image are sampled in chronological order over a 5-second period, with 2 seconds before and 3 seconds after the command was given, "
                    "providing context for the user's activity. "
                    f"The command is: \"{text}\". "
                    "Based on the frames and the command, identify the locomotion activity the user is performing."
                )
        elif self.strategy == 'cot':
            if self.mode == 'image_only':
                return (
                    "You are provided with an image containing field-of-view (FOV) frames from smart glasses worn by a user performing a locomotion activity "
                    "in an industrial environment. The 9 frames in the image are sampled in chronological order over a 5-second period, with 2 seconds before and 3 seconds after the command was given, "
                    "Think step by step: analyze the frames one by one to understand how the user's perspective changes over time, and infer the locomotion activity based on the sequence."
                )
            elif self.mode == 'text_only':
                return (
                    f"The user is performing a locomotion activity in an industrial environment and has issued the following command: \"{text}\". "
                    "Think step by step: break down the meaning of the command, understand the user's intention, and determine the locomotion activity the user is performing."
                )
            elif self.mode == 'image_text':
                return (
                    "You are provided with an image containing field-of-view (FOV) frames from smart glasses worn by a user performing a locomotion activity "
                    "in an industrial environment, along with a spoken command issued by the user. "
                    "The 9 frames in the image are sampled in chronological order over a 5-second period, with 2 seconds before and 3 seconds after the command was given, "
                    f"The command is: \"{text}\". "
                    "Think step by step: analyze the sequence of frames to understand how the user's perspective changes over time, and interpret the command to understand their intention. "
                    "Combine information from both the frames and the command to predict the locomotion activity the user is performing."
                )

    def predict(self, text, encoded_image=None, temperature=0.7, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, max_tokens=50, model_name=None):
        if model_name is not None:
            self.model_name = model_name
        function = {
            "name": "predict_locomotion_activity",
            "description": "Predict the locomotion activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prediction": {
                        "type": "string",
                        "enum": self.valid_labels,
                        "description": "The predicted locomotion activity."
                    }
                },
                "required": ["prediction"]
            }
        }
        prompt = self.construct_prompt(text=text)
        
        system_content = (
            f"You are an AI specialized in a classification task. "
            f"The task involves identifying locomotion activities based on the provided information."
        )

        if self.mode in ['image_only', 'image_text']:
            message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"}
                    }
                ]}
            ]
        elif self.mode == 'text_only':
            message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            functions=[function],
            function_call={"name": "predict_locomotion_activity"},
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        function_call_arguments = json.loads(response.choices[0].message.function_call.arguments)
        predicted_label = function_call_arguments["prediction"]
        return predicted_label

