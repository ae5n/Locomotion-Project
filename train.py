import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltModel, AutoProcessor, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import wandb
import os
import pandas as pd

from data_loader import CLIPLocomotionDataset, ViLTLocomotionDataset, clip_collate_fn, vilt_collate_fn, FlorenceLocomotionDataset, florence_collate_fn
from models import CustomCLIPModel, CustomViLTModel

def train_model(args):
    # Load the JSON files
    with open(args.train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)

    # Initialize processor and model based on the selected model
    if args.model_name == "clip-vit-large-patch14-336":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        train_dataset = CLIPLocomotionDataset(train_data, args.image_folder, processor, mode=args.mode)
        test_dataset = CLIPLocomotionDataset(test_data, args.image_folder, processor, mode=args.mode)
        custom_model = CustomCLIPModel(base_model, len(train_dataset.get_label_map()), mode=args.mode, fusion_method=args.fusion_method)
        collate_fn = lambda x: clip_collate_fn(x, mode=args.mode)

    elif args.model_name == "vilt-b32-mlm":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        train_dataset = ViLTLocomotionDataset(train_data, args.image_folder, processor)
        test_dataset = ViLTLocomotionDataset(test_data, args.image_folder, processor)
        custom_model = CustomViLTModel(base_model, len(train_dataset.get_label_map()))
        collate_fn = vilt_collate_fn

    elif args.model_name == "imagebind_huge":
        from data_loader import ImageBindLocomotionDataset
        from models import CustomImageBindModel
        from imagebind.models import imagebind_model
        base_model = imagebind_model.imagebind_huge(pretrained=True)
        train_dataset = ImageBindLocomotionDataset(train_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
        test_dataset = ImageBindLocomotionDataset(test_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
        custom_model = CustomImageBindModel(base_model, len(train_dataset.get_label_map()), mode=args.mode, fusion_method=args.fusion_method)
        collate_fn = None  # ImageBind doesn't require a special collate function
    
    elif args.model_name == "microsoft/Florence-2-large":
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
        train_dataset = FlorenceLocomotionDataset(train_data, args.image_folder, use_prompt=args.use_prompt, mode=args.mode)
        test_dataset = FlorenceLocomotionDataset(test_data, args.image_folder, use_prompt=args.use_prompt, mode=args.mode)
        collate_fn = lambda x: florence_collate_fn(x, processor, mode=args.mode)
        custom_model = base_model  # Use the base model directly for Florence-2

        # Optionally freeze the vision encoder
        if args.freeze_vision_encoder:
            for param in custom_model.vision_tower.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    custom_model.to(args.device) 

    fusion_suffix = f"_{args.fusion_method}" if args.model_name not in ["vilt-b32-mlm", "microsoft/Florence-2-large"] and args.mode in ["image_text", "image_audio"] else ""

    # Initialize WandB if the project name is provided
    if args.wandb_project:
        wandb.init(project=args.wandb_project)
        
        # Create a filtered args dictionary for logging
        filtered_args = vars(args).copy()

        # Remove irrelevant args based on the selected model
        if args.model_name == "microsoft/Florence-2-large":
            filtered_args.pop('fusion_method', None)  # Florence doesn't use fusion_method
            filtered_args['use_prompt'] = args.use_prompt  # Florence uses use_prompt
            filtered_args['freeze_vision_encoder'] = args.freeze_vision_encoder  # Florence uses freeze_vision_encoder
        else:
            filtered_args.pop('use_prompt', None)  # Only Florence uses use_prompt
            filtered_args.pop('freeze_vision_encoder', None)  # Only Florence uses freeze_vision_encoder

        if args.model_name not in ["clip-vit-large-patch14-336", "imagebind_huge"]:
            filtered_args.pop('fusion_method', None)  # Only CLIP and ImageBind use fusion_method

        # Save the filtered args to WandB config
        wandb.config.update(filtered_args)

        # Calculate and log model parameters once
        model_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in custom_model.parameters())
        
        wandb.config.update({
            "trainable_parameters": model_params,  # Log trainable parameters
            "total_parameters": total_params,  # Log total parameters
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if args.device == 'cuda' else "CPU"
        })
        
        # Log the model architecture and gradients/weights
        wandb.watch(custom_model, log="all", log_freq=10)  # Log gradients and parameters at each 10th step

    # Define optimizer, criterion, and scheduler
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=args.learning_rate)
    if args.model_name != "microsoft/Florence-2-large":
        criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training loop with early stopping
    best_loss = float('inf')
    trials = 0

    for epoch in range(args.num_epochs):
        custom_model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            if args.model_name == "clip-vit-large-patch14-336":
                ids, images, input_ids, attention_mask, labels = batch
                if args.mode == "image_only":
                    images = images.to(args.device)
                    logits = custom_model(pixel_values=images)
                elif args.mode == "text_only":
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits = custom_model(input_ids=input_ids, attention_mask=attention_mask)
                elif args.mode == "image_text":
                    images = images.to(args.device)
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
            elif args.model_name == "vilt-b32-mlm":
                ids, images, input_ids, attention_mask, labels = batch
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits, _ = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            elif args.model_name == "imagebind_huge":
                ids, inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                logits = custom_model(inputs)
            
            elif args.model_name == "microsoft/Florence-2-large":
                ids, inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                outputs = custom_model(**inputs, labels=labels.to(args.device))
                loss = outputs.loss

            if args.model_name != "microsoft/Florence-2-large":
                # Convert labels list to tensor
                labels = torch.tensor(labels).to(args.device)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(custom_model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss}")

        # Log training metrics to WandB
        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gpu_memory_usage": torch.cuda.max_memory_allocated(args.device) / (1024 ** 2) if args.device == 'cuda' else 0,
                "cpu_memory_usage": torch.cuda.memory_reserved(args.device) / (1024 ** 2) if args.device == 'cuda' else 0
            })

        # Learning rate scheduling
        scheduler.step(train_loss)

        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            trials = 0
            # Save the model when it achieves the best loss
            model_filename = f'best_{args.model_name.replace("/", "_")}_{args.mode}_{fusion_suffix}_btch{args.batch_size}epch{args.num_epochs}.pth'
            torch.save(custom_model.state_dict(), f'{args.output_dir}/{model_filename}')
            if args.model_name == "microsoft/Florence-2-large":
                processor_filename = model_filename.replace('.pth', '_processor.pth')
                processor.save_pretrained(os.path.join(args.output_dir, processor_filename))
        else:
            trials += 1
            if trials >= args.patience:
                print(f"Early stopping on epoch {epoch+1}")
                break
    
    final_loss = train_loss

    # If the final model is better than the saved best model, overwrite it
    if final_loss <= best_loss:
        print("Final model is better than or equal to the best model during training. Overwriting the best model.")
        torch.save(custom_model.state_dict(), f'{args.output_dir}/{model_filename}')
        if args.model_name == "microsoft/Florence-2-large":
            processor_filename = model_filename.replace('.pth', '_processor.pth')
            processor.save_pretrained(os.path.join(args.output_dir, processor_filename))
    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, train_dataset.get_label_map(), args, processor)

def evaluate_model(model, dataloader, label_mapping, args, processor):
    model.eval()
    true_labels = []
    predicted_labels = []
    ids_list = []

    # Reverse the label map for lookup
    inverse_label_map = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for batch in dataloader:
            if args.model_name == "clip-vit-large-patch14-336":
                ids, images, input_ids, attention_mask, labels = batch
                if args.mode == "image_only":
                    images = images.to(args.device)
                    logits = model(pixel_values=images)
                elif args.mode == "text_only":
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                elif args.mode == "image_text":
                    images = images.to(args.device)
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            elif args.model_name == "vilt-b32-mlm":
                ids, images, input_ids, attention_mask, labels = batch
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits, _ = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            elif args.model_name == "imagebind_huge":
                ids, inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                logits = model(inputs)
                labels = labels.cpu().tolist()  # Convert to list of indices for ImageBind
            
            elif args.model_name == "microsoft/Florence-2-large":
                ids, inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                generated_ids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], max_new_tokens=50)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                true_text = processor.batch_decode(labels, skip_special_tokens=True)
                print(f"Generated text: {generated_text}, True label: {true_text}")
                predicted_labels.extend(generated_text)
                true_labels.extend(true_text)
                ids_list.extend(ids)
            
            if args.model_name != "microsoft/Florence-2-large":
                preds = torch.argmax(logits, dim=1)
                if isinstance(labels, torch.Tensor):  # Handle labels for ImageBind
                    true_labels.extend(labels.cpu().tolist())
                else:  # Handle labels for CLIP and ViLT
                    true_labels.extend(labels)

                predicted_labels.extend(preds.cpu().numpy())
                ids_list.extend(ids)

    # Convert indices to labels using the inverse label map
    if args.model_name != "microsoft/Florence-2-large":
        true_labels = [inverse_label_map[label] for label in true_labels]
        predicted_labels = [inverse_label_map[label] for label in predicted_labels]

    # Create a DataFrame with ID, true labels, and predicted labels
    eval_df = pd.DataFrame({
        'ID': ids_list,
        'True Label': true_labels,
        'Predicted Label': predicted_labels
    })

    # Log the DataFrame to WandB
    if args.wandb_project:
        wandb.log({"eval_dataframe": wandb.Table(dataframe=eval_df)}) 

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    print(f"Accuracy: {accuracy}")
    print(classification_report(true_labels, predicted_labels))

    # Log evaluation metrics to WandB
    if args.wandb_project:
        wandb.log({
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score'],
            "classification_report": report
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for configuration
    parser.add_argument("model_name", type=str, choices=["clip-vit-large-patch14-336", "vilt-b32-mlm", "imagebind_huge", "microsoft/Florence-2-large"], help="Model name to use (e.g., 'clip-vit-large-patch14-336', 'vilt-b32-mlm', 'imagebind_huge', 'microsoft/Florence-2-large').")
    parser.add_argument("mode", type=str, choices=["image_only", "text_only", "image_text", "image_audio", "audio_only"], help="Mode of training (e.g., 'image_only', 'text_only', 'image_text', 'image_audio').")
    parser.add_argument("--fusion_method", type=str, choices=["average", "concat", "cross_attention"], default="average", help="Fusion method to use for combining embeddings.")
    parser.add_argument("--train_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/train_data.json', help="Path to the training JSON file.")
    parser.add_argument("--test_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/test_data.json', help="Path to the test JSON file.")
    parser.add_argument("--image_folder", type=str, default='/content/drive/MyDrive/My Data/processed data/images', help="Path to the image folder.")
    parser.add_argument("--audio_folder", type=str, default='/content/drive/MyDrive/My Data/processed data/audio', help="Path to the audio folder (only required for ImageBind).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=22, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--output_dir", type=str, default='/content/drive/MyDrive/My Data', help="Directory to save the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--wandb_project", type=str, help="Weights and Biases project name. If provided, metrics will be logged to WandB.")
    parser.add_argument("--use_prompt", action="store_true", help="Use the prompt for text input if set; otherwise, use the direct command text.")  # Only for Florence
    parser.add_argument("--freeze_vision_encoder", action='store_true', help="Whether to freeze the vision encoder during training.")  # Only for Florence

    args = parser.parse_args()

    # Train and test the model with the provided arguments
    train_model(args)
