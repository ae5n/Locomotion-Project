import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltModel, AutoProcessor, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import yaml
import wandb
import os
import pandas as pd

from data_loader import CLIPLocomotionDataset, ViLTLocomotionDataset, clip_collate_fn, vilt_collate_fn, FlorenceLocomotionDataset, florence_collate_fn
from models import CustomCLIPModel, CustomViLTModel

def load_config(base_yaml, model_yaml):
    with open(base_yaml, 'r') as base_file:
        base_config = yaml.safe_load(base_file)
    
    with open(model_yaml, 'r') as model_file:
        model_config = yaml.safe_load(model_file)
    
    # Merge the base config with the model-specific config
    config = {**base_config, **model_config}
    
    return config

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
        custom_model = CustomCLIPModel(base_model, len(train_dataset.get_label_map()), mode=args.mode, fusion_method=getattr(args, 'fusion_method', None))
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
        custom_model = CustomImageBindModel(base_model, len(train_dataset.get_label_map()), mode=args.mode, fusion_method=getattr(args, 'fusion_method', None))
        collate_fn = None  # ImageBind doesn't require a special collate function
    
    elif args.model_name == "microsoft/Florence-2-large":
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
        train_dataset = FlorenceLocomotionDataset(train_data, args.image_folder, use_prompt=getattr(args, 'use_prompt', None), mode=args.mode)
        test_dataset = FlorenceLocomotionDataset(test_data, args.image_folder, use_prompt=getattr(args, 'use_prompt', None), mode=args.mode)
        collate_fn = lambda x: florence_collate_fn(x, processor, mode=args.mode)
        custom_model = base_model  # Use the base model directly for Florence-2

        # Optionally freeze the vision encoder
        if getattr(args, 'freeze_vision_encoder', None):
            for param in custom_model.vision_tower.parameters():
                param.requires_grad = False

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    custom_model.to(args.device) 

    fusion_suffix = f"_{getattr(args, 'fusion_method', None)}" if args.model_name not in ["vilt-b32-mlm", "microsoft/Florence-2-large"] and args.mode in ["image_text", "image_audio"] else ""

    # Initialize WandB if the project name is provided
    if args.wandb_project:
        wandb.init(project=args.wandb_project)
        
        # Create a filtered args dictionary for logging
        filtered_args = vars(args).copy()

        # Remove irrelevant args based on the selected model
        if args.model_name == "microsoft/Florence-2-large":
            filtered_args.pop('fusion_method', None)  # Florence doesn't use fusion_method
            filtered_args['use_prompt'] = getattr(args, 'use_prompt', None)  # Florence uses use_prompt
            filtered_args['freeze_vision_encoder'] = getattr(args, 'freeze_vision_encoder', None)  # Florence uses freeze_vision_encoder
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

    model_folder_name = args.model_name.replace("/", "_")  # Replace '/' with '_' for the folder name
    experiment_dir = os.path.join(args.output_dir, model_folder_name, os.path.splitext(os.path.basename(args.model_config))[0])
    os.makedirs(experiment_dir, exist_ok=True)

    best_model_state = None  # Initialize best_model_state

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

        # Early stopping and best model tracking
        if train_loss < best_loss:
            best_loss = train_loss
            trials = 0
            best_model_state = custom_model.state_dict()  # Save the best model state in memory
            best_model_filename = os.path.join(experiment_dir, 'best_model.pth')
            torch.save(best_model_state, best_model_filename)
            if args.model_name == "microsoft/Florence-2-large":
                processor_dir = os.path.join(experiment_dir, 'florence_processor')
                processor.save_pretrained(processor_dir)
        else:
            trials += 1
            if trials >= args.patience:
                print(f"Early stopping on epoch {epoch+1}")
                break

    # At the end of training, check if the final model is better than the best model and save it if necessary
    final_loss = train_loss
    if final_loss < best_loss:
        print("Final model is better than the previously saved best model. Overwriting the best model.")
        best_model_state = custom_model.state_dict()  # Save the final model state
        best_model_filename = os.path.join(experiment_dir, 'best_model.pth')
        torch.save(custom_model.state_dict(), best_model_filename)
        if args.model_name == "microsoft/Florence-2-large":
            processor_dir = os.path.join(experiment_dir, 'florence_processor')
            processor.save_pretrained(processor_dir)
    else:
        # Ensure the final model is set to the best model
        if best_model_state is not None:
            custom_model.load_state_dict(best_model_state)
            print("Final model is set to the best model during training.")

    # Save the model-specific configuration file in the same folder with the original name
    config_filename = os.path.join(experiment_dir, os.path.basename(args.model_config))
    with open(config_filename, 'w') as config_file:
        yaml.dump(vars(args), config_file)

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

    # Add arguments for base config and model-specific config
    parser.add_argument("base_config", type=str, help="Path to the base configuration YAML file.")
    parser.add_argument("model_config", type=str, help="Path to the model-specific configuration YAML file.")

    args = parser.parse_args()

    # Load the configuration from the YAML files
    config = load_config(args.base_config, args.model_config)
    
    # Add the model_config attribute to the config dictionary
    config['model_config'] = args.model_config
    
    # Train and test the model with the provided configuration
    train_model(argparse.Namespace(**config))
