import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltModel, AutoProcessor, AutoModelForCausalLM
import openai
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import yaml
import logging
import wandb
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from data_loader import CLIPLocomotionDataset, ViLTLocomotionDataset, FlorenceLocomotionDataset, GPT4LocomotionDataset, clip_collate_fn, vilt_collate_fn, florence_collate_fn
from models import CustomCLIPModel, CustomViLTModel, CustomGPT4oModel

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(base_yaml, model_yaml):
    with open(base_yaml, 'r') as base_file:
        base_config = yaml.safe_load(base_file)
    
    with open(model_yaml, 'r') as model_file:
        model_config = yaml.safe_load(model_file)
    
    # Merge the base config with the model-specific config
    config = {**base_config, **model_config}
    
    return config

def train_model(args): 
    # Filter out paths that you don't want to log
    filtered_args = {k: v for k, v in vars(args).items() if k not in ['train_json_path', 'test_json_path', 'image_folder', 'audio_folder', 'output_dir']}
 
    # Log each argument on a separate line
    logger.info("Training/evaluating model with the following arguments:")
    for key, value in filtered_args.items():
        logger.info(f"{key}: {value}")

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
        processor = None
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

    elif args.model_name == "gpt-4o":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        test_dataset = GPT4LocomotionDataset(test_data, args.image_folder, mode=args.mode)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        custom_model = CustomGPT4oModel(num_classes=len(test_dataset.get_label_map()), mode=args.mode, model_name=args.model_name, client=client)
        # We only evaluate the GPT-4o model by utilizing zero-shot learning
        evaluate_model(custom_model, test_dataloader, test_dataset.get_label_map(), args=args, processor=None)
        return

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    custom_model.to(args.device) 

    # Log to WandB if the project name is provided
    if args.wandb_project:

        wandb.config.update(vars(args))

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
    logger.info("Training started.")
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
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss}")

        # Log training metrics to WandB
        if args.wandb_project:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gpu_memory_usage": torch.cuda.max_memory_allocated(args.device) / (1024 ** 2) if args.device == 'cuda' else 0,
                "cpu_memory_usage": torch.cuda.memory_reserved(args.device) / (1024 ** 2) if args.device == 'cuda' else 0
            })

        previous_lr = optimizer.param_groups[0]['lr']
        # Learning rate scheduling
        scheduler.step(train_loss)

        # Log when the learning rate is reduced
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < previous_lr:
            logger.info(f"Learning rate reduced from {previous_lr} to {current_lr}")

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
                logger.info(f"Early stopping on epoch {epoch+1}")
                break

    # At the end of training, check if the final model is better than the best model and save it if necessary
    final_loss = train_loss
    if final_loss < best_loss:
        logger.info("Final model is better than the previously saved best model. Overwriting the best model.")
        best_model_state = custom_model.state_dict()  # Save the final model state
        best_model_filename = os.path.join(experiment_dir, 'best_model.pth')
        torch.save(custom_model.state_dict(), best_model_filename)
        if args.model_name == "microsoft/Florence-2-large":
            processor_dir = os.path.join(experiment_dir, 'florence_processor')
            processor.save_pretrained(processor_dir)
    else:
        if best_model_state is not None:
            custom_model.load_state_dict(best_model_state)
            logger.info(f"Final model did not outperform the best model. Loaded the best model with loss: {best_loss:.4f}")
        else:
            logger.info(f"No better model found during training. Using the model from the last epoch.")

    # Save the model-specific configuration file in the same folder with the original name
    config_filename = os.path.join(experiment_dir, os.path.basename(args.model_config))
    with open(config_filename, 'w') as config_file:
        yaml.dump(vars(args), config_file)

    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, train_dataset.get_label_map(), args, processor)

def evaluate_model(model, dataloader, label_mapping, args, processor):
    if args.model_name != "gpt-4o":
        model.eval()
    true_labels = []
    predicted_labels = []
    ids_list = []

    # Reverse the label map for lookup
    inverse_label_map = {v: k for k, v in label_mapping.items()}
    
    logger.info("Evaluation started.")
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
                predicted_labels.extend(generated_text)
                true_labels.extend(true_text)
                ids_list.extend(ids)

            elif args.model_name == "gpt-4o":
                model.set_valid_labels(label_mapping)
                ids, texts, encoded_images, labels = batch
                for i in range(len(ids)):
                    predicted_label = model.predict(
                        text=texts[i], 
                        encoded_image=encoded_images[i] if args.mode in ['image_only', 'image_text'] else None,
                        model_name=args.model_name,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        frequency_penalty=args.frequency_penalty,
                        presence_penalty=args.presence_penalty,
                        max_tokens=args.max_tokens
                    )
                    predicted_labels.append(predicted_label)
                    true_labels.append(labels[i])
                    ids_list.append(ids[i])
                    logger.info(f"Processed {len(ids_list)} samples ... ")
                    time.sleep(2)  # To avoid hitting API rate limits

            if args.model_name not in ["microsoft/Florence-2-large", "gpt-4o"]:
                preds = torch.argmax(logits, dim=1)
                if isinstance(labels, torch.Tensor):  # Handle labels for ImageBind
                    true_labels.extend(labels.cpu().tolist())
                else:  # Handle labels for CLIP and ViLT
                    true_labels.extend(labels)

                predicted_labels.extend(preds.cpu().numpy())
                ids_list.extend(ids)
            # break
    # Convert indices to labels using the inverse label map
    if args.model_name not in ["microsoft/Florence-2-large", "gpt-4o"]:
        true_labels = [inverse_label_map[label] for label in true_labels]
        predicted_labels = [inverse_label_map[label] for label in predicted_labels]

    # Create a DataFrame with ID, true labels, and predicted labels
    eval_df = pd.DataFrame({
        'ID': ids_list,
        'True Label': true_labels,
        'Predicted Label': predicted_labels
    })

    # Filter out misclassified samples
    misclassified_df = eval_df[eval_df['True Label'] != eval_df['Predicted Label']]

    # Log both the evaluation DataFrame and the misclassified samples to WandB
    config_name = os.path.splitext(os.path.basename(args.model_config))[0]
    if args.wandb_project:
        wandb.log({f"{config_name}_predictions": wandb.Table(dataframe=eval_df)}) 
        wandb.log({f"{config_name}_misclassified_samples": wandb.Table(dataframe=misclassified_df)})

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(classification_report(true_labels, predicted_labels))
    
    # Flatten the classification report dictionary
    flattened_report = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_report[f"{key}_{sub_key}"] = sub_value
        else:
            flattened_report[key] = value
    
    # Log the flattened classification report to WandB
    if args.wandb_project:
        wandb.log(flattened_report)

    # Compute the confusion matrix
    valid_labels = unique_labels(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=valid_labels)

    # Generate the confusion matrix plot
    plt.figure(figsize=(20, 16), dpi=300)
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=valid_labels, yticklabels=valid_labels,
                        annot_kws={"size": 13}, cbar_kws={"shrink": 1})

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=14) 
    plt.yticks(rotation=0, fontsize=13)  

    plt.xlabel('Predicted label', fontsize=15) 
    plt.ylabel('True label', fontsize=15) 
    # plt.title('Confusion Matrix', fontsize=16) 
    plt.tight_layout()

    # Log the plot to WandB
    if args.wandb_project:
        wandb.log({"confusion_matrix": wandb.Image(plt)})

    plt.close()

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
    
    # Initialize WandB 
    if config.get('wandb_project', None):
        wandb.init(project=config['wandb_project'], name=os.path.splitext(os.path.basename(config['model_config']))[0])
        wandb.config.update(config)

    # Train and test the model with the provided configuration
    train_model(argparse.Namespace(**config))
    wandb.finish()