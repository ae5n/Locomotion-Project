import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltModel
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import wandb

from data_loader import CLIPLocomotionDataset, ViLTLocomotionDataset, clip_collate_fn, vilt_collate_fn
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

    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    custom_model.to(args.device) 

    fusion_suffix = f"_{args.fusion_method}" if args.model_name != "vilt-b32-mlm" and args.mode in ["image_text", "image_audio"] else ""

    # Initialize WandB if the project name is provided
    if args.wandb_project:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)  # Save the args to WandB config
    
        # Calculate and log model parameters once
        model_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in custom_model.parameters())
        
        wandb.config.update({
            "model_name": args.model_name,
            "mode": args.mode,  # Log the mode (e.g., 'image_only', 'text_only', etc.)
            "trainable_parameters": model_params,  # Log trainable parameters
            "total_parameters": total_params,  # Log total parameters
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if args.device == 'cuda' else "CPU"
        })
        
        # Log the model architecture and gradients/weights
        wandb.watch(custom_model, log="all", log_freq=10)  # Log gradients and parameters at each 10th step

    # Define optimizer, criterion, and scheduler
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=args.learning_rate)
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
                images, input_ids, attention_mask, labels = batch
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
                images, input_ids, attention_mask, labels = batch
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits, _ = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            elif args.model_name == "imagebind_huge":
                inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                logits = custom_model(inputs)

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
            model_filename = f'best_{args.model_name}_{args.mode}_{fusion_suffix}_btch{args.batch_size}epch{args.num_epochs}.pth'
            torch.save(custom_model.state_dict(), f'{args.output_dir}/{model_filename}')
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

    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, train_dataset.get_label_map(), args)

def evaluate_model(model, dataloader, label_mapping, args):
    model.eval()
    true_labels = []
    predicted_labels = []

    # Reverse the label map for lookup
    inverse_label_map = {v: k for k, v in label_mapping.items()}

    with torch.no_grad():
        for batch in dataloader:
            if args.model_name == "clip-vit-large-patch14-336":
                images, input_ids, attention_mask, labels = batch
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
                images, input_ids, attention_mask, labels = batch
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits, _ = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            elif args.model_name == "imagebind_huge":
                inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                logits = model(inputs)
                labels = labels.cpu().tolist()  # Convert to list of indices for ImageBind

            preds = torch.argmax(logits, dim=1)

            if isinstance(labels, torch.Tensor):  # Handle labels for ImageBind
                true_labels.extend(labels.cpu().tolist())
            else:  # Handle labels for CLIP and ViLT
                true_labels.extend(labels)

            predicted_labels.extend(preds.cpu().numpy())

    # Convert indices to labels using the inverse label map
    true_labels = [inverse_label_map[label] for label in true_labels]
    predicted_labels = [inverse_label_map[label] for label in predicted_labels]

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
    parser.add_argument("model_name", type=str, choices=["clip-vit-large-patch14-336", "vilt-b32-mlm", "imagebind_huge"], help="Model name to use (e.g., 'clip-vit-large-patch14-336', 'vilt-b32-mlm', 'imagebind_huge').")
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

    args = parser.parse_args()

    # Train and test the model with the provided arguments
    train_model(args)
