import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, ViltProcessor, ViltModel
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

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
        custom_model = CustomCLIPModel(base_model, len(set([item['label'] for item in train_data])), mode=args.mode)
        collate_fn = lambda x: clip_collate_fn(x, mode=args.mode)
    elif args.model_name == "vilt-b32-mlm":
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        train_dataset = ViLTLocomotionDataset(train_data, args.image_folder, processor)
        test_dataset = ViLTLocomotionDataset(test_data, args.image_folder, processor)
        custom_model = CustomViLTModel(base_model, len(set([item['label'] for item in train_data])))
        collate_fn = vilt_collate_fn
    elif args.model_name == "imagebind_huge":
        from data_loader import ImageBindLocomotionDataset
        from models import CustomImageBindModel

        train_dataset = ImageBindLocomotionDataset(train_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
        test_dataset = ImageBindLocomotionDataset(test_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
        custom_model = CustomImageBindModel(len(set([item['label'] for item in train_data])), mode=args.mode)
        collate_fn = None  # ImageBind doesn't require a special collate function
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    custom_model.to(args.device)

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
            if args.model_name == "imagebind_huge":
                inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                labels = labels.to(args.device)
                logits = custom_model(inputs)
            else:
                images, input_ids, attention_mask, labels = batch
                if args.model_name == "clip-vit-large-patch14-336":
                    if args.mode == "image_only":
                        images = images.to(args.device)
                        logits_per_image, _ = custom_model(pixel_values=images)
                        logits = logits_per_image
                    elif args.mode == "text_only":
                        input_ids = input_ids.to(args.device)
                        attention_mask = attention_mask.to(args.device)
                        _, logits_per_text = custom_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits_per_text
                    elif args.mode == "image_text":
                        images = images.to(args.device)
                        input_ids = input_ids.to(args.device)
                        attention_mask = attention_mask.to(args.device)
                        logits_per_image, logits_per_text = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                        logits = (logits_per_image + logits_per_text) / 2
                elif args.model_name == "vilt-b32-mlm":
                    images = images.to(args.device)
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits, _ = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

                # Convert labels to numeric indices
                label_mapping = {label: idx for idx, label in enumerate(set([item['label'] for item in train_data]))}
                labels = torch.tensor([label_mapping[label] for label in labels]).to(args.device)

            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(custom_model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {train_loss}")

        # Learning rate scheduling
        scheduler.step(train_loss)

        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            trials = 0
            # Save the model when it achieves the best loss
            model_filename = f'best_{args.model_name}_{args.mode}_btch{args.batch_size}epch{args.num_epochs}.pth'
            torch.save(custom_model.state_dict(), f'{args.output_dir}/{model_filename}')
        else:
            trials += 1
            if trials >= args.patience:
                print(f"Early stopping on epoch {epoch+1}")
                break

    # Save the final model
    final_model_filename = f'fine_tuned_{args.model_name}_{args.mode}_btch{args.batch_size}epch{args.num_epochs}.pth'
    torch.save(custom_model.state_dict(), f'{args.output_dir}/{final_model_filename}')

    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, label_mapping, args)

def evaluate_model(model, dataloader, label_mapping, args):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if args.model_name == "imagebind_huge":
                inputs, labels = batch
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                labels = labels.to(args.device)
                logits = model(inputs)
            else:
                images, input_ids, attention_mask, labels = batch
                if args.model_name == "clip-vit-large-patch14-336":
                    if args.mode == "image_only":
                        images = images.to(args.device)
                        logits_per_image, _ = model(pixel_values=images)
                        logits = logits_per_image
                    elif args.mode == "text_only":
                        input_ids = input_ids.to(args.device)
                        attention_mask = attention_mask.to(args.device)
                        _, logits_per_text = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = logits_per_text
                    elif args.mode == "image_text":
                        images = images.to(args.device)
                        input_ids = input_ids.to(args.device)
                        attention_mask = attention_mask.to(args.device)
                        logits_per_image, logits_per_text = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                        logits = (logits_per_image + logits_per_text) / 2
                elif args.model_name == "vilt-b32-mlm":
                    images = images.to(args.device)
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    logits, _ = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(logits, dim=1)

            # Ensure labels are tensors before calling .tolist()
            if isinstance(labels, torch.Tensor):
                true_labels.extend(labels.tolist())
            else:
                true_labels.extend(labels)
            predicted_labels.extend(preds.cpu().numpy())

    # Convert true labels to numerical indices
    true_labels = [label_mapping[label] for label in true_labels]

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")
    print(classification_report(true_labels, predicted_labels, target_names=list(label_mapping.keys())))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for configuration
    parser.add_argument("model_name", type=str, choices=["clip-vit-large-patch14-336", "vilt-b32-mlm", "imagebind_huge"], help="Model name to use (e.g., 'clip-vit-large-patch14-336', 'vilt-b32-mlm', 'imagebind_huge').")
    parser.add_argument("mode", type=str, choices=["image_only", "text_only", "image_text", "image_audio", "audio_only"], help="Mode of training (e.g., 'image_only', 'text_only', 'image_text', 'image_audio', 'audio_only').")
    parser.add_argument("--train_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/train_data.json', help="Path to the training JSON file.")
    parser.add_argument("--test_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/test_data.json', help="Path to the test JSON file.")
    parser.add_argument("--image_folder", type=str, default='/content/drive/MyDrive/My Data/processed data/images', help="Path to the image folder.")
    parser.add_argument("--audio_folder", type=str, help="Path to the audio folder (only required for ImageBind).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=22, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--output_dir", type=str, default='/content/drive/MyDrive/My Data', help="Directory to save the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")

    args = parser.parse_args()

    # Train and test the model with the provided arguments
    train_model(args)
