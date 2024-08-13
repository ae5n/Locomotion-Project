import torch
import argparse
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from data_loader import CLIPLocomotionDataset, clip_collate_fn
from models import CustomCLIPModel

def train_model(args):
    # Load the JSON files
    with open(args.train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)

    # Initialize CLIP processor and model
    if args.model_name == "clip-vit-large-patch14-336":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    # Set up dataset and dataloader
    train_dataset = CLIPLocomotionDataset(train_data, args.image_folder, processor, mode=args.mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: clip_collate_fn(x, mode=args.mode))

    test_dataset = CLIPLocomotionDataset(test_data, args.image_folder, processor, mode=args.mode)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: clip_collate_fn(x, mode=args.mode))

    # Instantiate the custom model
    num_classes = len(set([item['label'] for item in train_data]))
    custom_model = CustomCLIPModel(base_model, num_classes, mode=args.mode)
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
            images, input_ids, attention_mask, labels = batch
            if args.mode == "image_only":
                images = images.to(args.device)
                logits_per_image, _ = custom_model(pixel_values=images)
                logits = logits_per_image
            elif args.mode == "text_only":
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                _, logits_per_text = custom_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = logits_per_text
            elif args.mode == "image_and_text":
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits_per_image, logits_per_text = custom_model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                logits = (logits_per_image + logits_per_text) / 2

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
            torch.save(custom_model.state_dict(), f'{args.output_dir}/best_{args.model_name}_model.pth')
        else:
            trials += 1
            if trials >= args.patience:
                print(f"Early stopping on epoch {epoch+1}")
                break

    # Save the final model
    final_model_path = f'{args.output_dir}/fine_tuned_{args.model_name}_model.pth'
    torch.save(custom_model.state_dict(), final_model_path)

    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, label_mapping, args)

def evaluate_model(model, dataloader, label_mapping, args):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, input_ids, attention_mask, labels = batch
            if args.mode == "image_only":
                images = images.to(args.device)
                logits_per_image, _ = model(pixel_values=images)
                logits = logits_per_image
            elif args.mode == "text_only":
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                _, logits_per_text = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = logits_per_text
            elif args.mode == "image_and_text":
                images = images.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                logits_per_image, logits_per_text = model(pixel_values=images, input_ids=input_ids, attention_mask=attention_mask)
                logits = (logits_per_image + logits_per_text) / 2

            preds = torch.argmax(logits, dim=1)

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
    parser.add_argument("model_name", type=str, help="Name of the CLIP model to use (e.g., 'clip-vit-large-patch14-336').")
    parser.add_argument("mode", type=str, choices=["image_only", "text_only", "image_and_text"], help="Mode of training (e.g., 'image_only', 'text_only', 'image_and_text').")
    parser.add_argument("--train_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/train_data.json', help="Path to the training JSON file.")
    parser.add_argument("--test_json_path", type=str, default='/content/drive/MyDrive/My Data/processed data/test_data.json', help="Path to the test JSON file.")
    parser.add_argument("--image_folder", type=str, default='/content/drive/MyDrive/My Data/processed data/images', help="Path to the image folder.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=22, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--output_dir", type=str, default='/content/drive/MyDrive/My Data', help="Directory to save the model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")

    args = parser.parse_args()

    # Train and test the model with the provided arguments
    train_model(args)
