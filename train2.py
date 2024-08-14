import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from data_loader import ImageBindLocomotionDataset
from models import CustomImageBindModel

def train_model(args):
    # Load the JSON files
    with open(args.train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)

    # Initialize dataset and model
    train_dataset = ImageBindLocomotionDataset(train_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
    test_dataset = ImageBindLocomotionDataset(test_data, args.image_folder, args.audio_folder, mode=args.mode, device=args.device)
    
    # Here we define the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    custom_model = CustomImageBindModel(len(set([item['label'] for item in train_data])), mode=args.mode)
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
        for inputs, labels in train_dataloader:  # Make sure train_dataloader is defined
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = custom_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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
            torch.save(custom_model.state_dict(), f'{args.output_dir}/best_imagebind_model.pth')
        else:
            trials += 1
            if trials >= args.patience:
                print("Early stopping on epoch {epoch+1}")
                break

    # Save the final model
    final_model_path = f'{args.output_dir}/fine_tuned_imagebind_model.pth'
    torch.save(custom_model.state_dict(), final_model_path)

    # Evaluate the model after training
    evaluate_model(custom_model, test_dataloader, args)


def evaluate_model(model, dataloader, args):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            labels = labels.to(args.device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(preds.cpu().numpy())

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")
    print(classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments for configuration
    parser.add_argument("--train_json_path", type=str, required=True)
    parser.add_argument("--test_json_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--audio_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['image_text', 'image_audio', 'image_only', 'text_only', 'audio_only'], required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train_model(args)
