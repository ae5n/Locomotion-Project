import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json
import os
import re
from difflib import get_close_matches
from torch.nn.utils.rnn import pad_sequence
from data_loader import FlorenceLocomotionDataset  # Assuming this is where the dataset class is defined

def florence_collate_fn(batch, processor, mode):
    texts, labels, images = zip(*batch)
    
    if mode == 'image_text':
        inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=False)
    elif mode == 'image_only':
        if any(texts):  # If there is any text input, include it
            inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=False)
        else:
            inputs = processor(images=list(images), return_tensors="pt", padding=True, truncation=False)
    
    # Process labels for sequence-to-sequence learning
    tokenized_labels = processor.tokenizer(list(labels), return_tensors="pt", padding=True, truncation=False)
    
    return inputs, tokenized_labels['input_ids']


def train_model(args):
    # Load the JSON files
    with open(args.train_json_path, 'r') as f:
        train_data = json.load(f)
    with open(args.test_json_path, 'r') as f:
        test_data = json.load(f)

    # Extract labels from training data
    label_map = {label: idx for idx, label in enumerate({entry['label'] for entry in train_data})}
    labels = list(label_map.keys())

    # Initialize the processor and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # Optionally freeze the vision encoder
    if args.freeze_vision_encoder:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    # Set up datasets and dataloaders
    train_dataset = FlorenceLocomotionDataset(train_data, args.image_folder, use_prompt=args.use_prompt, mode=args.mode)
    test_dataset = FlorenceLocomotionDataset(test_data, args.image_folder, use_prompt=args.use_prompt, mode=args.mode)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: florence_collate_fn(batch, processor, args.mode))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: florence_collate_fn(batch, processor, args.mode))

    # Set up optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.num_epochs}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Debugging prints to check the shapes of inputs and labels
            print(f"input_ids shape: {inputs['input_ids'].shape}")
            print(f"pixel_values shape: {inputs['pixel_values'].shape}")
            print(f"labels shape: {labels.shape}")

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Training Loss: {avg_train_loss}")

        # Save model if it improves
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model.save_pretrained(os.path.join(args.output_dir, f"best_florence_model_epoch{epoch + 1}.pth"))
            processor.save_pretrained(os.path.join(args.output_dir, f"best_florence_processor_epoch{epoch + 1}.pth"))

        scheduler.step(avg_train_loss)

    # Save final model and processor
    model.save_pretrained(os.path.join(args.output_dir, "finetuned_florence_model.pth"))
    processor.save_pretrained(os.path.join(args.output_dir, "finetuned_florence_processor.pth"))

    # Evaluate by generating model output
    evaluate_model(model, processor, test_loader, device, label_map)

def evaluate_model(model, processor, dataloader, device, label_map):
    model.eval()
    generated_outputs = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            # Generate model outputs
            generated_ids = model.generate(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], max_new_tokens=50)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Convert true labels back to text for comparison
            true_text = processor.batch_decode(labels, skip_special_tokens=True)

            print(f"Generated text: {generated_text}, True label: {true_text}")
            
            generated_outputs.extend(generated_text)
            true_labels.extend(true_text)

    # Direct comparison for evaluation metrics
    accuracy = accuracy_score(true_labels, generated_outputs)
    report = classification_report(true_labels, generated_outputs, labels=list(label_map.keys()))

    print(f"Accuracy: {accuracy}")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments for configuration
    parser.add_argument("--model_name", type=str, default="microsoft/Florence-2-large", help="Model name to use for fine-tuning.")
    parser.add_argument("--mode", type=str, default="image_text", help="Mode for the dataset: image_text, image_only, or text_only.")
    parser.add_argument("--train_json_path", type=str, default="train_data.json", help="Path to the training JSON file.")
    parser.add_argument("--test_json_path", type=str, default="test_data.json", help="Path to the test JSON file.")
    parser.add_argument("--image_folder", type=str, default="images/", help="Path to the image folder.")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save the model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--freeze_vision_encoder", action='store_true', help="Whether to freeze the vision encoder during training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--use_prompt", action="store_true", help="Use the prompt for text input if set; otherwise, use the direct command text.")

    args = parser.parse_args()

    # Start training
    train_model(args)
