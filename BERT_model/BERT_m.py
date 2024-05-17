# Import requied modules
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Function to convert emotion text labels to integer labels
def convert_emotions_to_labels(text_labels):
    label_dict = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'surprise': 4, 'love': 5}
    return [label_dict[label.lower()] for label in text_labels]

# Data loading function
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['Text'].tolist()
    labels = convert_emotions_to_labels(df['Emotion'].tolist())
    return texts, labels


# Custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-cased', num_labels=6, ignore_mismatched_sizes=True)  # There are six types of emotion
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
max_len = 128
batch_size = 64  # 16, 32

# Load training, validation, and test data
train_texts, train_labels = load_data('../Datasets/train.csv')
val_texts, val_labels = load_data('../Datasets/validation.csv')
test_texts, test_labels = load_data('../Datasets/test.csv')

# Create datasets for further BERT model
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_len)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setting epochs, initialise optimizer and learning rate scheduler
n_epochs = 3  # 2, 4, 5
optimizer = Adam(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * n_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Function to calculate flat accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to evaluate the model on the validation dataset
def val(model, val_loader, device):
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    nb_val_steps = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        tmp_val_accuracy = flat_accuracy(logits, label_ids)

        total_val_loss += loss.item()
        total_val_accuracy += tmp_val_accuracy
        nb_val_steps += 1

    avg_val_accuracy = total_val_accuracy / nb_val_steps
    avg_val_loss = total_val_loss / nb_val_steps

    return avg_val_loss, avg_val_accuracy


# Initialise lists to store metrics for plotting
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []


# Run training and validation operation
for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, labels)

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    avg_val_loss, avg_val_accuracy = val(model, val_loader, device)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    print(f'Training: Epoch {epoch+1}/{n_epochs}, Accuracy: {avg_train_accuracy:.4f}, Loss: {avg_train_loss:.4f}')
    print(f'Validation: Epoch {epoch+1}/{n_epochs}, Accuracy: {avg_val_accuracy:.4f}, Loss: {avg_val_loss:.4f}')

print('Finish training!')
print()


# Plot training and validation accuracies figure
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.xticks(range(1, n_epochs + 1))
plt.legend()
plt.show()

# Plot training and validation losses figure
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(range(1, n_epochs + 1))
plt.show()


# Set model to evaluation mode
model.eval()

# Initialise lists to store predictions and true labels
all_preds = []
all_labels = []

# Iterate through the dataloader
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to NumPy arrays for accuracy calculation
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate overall accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)

# Calculate each class accuracy and other metrics
class_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
class_report = classification_report(all_labels, all_preds, target_names=class_labels)

print(f'Overall Accuracy: {overall_accuracy:.4f}')
print(f'Per Class Accuracy and other metrics:\n{class_report}')


# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
