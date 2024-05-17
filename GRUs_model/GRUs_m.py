# Import requied module
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define global variables
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function to convert emotion text labels to integer labels
def convert_emotions_to_labels(text_labels):
    label_dict = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'surprise': 4, 'love': 5}
    return [label_dict[label.lower()] for label in text_labels]

# Define a text preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.lower()  # Convert text to lowercase
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the tokens
    return ' '.join(tokens)

# Data loading function
def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['Text'].tolist()
    texts = [preprocess_text(text) for text in texts]
    labels = convert_emotions_to_labels(df['Emotion'].tolist())
    return texts, labels


# Load training, validation, and test data
train_texts, train_labels = load_data('../Datasets/train.csv')
val_texts, val_labels = load_data('../Datasets/validation.csv')
test_texts, test_labels = load_data('../Datasets/test.csv')

# Generate word embeddings using Word2Vec
sentences = [word_tokenize(text.lower()) for text in train_texts]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)  # sg=1表示使用Skip-Gram
word2vec_model.save("word2vec.model")

# Build vocabulary and add <unk> token
vocab = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
vocab['<unk>'] = len(vocab) + 1

# Build the embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
for word, i in vocab.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]


# Text encoding function
def encode_text(text, vocab, max_len):
    tokens = word_tokenize(text.lower())
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    padding_length = max_len - len(token_ids)
    return token_ids[:max_len] + [0] * padding_length


# Custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = [encode_text(text, vocab, max_len) for text in texts]
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


max_len = 128
batch_size = 64 # 16, 32

# Create datasets for further GRUs model
train_dataset = EmotionDataset(train_texts, train_labels, vocab, max_len=max_len)
val_dataset = EmotionDataset(val_texts, val_labels, vocab, max_len=max_len)
test_dataset = EmotionDataset(test_texts, test_labels, vocab, max_len=max_len)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define GRU model architecture
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, embedding_matrix, dropout=0.5):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)  # Initialise the embedding layer with pre-trained word vectors
        self.dropout = nn.Dropout(dropout)   # Initialise the Dropout layer for regularization
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)  # Initialise the GRU layer to get outputs and hidden states
        self.fc = nn.Linear(hidden_dim, output_dim)  # Initialise the fully connected layer for final output classification

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.gru(embedded)
        hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)


# Check emotion distribution in each dataset
def convert_labels_to_emotions(numeric_labels):
    reverse_label_dict = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'surprise', 5: 'love'}
    return [reverse_label_dict[label] for label in numeric_labels]

def plot_label_distribution(labels, title):
    label_counts = Counter(convert_labels_to_emotions(labels))
    sorted_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
    counts = [label_counts[label] for label in sorted_labels]
    plt.bar(sorted_labels, counts)
    plt.xlabel('Emotions')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

plot_label_distribution(train_labels, 'Training Distribution')
plot_label_distribution(val_labels, 'Validation Distribution')
plot_label_distribution(test_labels, 'Testing Distribution')


# Set the device and initialise the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRU(vocab_size=len(vocab)+1, embedding_dim=100, hidden_dim=128, output_dim=6, n_layers=1, embedding_matrix=embedding_matrix)
model.to(device)

# Set the optimizer, loss function and epochs
optimizer = optim.Adam(model.parameters(), lr=5e-3)
criterion = nn.CrossEntropyLoss()
n_epochs = 10


# Function to calculate flat accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Training function
def train_model(train_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    start_time = time.time()
    for batch in train_loader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()

    end_time = time.time()
    print(f"Epoch training time: {end_time - start_time:.2f} seconds")

    return total_loss / len(train_loader), total_accuracy / len(train_loader)

# Validation function
def val_model(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == labels).float().mean().item()

    return total_loss / len(val_loader), total_accuracy / len(val_loader)


# Run training and validation
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for epoch in range(n_epochs):

    train_loss, train_acc = train_model(train_loader, model, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_loss, val_acc = val_model(val_loader, model, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Training: Epoch {epoch+1}/{n_epochs}, Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}')
    print(f'Validation: Epoch {epoch+1}/{n_epochs}, Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}')


# Plot training and validation accuracies figure
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, n_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Training and Validation Accuracy')
plt.xticks(range(1, n_epochs + 1))
#plt.xticks(range(1, n_epochs+1, 3))  
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
#plt.xticks(range(1, n_epochs+1, 3))
plt.show()


# Set model to evaluation mode
model.eval()

# Initialise lists to store predictions and true labels
all_preds = []
all_labels = []

# Iterate through the dataloader
for batch in test_loader:
    input_ids = batch['text'].to(device)
    labels = batch['label'].to(device)

    # Disable gradient calculation
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs
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
