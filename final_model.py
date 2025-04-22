import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from docx import Document

# ============ CONFIG ============ #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5

RESUME_FOLDER = r"C:\Piramal_final_model\dataset\CV train"
CSV_PATH = r"C:\Piramal_final_model\train_preprocessed.csv"

# ============ Seed ============ #
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ============ Data Preprocessing ============ #
def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["CandidateID", "Performance"])
    y = df["Performance"].astype(int).values
    X_tabular = df.drop(columns=["Performance", "CandidateID"]).copy()

    # Encode object columns
    cat_cols = X_tabular.select_dtypes(include='object').columns
    for col in cat_cols:
        X_tabular[col] = LabelEncoder().fit_transform(X_tabular[col].astype(str))

    # Fill missing
    X_tabular.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_tabular.fillna(0, inplace=True)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tabular)
    return df["CandidateID"].values, X_scaled, y

def read_resume_text(candidate_id):
    filepath = os.path.join(RESUME_FOLDER, f"{candidate_id} Resume.docx")
    if not os.path.exists(filepath):
        return "[PAD]"
    try:
        doc = Document(filepath)
        return " ".join([p.text for p in doc.paragraphs if p.text.strip()]) or "[PAD]"
    except Exception as e:
        print(f"[WARN] Failed reading {filepath}: {e}")
        return "[PAD]"

# ============ Dataset ============ #
class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, candidate_ids, tabular_data, labels, tokenizer):
        self.ids = candidate_ids
        self.tabular = tabular_data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        text = read_resume_text(self.ids[idx])
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.tabular[idx], dtype=torch.float32),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )

def custom_collate_fn(batch):
    input_ids, attention_masks, tabular_data, labels = zip(*batch)
    return (
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(tabular_data),
        torch.stack(labels)
    )

# ============ Model ============ #
class MultimodalClassifier(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(32 + 768, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, tabular_data):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        tabular_out = self.tabular_net(tabular_data)
        fused = torch.cat((bert_out, tabular_out), dim=1)
        return self.fusion(fused)

# ============ Train and Eval ============ #
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, tabular_data, labels in tqdm(dataloader):
        input_ids, attention_mask, tabular_data, labels = (
            input_ids.to(DEVICE), attention_mask.to(DEVICE),
            tabular_data.to(DEVICE), labels.to(DEVICE)
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, tabular_data)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, tabular_data, labels in dataloader:
            input_ids, attention_mask, tabular_data, labels = (
                input_ids.to(DEVICE), attention_mask.to(DEVICE),
                tabular_data.to(DEVICE), labels.to(DEVICE)
            )
            outputs = model(input_ids, attention_mask, tabular_data)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# ============ Main ============ #
def main():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    candidate_ids, X_tabular, y = preprocess_csv(CSV_PATH)

    X_train_ids, X_val_ids, X_train_tab, X_val_tab, y_train, y_val = train_test_split(
        candidate_ids, X_tabular, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = torch.utils.data.DataLoader(
        ResumeDataset(X_train_ids, X_train_tab, y_train, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        ResumeDataset(X_val_ids, X_val_tab, y_val, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn
    )

    model = MultimodalClassifier(tabular_dim=X_tabular.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss = train_model(model, train_loader, optimizer, criterion)
        acc = evaluate_model(model, val_loader)
        print(f"Train Loss: {loss:.4f} | Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_multimodal_model.pt")
            print("✅ Model saved (best so far)")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("✅ CUDA Available:", torch.cuda.get_device_name(0))
    else:
        print("❌ CUDA not available. Using CPU.")
    main()
