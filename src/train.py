import pandas as pd
import torch
from torch.utils.data import Dataset
from adversarial import add_noise
from transformers import BertTokenizer


class SpamDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, use_noise=False):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_noise = use_noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.data.iloc[idx]["label"]

        if self.use_noise:
            text = add_noise(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_datasets():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = SpamDataset("data/processed/train.csv", tokenizer)
    test_dataset = SpamDataset("data/processed/test.csv", tokenizer)

    noisy_test_dataset = SpamDataset(
        "data/processed/test.csv",
        tokenizer,
        use_noise=True
    )

    return train_dataset, test_dataset, noisy_test_dataset, tokenizer


from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import transformers


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def train_model():
    train_dataset, test_dataset, noisy_test_dataset, tokenizer = load_datasets()

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",          
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate on clean data
    clean_results = trainer.evaluate(eval_dataset=test_dataset)

# Evaluate on noisy data
    noisy_results = trainer.evaluate(eval_dataset=noisy_test_dataset)

    print("\nClean Data Results:", clean_results)
    print("\nNoisy Data Results:", noisy_results)

# Robustness drop
    drop = clean_results["eval_accuracy"] - noisy_results["eval_accuracy"]
    print(f"\nRobustness Drop: {drop:.4f}")

    # Save model
    trainer.save_model("models/saved_model")


if __name__ == "__main__":
    train_model()