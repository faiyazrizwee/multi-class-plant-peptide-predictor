import pandas as pd
import numpy as np
import logging
import os
import torch
import joblib
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/04_train_plm.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

class PeptideDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.nn.functional.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    
    precision, recall, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    
    try:
        roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
        
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
        'mcc': mcc,
        'roc_auc': roc_auc
    }

def plot_cm(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/cm_{model_name.replace(" ", "_")}.png')
    plt.close()

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    model_name = "facebook/esm2_t6_8M_UR50D" # Fast small ESM-2 for baseline
    logging.info(f"Loading PLM: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    # Load Label Encoder if exists, or fit new
    if os.path.exists('models/label_encoder.pkl'):
        le = joblib.load('models/label_encoder.pkl')
    else:
        le = LabelEncoder()
        le.fit(train_df['Label'])
        joblib.dump(le, 'models/label_encoder.pkl')
        
    y_train = le.transform(train_df['Label']).tolist()
    y_test = le.transform(test_df['Label']).tolist()
    classes = le.classes_
    num_labels = len(classes)
    
    logging.info("Tokenizing...")
    train_encodings = tokenizer(train_df['Sequence'].tolist(), truncation=True, padding=True, max_length=200)
    test_encodings = tokenizer(test_df['Sequence'].tolist(), truncation=True, padding=True, max_length=200)
    
    train_dataset = PeptideDataset(train_encodings, y_train)
    test_dataset = PeptideDataset(test_encodings, y_test)
    
    logging.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Freeze encoder
    for param in model.esm.parameters():
        param.requires_grad = False
    logging.info("Encoder frozen. Only classifier head will be trained.")
    
    class_wts = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_wts_tensor = torch.tensor(class_wts, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    training_args = TrainingArguments(
        output_dir='./models/esm2_results',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True
    )
    
    trainer = CustomTrainer(
        class_weights=class_wts_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    logging.info("Starting training...")
    trainer.train()
    
    logging.info("Evaluating...")
    eval_res = trainer.evaluate()
    logging.info(f"Evaluation Results: {eval_res}")
    
    # Save final results to CSV
    res_df = pd.DataFrame([{
        'Model': 'ESM2-8M-HeadFT',
        'Accuracy': eval_res['eval_accuracy'],
        'Precision': eval_res['eval_precision_macro'],
        'Recall': eval_res['eval_recall_macro'],
        'F1_Macro': eval_res['eval_f1_macro'],
        'F1_Weighted': eval_res['eval_f1_weighted'],
        'MCC': eval_res['eval_mcc'],
        'ROC_AUC': eval_res['eval_roc_auc']
    }])
    res_df.to_csv('results/plm_results.csv', index=False)
    
    # Generate Predictions for CM
    preds = trainer.predict(test_dataset)
    y_pred = preds.predictions.argmax(-1)
    plot_cm(y_test, y_pred, classes, "ESM2-8M")
    
    model.save_pretrained('./models/best_esm2_model')
    tokenizer.save_pretrained('./models/best_esm2_model')
    logging.info("ESM2 Model saved.")

if __name__ == "__main__":
    main()
