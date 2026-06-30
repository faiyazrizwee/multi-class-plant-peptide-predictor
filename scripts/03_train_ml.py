import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/03_train_ml.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/cm_{model_name.replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # top 20
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(20), importances[indices], align='center')
        plt.xticks(range(20), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, 20])
        plt.tight_layout()
        plt.savefig(f'plots/feat_imp_{model_name.replace(" ", "_")}.png')
        plt.close()

def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    logging.info("Loading features...")
    train_df = pd.read_csv('features/train_features.csv')
    test_df = pd.read_csv('features/test_features.csv')
    
    # Check for NaN and remove or fill
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    X_train = train_df.drop(columns=['Entry', 'Label']).values
    y_train_raw = train_df['Label'].values
    
    X_test = test_df.drop(columns=['Entry', 'Label']).values
    y_test_raw = test_df['Label'].values
    
    feature_names = train_df.drop(columns=['Entry', 'Label']).columns
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    joblib.dump(le, 'models/label_encoder.pkl')
    
    classes = le.classes_
    logging.info(f"Classes: {classes}")
    
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42), # Note: MLP doesn't take sample weights easily in sklearn init
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0)
    }
    
    results = []
    best_f1 = -1
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        try:
            if name in ['XGBoost', 'CatBoost']:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            elif name == 'MLP':
                # For MLP, no direct sample_weight in fit for standard multi-class easily, just train
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
                
            logging.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            
            res_dict = {
                'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 
                'F1_Macro': f1_macro, 'F1_Weighted': f1_weighted, 'MCC': mcc, 'ROC_AUC': roc_auc
            }
            results.append(res_dict)
            
            logging.info(f"{name} F1_Macro: {f1_macro:.4f}")
            
            # Save artifacts
            plot_confusion_matrix(y_test, y_pred, classes, name)
            plot_feature_importance(model, feature_names, name)
            
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model_name = name
                best_model = model
                
        except Exception as e:
            logging.error(f"Failed to train {name}: {e}")
            
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/classical_ml_results.csv', index=False)
    logging.info(f"\n{results_df.to_string()}")
    
    logging.info(f"Best Model: {best_model_name} with F1_Macro: {best_f1:.4f}")
    joblib.dump(best_model, 'models/best_classical_model.pkl')
    
    # Save detailed classification report for best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best, target_names=classes)
    with open('results/best_model_classification_report.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n\n")
        f.write(report)

if __name__ == "__main__":
    main()
