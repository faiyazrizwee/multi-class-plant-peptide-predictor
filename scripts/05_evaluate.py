import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/05_evaluate.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def main():
    logging.info("Starting Evaluation and Comparison...")
    
    ml_results_path = 'results/classical_ml_results.csv'
    plm_results_path = 'results/plm_results.csv'
    
    if os.path.exists(ml_results_path):
        ml_df = pd.read_csv(ml_results_path)
    else:
        logging.warning("Classical ML results not found.")
        ml_df = pd.DataFrame()
        
    if os.path.exists(plm_results_path):
        plm_df = pd.read_csv(plm_results_path)
    else:
        logging.warning("PLM results not found.")
        plm_df = pd.DataFrame()
        
    if ml_df.empty and plm_df.empty:
        logging.error("No results to compare.")
        return
        
    all_results = pd.concat([ml_df, plm_df], ignore_index=True)
    all_results = all_results.sort_values(by='F1_Macro', ascending=False)
    
    all_results.to_csv('results/final_model_comparison.csv', index=False)
    
    logging.info(f"\nFinal Model Comparison:\n{all_results.to_string()}")
    
    # Plot Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_results, x='Model', y='F1_Macro')
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Comparison - Macro F1 Score')
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig('plots/model_comparison_f1.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=all_results, x='Model', y='ROC_AUC')
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Comparison - ROC AUC (OVR)')
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig('plots/model_comparison_roc_auc.png')
    plt.close()
    
    logging.info("Evaluation plots saved in 'plots/' directory.")

if __name__ == "__main__":
    main()
