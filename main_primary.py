import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from config.config_primary import PRIMARY_CONFIG
from src.utils import set_seed, calculate_metrics, display_metrics
from src.preprocessing import PreprocessData
from src.vae_model import build_VAE, train_vae
from src.integrated_gradients import IntegratedGradients
from src.feature_selection import EnsembleFeatureSelector
from src.validation import ValidationAnalysis

def main():
    config = PRIMARY_CONFIG
    set_seed(config['seed'])
    
    print("="*80)
    print("BIOMARKER IDENTIFICATION PIPELINE - PRIMARY DATASET")
    print("="*80)
    
    print("\n[1/7] Loading and Preprocessing Data...")
    preprocessor = PreprocessData(config)
    X, y = preprocessor.load_data()
    print(f"Raw data shape: X={X.shape}, y={y.shape}")
    
    data = preprocessor.preprocess(X, y)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_df = data['X_train_df']
    X_test_df = data['X_test_df']
    genes_names = data['genes_names']
    label_encoder = data['label_encoder']
    
    print(f"Preprocessed shape: X_train={X_train.shape}, X_test={X_test.shape}")
    
    print("\n[2/7] Training VAE with Supervised Head...")
    vae_model, encoder, decoder = build_VAE(config, X_train.shape[1], num_classes=5)
    history = train_vae(vae_model, X_train, y_train, config)
    
    y_pred_train = np.argmax(vae_model.predict(X_train, verbose=0), axis=1)
    y_pred_test = np.argmax(vae_model.predict(X_test, verbose=0), axis=1)
    
    conf_train = confusion_matrix(y_train, y_pred_train)
    conf_test = confusion_matrix(y_test, y_pred_test)
    
    print("\nTraining Performance:")
    train_metrics = calculate_metrics(conf_train, y_train, y_pred_train)
    display_metrics(train_metrics, label_encoder.classes_)
    
    print("\nTesting Performance:")
    test_metrics = calculate_metrics(conf_test, y_test, y_pred_test)
    display_metrics(test_metrics, label_encoder.classes_)
    
    print("\n[3/7] Computing Integrated Gradients...")
    ig_module = IntegratedGradients(vae_model, config, X_train)
    avg_ig_by_class, elbow_points = ig_module.compute_class_specific_genes(
        X_test, y_test, label_encoder
    )
    
    print("\nElbow points detected per class:")
    for cls, k in elbow_points.items():
        print(f"  {cls}: {k} genes")
    
    class_genes = ig_module.extract_top_genes(avg_ig_by_class, elbow_points, genes_names)
    global_scores = ig_module.compute_global_scores(avg_ig_by_class, genes_names, method='sum')
    union_genes = ig_module.rank_union_genes(class_genes, global_scores)
    
    print(f"\nTotal candidate genes after union: {len(union_genes)}")
    
    X_train_selected = X_train_df[union_genes].values
    X_test_selected = X_test_df[union_genes].values
    
    print("\n[4/7] Ensemble Feature Selection...")
    feature_selector = EnsembleFeatureSelector(config)
    ranked_df = feature_selector.rank_features(X_train_selected, y_train, union_genes)
    
    print("Computing optimal feature subset...")
    optimal_results = feature_selector.find_optimal_k(
        ranked_df, X_train_selected, y_train, union_genes
    )
    
    best_k = optimal_results['best_k']
    best_score = optimal_results['best_score']
    top_biomarkers = optimal_results['top_features']
    
    print(f"\nOptimal number of features: {best_k}")
    print(f"Best cross-validated score: {best_score:.2f}%")
    print(f"\nIdentified Biomarkers:")
    for i, gene in enumerate(top_biomarkers, 1):
        print(f"  {i}. {gene}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(optimal_results['all_ks'], optimal_results['all_scores'], 
             '-o', linewidth=2, markersize=6)
    plt.xlabel('Number of Genes')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance vs Number of Genes (Primary Dataset)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/primary/performance_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[5/7] ANOVA Analysis...")
    validator = ValidationAnalysis(config)
    
    X_full = pd.DataFrame(X_train, columns=genes_names)
    anova_results = validator.compute_anova(X_full, y_train, top_biomarkers, label_encoder)
    anova_results = validator.apply_corrections(anova_results)
    
    print("\nANOVA Results:")
    print(anova_results.to_string(index=False))
    
    anova_results.to_csv('results/primary/anova_results.csv', index=False)
    
    print("\n[6/7] Correlation Analysis...")
    X_full_with_genes = pd.DataFrame(
        np.column_stack([X_train, y_train]),
        columns=list(genes_names) + ['cancer_type']
    )
    X_biomarkers_df = X_full_with_genes[top_biomarkers]
    
    corr_results = validator.compute_correlation(
        X_biomarkers_df, y_train, top_biomarkers
    )
    
    print("\nCorrelation Results:")
    print(corr_results.to_string(index=False))
    corr_results.to_csv('results/primary/correlation_results.csv', index=False)
    
    print("\n[7/7] Saving Results...")
    results_summary = {
        'dataset': 'primary',
        'n_samples_train': X_train.shape[0],
        'n_samples_test': X_test.shape[0],
        'n_features_original': X_train.shape[1],
        'n_features_selected': best_k,
        'train_accuracy': train_metrics['Overall Accuracy'],
        'test_accuracy': test_metrics['Overall Accuracy'],
        'cv_accuracy': best_score,
        'biomarkers': top_biomarkers
    }
    
    pd.DataFrame([results_summary]).to_json(
        'results/primary/summary.json', orient='records', indent=2
    )
    
    pd.DataFrame({'biomarker': top_biomarkers}).to_csv(
        'results/primary/biomarkers.txt', index=False, header=False
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nFinal Biomarker Panel ({best_k} genes):")
    print(", ".join(top_biomarkers))
    print(f"\nCross-validated Accuracy: {best_score:.2f}%")
    print("\nResults saved in: results/primary/")

if __name__ == "__main__":
    os.makedirs('results/primary', exist_ok=True)
    main()