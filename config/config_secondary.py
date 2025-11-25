SECONDARY_CONFIG = {
    'seed': 42,
    'dataset_name': 'secondary',
    'data_path': 'data/secondary/data.csv',
    'labels_path': 'data/secondary/labels.csv',
    'gene_mapping_path': None,
    'test_size': 0.2,
    'validation_split': 0.1,
    'detection_threshold': 0.5,
    'vae': {
        'latent_dim': 128,
        'encoder_units': [1024, 512, 256],
        'decoder_units': [256, 512, 1024],
        'dropout_rate': 0.0,
        'l2_reg': 0.0,
        'beta_kl': 0.3,
        'recon_weight': 1.0,
        'clf_units': [64],
        'clf_dropout': 0.0,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'epochs': 100,
        'early_stopping_patience': 5
    },
    'ig': {
        'steps': 200,
        'runs': 3,
        'top_n_scree': 400,
        'smooth': True,
        'window_size': 7
    },
    'feature_selection': {
        'rfe_step': 1,
        'rf_estimators': 500,
        'max_iter': 5000,
        'cv_splits': 10,
        'cv_repeats': 10,
        'max_features_eval': 50
    },
    'color_map': {
        'PRAD': '#2C6B9E',
        'LUAD': '#CBBFA3',
        'BRCA': '#F582AE',
        'KIRC': '#FF6A13',
        'COAD': '#005B77'
    }
}