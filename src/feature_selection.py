import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

class EnsembleFeatureSelector:
    def __init__(self, config):
        self.config = config
        self.fs_config = config['feature_selection']
        self.seed = config['seed']
        
    def create_selectors(self):
        rfe_estimator = LinearSVC(
            dual=False,
            max_iter=self.fs_config['max_iter'],
            random_state=self.seed
        )
        
        l1_estimator = LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=self.fs_config['max_iter'],
            random_state=self.seed,
            n_jobs=1
        )
        
        rf_estimator = RandomForestClassifier(
            n_estimators=self.fs_config['rf_estimators'],
            random_state=self.seed,
            n_jobs=1
        )
        
        selectors = {
            'rfe': RFE(
                rfe_estimator,
                n_features_to_select=1,
                step=self.fs_config['rfe_step']
            ),
            'l1': SelectFromModel(l1_estimator, threshold='mean'),
            'rf': SelectFromModel(rf_estimator, threshold='median')
        }
        
        return selectors
    
    def compute_feature_scores(self, selector, X, y):
        selector.fit(X, y)
        
        if hasattr(selector, 'ranking_'):
            ranking = selector.ranking_
            return ranking.max() + 1 - ranking
        else:
            estimator = selector.estimator_
            if hasattr(estimator, 'coef_'):
                importance = estimator.coef_
            elif hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
            else:
                raise ValueError("Estimator has no coef_ or feature_importances_")
            
            return np.abs(importance).mean(axis=0) if importance.ndim > 1 else np.abs(importance)
    
    def rank_features(self, X, y, feature_names):
        selectors = self.create_selectors()
        scores_dict = {}
        
        for name, selector in selectors.items():
            scores = self.compute_feature_scores(selector, X, y)
            scores_dict[name] = scores
        
        df = pd.DataFrame(scores_dict, index=feature_names)
        df['mean'] = df.mean(axis=1)
        df = df.sort_values('mean', ascending=False)
        
        return df
    
    def evaluate_feature_subset(self, X, y, feature_indices, k):
        X_subset = X[:, feature_indices]
        
        cv = RepeatedStratifiedKFold(
            n_splits=self.fs_config['cv_splits'],
            n_repeats=self.fs_config['cv_repeats'],
            random_state=self.seed
        )
        
        estimator = SVC(kernel='linear', random_state=self.seed)
        
        scores = cross_val_score(
            estimator, X_subset, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=1
        )
        
        mean_accuracy = scores.mean() * 100
        
        return mean_accuracy
    
    def find_optimal_k(self, ranked_df, X, y, feature_names):
        max_k = min(self.fs_config['max_features_eval'], len(ranked_df))
        
        results = []
        feature_name_list = list(feature_names)
        
        for k in range(1, max_k + 1):
            top_genes = ranked_df.index[:k].tolist()
            indices = [feature_name_list.index(g) for g in top_genes]
            
            score = self.evaluate_feature_subset(X, y, indices, k)
            results.append((k, score))
        
        ks, scores = zip(*results)
        best_k, best_score = max(results, key=lambda x: x[1])
        
        top_features = ranked_df.index[:best_k].tolist()
        
        return {
            'best_k': best_k,
            'best_score': best_score,
            'top_features': top_features,
            'all_ks': ks,
            'all_scores': scores
        }