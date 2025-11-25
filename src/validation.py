import numpy as np
import pandas as pd
from scipy import stats

class ValidationAnalysis:
    def __init__(self, config):
        self.config = config
        
    def compute_anova(self, X, y, biomarkers, label_encoder):
        results = []
        
        n_samples = len(y)
        n_classes = len(label_encoder.classes_)
        
        for biomarker in biomarkers:
            if biomarker not in X.columns:
                continue
                
            groups = [X[biomarker][y == i].values for i in range(n_classes)]
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            grand_mean = X[biomarker].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_within = sum(((g - g.mean())**2).sum() for g in groups)
            ss_total = ss_between + ss_within
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            df_between = n_classes - 1
            df_within = n_samples - n_classes
            
            ncp = f_stat * df_between
            power = 1 - stats.ncf.cdf(
                stats.f.ppf(0.95, df_between, df_within),
                df_between,
                df_within,
                ncp
            ) if f_stat > 0 else 0
            
            results.append({
                'Biomarker': biomarker,
                'F-statistic': f_stat,
                'P-value': p_value,
                'Eta-squared': eta_squared,
                'Power': power
            })
        
        return pd.DataFrame(results)
    
    def apply_corrections(self, anova_results, alpha_bonferroni=0.05):
        n_tests = len(anova_results)
        bonferroni_threshold = alpha_bonferroni / n_tests
        
        p_values = anova_results['P-value'].values
        fdr_results = self.benjamini_hochberg(p_values)
        
        anova_results['Bonferroni_Significant'] = anova_results['P-value'] < bonferroni_threshold
        anova_results['FDR_Significant'] = fdr_results
        anova_results['Bonferroni_Threshold'] = bonferroni_threshold
        
        return anova_results
    
    def benjamini_hochberg(self, p_values, alpha=0.05):
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        thresholds = (np.arange(1, n + 1) / n) * alpha
        significant = sorted_p <= thresholds
        
        if not np.any(significant):
            return np.zeros(n, dtype=bool)
        
        max_idx = np.where(significant)[0][-1]
        threshold = thresholds[max_idx]
        
        result = np.zeros(n, dtype=bool)
        result[sorted_indices[:max_idx + 1]] = True
        
        return result
    
    def compute_correlation(self, X, y, biomarkers):
        from scipy.stats import pearsonr
        
        results = []
        for biomarker in biomarkers:
            if biomarker not in X.columns:
                continue
            
            corr, p_val = pearsonr(X[biomarker].values, y)
            results.append({
                'Biomarker': biomarker,
                'Correlation': corr,
                'P-value': p_val
            })
        
        return pd.DataFrame(results)
    
    def permutation_test(self, X, y, biomarker, n_permutations=10000):
        from scipy.stats import pearsonr
        
        if biomarker not in X.columns:
            return None, None
        
        obs_corr = pearsonr(X[biomarker].values, y)[0]
        
        perm_corrs = []
        for _ in range(n_permutations):
            perm_y = np.random.permutation(y)
            perm_corrs.append(pearsonr(X[biomarker].values, perm_y)[0])
        
        return obs_corr, np.array(perm_corrs)