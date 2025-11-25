import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter1d

class IntegratedGradients:
    def __init__(self, model, config, X_train):
        self.model = model
        self.config = config
        self.ig_config = config['ig']
        self.baseline = tf.constant(
            np.median(X_train, axis=0, keepdims=True), 
            dtype=tf.float32
        )
        self.seed = config['seed']
        
    def interpolate_inputs(self, baseline, x, steps):
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        alphas = tf.expand_dims(alphas, axis=1)
        return baseline + alphas * (x - baseline)
    
    def compute_gradients(self, x, target_class_idx):
        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = self.model(x, training=False)
            target = predictions[:, target_class_idx]
        return tape.gradient(target, x)
    
    def integrated_gradients(self, input_tensor, target_class_idx, steps):
        interpolated = self.interpolate_inputs(self.baseline, input_tensor, steps)
        grads = self.compute_gradients(interpolated, target_class_idx)
        avg_grads = tf.reduce_mean(grads[:-1], axis=0)
        return (input_tensor - self.baseline) * avg_grads
    
    def stable_ig(self, input_tensor, target_class_idx, runs):
        steps = self.ig_config['steps']
        igs = []
        for run in range(runs):
            tf.random.set_seed(self.seed + run)
            ig = self.integrated_gradients(input_tensor, target_class_idx, steps)
            igs.append(ig.numpy())
        return np.mean(igs, axis=0)
    
    def get_class_attributions(self, X_samples, target_class_idx):
        runs = self.ig_config['runs']
        all_igs = []
        
        for i, x in enumerate(X_samples):
            tf.random.set_seed(self.seed + i)
            x_tensor = tf.convert_to_tensor(x[None, :], dtype=tf.float32)
            ig = self.stable_ig(x_tensor, target_class_idx, runs)
            all_igs.append(np.abs(ig.flatten()))
        
        return np.mean(all_igs, axis=0)
    
    def detect_elbow(self, scores):
        scores = np.array(scores)
        n = len(scores)
        coords = np.vstack((np.arange(n), scores)).T
        first, last = coords[0], coords[-1]
        
        vec = last - first
        vec_norm = vec / np.linalg.norm(vec)
        
        vecs = coords - first
        proj = np.dot(vecs, vec_norm)
        parallel = np.outer(proj, vec_norm)
        
        dist = np.linalg.norm(vecs - parallel, axis=1)
        return np.argmax(dist) + 1
    
    def compute_class_specific_genes(self, X_data, y_data, label_encoder):
        avg_ig_by_class = {}
        elbow_points = {}
        class_genes = {}
        genes_names = None
        
        top_n = self.ig_config['top_n_scree']
        
        for cls_idx in np.unique(y_data):
            cls_name = label_encoder.inverse_transform([cls_idx])[0]
            X_class = X_data[y_data == cls_idx]
            
            avg_ig = self.get_class_attributions(X_class, cls_idx)
            sorted_scores = np.sort(np.abs(avg_ig))[::-1][:top_n]
            
            if self.ig_config['smooth']:
                window = self.ig_config['window_size']
                sorted_scores = uniform_filter1d(sorted_scores, size=window)
            
            k = self.detect_elbow(sorted_scores)
            
            avg_ig_by_class[cls_name] = avg_ig
            elbow_points[cls_name] = k
        
        return avg_ig_by_class, elbow_points
    
    def extract_top_genes(self, avg_ig_by_class, elbow_points, genes_names):
        class_genes = {}
        
        for cls, ig_scores in avg_ig_by_class.items():
            k = elbow_points.get(cls, 0)
            top_indices = np.argsort(np.abs(ig_scores))[-k:][::-1]
            class_genes[cls] = [genes_names[i] for i in top_indices]
        
        return class_genes
    
    def compute_global_scores(self, avg_ig_by_class, genes_names, method='sum'):
        ig_array = np.array([np.abs(ig) for ig in avg_ig_by_class.values()])
        
        if method == 'sum':
            global_scores = np.sum(ig_array, axis=0)
        elif method == 'mean':
            global_scores = np.mean(ig_array, axis=0)
        elif method == 'max':
            global_scores = np.max(ig_array, axis=0)
        else:
            global_scores = np.sum(ig_array, axis=0)
        
        return dict(zip(genes_names, global_scores))
    
    def rank_union_genes(self, class_genes, global_scores):
        all_genes = set()
        for gene_list in class_genes.values():
            all_genes.update(gene_list)
        
        return sorted(all_genes, key=lambda g: global_scores[g], reverse=True)