# Cancer Biomarkers Identification Using Explainable Variational Autoencoders With Ensemble Feature Selection

An interpretable and generalized pipeline integrated VAE with integrated gradients and ensemble feature selection to address challenges such as RNA-Seq's high dimensionality and noise,  models overfitting, feature selection instability, dataset bias, and poor generalization.
## Overview

This pipeline implements a generalizable methodology for identifying cancer biomarkers from high-dimensional RNA-Seq data. The approach combines:

- **VAE-based representation learning** for dimensionality reduction
- **Integrated Gradients** for interpretable feature attribution
- **Ensemble Feature Selection** for robust gene ranking and parsimonous feature selection
- **Statistical validation** through ANOVA and correlation analysis

The methodology has been validated on two independent datasets:
- **Primary Dataset (UCI)**: 801 samples, 20,531 genes → 12-gene panel (99.97% accuracy)
- **Secondary Dataset (TCGA)**: 2,403 samples, 66,660 genes → 34-gene panel (99.31% accuracy)

## Project Structure

```
biomarker_identification/
├── config/
│   ├── config_primary.py       # Configuration for primary dataset
│   └── config_secondary.py     # Configuration for secondary dataset
├── data/
│   ├── data guide.txt                # Primary and secondary dataset detailed guide
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing pipeline
│   ├── vae_model.py           # VAE architecture
│   ├── integrated_gradients.py # IG implementation
│   ├── feature_selection.py    # Ensemble feature selection
│   ├── validation.py           # Statistical validation
│   └── utils.py                # Utility functions
├── results/                   # Research/project results  
├── main_primary.py             # Main script for primary dataset
├── main_secondary.py           # Main script for secondary dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create data directories:
```bash
mkdir -p data/primary data/secondary results/primary results/secondary
```

## Data Preparation

### Primary Dataset (UCI)

Place the following files in `data/primary/`:
- `data.csv` - Gene expression matrix
- `labels.csv` - Cancer type labels
- `Genetic_Data.csv` - Gene name mapping

### Secondary Dataset (TCGA)

Place the following files in `data/secondary/`:
- `data.csv` - Gene expression matrix
- `labels.csv` - Cancer type labels

## Usage

### Running the Pipeline

For the primary dataset:
```bash
python main_primary.py
```

For the secondary dataset:
```bash
python main_secondary.py
```

### Pipeline Stages

The pipeline executes seven stages:

1. **Data Loading and Preprocessing**
   - Loads raw RNA-Seq data
   - Removes constant features
   - Applies log transformation
   - Performs dual normalization (sparse/dense)

2. **VAE Training**
   - Trains VAE with supervised head
   - Learns latent representations
   - Evaluates classification performance

3. **Integrated Gradients**
   - Computes attribution scores
   - Identifies class-specific genes
   - Applies elbow method for selection

4. **Ensemble Feature Selection**
   - Ranks genes using SVM-RFE, L1-LR, RF
   - Creates consensus ranking
   - Finds optimal feature subset with cross validated linear SVM

5. **ANOVA Analysis**
   - Tests differential expression
   - Applies multiple testing corrections
   - Computes effect sizes

6. **Correlation Analysis**
   - Evaluates gene-phenotype associations
   - Computes significance

7. **Results Export**
   - Saves biomarker lists
   - Exports statistical results
   - Generates summary reports

## Configuration

### Key Parameters

Both configuration files (`config_primary.py` and `config_secondary.py`) contain:

- **VAE Architecture**: Encoder/decoder dimensions, latent space size
- **Training**: Learning rate, batch size, epochs, early stopping
- **IG Parameters**: Integration steps, stability runs, smoothing
- **Feature Selection**: Cross-validation folds, max features to evaluate

### Primary vs Secondary Differences

The main architectural difference between datasets:

**Primary Dataset:**
- Input: 20,254 features
- Encoder: [512, 256] → 128
- Decoder: [256, 512]

**Secondary Dataset:**
- Input: 61,958 features
- Encoder: [1024, 512, 256] → 128
- Decoder: [256, 512, 1024]

The deeper architecture for the secondary dataset accommodates the larger feature space.

## Output Files

Results are saved in `results/`:

## Methodology Details

### Preprocessing Pipeline

1. **Train/Test Split**: 80/20 stratified split
2. **Constant Removal**: Features with zero variance excluded
3. **Log Transformation**: `log(1 + x)` applied to all features
4. **Dual Normalization**:
   - Sparse genes (detection rate ≤ 0.5): No additional scaling
   - Dense genes (detection rate > 0.5): RobustScaler normalization

### VAE Architecture

- **Encoder**: Multilayer feedforward network with ReLU activations
- **Latent Space**: 128-dimensional continuous representation
- **Decoder**: Mirrors encoder structure
- **Supervised Head**: Shallow feedforward network for classification
- **Loss Function**: Combined reconstruction (MSE) + β-weighted KL divergence + classification loss

### Integrated Gradients

- **Baseline**: Median expression profile across training set
- **Integration**: multiple steps along interpolation path
- **Stabilization**: Averaged over multiple runs with different random seeds
- **Attribution**: Mean absolute IG scores per class
- **Selection**: Elbow method on sorted attribution curves

### Ensemble Feature Selection

Three complementary methods:
1. **SVM-RFE**: Recursive elimination with linear SVM
2. **L1-LR**: L1-regularized logistic regression (SAGA solver)
3. **RF Importance**: Random Forest impurity-based importance

Consensus ranking combines all three methods, followed by iterative evaluation via 10-fold cross-validation (repeated 10 times) using linear svm.

### Statistical Validation

- **ANOVA**: One-way analysis of variance across cancer types
- **Multiple Testing**: Bonferroni and FDR corrections
- **Effect Size**: Eta-squared calculation
- **Power Analysis**: Statistical power computation
- **Correlation**: Pearson correlation with phenotype

## Expected Results

### Primary Dataset
- **Preprocessing**: ~20,254 features after filtering
- **Candidate Genes**: ~200-300 from IG selection
- **Final Panel**: ~12 genes
- **Accuracy**: ~99.97% (cross-validated)

### Secondary Dataset
- **Preprocessing**: ~61,958 features after filtering
- **Candidate Genes**: ~250-550 from IG selection
- **Final Panel**: ~34 genes
- **Accuracy**: ~99.31% (cross-validated)

## Troubleshooting

### Memory Issues
- Reduce batch size in config
- Decrease VAE architecture dimensions
- Process data in smaller chunks

### Convergence Issues
- Increase training epochs
- Adjust learning rate
- Modify β parameter for KL divergence

### Feature Selection Taking Too Long
- Reduce `max_features_eval` in config
- Decrease `cv_repeats` parameter
- Use fewer estimators for Random Forest

## Citation

If you use this pipeline in your research, please cite:

```
cancer biomarkers identification using Explainable VAE with EFS ( Abdullah Khan, University of Malakand, Pakistan)
```

## License

MIT

## Contact

abdullahkhan.prof@gmail.com

## Acknowledgments

This is the implementation code of methodology described in the research paper:
"Cancer Biomarker Identification Pipeline using Explainable VAEs with Ensemble Feature Selection"
