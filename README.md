
# Marketing Mix Modeling (MMM) Analysis

A comprehensive Marketing Mix Modeling pipeline implementing three complementary approaches for analyzing media effectiveness and revenue attribution. This project combines causal inference through Instrumental Variables (IV/2SLS), regularized regression with ElasticNet, and gradient boosting with XGBoost to provide robust insights for marketing decision-making.

## 🎯 Project Overview

This MMM implementation focuses on:
- **Causal Inference**: IV/2SLS analysis with Google as mediator, instrumented by social channels
- **Predictive Modeling**: ElasticNet with feature selection and XGBoost with SHAP explanations  
- **Time Series Validation**: Proper temporal splits to avoid data leakage
- **Comprehensive Diagnostics**: Residual analysis, multicollinearity checks, sensitivity analysis
- **Business Insights**: Scenario planning and actionable recommendations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SruthiJayaram/revenue_analyser.git
cd revenue_analyser
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the analysis:**
```bash
# Run notebooks in order for complete analysis
jupyter notebook notebooks/01_EDA_and_prep.ipynb
jupyter notebook notebooks/02_modeling_iv_elasticnet_xgb.ipynb

# Or launch Jupyter and run cells interactively
jupyter notebook
```

## 📁 Project Structure

```
revenue_analyser/
├── 📊 data/
│   └── weekly_data.zip              # Original source data (4KB)
├── 📓 notebooks/
│   ├── 01_EDA_and_prep.ipynb       # Data exploration & feature engineering
│   └── 02_modeling_iv_elasticnet_xgb.ipynb  # Complete MMM modeling pipeline
├── 🐍 src/                         # Reusable Python modules
│   ├── preprocess.py               # Data loading and preprocessing
│   ├── features.py                 # Feature engineering and transformations
│   ├── models.py                   # Modeling functions (IV, ElasticNet, XGBoost)
│   └── diagnostics.py              # Model diagnostics and validation
├── 📈 reports/                     # Generated outputs (git ignored)
│   ├── *.png                       # Visualizations
│   ├── *.pkl                       # Saved models and summaries
│   └── analysis_summary.pkl        # Complete findings
├── 📋 requirements.txt             # Python dependencies
├──  .gitignore                   # Comprehensive exclusions
└── 📖 README.md                    # This file
```

## 🔬 Analysis Workflow

### 1. Data Exploration & Preparation (`01_EDA_and_prep.ipynb`)
- **Data Loading**: Extracts and cleans weekly data from zip file
- **Exploratory Analysis**: Revenue trends, media spend patterns, correlations
- **Feature Engineering**: 
  - Adstock transformations for media carryover effects
  - Hill saturation curves for diminishing returns
  - Time-based features (seasonality, trends)
  - Lag variables and promotional flags
- **Data Validation**: Missing value checks, outlier detection
- **Output**: `processed_weekly_data.csv`, feature lists, visualizations

### 2. Comprehensive Modeling (`02_modeling_iv_elasticnet_xgb.ipynb`)

#### **Time-Based Data Splits**
- Training: 70% (model development)
- Validation: 15% (hyperparameter tuning) 
- Test: 15% (final evaluation)
- Temporal order preserved to avoid data leakage

#### **Three Modeling Approaches:**

**🎯 Instrumental Variables (IV/2SLS)**
- **Purpose**: Causal inference for media incrementality
- **Setup**: Google spend as endogenous variable, social channels as instruments
- **Method**: Manual 2SLS implementation with rank deficiency checks
- **Output**: Causal coefficients, first-stage F-statistics, weak instrument diagnostics

**🎯 ElasticNet Regression**
- **Purpose**: Regularized regression with automatic feature selection
- **Method**: Cross-validated L1/L2 regularization with time series splits
- **Features**: Handles multicollinearity, provides sparse solutions
- **Output**: Selected features, coefficient plots, performance metrics

**🎯 XGBoost Gradient Boosting**
- **Purpose**: Non-linear pattern detection and feature importance
- **Method**: Early stopping with validation set, SHAP explanations
- **Features**: Captures complex interactions, robust to outliers
- **Output**: Feature importance rankings, SHAP analysis, learning curves

#### **Model Diagnostics**
- **Residual Analysis**: Normality tests, autocorrelation, heteroscedasticity
- **Multicollinearity**: VIF calculations and high-correlation detection
- **Performance Comparison**: RMSE, R², MAE across all models
- **Sensitivity Analysis**: Revenue impact scenarios with bootstrap confidence intervals

## 📊 Key Outputs

### Generated Files (in `reports/` directory):
- **Models**: `*.pkl` files for ElasticNet, XGBoost, and IV results
- **Visualizations**: 
  - Data splits and time series plots
  - Feature importance and coefficient plots
  - Residual diagnostics and model comparisons
  - SHAP explanations and sensitivity analysis
- **Summaries**: Analysis findings and business recommendations

### Business Insights:
- **Media Attribution**: Channel-specific revenue contribution
- **Incrementality**: Causal estimates for each media channel
- **Budget Optimization**: Guidance for media spend allocation
- **Scenario Planning**: Revenue impact of ±10% spend changes
- **Promotional Effects**: Impact analysis with confidence intervals

## 🔧 Technical Requirements

### Dependencies (from `requirements.txt`):
```txt
pandas              # Data manipulation
numpy               # Numerical computing
scikit-learn        # Machine learning
statsmodels         # Statistical modeling
linearmodels        # IV/2SLS regression
xgboost             # Gradient boosting
matplotlib          # Plotting
seaborn             # Statistical visualization
shap                # Model explainability
jupyter             # Interactive notebooks
notebook            # Jupyter notebook server
joblib              # Model persistence
```

### System Requirements:
- **RAM**: 4GB+ recommended for data processing
- **Storage**: 50MB for project + generated outputs
- **Python**: 3.8+ with pip package manager

## 📈 Data Requirements

### Expected Input Format:
Your dataset should be a weekly time series with:

```csv
week,revenue,facebook_spend,google_spend,tiktok_spend,instagram_spend,snapchat_spend,promotions
2023-09-17,83124.16,6030.80,3130.14,2993.22,1841.08,2204.72,0
2023-09-24,373.02,5241.44,2704.00,0.00,0.00,0.00,0
...
```

**Required Columns:**
- `week`: Date column (weekly frequency)
- `revenue`: Target variable (continuous)
- Media channels: `*_spend` columns (e.g., `facebook_spend`, `google_spend`)
- `promotions`: Promotional activity (binary or count)

**Optional Enhancements:**
- Additional control variables (seasonality, external factors)
- Customer metrics (emails sent, social followers)
- Pricing information (`average_price`)

### Sample Data Generation:
If working with sensitive data, the notebooks automatically generate realistic sample data with:
- Correlated media spend patterns
- Diminishing returns relationships
- Seasonal trends and promotional effects
- Realistic noise and variation

## 🏆 Model Performance

Recent validation results on sample data:

| Model | Validation RMSE | Validation R² | Key Strengths |
|-------|----------------|---------------|---------------|
| **ElasticNet** | $102,216 | -0.740 | Feature selection, interpretability |
| **XGBoost** | $81,699 | -0.112 | Non-linear patterns, robustness |
| **IV/2SLS** | - | 0.116 | Causal inference, unbiased estimates |

*Note: Negative R² values indicate challenging prediction scenarios typical in MMM with limited sample sizes.*

## 📋 Usage Examples

### Running Individual Components:
```python
# Load and preprocess data
from src.preprocess import preprocess_data
df = preprocess_data('../data/weekly_data.zip')

# Engineer features
from src.features import build_features
df_featured = build_features(df)

# Fit models
from src.models import fit_elasticnet_cv, fit_xgboost_cv
elasticnet_model = fit_elasticnet_cv(X_train, y_train)
xgb_model = fit_xgboost_cv(X_train, y_train)

# Diagnostics
from src.diagnostics import plot_residuals, compute_vif
plot_residuals(y_true, y_pred)
vif_results = compute_vif(X_train)
```

### Customizing Analysis:
- **Media Channels**: Modify `media_channels` list in notebooks
- **Time Splits**: Adjust `train_pct`, `val_pct`, `test_pct` ratios
- **Feature Engineering**: Add custom transformations in `features.py`
- **Model Parameters**: Update hyperparameters in modeling cells

## 🚨 Troubleshooting

### Common Issues:

**Import Errors:**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Data Loading Issues:**
- Verify `weekly_data.zip` exists in `data/` directory
- Check file permissions and zip file integrity
- Review data format matches expected schema

**Memory Issues:**
- Reduce sample size for SHAP analysis
- Use fewer bootstrap samples in sensitivity analysis
- Clear notebook outputs between runs

**Model Performance:**
- Negative R² values are common in MMM with small datasets
- Focus on model rankings and business insights rather than absolute metrics
- Consider data quality and feature engineering improvements

## 📝 Contributing

To extend this analysis:

1. **Add New Features**: Enhance `src/features.py` with custom transformations
2. **Include New Models**: Extend `src/models.py` with additional algorithms
3. **Improve Diagnostics**: Add new validation methods to `src/diagnostics.py`
4. **Enhance Visualizations**: Create additional plots and dashboards

## 📄 License & Contact

**Project**: Marketing Mix Modeling Analysis  
**Author**: Sruthi Jayaram  
**Email**: sruthijayaram987@email.com  

---

*This project demonstrates advanced MMM techniques for marketing attribution and incrementality measurement. The comprehensive approach combines causal inference, predictive modeling, and robust validation for actionable business insights.*
