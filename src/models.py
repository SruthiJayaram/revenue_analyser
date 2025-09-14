"""
Model training module for MMM analysis.

This module provides model wrappers for IV (2SLS), ElasticNet, and XGBoost approaches
with proper cross-validation and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

# Import IV models
try:
    from linearmodels.iv import IV2SLS
    IV_AVAILABLE = True
except ImportError:
    IV_AVAILABLE = False
    warnings.warn("linearmodels not available. IV2SLS functionality will be disabled.")

# Import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not available. XGBoost functionality will be disabled.")

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap not available. SHAP explanation functionality will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)


def fit_iv2sls(df: pd.DataFrame,
               endog: str,
               exog: List[str],
               instruments: List[str],
               config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Fit IV (2SLS) model using linearmodels.
    
    Estimates causal effects using instrumental variables to address endogeneity.
    Prints first-stage diagnostics and IV coefficient with robust standard errors.
    
    Args:
        df: DataFrame containing all variables
        endog: Name of endogenous (dependent) variable
        exog: List of exogenous variable names
        instruments: List of instrumental variable names
        config: Optional configuration dict with keys:
               - 'random_state': int (default 42)
               - 'cov_type': str (default 'robust')
               - 'print_summary': bool (default True)
    
    Returns:
        Fitted IV2SLS model object
        
    Raises:
        ImportError: If linearmodels package not available
        ValueError: If required columns missing or instruments weak
    """
    if not IV_AVAILABLE:
        raise ImportError("linearmodels package required for IV2SLS. Install with: pip install linearmodels")
    
    # Set default config
    default_config = {
        'random_state': 42,
        'cov_type': 'robust',
        'print_summary': True
    }
    config = {**default_config, **(config or {})}
    
    # Set random seed
    np.random.seed(config['random_state'])
    
    # Validate inputs
    missing_cols = []
    all_vars = [endog] + exog + instruments
    for var in all_vars:
        if var not in df.columns:
            missing_cols.append(var)
    
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")
    
    # Prepare data
    y = df[endog]
    X = df[exog]
    Z = df[instruments]
    
    logger.info(f"Fitting IV2SLS model:")
    logger.info(f"  Endogenous: {endog}")
    logger.info(f"  Exogenous: {exog}")
    logger.info(f"  Instruments: {instruments}")
    
    # Fit IV2SLS model
    model = IV2SLS(dependent=y, exog=X, instruments=Z)
    iv_results = model.fit(cov_type=config['cov_type'])
    
    # Extract and print first-stage diagnostics
    _print_first_stage_diagnostics(iv_results)
    
    # Print IV results summary
    if config['print_summary']:
        _print_iv_summary(iv_results, endog, exog, instruments)
    
    logger.info("IV2SLS model fitting completed successfully")
    
    return iv_results


def fit_elasticnet_cv(X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     tscv_splits: int = 5,
                     random_state: int = 42,
                     config: Optional[Dict[str, Any]] = None) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fit ElasticNet with time series cross-validation and hyperparameter tuning.
    
    Uses TimeSeriesSplit to respect temporal order and StandardScaler for preprocessing.
    Automatically tunes l1_ratio and alpha parameters via cross-validation.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        tscv_splits: Number of time series CV splits
        random_state: Random seed for reproducibility
        config: Optional configuration dict with keys:
               - 'l1_ratios': List[float] (default [0.1, 0.5, 0.7, 0.9, 0.95])
               - 'alphas': List[float] (default auto-generated)
               - 'cv_scoring': str (default 'neg_mean_squared_error')
               - 'max_iter': int (default 2000)
    
    Returns:
        Tuple[Pipeline, Dict]: (fitted_pipeline, best_hyperparameters)
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Set default config
    default_config = {
        'l1_ratios': [0.1, 0.5, 0.7, 0.9, 0.95],
        'alphas': None,  # Will use ElasticNetCV default
        'cv_scoring': 'neg_mean_squared_error',
        'max_iter': 2000
    }
    config = {**default_config, **(config or {})}
    
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y = y.values
    
    logger.info(f"Fitting ElasticNet with TimeSeriesSplit (n_splits={tscv_splits})")
    logger.info(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"  L1 ratios: {config['l1_ratios']}")
    
    # Create time series cross-validator
    tscv = TimeSeriesSplit(n_splits=tscv_splits)
    
    # Create ElasticNetCV with time series CV
    elasticnet = ElasticNetCV(
        l1_ratio=config['l1_ratios'],
        alphas=config['alphas'],
        cv=tscv,
        scoring=config['cv_scoring'],
        random_state=random_state,
        max_iter=config['max_iter'],
        selection='random'  # For faster convergence
    )
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', elasticnet)
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Extract best hyperparameters
    best_params = {
        'best_alpha': elasticnet.alpha_,
        'best_l1_ratio': elasticnet.l1_ratio_,
        'n_iter': elasticnet.n_iter_,
        'cv_scores_mean': np.mean(elasticnet.mse_path_, axis=1).min(),
        'n_features_selected': np.sum(elasticnet.coef_ != 0)
    }
    
    # Print results summary
    _print_elasticnet_summary(pipeline, best_params, feature_names)
    
    logger.info("ElasticNet CV fitting completed successfully")
    
    return pipeline, best_params


def rolling_time_series_cv(X: Union[pd.DataFrame, np.ndarray],
                          y: Union[pd.Series, np.ndarray],
                          model_func: callable,
                          initial_train_size: int,
                          horizon: int = 1,
                          step: int = 1) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Perform rolling time series cross-validation.
    
    Splits time series into rolling train/test windows, respecting temporal order.
    Useful for evaluating model performance on out-of-sample forecasts.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_func: Function that takes (X_train, y_train) and returns fitted model
        initial_train_size: Size of initial training window
        horizon: Forecast horizon (number of periods to predict)
        step: Step size between folds
        
    Returns:
        Tuple[List[Dict], Dict]: (fold_metrics, aggregated_metrics)
    """
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    n_samples = len(X)
    fold_metrics = []
    
    logger.info(f"Starting rolling CV: initial_size={initial_train_size}, horizon={horizon}, step={step}")
    
    # Rolling window splits
    for start_idx in range(0, n_samples - initial_train_size - horizon + 1, step):
        train_end = start_idx + initial_train_size
        test_start = train_end
        test_end = min(test_start + horizon, n_samples)
        
        if test_end <= test_start:
            break
            
        # Split data
        X_train = X[start_idx:train_end]
        y_train = y[start_idx:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Fit model and predict
        try:
            model = model_func(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            fold_metrics.append({
                'fold': len(fold_metrics) + 1,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'rmse': rmse,
                'mae': mae
            })
            
        except Exception as e:
            logger.warning(f"Fold {len(fold_metrics) + 1} failed: {e}")
            continue
    
    # Aggregate metrics
    if fold_metrics:
        rmse_scores = [f['rmse'] for f in fold_metrics]
        mae_scores = [f['mae'] for f in fold_metrics]
        
        aggregated = {
            'n_folds': len(fold_metrics),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores)
        }
    else:
        aggregated = {'n_folds': 0, 'rmse_mean': np.nan, 'rmse_std': np.nan, 
                     'mae_mean': np.nan, 'mae_std': np.nan}
    
    logger.info(f"Rolling CV completed: {aggregated['n_folds']} successful folds")
    
    return fold_metrics, aggregated


def fit_xgboost_cv(X: Union[pd.DataFrame, np.ndarray],
                   y: Union[pd.Series, np.ndarray],
                   tscv: TimeSeriesSplit,
                   params: Optional[Dict[str, Any]] = None,
                   num_boost_round: int = 1000,
                   early_stopping_rounds: int = 20,
                   random_state: int = 42) -> Tuple[Any, Dict[str, Any], int]:
    """
    Fit XGBoost with time series cross-validation and early stopping.
    
    Uses provided TimeSeriesSplit folds to perform cross-validation with XGBoost,
    incorporating early stopping to prevent overfitting and find optimal boosting rounds.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target variable (n_samples,)
        tscv: TimeSeriesSplit object for cross-validation
        params: XGBoost parameters dict. If None, uses sensible defaults
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Stop if no improvement for this many rounds
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple[xgb.Booster, Dict[str, Any], int]: (trained_booster, cv_metrics, best_iteration)
        
    Raises:
        ImportError: If xgboost package not available
        ValueError: If invalid parameters provided
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost package required. Install with: pip install xgboost")
    
    # Set default parameters
    default_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'verbosity': 0  # Reduce output noise
    }
    
    if params is None:
        params = default_params
    else:
        # Merge with defaults, user params override
        merged_params = default_params.copy()
        merged_params.update(params)
        params = merged_params
    
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y = y.values
    
    logger.info(f"Fitting XGBoost with TimeSeriesSplit CV")
    logger.info(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    logger.info(f"  Max boosting rounds: {num_boost_round}")
    logger.info(f"  Early stopping: {early_stopping_rounds} rounds")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    
    # Prepare CV folds for XGBoost
    cv_folds = []
    for train_idx, val_idx in tscv.split(X):
        cv_folds.append((train_idx.tolist(), val_idx.tolist()))
    
    logger.info(f"Running {len(cv_folds)}-fold time series cross-validation")
    
    # Run cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        folds=cv_folds,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,  # Reduce output noise
        return_trainpreds=False,
        as_pandas=True,
        seed=random_state
    )
    
    # Find best iteration
    best_iteration = len(cv_results)
    if f'test-{params["eval_metric"]}-mean' in cv_results.columns:
        metric_col = f'test-{params["eval_metric"]}-mean'
        best_iteration = cv_results[metric_col].idxmin() + 1  # XGBoost uses 1-based indexing
    
    logger.info(f"Best iteration: {best_iteration}")
    logger.info(f"Best CV score: {cv_results.iloc[best_iteration-1].to_dict()}")
    
    # Train final model with best iteration
    final_booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_iteration,
        verbose_eval=False
    )
    
    # Extract metrics summary
    cv_metrics = {
        'best_iteration': best_iteration,
        'cv_scores': cv_results.to_dict('list'),
        'best_train_score': cv_results.iloc[best_iteration-1].get(f'train-{params["eval_metric"]}-mean', np.nan),
        'best_test_score': cv_results.iloc[best_iteration-1].get(f'test-{params["eval_metric"]}-mean', np.nan),
        'best_test_std': cv_results.iloc[best_iteration-1].get(f'test-{params["eval_metric"]}-std', np.nan)
    }
    
    # Print summary
    _print_xgboost_summary(params, cv_metrics, feature_names)
    
    logger.info("XGBoost CV fitting completed successfully")
    
    return final_booster, cv_metrics, best_iteration


def explain_with_shap(model: Any, 
                     X: Union[pd.DataFrame, np.ndarray],
                     max_display: int = 20,
                     plot_type: str = 'summary') -> Dict[str, Any]:
    """
    Generate SHAP explanations for tree-based models.
    
    Uses SHAP TreeExplainer to compute feature importance and create summary plots
    for understanding model predictions and feature contributions.
    
    Args:
        model: Trained tree-based model (XGBoost, LightGBM, RandomForest, etc.)
        X: Feature matrix for explanation (typically validation/test set)
        max_display: Maximum number of features to display in plots
        plot_type: Type of SHAP plot ('summary', 'bar', 'waterfall', or 'all')
        
    Returns:
        Dict[str, Any]: Dictionary containing SHAP values and plot objects
        
    Raises:
        ImportError: If shap package not available
        ValueError: If model type not supported by TreeExplainer
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap package required. Install with: pip install shap")
    
    # Convert to numpy array if needed, but preserve feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_array = X
    
    logger.info(f"Computing SHAP explanations for {X_array.shape[0]} samples")
    logger.info(f"Features: {len(feature_names)}")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_array)
        
        # Handle XGBoost output format (sometimes returns list)
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]
        
        logger.info(f"SHAP values computed with shape: {shap_values.shape}")
        
        # Create explanation object for better plotting
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X_array,
                feature_names=feature_names
            )
        else:
            explanation = None
        
        # Generate plots
        plots = {}
        
        if plot_type in ['summary', 'all']:
            try:
                if explanation is not None:
                    plots['summary'] = shap.plots.summary(explanation, max_display=max_display, show=False)
                else:
                    plots['summary'] = shap.summary_plot(shap_values, X_array, 
                                                       feature_names=feature_names,
                                                       max_display=max_display, show=False)
                logger.info("✓ Created SHAP summary plot")
            except Exception as e:
                logger.warning(f"Could not create summary plot: {e}")
        
        if plot_type in ['bar', 'all']:
            try:
                if explanation is not None:
                    plots['bar'] = shap.plots.bar(explanation, max_display=max_display, show=False)
                else:
                    # Calculate mean absolute SHAP values for bar plot
                    mean_shap = np.mean(np.abs(shap_values), axis=0)
                    plots['bar'] = shap.summary_plot(shap_values, X_array,
                                                   feature_names=feature_names,
                                                   plot_type='bar', max_display=max_display, show=False)
                logger.info("✓ Created SHAP bar plot")
            except Exception as e:
                logger.warning(f"Could not create bar plot: {e}")
        
        # Compute feature importance summary
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        results = {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_importance': importance_df,
            'plots': plots,
            'explanation': explanation
        }
        
        # Print feature importance summary
        _print_shap_summary(importance_df, max_display)
        
        logger.info("SHAP explanation completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise ValueError(f"Model type may not be supported by TreeExplainer: {e}")


def _print_xgboost_summary(params: Dict[str, Any], cv_metrics: Dict[str, Any], feature_names: List[str]) -> None:
    """Print XGBoost CV results summary."""
    print("\n=== XGBOOST CV RESULTS ===")
    print(f"Best iteration: {cv_metrics['best_iteration']}")
    print(f"Best CV score ({params['eval_metric']}): {cv_metrics['best_test_score']:.4f} ± {cv_metrics['best_test_std']:.4f}")
    print(f"Training score: {cv_metrics['best_train_score']:.4f}")
    print(f"Features used: {len(feature_names)}")
    
    # Show key parameters
    key_params = ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree']
    print("\nKey parameters:")
    for param in key_params:
        if param in params:
            print(f"  {param}: {params[param]}")


def _print_shap_summary(importance_df: pd.DataFrame, max_display: int) -> None:
    """Print SHAP feature importance summary."""
    print("\n=== SHAP FEATURE IMPORTANCE ===")
    print("Top features by mean absolute SHAP value:")
    
    display_df = importance_df.head(max_display)
    for idx, row in display_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    if len(importance_df) > max_display:
        print(f"  ... and {len(importance_df) - max_display} more features")


def _print_first_stage_diagnostics(iv_results) -> None:
    """Print first-stage regression diagnostics."""
    try:
        first_stage = iv_results.first_stage
        
        print("\n=== FIRST-STAGE DIAGNOSTICS ===")
        
        # Extract F-statistics if available
        if hasattr(first_stage, 'f_statistic'):
            f_stat = first_stage.f_statistic
            print(f"First-stage F-statistic: {f_stat:.3f}")
            
            # Rule of thumb: F > 10 indicates strong instruments
            if f_stat > 10:
                print("✓ Instruments appear strong (F > 10)")
            else:
                print("⚠ Instruments may be weak (F < 10)")
        
        print("First-stage regression summary:")
        print(first_stage.summary.tables[1])  # Coefficient table
        
    except Exception as e:
        logger.warning(f"Could not extract first-stage diagnostics: {e}")


def _print_iv_summary(iv_results, endog: str, exog: List[str], instruments: List[str]) -> None:
    """Print IV estimation results summary."""
    print("\n=== IV (2SLS) ESTIMATION RESULTS ===")
    print(f"Dependent variable: {endog}")
    print(f"Exogenous variables: {', '.join(exog)}")
    print(f"Instruments: {', '.join(instruments)}")
    
    # Print main coefficient table
    print("\nCoefficient estimates (with robust standard errors):")
    print(iv_results.summary.tables[1])
    
    # Print key statistics
    print(f"\nR-squared: {iv_results.rsquared:.4f}")
    print(f"Number of observations: {iv_results.nobs}")


def _print_elasticnet_summary(pipeline: Pipeline, best_params: Dict[str, Any], feature_names: List[str]) -> None:
    """Print ElasticNet CV results summary."""
    print("\n=== ELASTICNET CV RESULTS ===")
    print(f"Best alpha: {best_params['best_alpha']:.6f}")
    print(f"Best L1 ratio: {best_params['best_l1_ratio']:.3f}")
    print(f"Features selected: {best_params['n_features_selected']} / {len(feature_names)}")
    print(f"CV score (neg_MSE): {best_params['cv_scores_mean']:.4f}")
    print(f"Iterations to convergence: {best_params['n_iter']}")
    
    # Show selected features
    elasticnet = pipeline.named_steps['elasticnet']
    selected_features = []
    for i, coef in enumerate(elasticnet.coef_):
        if coef != 0:
            selected_features.append(f"{feature_names[i]}: {coef:.4f}")
    
    print(f"\nSelected features and coefficients:")
    for feature in selected_features[:10]:  # Show top 10
        print(f"  {feature}")
    
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more")


if __name__ == "__main__":
    # Example usage and testing
    print("=== MODELS.PY MODULE TEST ===")
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 100
    
    # Create time series data
    time = np.arange(n_samples)
    trend = 0.1 * time
    seasonality = 2 * np.sin(2 * np.pi * time / 12)
    
    # Create features
    X1 = np.random.normal(0, 1, n_samples) + 0.3 * trend  # Correlated with trend
    X2 = np.random.normal(0, 1, n_samples) + seasonality   # Correlated with seasonality
    X3 = np.random.normal(0, 1, n_samples)                 # Random
    
    # Create instruments (for IV testing)
    Z1 = np.random.normal(0, 1, n_samples) + 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
    Z2 = np.random.normal(0, 1, n_samples) + 0.4 * X2 + np.random.normal(0, 0.5, n_samples)
    
    # Create target with endogeneity
    error = np.random.normal(0, 1, n_samples)
    y = 1.5 + 2.0 * X1 + 1.0 * X2 + 0.5 * X3 + trend + seasonality + error
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'y': y,
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'Z1': Z1,
        'Z2': Z2
    })
    
    print(f"Created test data: {test_df.shape}")
    
    # Test 1: ElasticNet CV
    print("\n=== TEST 1: ElasticNet CV ===")
    X_features = test_df[['X1', 'X2', 'X3']]
    y_target = test_df['y']
    
    try:
        pipeline, best_params = fit_elasticnet_cv(X_features, y_target, tscv_splits=3)
        print("✓ ElasticNet CV completed successfully")
    except Exception as e:
        print(f"✗ ElasticNet CV failed: {e}")
    
    # Test 2: Rolling CV
    print("\n=== TEST 2: Rolling Time Series CV ===")
    
    def simple_model_func(X_train, y_train):
        """Simple linear model for testing rolling CV."""
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    try:
        fold_results, agg_results = rolling_time_series_cv(
            X_features, y_target, 
            simple_model_func,
            initial_train_size=50,
            horizon=5,
            step=10
        )
        print(f"✓ Rolling CV completed: {agg_results['n_folds']} folds")
        print(f"  Average RMSE: {agg_results['rmse_mean']:.3f} ± {agg_results['rmse_std']:.3f}")
    except Exception as e:
        print(f"✗ Rolling CV failed: {e}")
    
    # Test 3: XGBoost CV (if available)
    print("\n=== TEST 3: XGBoost CV ===")
    if XGBOOST_AVAILABLE:
        try:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            xgb_params = {
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8
            }
            
            booster, cv_metrics, best_iter = fit_xgboost_cv(
                X_features, y_target, 
                tscv, 
                params=xgb_params,
                num_boost_round=100,
                early_stopping_rounds=10
            )
            print("✓ XGBoost CV completed successfully")
            
            # Test SHAP explanations (if available)
            if SHAP_AVAILABLE:
                print("\n=== TEST 3b: SHAP Explanations ===")
                try:
                    # Use subset for faster testing
                    X_subset = X_features.iloc[:20] if len(X_features) > 20 else X_features
                    shap_results = explain_with_shap(booster, X_subset, max_display=5)
                    print("✓ SHAP explanations completed successfully")
                except Exception as e:
                    print(f"✗ SHAP explanations failed: {e}")
            else:
                print("✗ SHAP explanations skipped (shap not available)")
                
        except Exception as e:
            print(f"✗ XGBoost CV failed: {e}")
    else:
        print("✗ XGBoost CV skipped (xgboost not available)")
    
    # Test 4: IV2SLS (if available)
    print("\n=== TEST 4: IV2SLS ===")
    if IV_AVAILABLE:
        try:
            iv_model = fit_iv2sls(
                test_df,
                endog='y',
                exog=['X2', 'X3'],
                instruments=['Z1', 'Z2'],
                config={'print_summary': True}
            )
            print("✓ IV2SLS completed successfully")
        except Exception as e:
            print(f"✗ IV2SLS failed: {e}")
    else:
        print("✗ IV2SLS skipped (linearmodels not available)")
    
    print("\n=== MODULE TEST COMPLETE ===")
