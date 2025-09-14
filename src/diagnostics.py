"""
Diagnostic utilities for MMM analysis.

This module provides diagnostic functions for model validation, residual analysis,
multicollinearity detection, and instrumental variable assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
import warnings

# Optional imports for advanced diagnostics
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.tsa.stattools import acf
    from statsmodels.graphics.tsaplots import plot_acf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Some diagnostic functions will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def plot_residuals(y_true: Union[pd.Series, np.ndarray], 
                  y_pred: Union[pd.Series, np.ndarray],
                  title: str = "Residual Diagnostics",
                  figsize: Tuple[int, int] = (15, 10),
                  save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive residual diagnostic plots and compute statistics.
    
    Generates histogram, Q-Q plot, ACF of residuals, and computes Durbin-Watson statistic
    to assess model assumptions and residual properties.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Title for the plot
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        
    Returns:
        Dict[str, Any]: Dictionary containing diagnostic statistics and test results
        
    Raises:
        ValueError: If input arrays have different lengths
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Input arrays must have same length: {len(y_true)} vs {len(y_pred)}")
    
    # Calculate residuals
    residuals = y_true - y_pred
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    logger.info(f"Computing residual diagnostics for {len(residuals)} observations")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add LOWESS line for trend detection
    try:
        from scipy.signal import savgol_filter
        if len(y_pred) > 10:
            sorted_indices = np.argsort(y_pred)
            smoothed = savgol_filter(residuals[sorted_indices], 
                                   min(51, len(residuals)//3 if len(residuals) > 3 else 3), 
                                   1 if len(residuals) < 4 else 3)
            axes[0, 0].plot(y_pred[sorted_indices], smoothed, color='blue', linewidth=2, alpha=0.8)
    except:
        pass  # Skip smoothing if not available
    
    # 2. Histogram of residuals
    axes[0, 1].hist(residuals, bins=min(30, len(residuals)//5), alpha=0.7, density=True, color='skyblue')
    
    # Overlay normal distribution
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mu, sigma)
    axes[0, 1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal')
    
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot (Normal)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Scale-Location plot (sqrt of standardized residuals vs fitted)
    sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))
    axes[1, 0].scatter(y_pred, sqrt_std_resid, alpha=0.6, s=30)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. ACF of residuals
    if STATSMODELS_AVAILABLE and len(residuals) > 10:
        try:
            plot_acf(residuals, ax=axes[1, 1], lags=min(20, len(residuals)//4), alpha=0.05)
            axes[1, 1].set_title('ACF of Residuals')
        except Exception as e:
            logger.warning(f"Could not plot ACF: {e}")
            axes[1, 1].text(0.5, 0.5, 'ACF plot unavailable', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ACF of Residuals (unavailable)')
    else:
        axes[1, 1].text(0.5, 0.5, 'ACF requires statsmodels', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ACF of Residuals (unavailable)')
    
    # 6. Residuals vs Order (time series plot)
    axes[1, 2].plot(range(len(residuals)), residuals, alpha=0.7, linewidth=1)
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[1, 2].set_xlabel('Observation Order')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].set_title('Residuals vs Order')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual plots saved to: {save_path}")
    
    # Compute diagnostic statistics
    diagnostics = _compute_residual_statistics(residuals, y_true, y_pred)
    
    # Print summary
    _print_residual_summary(diagnostics)
    
    return {
        'diagnostics': diagnostics,
        'residuals': residuals,
        'standardized_residuals': standardized_residuals,
        'figure': fig
    }


def compute_vif(X: Union[pd.DataFrame, np.ndarray], 
               feature_names: Optional[List[str]] = None,
               threshold: float = 10.0) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for multicollinearity detection.
    
    VIF measures how much the variance of a coefficient increases due to collinearity.
    Values > 10 typically indicate problematic multicollinearity.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: Feature names. If None, auto-generated for numpy arrays
        threshold: VIF threshold for flagging high multicollinearity
        
    Returns:
        pd.DataFrame: VIF scores for each feature, sorted by VIF value
        
    Raises:
        ImportError: If statsmodels not available
        ValueError: If input has insufficient variation or rank deficiency
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for VIF computation. Install with: pip install statsmodels")
    
    # Convert to DataFrame if needed
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
        feature_names = X_df.columns.tolist()
    
    logger.info(f"Computing VIF for {len(feature_names)} features")
    
    # Check for constant columns
    constant_cols = []
    for col in X_df.columns:
        if X_df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.warning(f"Removing constant columns: {constant_cols}")
        X_df = X_df.drop(columns=constant_cols)
    
    if X_df.shape[1] < 2:
        raise ValueError("Need at least 2 non-constant features for VIF computation")
    
    # Compute VIF for each feature
    vif_data = []
    
    for i, feature in enumerate(X_df.columns):
        try:
            vif_value = variance_inflation_factor(X_df.values, i)
            vif_data.append({
                'feature': feature,
                'vif': vif_value
            })
        except Exception as e:
            logger.warning(f"Could not compute VIF for {feature}: {e}")
            vif_data.append({
                'feature': feature,
                'vif': np.nan
            })
    
    # Create DataFrame and sort by VIF
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('vif', ascending=False)
    
    # Add flags for high VIF
    vif_df['high_vif'] = vif_df['vif'] > threshold
    
    # Print summary
    _print_vif_summary(vif_df, threshold)
    
    logger.info("VIF computation completed")
    
    return vif_df


def first_stage_diagnostic(first_stage_model: Any,
                          instruments: List[str],
                          endogenous: str,
                          significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Extract and analyze first-stage regression diagnostics for IV estimation.
    
    Evaluates instrument strength using F-statistics and other first-stage metrics
    to assess the validity of the instrumental variable approach.
    
    Args:
        first_stage_model: First-stage regression model object (from IV estimation)
        instruments: List of instrument variable names
        endogenous: Name of endogenous variable being instrumented
        significance_level: Significance level for statistical tests
        
    Returns:
        Dict[str, Any]: Dictionary containing first-stage diagnostics and assessments
    """
    logger.info(f"Analyzing first-stage diagnostics for {endogenous}")
    logger.info(f"Instruments: {', '.join(instruments)}")
    
    diagnostics = {
        'endogenous_var': endogenous,
        'instruments': instruments,
        'n_instruments': len(instruments)
    }
    
    try:
        # Extract basic statistics
        if hasattr(first_stage_model, 'rsquared'):
            diagnostics['r_squared'] = first_stage_model.rsquared
        
        if hasattr(first_stage_model, 'nobs'):
            diagnostics['n_observations'] = first_stage_model.nobs
        
        # Extract F-statistic
        f_statistic = None
        f_pvalue = None
        
        if hasattr(first_stage_model, 'f_statistic'):
            f_statistic = first_stage_model.f_statistic
            diagnostics['f_statistic'] = f_statistic
        
        if hasattr(first_stage_model, 'f_pvalue'):
            f_pvalue = first_stage_model.f_pvalue
            diagnostics['f_pvalue'] = f_pvalue
        
        # Alternative: extract from summary tables
        if f_statistic is None and hasattr(first_stage_model, 'summary'):
            try:
                summary_str = str(first_stage_model.summary)
                # Try to parse F-statistic from summary
                import re
                f_match = re.search(r'F-statistic:\s*([0-9.]+)', summary_str)
                if f_match:
                    f_statistic = float(f_match.group(1))
                    diagnostics['f_statistic'] = f_statistic
            except:
                pass
        
        # Instrument strength assessment
        if f_statistic is not None:
            # Rule of thumb: F > 10 indicates strong instruments
            diagnostics['instrument_strength'] = 'strong' if f_statistic > 10 else 'weak'
            diagnostics['staiger_stock_rule'] = f_statistic > 10
            
            # More conservative critical values for multiple instruments
            if len(instruments) == 1:
                critical_value = 16.38  # 10% worst-case bias
            elif len(instruments) == 2:
                critical_value = 19.93
            else:
                critical_value = 22.30  # Conservative for 3+ instruments
            
            diagnostics['critical_value_10pct'] = critical_value
            diagnostics['passes_critical_value'] = f_statistic > critical_value
        
        # Extract individual instrument coefficients if available
        if hasattr(first_stage_model, 'params'):
            instrument_coeffs = {}
            for instrument in instruments:
                if instrument in first_stage_model.params.index:
                    instrument_coeffs[instrument] = {
                        'coeff': first_stage_model.params[instrument],
                        'pvalue': first_stage_model.pvalues[instrument] if hasattr(first_stage_model, 'pvalues') else None
                    }
            diagnostics['instrument_coefficients'] = instrument_coeffs
        
        # Partial R-squared (if computable)
        try:
            if hasattr(first_stage_model, 'rsquared') and hasattr(first_stage_model, 'rsquared_adj'):
                # This would require additional computation for true partial R-squared
                diagnostics['partial_r_squared_approx'] = first_stage_model.rsquared
        except:
            pass
        
    except Exception as e:
        logger.warning(f"Error extracting first-stage diagnostics: {e}")
        diagnostics['error'] = str(e)
    
    # Print diagnostic summary
    _print_first_stage_summary(diagnostics)
    
    return diagnostics


def _compute_residual_statistics(residuals: np.ndarray, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive residual statistics."""
    stats_dict = {
        'n_observations': len(residuals),
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals)
    }
    
    # Normality tests
    if len(residuals) > 7:  # Minimum for Shapiro-Wilk
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for performance
            stats_dict['shapiro_wilk_statistic'] = shapiro_stat
            stats_dict['shapiro_wilk_pvalue'] = shapiro_p
            stats_dict['residuals_normal'] = shapiro_p > 0.05
        except:
            stats_dict['shapiro_wilk_statistic'] = np.nan
            stats_dict['shapiro_wilk_pvalue'] = np.nan
    
    # Durbin-Watson test for autocorrelation
    if STATSMODELS_AVAILABLE:
        try:
            dw_stat = durbin_watson(residuals)
            stats_dict['durbin_watson'] = dw_stat
            stats_dict['autocorrelation_concern'] = dw_stat < 1.5 or dw_stat > 2.5
        except:
            stats_dict['durbin_watson'] = np.nan
    
    # Ljung-Box test for autocorrelation
    if STATSMODELS_AVAILABLE and len(residuals) > 10:
        try:
            lb_stat, lb_p = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=False)
            stats_dict['ljung_box_statistic'] = lb_stat[-1]
            stats_dict['ljung_box_pvalue'] = lb_p[-1]
            stats_dict['serial_correlation'] = lb_p[-1] < 0.05
        except:
            pass
    
    # Homoscedasticity assessment (simple)
    try:
        # Split into low and high fitted value groups
        median_fitted = np.median(y_pred)
        low_group = residuals[y_pred <= median_fitted]
        high_group = residuals[y_pred > median_fitted]
        
        if len(low_group) > 1 and len(high_group) > 1:
            levene_stat, levene_p = stats.levene(low_group, high_group)
            stats_dict['levene_statistic'] = levene_stat
            stats_dict['levene_pvalue'] = levene_p
            stats_dict['homoscedasticity'] = levene_p > 0.05
    except:
        pass
    
    return stats_dict


def _print_residual_summary(diagnostics: Dict[str, Any]) -> None:
    """Print residual diagnostic summary."""
    print("\n=== RESIDUAL DIAGNOSTICS SUMMARY ===")
    print(f"Observations: {diagnostics.get('n_observations', 'N/A')}")
    print(f"Mean residual: {diagnostics.get('mean_residual', np.nan):.6f}")
    print(f"Std residual: {diagnostics.get('std_residual', np.nan):.4f}")
    
    # Normality
    if 'shapiro_wilk_pvalue' in diagnostics:
        shapiro_p = diagnostics['shapiro_wilk_pvalue']
        print(f"Shapiro-Wilk p-value: {shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else '(Non-normal)'}")
    
    # Autocorrelation
    if 'durbin_watson' in diagnostics:
        dw = diagnostics['durbin_watson']
        if 1.5 <= dw <= 2.5:
            dw_msg = "(No autocorrelation)"
        else:
            dw_msg = "(Possible autocorrelation)"
        print(f"Durbin-Watson: {dw:.3f} {dw_msg}")
    
    # Homoscedasticity
    if 'levene_pvalue' in diagnostics:
        levene_p = diagnostics['levene_pvalue']
        print(f"Levene test p-value: {levene_p:.4f} {'(Homoscedastic)' if levene_p > 0.05 else '(Heteroscedastic)'}")


def _print_vif_summary(vif_df: pd.DataFrame, threshold: float) -> None:
    """Print VIF summary."""
    print("\n=== VARIANCE INFLATION FACTOR (VIF) SUMMARY ===")
    print(f"VIF Threshold: {threshold}")
    
    high_vif = vif_df[vif_df['high_vif'] == True]
    if len(high_vif) > 0:
        print(f"\n⚠ High VIF features ({len(high_vif)}):")
        for _, row in high_vif.iterrows():
            print(f"  {row['feature']}: {row['vif']:.2f}")
    else:
        print("\n✓ No high VIF features detected")
    
    print(f"\nTop 5 VIF scores:")
    for _, row in vif_df.head().iterrows():
        print(f"  {row['feature']}: {row['vif']:.2f}")


def _print_first_stage_summary(diagnostics: Dict[str, Any]) -> None:
    """Print first-stage diagnostic summary."""
    print("\n=== FIRST-STAGE DIAGNOSTICS ===")
    print(f"Endogenous variable: {diagnostics.get('endogenous_var', 'N/A')}")
    print(f"Number of instruments: {diagnostics.get('n_instruments', 'N/A')}")
    
    if 'f_statistic' in diagnostics:
        f_stat = diagnostics['f_statistic']
        print(f"F-statistic: {f_stat:.3f}")
        
        strength = diagnostics.get('instrument_strength', 'unknown')
        print(f"Instrument strength: {strength}")
        
        if 'critical_value_10pct' in diagnostics:
            critical = diagnostics['critical_value_10pct']
            passes = diagnostics['passes_critical_value']
            print(f"Critical value (10% bias): {critical:.2f} {'✓' if passes else '✗'}")
    
    if 'r_squared' in diagnostics:
        print(f"First-stage R²: {diagnostics['r_squared']:.4f}")
    
    # Print individual instrument significance
    if 'instrument_coefficients' in diagnostics:
        print("\nInstrument coefficients:")
        for instr, stats in diagnostics['instrument_coefficients'].items():
            pval = stats.get('pvalue', np.nan)
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"  {instr}: {stats['coeff']:.4f} {sig}")


if __name__ == "__main__":
    # Example usage and testing
    print("=== DIAGNOSTICS.PY MODULE TEST ===")
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 200
    
    # Create features with some multicollinearity
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = 0.8 * X1 + 0.6 * X2 + 0.2 * np.random.normal(0, 1, n_samples)  # Correlated
    X4 = np.random.normal(0, 1, n_samples)
    
    X = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2, 
        'feature_3': X3,  # Should have high VIF
        'feature_4': X4
    })
    
    # Create target with heteroscedasticity and some autocorrelation
    time_trend = np.arange(n_samples) * 0.01
    error = np.random.normal(0, 1 + 0.5 * np.abs(X1), n_samples)  # Heteroscedastic
    y_true = 2 + 1.5*X1 + 1.0*X2 + 0.5*X3 + 0.3*X4 + time_trend + error
    
    # Create predictions with some bias
    y_pred = 1.8 + 1.4*X1 + 1.1*X2 + 0.4*X3 + 0.35*X4 + time_trend + 0.1*np.random.normal(0, 1, n_samples)
    
    print(f"Created test data: {n_samples} samples, {X.shape[1]} features")
    
    # Test 1: Residual diagnostics
    print("\n=== TEST 1: Residual Diagnostics ===")
    try:
        residual_results = plot_residuals(y_true, y_pred, title="Test Residual Analysis")
        print("✓ Residual diagnostics completed successfully")
        plt.show()
    except Exception as e:
        print(f"✗ Residual diagnostics failed: {e}")
    
    # Test 2: VIF computation
    print("\n=== TEST 2: VIF Computation ===")
    if STATSMODELS_AVAILABLE:
        try:
            vif_results = compute_vif(X, threshold=5.0)
            print("✓ VIF computation completed successfully")
            print(f"VIF results:\n{vif_results}")
        except Exception as e:
            print(f"✗ VIF computation failed: {e}")
    else:
        print("✗ VIF computation skipped (statsmodels not available)")
    
    # Test 3: Mock first-stage diagnostics
    print("\n=== TEST 3: First-Stage Diagnostics ===")
    
    class MockFirstStageModel:
        def __init__(self):
            self.f_statistic = 15.6
            self.f_pvalue = 0.001
            self.rsquared = 0.45
            self.nobs = n_samples
            self.params = pd.Series({'instrument_1': 0.8, 'instrument_2': 0.6})
            self.pvalues = pd.Series({'instrument_1': 0.01, 'instrument_2': 0.03})
    
    try:
        mock_model = MockFirstStageModel()
        first_stage_results = first_stage_diagnostic(
            mock_model, 
            instruments=['instrument_1', 'instrument_2'],
            endogenous='endogenous_var'
        )
        print("✓ First-stage diagnostics completed successfully")
    except Exception as e:
        print(f"✗ First-stage diagnostics failed: {e}")
    
    print("\n=== MODULE TEST COMPLETE ===")
