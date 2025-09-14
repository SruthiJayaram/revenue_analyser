"""
Feature engineering module for MMM analysis.

This module handles time series feature creation, media transformations (adstock, saturation),
and prepares the final feature matrix for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

# Import from absolute module path or local import
try:
    from .preprocess import adstock, hill_saturation
except ImportError:
    # Fallback for direct execution
    from preprocess import adstock, hill_saturation

# Configure logging
logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, 
                  adstock_params: Optional[Dict[str, float]] = None,
                  saturation_params: Optional[Dict[str, Dict[str, float]]] = None,
                  saturation_method: str = 'hill',
                  media_channels: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build comprehensive feature set for MMM modeling.
    
    Creates time features, applies media transformations (adstock + saturation),
    and prepares target variables for analysis.
    
    Args:
        df: Input dataframe with weekly data (must contain date column)
        adstock_params: Dict of {channel: decay_rate}. Default decay=0.5 for all channels
        saturation_params: Dict of {channel: {param: value}} for saturation. 
                          For 'hill': {alpha: 1.0, k: 1.0}
                          For 'log1p': {} (no params needed)
        saturation_method: 'hill' or 'log1p' for saturation transformation
        media_channels: List of media spend columns. If None, uses default channels
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (enhanced_dataframe, feature_column_names)
        
    Raises:
        ValueError: If required columns are missing or invalid parameters
        KeyError: If expected date column not found
    """
    if media_channels is None:
        media_channels = ['facebook_spend', 'tiktok_spend', 'snapchat_spend', 'google_spend']
    
    if saturation_method not in ['hill', 'log1p']:
        raise ValueError(f"saturation_method must be 'hill' or 'log1p', got '{saturation_method}'")
    
    # Create a copy to avoid modifying original
    df_features = df.copy()
    
    # Step 1: Sort by date and create time features
    df_features = _create_time_features(df_features)
    
    # Step 2: Apply media transformations
    df_features, media_feature_cols = _transform_media_channels(
        df_features, 
        media_channels,
        adstock_params or {},
        saturation_params or {},
        saturation_method
    )
    
    # Step 3: Create target and auxiliary features
    df_features, target_cols = _create_target_features(df_features)
    
    # Step 4: Collect all new feature columns
    time_cols = ['t', 'weekofyear', 'year', 'week_sin', 'week_cos']
    feature_columns = time_cols + media_feature_cols + target_cols
    
    logger.info(f"Feature engineering complete. Created {len(feature_columns)} new features.")
    
    return df_features, feature_columns


def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from date column.
    
    Args:
        df: DataFrame with date column (week, date, etc.)
        
    Returns:
        pd.DataFrame: DataFrame with added time features
        
    Raises:
        KeyError: If no suitable date column found
    """
    # Find date column
    date_col = None
    for col in df.columns:
        if col.lower() in ['week', 'date', 'week_start', 'week_end'] and pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
    
    if date_col is None:
        raise KeyError("No datetime column found. Expected 'week', 'date', 'week_start', or 'week_end'")
    
    # Sort by date
    df_sorted = df.sort_values(date_col).copy()
    
    # Create time index (starting from 1)
    df_sorted['t'] = range(1, len(df_sorted) + 1)
    
    # Extract date components
    df_sorted['weekofyear'] = df_sorted[date_col].dt.isocalendar().week
    df_sorted['year'] = df_sorted[date_col].dt.year
    
    # Create cyclical week features (week of year as sine/cosine)
    week_angle = 2 * np.pi * df_sorted['weekofyear'] / 52
    df_sorted['week_sin'] = np.sin(week_angle)
    df_sorted['week_cos'] = np.cos(week_angle)
    
    logger.info(f"Created time features based on column '{date_col}'")
    
    return df_sorted


def _transform_media_channels(df: pd.DataFrame,
                            media_channels: List[str],
                            adstock_params: Dict[str, float],
                            saturation_params: Dict[str, Dict[str, float]],
                            saturation_method: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply adstock and saturation transformations to media channels.
    
    Args:
        df: DataFrame containing media spend columns
        media_channels: List of channel column names
        adstock_params: Channel-specific decay parameters
        saturation_params: Channel-specific saturation parameters
        saturation_method: 'hill' or 'log1p'
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (enhanced_df, new_column_names)
    """
    df_transformed = df.copy()
    new_columns = []
    
    for channel in media_channels:
        if channel not in df.columns:
            logger.warning(f"Channel '{channel}' not found in data, skipping...")
            continue
        
        # Get adstock parameter (default 0.5)
        decay = adstock_params.get(channel, 0.5)
        
        # Apply adstock transformation
        adstock_col = f"{channel}_adstock"
        df_transformed[adstock_col] = adstock(df_transformed[channel], decay)
        new_columns.append(adstock_col)
        
        # Apply saturation transformation
        if saturation_method == 'hill':
            # Get Hill parameters (default alpha=1.0, k=1.0)
            hill_params = saturation_params.get(channel, {'alpha': 1.0, 'k': 1.0})
            alpha = hill_params.get('alpha', 1.0)
            k = hill_params.get('k', 1.0)
            
            saturated_col = f"{channel}_saturated"
            df_transformed[saturated_col] = hill_saturation(
                df_transformed[adstock_col].values, alpha=alpha, k=k
            )
            
        elif saturation_method == 'log1p':
            saturated_col = f"{channel}_saturated"
            df_transformed[saturated_col] = np.log1p(df_transformed[adstock_col])
        
        new_columns.append(saturated_col)
        
        logger.info(f"Transformed {channel}: adstock(decay={decay}) + {saturation_method} saturation")
    
    return df_transformed, new_columns


def _create_target_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create target variable and promotional features.
    
    Args:
        df: DataFrame containing 'revenue' and 'promotions' columns
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: (enhanced_df, new_column_names)
    """
    df_targets = df.copy()
    new_columns = []
    
    # Create promotional flag
    if 'promotions' in df.columns:
        df_targets['promo_flag'] = (df_targets['promotions'] > 0).astype(int)
        new_columns.append('promo_flag')
        logger.info("Created promo_flag from promotions column")
    else:
        logger.warning("'promotions' column not found, skipping promo_flag creation")
    
    # Create log revenue
    if 'revenue' in df.columns:
        df_targets['log_revenue'] = np.log1p(df_targets['revenue'])
        new_columns.append('log_revenue')
        logger.info("Created log_revenue from revenue column")
    else:
        logger.warning("'revenue' column not found, skipping log_revenue creation")
    
    return df_targets, new_columns


def get_default_adstock_params(channels: List[str], default_decay: float = 0.5) -> Dict[str, float]:
    """
    Generate default adstock parameters for all channels.
    
    Args:
        channels: List of channel names
        default_decay: Default decay rate for all channels
        
    Returns:
        Dict[str, float]: Channel to decay rate mapping
    """
    return {channel: default_decay for channel in channels}


def get_default_saturation_params(channels: List[str], 
                                saturation_method: str = 'hill') -> Dict[str, Dict[str, float]]:
    """
    Generate default saturation parameters for all channels.
    
    Args:
        channels: List of channel names
        saturation_method: 'hill' or 'log1p'
        
    Returns:
        Dict[str, Dict[str, float]]: Channel to parameter mapping
    """
    if saturation_method == 'hill':
        return {channel: {'alpha': 1.0, 'k': 1.0} for channel in channels}
    elif saturation_method == 'log1p':
        return {channel: {} for channel in channels}
    else:
        raise ValueError(f"Unknown saturation method: {saturation_method}")


def validate_feature_data(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, any]:
    """
    Validate the feature-engineered dataset and return summary statistics.
    
    Args:
        df: DataFrame with engineered features
        feature_columns: List of feature column names
        
    Returns:
        Dict: Validation summary with statistics and warnings
    """
    validation_summary = {
        'total_features': len(feature_columns),
        'missing_values': {},
        'infinite_values': {},
        'zero_variance': [],
        'warnings': []
    }
    
    for col in feature_columns:
        if col not in df.columns:
            validation_summary['warnings'].append(f"Feature column '{col}' not found in dataframe")
            continue
        
        # Check missing values
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            validation_summary['missing_values'][col] = missing_count
        
        # Check infinite values
        if df[col].dtype in [np.float64, np.float32]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validation_summary['infinite_values'][col] = inf_count
        
        # Check zero variance
        if df[col].nunique() <= 1:
            validation_summary['zero_variance'].append(col)
    
    return validation_summary


if __name__ == "__main__":
    # Example usage and testing
    print("=== FEATURES.PY MODULE TEST ===")
    
    # Create sample data for testing
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample weekly data
    start_date = datetime(2023, 1, 1)
    weeks = pd.date_range(start_date, periods=26, freq='W')  # 6 months of data
    
    sample_data = {
        'week': weeks,
        'revenue': np.random.lognormal(11, 0.3, 26),  # Log-normal revenue
        'facebook_spend': np.random.exponential(5000, 26),
        'tiktok_spend': np.random.exponential(3000, 26),
        'snapchat_spend': np.random.exponential(2000, 26),
        'google_spend': np.random.exponential(8000, 26),
        'promotions': np.random.poisson(2, 26),  # Promotion intensity
    }
    
    sample_df = pd.DataFrame(sample_data)
    print(f"Created sample data with shape: {sample_df.shape}")
    print(f"Date range: {sample_df['week'].min()} to {sample_df['week'].max()}")
    
    # Test 1: Basic feature engineering
    print("\n=== TEST 1: Basic Feature Engineering ===")
    enhanced_df, feature_cols = build_features(sample_df)
    
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Enhanced columns: {len(enhanced_df.columns)}")
    print(f"New feature columns: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    
    # Test 2: Custom parameters
    print("\n=== TEST 2: Custom Parameters ===")
    custom_adstock = {
        'facebook_spend': 0.3,
        'google_spend': 0.7
    }
    
    custom_saturation = {
        'facebook_spend': {'alpha': 2.0, 'k': 0.5},
        'google_spend': {'alpha': 1.5, 'k': 2.0}
    }
    
    enhanced_df_custom, feature_cols_custom = build_features(
        sample_df,
        adstock_params=custom_adstock,
        saturation_params=custom_saturation,
        saturation_method='hill'
    )
    
    print(f"Custom features created: {len(feature_cols_custom)}")
    
    # Test 3: Log saturation method
    print("\n=== TEST 3: Log1p Saturation ===")
    enhanced_df_log, feature_cols_log = build_features(
        sample_df,
        saturation_method='log1p'
    )
    
    print(f"Log saturation features: {len(feature_cols_log)}")
    
    # Test 4: Validation
    print("\n=== TEST 4: Feature Validation ===")
    validation_results = validate_feature_data(enhanced_df, feature_cols)
    
    print(f"Total features: {validation_results['total_features']}")
    print(f"Missing values: {validation_results['missing_values']}")
    print(f"Zero variance features: {validation_results['zero_variance']}")
    print(f"Warnings: {validation_results['warnings']}")
    
    # Test 5: Show sample of transformed features
    print("\n=== TEST 5: Sample Transformed Features ===")
    display_cols = ['week', 'revenue', 'log_revenue', 'promo_flag', 
                   'facebook_spend', 'facebook_spend_adstock', 'facebook_spend_saturated']
    
    available_cols = [col for col in display_cols if col in enhanced_df.columns]
    print(enhanced_df[available_cols].head().round(3))
    
    print("\n=== MODULE TEST COMPLETE ===")
