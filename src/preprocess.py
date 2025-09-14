"""
Data preprocessing module for MMM analysis.

This module handles data extraction, loading, and basic cleaning for weekly marketing data.
Supports both zip files and folder inputs containing CSV files.
"""

import os
import zipfile
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_data(input_path: Union[str, Path], output_dir: str = "./data/") -> str:
    """
    Extract CSV files from a zip archive or copy from a folder to the output directory.
    
    Args:
        input_path: Path to zip file or folder containing CSV files
        output_dir: Directory to extract/copy CSV files to
        
    Returns:
        str: Path to the output directory containing extracted files
        
    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If input is neither a zip file nor a directory
    """
    input_path = Path(input_path)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if input_path.is_file() and input_path.suffix.lower() == '.zip':
        logger.info(f"Extracting zip file: {input_path}")
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            # Extract only CSV files
            csv_files = [f for f in zip_ref.namelist() if f.lower().endswith('.csv')]
            for csv_file in csv_files:
                zip_ref.extract(csv_file, output_path)
                logger.info(f"Extracted: {csv_file}")
        
    elif input_path.is_dir():
        logger.info(f"Copying CSV files from directory: {input_path}")
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in directory: {input_path}")
        
        for csv_file in csv_files:
            destination = output_path / csv_file.name
            destination.write_bytes(csv_file.read_bytes())
            logger.info(f"Copied: {csv_file.name}")
    
    else:
        raise ValueError(f"Input must be a zip file or directory, got: {input_path}")
    
    return str(output_path)


def detect_main_csv(data_dir: str) -> Optional[str]:
    """
    Detect the main weekly CSV file containing date column ('week' or 'date').
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        str or None: Path to the main CSV file, or None if not found
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files, detecting main weekly data...")
    
    for csv_file in csv_files:
        try:
            # Read first few rows to check for date columns
            sample_df = pd.read_csv(csv_file, nrows=5)
            columns_lower = [col.lower() for col in sample_df.columns]
            
            if 'week' in columns_lower or 'date' in columns_lower:
                logger.info(f"Detected main weekly CSV: {csv_file.name}")
                return str(csv_file)
                
        except Exception as e:
            logger.warning(f"Could not read {csv_file.name}: {e}")
            continue
    
    # If no date column found, return the first CSV as fallback
    if csv_files:
        logger.warning("No date column found, using first CSV file as fallback")
        return str(csv_files[0])
    
    return None


def load_weekly_data(csv_path: str) -> pd.DataFrame:
    """
    Load weekly CSV data with proper date parsing.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe with parsed dates
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Loading CSV file: {csv_path}")
    
    # Try to detect date columns and parse them
    df = pd.read_csv(csv_path)
    
    # Find date columns
    date_columns = []
    for col in df.columns:
        if col.lower() in ['week', 'date', 'week_start', 'week_end']:
            date_columns.append(col)
    
    if date_columns:
        logger.info(f"Parsing date columns: {date_columns}")
        df = pd.read_csv(csv_path, parse_dates=date_columns)
    
    logger.info(f"Successfully loaded dataframe with shape: {df.shape}")
    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print comprehensive data summary including shape, sample, dtypes, and missing values.
    
    Args:
        df: DataFrame to summarize
    """
    logger.info("=== DATA SUMMARY ===")
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_pct
    })
    # Only show columns with missing values
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
    
    if not missing_summary.empty:
        print(missing_summary)
    else:
        print("No missing values found!")
    
    print("\n=== SUMMARY COMPLETE ===")


def save_cleaned_data(df: pd.DataFrame, output_path: str = "data/clean_weekly.csv") -> str:
    """
    Save cleaned dataframe to CSV.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the cleaned CSV
        
    Returns:
        str: Path where file was saved
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to: {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    return str(output_path)


def preprocess_data(input_path: Union[str, Path], 
                   output_dir: str = "./data/",
                   save_cleaned: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Complete preprocessing pipeline: extract, detect, load, summarize, and save.
    
    Args:
        input_path: Path to zip file or folder containing CSV files
        output_dir: Directory to extract files to
        save_cleaned: Whether to save cleaned CSV
        
    Returns:
        Tuple[pd.DataFrame, str]: (loaded_dataframe, path_to_main_csv)
        
    Raises:
        ValueError: If no suitable CSV file is found
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Step 1: Extract data
    extracted_dir = extract_data(input_path, output_dir)
    
    # Step 2: Detect main CSV
    main_csv_path = detect_main_csv(extracted_dir)
    if main_csv_path is None:
        raise ValueError("No suitable CSV file found with date columns")
    
    # Step 3: Load data
    df = load_weekly_data(main_csv_path)
    
    # Step 4: Print summary
    print_data_summary(df)
    
    # Step 5: Save cleaned data
    if save_cleaned:
        cleaned_path = save_cleaned_data(df)
        logger.info(f"Preprocessing complete. Cleaned data saved to: {cleaned_path}")
    
    return df, main_csv_path


def adstock(series: pd.Series, decay: float) -> pd.Series:
    """
    Compute exponential adstock transformation for media spend data.
    
    Adstock models the carryover effect of advertising where current period impact
    includes a decayed effect from previous periods: Ad_t = spend_t + decay * Ad_{t-1}
    
    Args:
        series: Time series of media spend values (must be chronologically ordered)
        decay: Decay parameter in [0,1]. 0 = no carryover, 1 = perfect carryover
        
    Returns:
        pd.Series: Adstocked series with same index as input
        
    Raises:
        ValueError: If decay is not in [0,1] or series is empty
        TypeError: If series is not a pandas Series
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if not 0 <= decay <= 1:
        raise ValueError(f"Decay parameter must be in [0,1], got {decay}")
    
    if len(series) == 0:
        raise ValueError("Input series cannot be empty")
    
    # Initialize adstocked series
    adstocked = pd.Series(index=series.index, dtype=float)
    adstocked.iloc[0] = series.iloc[0]  # First period has no carryover
    
    # Compute adstock using iterative formula
    for i in range(1, len(series)):
        adstocked.iloc[i] = series.iloc[i] + decay * adstocked.iloc[i-1]
    
    return adstocked


def hill_saturation(x: np.ndarray, alpha: float = 1.0, k: float = 1.0) -> np.ndarray:
    """
    Apply Hill saturation transformation to model diminishing returns.
    
    The Hill function models saturation curves commonly used in marketing mix modeling:
    f(x) = x^alpha / (x^alpha + k^alpha)
    
    Args:
        x: Input array (typically media spend)
        alpha: Shape parameter (>0). Higher values = steeper saturation curve
        k: Half-saturation point (>0). Point where f(k) = 0.5
        
    Returns:
        np.ndarray: Saturated values with same shape as input
        
    Raises:
        ValueError: If alpha or k are not positive
        TypeError: If x cannot be converted to numpy array
    """
    try:
        x = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        raise TypeError("Input x must be convertible to numpy array")
    
    if alpha <= 0:
        raise ValueError(f"Alpha parameter must be positive, got {alpha}")
    
    if k <= 0:
        raise ValueError(f"K parameter must be positive, got {k}")
    
    # Handle zero and negative values (set to small positive to avoid division issues)
    x_safe = np.where(x <= 0, 1e-10, x)
    
    # Compute Hill saturation
    x_alpha = np.power(x_safe, alpha)
    k_alpha = np.power(k, alpha)
    
    saturated = x_alpha / (x_alpha + k_alpha)
    
    # Preserve zeros in original input
    result = np.where(x <= 0, 0.0, saturated)
    
    return result


if __name__ == "__main__":
    # Example usage and basic tests
    print("=== PREPROCESS.PY MODULE TEST ===")
    
    # Test with a sample dataframe (simulating weekly data)
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample weekly data
    start_date = datetime(2023, 1, 1)
    weeks = pd.date_range(start_date, periods=52, freq='W')
    
    sample_data = {
        'week': weeks,
        'revenue': np.random.normal(100000, 20000, 52),
        'facebook_spend': np.random.normal(5000, 1000, 52),
        'google_spend': np.random.normal(8000, 1500, 52),
        'promotions': np.random.binomial(1, 0.3, 52)
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Add some missing values for testing
    sample_df.loc[0:2, 'facebook_spend'] = np.nan
    
    print("Sample data created for testing:")
    print_data_summary(sample_df)
    
    # Test save function
    test_output_path = "data/test_sample.csv"
    saved_path = save_cleaned_data(sample_df, test_output_path)
    print(f"\nTest save completed: {saved_path}")
    
    print("\n=== ADSTOCK FUNCTION TESTS ===")
    
    # Test adstock function
    test_spend = pd.Series([100, 0, 50, 200, 0], name='test_spend')
    print(f"Original spend: {test_spend.tolist()}")
    
    # Test different decay rates
    for decay in [0.0, 0.3, 0.6, 0.9]:
        adstocked = adstock(test_spend, decay)
        print(f"Adstock (decay={decay}): {adstocked.round(2).tolist()}")
    
    # Test parameter validation
    try:
        adstock(test_spend, 1.5)  # Should raise error
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n=== HILL SATURATION FUNCTION TESTS ===")
    
    # Test hill saturation function
    test_x = np.array([0, 1, 2, 5, 10, 50, 100])
    print(f"Original values: {test_x}")
    
    # Test different parameters
    params = [
        (1.0, 1.0),  # Default
        (2.0, 1.0),  # Steeper curve
        (1.0, 5.0),  # Higher half-saturation
        (0.5, 2.0)   # Gentler curve
    ]
    
    for alpha, k in params:
        saturated = hill_saturation(test_x, alpha, k)
        print(f"Hill (α={alpha}, k={k}): {saturated.round(3)}")
    
    # Test parameter validation
    try:
        hill_saturation(test_x, alpha=-1)  # Should raise error
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test with zeros and negatives
    test_mixed = np.array([-1, 0, 1, 5])
    saturated_mixed = hill_saturation(test_mixed)
    print(f"Mixed input {test_mixed} -> {saturated_mixed.round(3)}")
    
    print("\n=== MODULE TEST COMPLETE ===")


def generate_sample_mmm_data(start_date='2022-01-01', periods=104) -> pd.DataFrame:
    """
    Generate realistic sample MMM data for testing and demonstrations.
    
    Args:
        start_date: Start date for the time series (default: '2022-01-01')
        periods: Number of weeks to generate (default: 104 = 2 years)
        
    Returns:
        pd.DataFrame: Sample data with realistic MMM structure and relationships
    """
    np.random.seed(42)  # For reproducibility
    
    # Create date range (weekly data)
    dates = pd.date_range(start=start_date, periods=periods, freq='W')
    
    # Generate media spend data with realistic patterns
    google_base = 15000
    facebook_base = 8000
    tiktok_base = 5000
    snapchat_base = 3000
    
    # Add seasonal patterns and noise
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(periods) / 52)
    
    data = {
        'date': dates,
        'google_spend': np.maximum(0, google_base * seasonal_factor + 
                                 np.random.normal(0, 3000, periods)),
        'facebook_spend': np.maximum(0, facebook_base * seasonal_factor + 
                                   np.random.normal(0, 2000, periods)),
        'tiktok_spend': np.maximum(0, tiktok_base + 
                                 np.random.normal(0, 1500, periods)),
        'snapchat_spend': np.maximum(0, snapchat_base + 
                                   np.random.normal(0, 1000, periods)),
        'promotions': np.random.binomial(1, 0.15, periods),
        'competitor_index': 1 + 0.2 * np.random.normal(0, 1, periods),
    }
    
    # Generate revenue with realistic MMM relationships
    base_revenue = 45000
    
    # Create DataFrame first so we can access Series
    df_temp = pd.DataFrame(data)
    
    # Apply adstock and saturation to media channels
    google_adstocked = adstock(df_temp['google_spend'], decay=0.5)
    facebook_adstocked = adstock(df_temp['facebook_spend'], decay=0.3)
    tiktok_adstocked = adstock(df_temp['tiktok_spend'], decay=0.2)
    snapchat_adstocked = adstock(df_temp['snapchat_spend'], decay=0.2)
    
    google_saturated = hill_saturation(google_adstocked, alpha=2, k=0.3)
    facebook_saturated = hill_saturation(facebook_adstocked, alpha=1.5, k=0.4)
    tiktok_saturated = hill_saturation(tiktok_adstocked, alpha=1.2, k=0.5)
    snapchat_saturated = hill_saturation(snapchat_adstocked, alpha=1, k=0.6)
    
    # Media contribution to revenue
    google_effect = 2000 * google_saturated
    facebook_effect = 1500 * facebook_saturated  
    tiktok_effect = 1000 * tiktok_saturated
    snapchat_effect = 800 * snapchat_saturated
    
    # Other effects
    promo_effect = df_temp['promotions'] * 8000
    seasonal_effect = 15000 * np.sin(2 * np.pi * np.arange(periods) / 52)
    competitor_effect = -5000 * (df_temp['competitor_index'] - 1)
    noise = np.random.normal(0, 3000, periods)
    
    # Total revenue
    df_temp['revenue'] = (base_revenue + google_effect + facebook_effect + 
                         tiktok_effect + snapchat_effect + promo_effect + 
                         seasonal_effect + competitor_effect + noise)
    
    # Ensure positive revenue
    df_temp['revenue'] = np.maximum(df_temp['revenue'], 1000)
    
    return df_temp
