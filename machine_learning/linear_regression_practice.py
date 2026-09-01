"""
Modular Heart Disease ML Exercise - Linear, Ridge, Lasso Regression with Polynomial Features

This module provides separate functions for each component to enable future skill extraction.
Each function is designed to be independently callable and testable.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_heart_disease_data(filepath: str = "data/Heart_Disease_Prediction.csv") -> pd.DataFrame:
    """
    Load and preprocess the heart disease dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        Preprocessed DataFrame with 'Heart Disease' mapped to binary (1/0)
    """
    df = pd.read_csv(filepath)
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return df


def prepare_features_target(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'Max HR'
) -> tuple:
    """
    Extract features and target from the dataset.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names (default: ['BP'])
        target_col: Target column name (default: 'Max HR')

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if feature_cols is None:
        feature_cols = ['BP']

    features = df[feature_cols]
    target = df[target_col]
    return features, target


# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets.

    Args:
        features: Feature DataFrame
        target: Target Series
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)


# =============================================================================
# POLYNOMIAL FEATURES
# =============================================================================

def create_polynomial_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    degree: int = 2,
    include_bias: bool = False
) -> tuple:
    """
    Create polynomial features for train and test sets.

    Args:
        X_train: Training features
        X_test: Test features
        degree: Polynomial degree
        include_bias: Whether to include bias column

    Returns:
        Tuple of (X_train_poly, X_test_poly, poly_features_transformer)
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)  # Use transform, not fit_transform
    return X_train_poly, X_test_poly, poly_features


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_linear_regression(
    X_train: np.ndarray,
    y_train: pd.Series
) -> LinearRegression:
    """
    Train a Linear Regression model.

    Args:
        X_train: Training features (can be polynomial)
        y_train: Training target

    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    alpha: float = 1.0
) -> Ridge:
    """
    Train a Ridge Regression model.

    Args:
        X_train: Training features (can be polynomial)
        y_train: Training target
        alpha: Regularization strength

    Returns:
        Trained Ridge model
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_lasso_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    alpha: float = 1.0
) -> Lasso:
    """
    Train a Lasso Regression model.

    Args:
        X_train: Training features (can be polynomial)
        y_train: Training target
        alpha: Regularization strength

    Returns:
        Trained Lasso model
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: pd.Series
) -> dict:
    """
    Evaluate a regression model and return metrics.

    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with mse, r2, coefficients, intercept
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Handle coefficients (different shapes for different models)
    if hasattr(model, 'coef_'):
        coef = model.coef_
    else:
        coef = None

    return {
        'mse': mse,
        'r2': r2,
        'coefficients': coef,
        'intercept': model.intercept_,
        'predictions': y_pred
    }


def print_model_results(model_name: str, results: dict) -> None:
    """
    Print formatted model evaluation results.

    Args:
        model_name: Name of the model
        results: Dictionary from evaluate_model()
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Results")
    print(f"{'='*50}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"R² Score: {results['r2']:.4f}")
    if results['coefficients'] is not None:
        print(f"Coefficients: {results['coefficients']}")
    print(f"Intercept: {results['intercept']:.4f}")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_regression_results(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    x_label: str = "BP",
    y_label: str = "Max HR",
    show: bool = True
) -> plt.Figure:
    """
    Plot regression results with training data, test data, and predictions.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        y_pred: Model predictions on test set
        model_name: Name for the plot title
        x_label: X-axis label
        y_label: Y-axis label
        show: Whether to display the plot

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(X_test, y_test, color="blue", label="Test Data", alpha=0.6)
    ax.scatter(X_train, y_train, color="green", label="Training Data", alpha=0.6)

    # Sort for clean line plot
    sort_idx = np.argsort(X_test.values.flatten())
    ax.plot(X_test.iloc[sort_idx], y_pred[sort_idx], color="red", label="Predictions", linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{model_name} - {x_label} vs {y_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


def plot_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: dict,
    x_label: str = "BP",
    y_label: str = "Max HR",
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of multiple models on the same graph.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        predictions: Dict of {model_name: y_pred_array}
        x_label: X-axis label
        y_label: Y-axis label
        show: Whether to display the plot

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'Linear': 'red',
        'Ridge': 'orange',
        'Lasso': 'yellow',
        'Polynomial': 'purple'
    }

    ax.scatter(X_test, y_test, color="blue", label="Test Data", alpha=0.5, s=30)
    ax.scatter(X_train, y_train, color="green", label="Training Data", alpha=0.5, s=30)

    # Sort for clean line plots
    sort_idx = np.argsort(X_test.values.flatten())
    X_test_sorted = X_test.iloc[sort_idx]

    for model_name, y_pred in predictions.items():
        color = colors.get(model_name, 'black')
        ax.plot(
            X_test_sorted,
            y_pred[sort_idx],
            color=color,
            label=model_name,
            linewidth=2
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Model Comparison - {x_label} vs {y_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return fig


# =============================================================================
# HIGH-LEVEL PIPELINE FUNCTIONS
# =============================================================================

def run_linear_regression_pipeline(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'Max HR',
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Run complete linear regression pipeline.

    Returns:
        Dictionary with model, metrics, and data splits
    """
    features, target = prepare_features_target(df, feature_cols, target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size, random_state)

    model = train_linear_regression(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    return {
        'model': model,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def run_polynomial_regression_pipeline(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'Max HR',
    degree: int = 2,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Run complete polynomial regression pipeline.

    Returns:
        Dictionary with model, metrics, data splits, and polynomial transformer
    """
    features, target = prepare_features_target(df, feature_cols, target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size, random_state)

    X_train_poly, X_test_poly, poly_transformer = create_polynomial_features(
        X_train, X_test, degree
    )

    model = train_linear_regression(X_train_poly, y_train)
    results = evaluate_model(model, X_test_poly, y_test)

    return {
        'model': model,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_poly': X_train_poly,
        'X_test_poly': X_test_poly,
        'y_train': y_train,
        'y_test': y_test,
        'poly_transformer': poly_transformer
    }


def run_ridge_regression_pipeline(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'Max HR',
    degree: int = 2,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Run complete Ridge regression pipeline with polynomial features.

    Returns:
        Dictionary with model, metrics, data splits, and polynomial transformer
    """
    features, target = prepare_features_target(df, feature_cols, target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size, random_state)

    X_train_poly, X_test_poly, poly_transformer = create_polynomial_features(
        X_train, X_test, degree
    )

    model = train_ridge_regression(X_train_poly, y_train, alpha)
    results = evaluate_model(model, X_test_poly, y_test)

    return {
        'model': model,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_poly': X_train_poly,
        'X_test_poly': X_test_poly,
        'y_train': y_train,
        'y_test': y_test,
        'poly_transformer': poly_transformer
    }


def run_lasso_regression_pipeline(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'Max HR',
    degree: int = 2,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Run complete Lasso regression pipeline with polynomial features.

    Returns:
        Dictionary with model, metrics, data splits, and polynomial transformer
    """
    features, target = prepare_features_target(df, feature_cols, target_col)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size, random_state)

    X_train_poly, X_test_poly, poly_transformer = create_polynomial_features(
        X_train, X_test, degree
    )

    model = train_lasso_regression(X_train_poly, y_train, alpha)
    results = evaluate_model(model, X_test_poly, y_test)

    return {
        'model': model,
        'results': results,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_poly': X_train_poly,
        'X_test_poly': X_test_poly,
        'y_train': y_train,
        'y_test': y_test,
        'poly_transformer': poly_transformer
    }


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """Main function to run all regression models and compare them."""
    print("Loading Heart Disease dataset...")
    df = load_heart_disease_data()
    print(f"Dataset shape: {df.shape}")

    # Run Linear Regression (simple, no polynomial)
    print("\n" + "="*60)
    print("RUNNING LINEAR REGRESSION (Simple)")
    print("="*60)
    linear_result = run_linear_regression_pipeline(df)
    print_model_results("Linear Regression", linear_result['results'])
    plot_regression_results(
        linear_result['X_train'], linear_result['y_train'],
        linear_result['X_test'], linear_result['y_test'],
        linear_result['results']['predictions'],
        "Linear Regression"
    )

    # Run Polynomial Regression
    print("\n" + "="*60)
    print("RUNNING POLYNOMIAL REGRESSION (Degree 2)")
    print("="*60)
    poly_result = run_polynomial_regression_pipeline(df)
    print_model_results("Polynomial Regression", poly_result['results'])
    plot_regression_results(
        poly_result['X_train'], poly_result['y_train'],
        poly_result['X_test'], poly_result['y_test'],
        poly_result['results']['predictions'],
        "Polynomial Regression (Degree 2)"
    )

    # Run Ridge Regression
    print("\n" + "="*60)
    print("RUNNING RIDGE REGRESSION (Polynomial Degree 2)")
    print("="*60)
    ridge_result = run_ridge_regression_pipeline(df, alpha=1.0)
    print_model_results("Ridge Regression", ridge_result['results'])
    plot_regression_results(
        ridge_result['X_train'], ridge_result['y_train'],
        ridge_result['X_test'], ridge_result['y_test'],
        ridge_result['results']['predictions'],
        "Ridge Regression"
    )

    # Run Lasso Regression
    print("\n" + "="*60)
    print("RUNNING LASSO REGRESSION (Polynomial Degree 2)")
    print("="*60)
    lasso_result = run_lasso_regression_pipeline(df, alpha=1.0)
    print_model_results("Lasso Regression", lasso_result['results'])
    plot_regression_results(
        lasso_result['X_train'], lasso_result['y_train'],
        lasso_result['X_test'], lasso_result['y_test'],
        lasso_result['results']['predictions'],
        "Lasso Regression"
    )

    # Comparison Plot
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    # Use polynomial test set for comparison (since Ridge/Lasso use polynomial features)
    X_test_poly = poly_result['X_test_poly']

    # Get predictions on the same polynomial test set for all models
    linear_poly_model = train_linear_regression(
        poly_result['X_train_poly'], poly_result['y_train']
    )
    linear_poly_predictions = linear_poly_model.predict(X_test_poly)

    predictions = {
        'Linear': poly_result['results']['predictions'],
        'Ridge': ridge_result['results']['predictions'],
        'Lasso': lasso_result['results']['predictions'],
    }

    plot_model_comparison(
        poly_result['X_train'], poly_result['y_train'],
        poly_result['X_test'], poly_result['y_test'],
        predictions
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    models_summary = {
        'Linear (Simple)': linear_result['results'],
        'Polynomial': poly_result['results'],
        'Ridge': ridge_result['results'],
        'Lasso': lasso_result['results'],
    }

    for name, res in models_summary.items():
        print(f"{name:20s} - MSE: {res['mse']:.4f}, R²: {res['r2']:.4f}")


if __name__ == "__main__":
    main()