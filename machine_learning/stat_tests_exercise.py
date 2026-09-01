"""
Heart Disease Prediction - Data Analysis & Modelling
Modular refactor of the original script.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LinearRegression


def load_data(filepath: str) -> pd.DataFrame:
    """Load the heart disease dataset from CSV."""
    return pd.read_csv(filepath)


def inspect_data(df: pd.DataFrame) -> None:
    """Print data info and descriptive statistics."""
    print(df.info())
    print(df.describe())


def visualize_bp_distribution(df: pd.DataFrame) -> None:
    """Plot distribution of Blood Pressure with KDE."""
    sns.histplot(df["BP"], kde=True)
    plt.title("Distribution of Blood Pressure Values in Given Data Set")
    plt.show()


def preprocess_heart_disease(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'Presence'/'Absence' to 1/0 in Heart Disease column."""
    df = df.copy()
    df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return df


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for all numeric columns."""
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heat Map")
    plt.show()


def perform_bp_ttest(df: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float]:
    """
    Perform independent t-test comparing BP between heart disease presence and absence.
    Returns (t_statistic, p_value).
    """
    presence_bp = df[df['Heart Disease'] == 1]['BP']
    absence_bp = df[df['Heart Disease'] == 0]['BP']

    t_stat, p_value = ttest_ind(presence_bp, absence_bp)
    print(f"T-Stat: {t_stat}, P-value: {p_value}")

    if p_value <= alpha:
        print("Reject Null hypothesis - there is significant difference or effect")
    else:
        print("Failed to reject Null hypothesis - No significant difference or effect")

    return t_stat, p_value


def fit_age_cholesterol_regression(df: pd.DataFrame) -> LinearRegression:
    """Fit linear regression: Age -> Cholesterol. Returns fitted model."""
    x = np.array(df['Age']).reshape(-1, 1)
    y = np.array(df['Cholesterol'])
    model = LinearRegression()
    model.fit(x, y)
    print(f"Cholesterol Model - Slope: {model.coef_[0]}, Intercept: {model.intercept_}, R-squared: {model.score(x, y)}")
    return model


def fit_age_bp_regression(df: pd.DataFrame) -> LinearRegression:
    """Fit linear regression: Age -> Blood Pressure. Returns fitted model."""
    x = np.array(df['Age']).reshape(-1, 1)
    z = np.array(df['BP'])
    model = LinearRegression()
    model.fit(x, z)
    print(f"BP Model - Slope: {model.coef_[0]}, Intercept: {model.intercept_}, R-squared: {model.score(x, z)}")
    return model


def plot_regression_results(df: pd.DataFrame, cholesterol_model: LinearRegression, bp_model: LinearRegression) -> None:
    """Plot scatter and regression lines for Age vs Cholesterol and Age vs BP."""
    x = np.array(df['Age']).reshape(-1, 1)
    y = np.array(df['Cholesterol'])
    z = np.array(df['BP'])

    plt.scatter(x, y, color="blue", label="Cholesterol")
    plt.scatter(x, z, color="green", label="Blood Pressure")
    plt.plot(x, cholesterol_model.predict(x), color="red", label="Cholesterol Regression")
    plt.plot(x, bp_model.predict(x), color="orange", label="Blood Pressure Regression")
    plt.xlabel("Age")
    plt.ylabel("Cholesterol / Blood Pressure")
    plt.legend()
    plt.title("Linear Regression - Age vs Cholesterol vs Blood Pressure")
    plt.show()


def perform_chi_squared_test(df: pd.DataFrame, alpha: float = 0.05) -> tuple[float, float, int, np.ndarray]:
    """
    Perform chi-squared test on Age vs BP contingency table.
    Returns (chi2, p_value, dof, expected_frequencies).
    """
    contingency_table = pd.crosstab(df['Age'], df['BP'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected frequencies:\n{expected}")

    if p_value <= alpha:
        print("Reject null hypothesis - variables are dependent")
    else:
        print("Failed to reject Null hypothesis - variables are independent")

    return chi2, p_value, dof, expected


def main():
    """Run the full heart disease analysis pipeline."""
    filepath = "data/Heart_Disease_Prediction.csv"

    # Load and inspect
    df = load_data(filepath)
    inspect_data(df)

    # Visualize BP distribution
    visualize_bp_distribution(df)

    # Preprocess and visualize correlations
    df = preprocess_heart_disease(df)
    plot_correlation_heatmap(df)

    # Hypothesis testing
    perform_bp_ttest(df)

    # Linear regression models
    cholesterol_model = fit_age_cholesterol_regression(df)
    bp_model = fit_age_bp_regression(df)
    plot_regression_results(df, cholesterol_model, bp_model)

    # Chi-squared test
    perform_chi_squared_test(df)


if __name__ == "__main__":
    main()
