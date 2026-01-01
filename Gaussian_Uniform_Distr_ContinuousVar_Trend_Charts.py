import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime, timedelta

# ==========================================
# 1. PARAMETRIC CONFIGURATION (HYDERABAD WINTER)
# ==========================================
# Defining the simulation constants based on climatological research.
# Location: Hyderabad, India
# Season: Winter (January)
# Data Sources: [2, 3, 5]

# Temperature Extremes (Degrees Celsius)
T_MIN = 16.0  # Average Low (Sunrise)
T_MAX = 29.0  # Average High (Afternoon)
T_MEAN = (T_MAX + T_MIN) / 2
AMPLITUDE = (T_MAX - T_MIN) / 2

# Temporal Resolution
# We simulate a "continuous" variable by using 1-minute intervals.
# 24 hours * 60 minutes = 1440 data points.
POINTS_PER_DAY = 1440
TIME_STEP_MIN = 1

# Noise Parameters (The Stochastic Component)
# Standard deviation of the random noise (sensor error + micro-climate).
NOISE_STD = 0.6 

# Seed for Reproducibility
np.random.seed(2024)

# ==========================================
# 2. DATA GENERATION ENGINE
# ==========================================

def generate_diurnal_cycle(n_points, t_min, t_max):
    """
    Generates a synthetic temperature curve using a shifted sinusoidal model.
    Note: Real diurnal cycles are asymmetric (Parton-Logan model), but a 
    shifted cosine is sufficient for distributional demonstration.
    
    The peak is shifted to occur at 15:00 hours (index ~900).
    The trough is shifted to occur at 06:00 hours (index ~360).
    """
    # Time vector (0 to 2pi)
    t_radians = np.linspace(0, 2 * np.pi, n_points)
    
    # Phase Shift Calculation:
    # We want -cos(x) because it starts at min, goes to max.
    # We want the min at 06:00 (approx 1/4 of the day).
    # Normal -cos starts min at t=0. We need to shift it.
    # Shift amount: To move min from 00:00 to 06:00, we shift by -6 hours.
    
    # 24 hours corresponds to 2pi.
    # 6 hours corresponds to pi/2.
    shift = np.pi / 2 + 1.2 # Tuned to align max with 15:00 roughly
    
    deterministic_signal = T_MEAN + AMPLITUDE * np.sin(t_radians - shift)
    
    return deterministic_signal

# Generate the Deterministic Signal
time_vector = np.linspace(0, 24, POINTS_PER_DAY)
temp_deterministic = generate_diurnal_cycle(POINTS_PER_DAY, T_MIN, T_MAX)

# Generate Stochastic Noise (Gaussian)
# This represents the random fluctuations requested for the Gaussian plot.
noise_signal = np.random.normal(loc=0, scale=NOISE_STD, size=POINTS_PER_DAY)

# Generate Control Variable (Uniform)
# This represents a variable like "Humidity Probability" or "Cloud Fraction"
# drawn from a uniform distribution .
uniform_variable = np.random.uniform(low=0, high=1, size=POINTS_PER_DAY)

# Combine for Final "Observed" Temperature
temp_observed = temp_deterministic + noise_signal

# ==========================================
# 3. DATAFRAME CONSTRUCTION (PANDAS)
# ==========================================
# Creating a DatetimeIndex for the simulation date (Jan 15)
start_date = datetime(2024, 1, 15, 0, 0, 0)
time_index = datetime(2024,1,15,0,0,0)

df_weather = pd.DataFrame({
    'timestamp': time_index,
    'hours': time_vector,
    'temp_ideal': temp_deterministic,
    'temp_observed': temp_observed,
    'noise': noise_signal,
    'cloud_prob': uniform_variable
})

# ==========================================
# 4. VISUALIZATION ARCHITECTURE (MATPLOTLIB)
# ==========================================
# We create a layout with:
# - Top: Time Series of Temperature (The "Continuous Variable")
# - Bottom Left: Gaussian Histogram of the Noise
# - Bottom Right: Uniform Histogram of the Control Variable

fig = plt.figure(figsize=(14, 10))
# Using GridSpec for flexible layout (2 rows, 2 columns)
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3)

# --- PLOT A: Diurnal Temperature Cycle ---
ax_ts = fig.add_subplot(gs[0, 0]) # Span both columns
ax_ts.plot(df_weather['timestamp'], df_weather['temp_observed'], 
           color='#d62728', alpha=0.6, linewidth=1, label='Simulated Temp (with Noise)')
ax_ts.plot(df_weather['timestamp'], df_weather['temp_ideal'], 
           color='#000000', linewidth=2, linestyle='--', label='Deterministic Model (Mean)')

# Annotations for Hyderabad Context
ax_ts.set_title('Simulated Diurnal Temperature Cycle: Hyderabad (Jan 15)', fontsize=14, fontweight='bold')
ax_ts.set_ylabel('Temperature (°C)', fontsize=12)
ax_ts.set_xlabel('Local Time', fontsize=12)
ax_ts.grid(True, which='major', linestyle='-', alpha=0.6)
ax_ts.legend(loc='upper right')

# Highlighting the Diurnal Range
ax_ts.axhspan(T_MIN, T_MAX, color='yellow', alpha=0.1, label='Diurnal Range')
text_str = f'Max: {df_weather["temp_observed"].max():.1f}°C\nMin: {df_weather["temp_observed"].min():.1f}°C'
ax_ts.text(0.02, 0.9, text_str, transform=ax_ts.transAxes, 
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# --- PLOT B: Gaussian Distribution (The Noise) ---
ax_gauss = fig.add_subplot(gs[0, 1])
# Plot Histogram
count, bins, ignored = ax_gauss.hist(df_weather['noise'], bins=40, density=True, 
                                     color='#1f77b4', alpha=0.7, edgecolor='black', label='Noise Histogram')

# Plot Theoretical Gaussian PDF
mu, sigma = 0, NOISE_STD
x_gauss = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax_gauss.plot(x_gauss, stats.norm.pdf(x_gauss, mu, sigma), 
              linewidth=2, color='r', label='Theoretical Gaussian PDF')

ax_gauss.set_title('Gaussian Distribution\n(Residual Noise Component)', fontsize=12, fontweight='bold')
ax_gauss.set_xlabel('Temperature Deviation (°C)')
ax_gauss.set_ylabel('Probability Density')
ax_gauss.legend()
ax_gauss.grid(True, alpha=0.3)

# --- PLOT C: Uniform Distribution (The Control Var) ---
ax_unif = fig.add_subplot(gs[1,0])
# Plot Histogram
count, bins, ignored = ax_unif.hist(df_weather['cloud_prob'], bins=40, density=True, 
                                    color='#2ca02c', alpha=0.7, edgecolor='black', label='Uniform Histogram')

# Plot Theoretical Uniform PDF
# For Uniform, PDF = 1 for 0<=x<=1 ==?
x_uniform=np.linspace(0, 100, 1)
ax_unif.plot(x_uniform, 1 , linewidth=3, color='r', label='Theoretical Uniform PDF')

ax_unif.set_title('Uniform Distribution\n(Stochastic Control Variable)', fontsize=12, fontweight='bold')
ax_unif.set_xlabel('Variable Value (0 to 1)')
ax_unif.set_ylabel('Probability Density')
ax_unif.set_ylim(0, 1.5)
ax_unif.legend(loc='lower center')
ax_unif.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()