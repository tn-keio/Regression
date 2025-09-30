import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, Data
from sklearn.linear_model import LinearRegression

# Matplotlib setting to prevent garbled text for the minus sign.
plt.rcParams['axes.unicode_minus'] = False

# --- Simulation Parameters ---
N_TOTAL = 10000       # Total number of data points in the dataset
N_SAMPLE = 100        # Number of data points to sample
N_TRIALS = 100        # Number of sampling trials to repeat
OUTLIER_MAGNITUDE = 10 # Magnitude of outliers
ERROR_STD = 0.5       # Standard deviation of the data noise

# Error variance ratio for Deming Regression: σ²_y / σ²_x
ERROR_VAR_RATIO = 1.0  # 1.0 means equal variance (Orthogonal Regression)

def passing_bablok_manual(x, y):
    """
    Calculates Passing-Bablok regression using only Numpy.
    """
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope_ij = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope_ij)
    
    slope_median = np.median(slopes)
    intercepts = y - slope_median * x
    intercept_median = np.median(intercepts)
    
    return {'slope': slope_median, 'intercept': intercept_median}

def generate_dataset(n_total, dist_type='normal', outlier_magnitude=10, error_std=0.5):
    """
    Generates a dataset with a Y=X relationship, including outliers in both X and Y.
    """
    np.random.seed(42)

    if dist_type == 'normal':
        x = np.random.normal(loc=10, scale=3, size=n_total)
    elif dist_type == 'lognormal':
        x = np.random.lognormal(mean=2, sigma=0.5, size=n_total)
    else:
        raise ValueError("dist_type must be 'normal' or 'lognormal'.")

    error = np.random.normal(loc=0, scale=error_std, size=n_total)
    y = x.copy() + error

    n_outliers_each = int(n_total * 0.025)
    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)
    
    y_outlier_indices = all_indices[:n_outliers_each]
    x_outlier_indices = all_indices[n_outliers_each : n_outliers_each * 2]
    
    y[y_outlier_indices] += (np.random.rand(n_outliers_each) - 0.5) * 2 * outlier_magnitude
    x[x_outlier_indices] += (np.random.rand(n_outliers_each) - 0.5) * 2 * outlier_magnitude
    
    return x, y

# --- New Function to Plot Initial Data ---
def plot_initial_data(x_total, y_total, dist_name):
    """
    Plots the scatter plot and histograms for the entire dataset.
    """
    print(f"\nPlotting initial distribution for {dist_name} data...")

    # 1. Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_total, y_total, alpha=0.2, s=10) # s=10 for smaller points
    # Add a reference line
    lim_min = min(np.min(x_total), np.min(y_total))
    lim_max = max(np.max(x_total), np.max(y_total))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='Y = X')
    plt.title(f'Scatter Plot of Full Dataset ({dist_name})', fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

    # 2. Histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(x_total, bins=100, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Histogram of X ({dist_name})')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(y_total, bins=100, color='salmon', edgecolor='black')
    axes[1].set_title(f'Histogram of Y ({dist_name})')
    axes[1].set_xlabel('Value')
    
    plt.tight_layout()
    plt.show()


def run_simulation_and_plot(x_total, y_total, dist_name):
    """
    Runs the 100-trial simulation and plots the regression results.
    """
    results = {'ols': [], 'gmr': [], 'deming': [], 'pb': []}
    x_min, x_max = np.percentile(x_total, [1, 99])
    x_fit = np.linspace(x_min, x_max, 100)
    
    # For scipy.odr: wd = 1/σ²_x, we = 1/σ²_y
    # If σ²_y / σ²_x = ERROR_VAR_RATIO
    SIGMA_X_SQUARED = 1.0
    SIGMA_Y_SQUARED = ERROR_VAR_RATIO * SIGMA_X_SQUARED

    print(f"\nRunning {N_TRIALS} regression trials for {dist_name} data...")
    for i in range(N_TRIALS):
        sample_indices = np.random.choice(N_TOTAL, N_SAMPLE, replace=False)
        x_sample, y_sample = x_total[sample_indices], y_total[sample_indices]

        # Regression calculations...
        ols_model = LinearRegression().fit(x_sample.reshape(-1, 1), y_sample)
        ols_slope, ols_intercept = ols_model.coef_[0], ols_model.intercept_
        results['ols'].append(ols_slope * x_fit + ols_intercept)
        
        s_x, s_y = np.std(x_sample), np.std(y_sample)
        r = np.corrcoef(x_sample, y_sample)[0, 1] if s_x > 0 and s_y > 0 else 0
        gmr_slope = (s_y / s_x) * np.sign(r) if s_x > 0 else 0
        gmr_intercept = np.mean(y_sample) - gmr_slope * np.mean(x_sample)
        results['gmr'].append(gmr_slope * x_fit + gmr_intercept)
        
        # Deming Regression with explicit error variance ratio
        # wd = weight for x-direction = 1/σ²_x
        # we = weight for y-direction = 1/σ²_y
        linear_model = Model(lambda p, x: p[0] * x + p[1])
        data_deming = Data(x_sample, y_sample, wd=1/SIGMA_X_SQUARED, we=1/SIGMA_Y_SQUARED)
        odr_run = ODR(data_deming, linear_model, beta0=[1.0, 0.0]).run()
        deming_slope, deming_intercept = odr_run.beta
        results['deming'].append(deming_slope * x_fit + deming_intercept)

        pb_res = passing_bablok_manual(x_sample, y_sample)
        pb_slope, pb_intercept = pb_res['slope'], pb_res['intercept']
        results['pb'].append(pb_slope * x_fit + pb_intercept)
    
    print("Plotting regression results...")
    
    plot_titles = {
        'ols': 'Ordinary Least Squares',
        'gmr': 'Geometric Mean Regression',
        'deming': f'Orthogonal / Deming Regression (λ = {ERROR_VAR_RATIO})',
        'pb': 'Passing-Bablok Regression'
    }

    for method, lines in results.items():
        plt.figure(figsize=(10, 8))
        for line in lines:
            plt.plot(x_fit, line, color='blue', alpha=0.1)
        
        plt.plot(x_fit, x_fit, 'r--', linewidth=2, label='Y = X (Ideal Line)')
        plt.title(f'{plot_titles[method]} on {dist_name} Data (100 Trials)', fontsize=16)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

def main():
    """
    Main execution function to run both simulations.
    """
    # --- Part 1: Simulation with Normal Distribution ---
    print("--- Starting Simulation with Normal Distribution Dataset ---")
    x_total_norm, y_total_norm = generate_dataset(
        N_TOTAL, dist_type='normal', outlier_magnitude=OUTLIER_MAGNITUDE, error_std=ERROR_STD
    )
    # Plot initial data before simulation
    plot_initial_data(x_total_norm, y_total_norm, "Normal")
    # Run regression simulation
    run_simulation_and_plot(x_total_norm, y_total_norm, "Normal")

    # --- Part 2: Simulation with Non-Normal (Lognormal) Distribution ---
    print("\n--- Starting Simulation with Non-Normal (Lognormal) Distribution Dataset ---")
    x_total_lognorm, y_total_lognorm = generate_dataset(
        N_TOTAL, dist_type='lognormal', outlier_magnitude=OUTLIER_MAGNITUDE, error_std=ERROR_STD
    )
    # Plot initial data before simulation
    plot_initial_data(x_total_lognorm, y_total_lognorm, "Non-Normal (Lognormal)")
    # Run regression simulation
    run_simulation_and_plot(x_total_lognorm, y_total_lognorm, "Non-Normal (Lognormal)")

if __name__ == '__main__':
    main()