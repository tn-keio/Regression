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
OUTLIER_RATIO = 0.05  # Percentage of outliers (5%)
OUTLIER_MAGNITUDE = 10 # Magnitude of outliers
ERROR_STD = 0.5       # Standard deviation of the data noise

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

def generate_dataset(n_total, dist_type='lognormal', outlier_ratio=0.05, outlier_magnitude=10, error_std=0.5):
    """
    Generates a dataset with a Y=X relationship.
    """
    np.random.seed(42) # for reproducibility

    if dist_type == 'lognormal':
        x = np.random.lognormal(mean=2, sigma=0.5, size=n_total)
    else:
        raise ValueError("This script is designed for 'lognormal' distribution.")

    error = np.random.normal(loc=0, scale=error_std, size=n_total)
    y = x + error

    n_outliers = int(n_total * outlier_ratio)
    outlier_indices = np.random.choice(n_total, n_outliers, replace=False)
    outlier_shift = (np.random.rand(n_outliers) - 0.5) * 2 * outlier_magnitude
    y[outlier_indices] += outlier_shift
    
    return x, y

def main():
    """
    Main execution function to run the simulation.
    """
    print("--- Simulation with Non-Normal (Lognormal) Distribution Dataset ---")
    
    # 1. Generate the full dataset once
    x_total, y_total = generate_dataset(
        N_TOTAL, 
        dist_type='lognormal', 
        outlier_ratio=OUTLIER_RATIO, 
        outlier_magnitude=OUTLIER_MAGNITUDE, 
        error_std=ERROR_STD
    )
    
    # Create lists to store the results of each trial for each method
    results = {
        'ols': [],
        'gmr': [],
        'ortho': [],
        'pb': []
    }

    # Determine the overall range for plotting
    x_min, x_max = np.percentile(x_total, [1, 99])
    x_fit = np.linspace(x_min, x_max, 100)

    # 2. Repeat the sampling and regression process 100 times
    print(f"Running {N_TRIALS} trials...")
    for i in range(N_TRIALS):
        # Randomly sample 100 data points without replacement
        sample_indices = np.random.choice(N_TOTAL, N_SAMPLE, replace=False)
        x_sample = x_total[sample_indices]
        y_sample = y_total[sample_indices]

        # --- Perform all regression analyses for the current sample ---
        
        # OLS
        ols_model = LinearRegression().fit(x_sample.reshape(-1, 1), y_sample)
        ols_slope, ols_intercept = ols_model.coef_[0], ols_model.intercept_
        results['ols'].append(ols_slope * x_fit + ols_intercept)
        
        # GMR
        s_x, s_y = np.std(x_sample), np.std(y_sample)
        r = np.corrcoef(x_sample, y_sample)[0, 1]
        gmr_slope = (s_y / s_x) * np.sign(r)
        gmr_intercept = np.mean(y_sample) - gmr_slope * np.mean(x_sample)
        results['gmr'].append(gmr_slope * x_fit + gmr_intercept)
        
        # Orthogonal/Deming
        linear_model = Model(lambda p, x: p[0] * x + p[1])
        data = Data(x_sample, y_sample)
        odr_run = ODR(data, linear_model, beta0=[1.0, 0.0]).run()
        ortho_slope, ortho_intercept = odr_run.beta
        results['ortho'].append(ortho_slope * x_fit + ortho_intercept)

        # Passing-Bablok
        pb_res = passing_bablok_manual(x_sample, y_sample)
        pb_slope, pb_intercept = pb_res['slope'], pb_res['intercept']
        results['pb'].append(pb_slope * x_fit + pb_intercept)
    
    print("Plotting results...")

    # 3. Plot the results for each method in a separate figure
    
    # Plot titles and regression method names mapping
    plot_titles = {
        'ols': 'Ordinary Least Squares',
        'gmr': 'Geometric Mean Regression',
        'ortho': 'Orthogonal / Deming Regression',
        'pb': 'Passing-Bablok Regression'
    }

    for method, lines in results.items():
        plt.figure(figsize=(10, 8))
        
        # Plot all 100 regression lines with transparency
        for line in lines:
            plt.plot(x_fit, line, color='blue', alpha=0.1)
        
        # Plot the ideal Y=X line for reference
        plt.plot(x_fit, x_fit, 'r--', linewidth=2, label='Y = X (Ideal Line)')
        
        plt.title(f'{plot_titles[method]} (100 Trials)', fontsize=16)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True)
        plt.legend()
        # Set axis limits to be consistent across all plots
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

if __name__ == '__main__':
    main()