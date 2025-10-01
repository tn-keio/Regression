import time
import numpy as np
from scipy.stats import theilslopes

def generate_data(n_samples=10000, slope=1.0, intercept=0.0, noise_level=0.5):
    """
    シミュレーション用のデータを生成します。
    y = slope * x + intercept + noise の関係を持つデータを生成します。

    Args:
        n_samples (int): 生成するデータ点の数。
        slope (float): データの真の傾き。
        intercept (float): データの真の切片。
        noise_level (float): データに加えるばらつき（ノイズ）の標準偏差。

    Returns:
        tuple: x座標の配列とy座標の配列。
    """
    print(f"--- データ生成開始 (サンプル数: {n_samples}) ---")
    # xデータを生成 (0から100の範囲でランダム)
    x = np.random.rand(n_samples) * 100
    # yデータを生成 (y = slope * x + intercept)
    y_true = slope * x + intercept
    # yデータに正規分布に従うノイズを加える
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    print("データ生成完了。\n")
    return x, y

def passing_bablok_classic(x, y):
    """
    従来法（ brute-force / O(n^2) ）でPassing-Bablok回帰を計算します。
    すべての2点間の傾きを計算するため、データ数が多いと非常に時間がかかります。

    Args:
        x (np.ndarray): x座標のデータ。
        y (np.ndarray): y座標のデータ。

    Returns:
        tuple: 推定された傾き, 推定された切片, 計算時間(秒)。
    """
    print("--- 従来法によるPassing-Bablok回帰開始 ---")
    start_time = time.time()
    
    n = len(x)
    slopes = []
    # すべての点のペア (i, j) について傾きを計算
    for i in range(n):
        for j in range(i + 1, n):
            # x_i == x_j の場合は傾きを計算しない
            if x[j] - x[i] != 0:
                slope_ij = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope_ij)
    
    # 傾きの中央値を計算
    estimated_slope = np.median(slopes)
    
    # 切片の候補 (y_i - slope * x_i) の中央値を計算
    intercept_candidates = y - estimated_slope * x
    estimated_intercept = np.median(intercept_candidates)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"従来法による計算完了。")
    return estimated_slope, estimated_intercept, elapsed_time

def passing_bablok_fast(x, y):
    """
    高速な手法（SciPyのtheilslopes）で回帰を計算します。
    Theil-Sen推定器はPassing-Bablokと同様に傾きの中央値に基づいています。
    内部で計算量を削減するアルゴリズム（ePBと同様の高速化）が使われています。

    Args:
        x (np.ndarray): x座標のデータ。
        y (np.ndarray): y座標のデータ。

    Returns:
        tuple: 推定された傾き, 推定された切片, 計算時間(秒)。
    """
    print("--- 高速な手法 (SciPy theilslopes) による回帰開始 ---")
    start_time = time.time()
    
    # theilslopesは傾き、切片、傾きの信頼区間の下限・上限を返す
    estimated_slope, estimated_intercept, _, _ = theilslopes(y, x, 0.95)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"高速な手法による計算完了。")
    return estimated_slope, estimated_intercept, elapsed_time

def format_time(seconds):
    """
    秒を、可読性の高い「X年Y日Z時間...」の形式の文字列に変換します。

    Args:
        seconds (float): 変換する秒数。

    Returns:
        str: フォーマットされた時間文字列。
    """
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365) # 閏年は考慮しない簡易計算

    if years > 0:
        return f"{int(years)}年 {int(days)}日"
    elif days > 0:
        return f"{int(days)}日 {int(hours)}時間"
    elif hours > 0:
        return f"{int(hours)}時間 {int(minutes)}分"
    else:
        return f"{int(minutes)}分 {sec:.1f}秒"


def main():
    """
    メイン処理。データの生成、2つの手法による回帰分析、結果の表示を行います。
    """
    # --- シミュレーション設定 ---
    # 高速法の測定用データ数
    N_SAMPLES_FAST = 10000 
    # 従来法の測定用データ数
    N_SAMPLES_CLASSIC = 2000
    # 推定対象のデータ数
    N_SAMPLES_TARGET = 1_000_000

    # データのばらつきを設定
    NOISE_LEVEL = 2.0
    
    # --- シミュレーション実行 ---
    # 高速法はより多くのデータで測定するため、多めに生成
    x, y = generate_data(n_samples=N_SAMPLES_FAST, noise_level=NOISE_LEVEL)
    
    # 従来法は計算量が多いため、データの一部 (N_SAMPLES_CLASSIC) で実行
    x_subset = x[:N_SAMPLES_CLASSIC]
    y_subset = y[:N_SAMPLES_CLASSIC]
    
    classic_slope, classic_intercept, classic_time = passing_bablok_classic(x_subset, y_subset)
    fast_slope, fast_intercept, fast_time = passing_bablok_fast(x, y)
    
    # --- 推定時間の計算 ---
    # 従来法 (O(n^2)) で100万件処理した場合の推定時間
    estimated_classic_time_sec = classic_time * (N_SAMPLES_TARGET / N_SAMPLES_CLASSIC) ** 2
    formatted_estimated_classic_time = format_time(estimated_classic_time_sec)

    # 高速な手法 (O(n log n)) で100万件処理した場合の推定時間
    # 計算式: 測定時間 * (ターゲット件数 / 測定件数) * log(ターゲット件数) / log(測定件数)
    log_scaling = np.log(N_SAMPLES_TARGET) / np.log(N_SAMPLES_FAST)
    n_scaling = N_SAMPLES_TARGET / N_SAMPLES_FAST
    estimated_fast_time_sec = fast_time * n_scaling * log_scaling
    formatted_estimated_fast_time = format_time(estimated_fast_time_sec)


    # --- 結果表示 ---
    print("\n\n" + "="*50)
    print("      Passing-Bablok回帰 シミュレーション結果")
    print("="*50)
    
    print("\n[従来法]")
    print(f"測定データ数: {N_SAMPLES_CLASSIC:,} 件")
    print(f"  推定された回帰式: y = {classic_slope:.4f}x + {classic_intercept:.4f}")
    print(f"  計算所要時間:      {classic_time:.4f} 秒")
    print(f"  └─ (参考) 全{N_SAMPLES_TARGET:,}件を処理した場合の推定時間: 約 {formatted_estimated_classic_time}")
    
    print("\n[高速な手法 (ePBに相当)]")
    print(f"測定データ数: {N_SAMPLES_FAST:,} 件")
    print(f"  推定された回帰式: y = {fast_slope:.4f}x + {fast_intercept:.4f}")
    print(f"  計算所要時間:      {fast_time:.4f} 秒")
    print(f"  └─ (参考) 全{N_SAMPLES_TARGET:,}件を処理した場合の推定時間: 約 {formatted_estimated_fast_time}")
    
    print("\n" + "="*50)
    print("考察:")
    print("・従来法で100万件のデータを処理するのは非現実的な時間がかかると推定されます。")
    print("・高速な手法は100万件のデータでも、従来法とは比較にならないほど高速に処理可能と推定されます。")
    print("="*50)


if __name__ == "__main__":
    main()

