import sys
import numpy
import matplotlib
import scipy
import sklearn

# sys.version でPythonのバージョン情報を取得して表示
print(f"Python version: {sys.version}")
print("-" * 30) # 区切り線

# 各ライブラリのバージョンを表示
print(f"numpy version: {numpy.__version__}")
print(f"scipy version: {scipy.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")