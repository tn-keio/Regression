# 概要
医学系研究等で使用される方法間比較用回帰式の性能比較のためのスクリプト

# 環境
基本的にはPythonで動作するように作成しています

# 各ファイルの詳細
## 20250922_VerCheck.py
使用環境の取得

## 20250923_TwoDist_Repeated_withHistogram.py
正規分布、非正規分布のシミュレーションデータから、4つの回帰式でのグラフを作成
全体の相関図、ヒストグラム、各回帰式でのシミュレーションデータを順に表示する
なお、Demingの誤差分散比は1.0で計算しているため、厳密にはDeming回帰ではなく、この場合に限り直交回帰 Orthogonal Regression である。

## 20250923_TwoDist_Repeated_withHistogram_unequal-V.py
”20250923_TwoDist_Repeated_withHistogram.py”と同じだが、誤差が等分散ではないデータを作成する
Deming用の誤差分散比は0.9で設定
