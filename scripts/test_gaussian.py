# NumPyとMatplotlibをインポート
import numpy as np
import matplotlib.pyplot as plt


# ガウス関数を定義
def gauss(x_g, a=1, mu=0, sigma=90):
    return a * np.exp(-(x_g - mu) ** 2 / (2 * sigma ** 2))


# Figureを作成
fig = plt.figure(figsize=(8, 6))

# FigureにAxesを追加
ax = fig.add_subplot(111)

# Axesのタイトルを'Gaussian Function'に設定
ax.set_title("Gaussian Function", fontsize=16)

# 目盛線を表示
ax.grid()

# 軸ラベルを設定
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)

# 軸範囲を設定
ax.set_xlim([-90, 90])
ax.set_ylim([0, 1.2])

# -4～8まで0.1刻みの数値の配列
x = np.arange(-90, 90, 0.1)

# グラフに描く関数
f1 = gauss(x)
f2 = gauss(x, mu=2, sigma=2)

# Axesにガウス関数を描画
ax.plot(x, f1, color="red", label="a=1.0, μ=0, σ=1")
ax.plot(x, f2, color="blue", label="a=0.5, μ=2, σ=2")

# 凡例の表示
ax.legend(fontsize=14)

plt.show()
