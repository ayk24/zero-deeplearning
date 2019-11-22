import sys
import os
sys.path.append(os.path.abspath('..'))  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient  # 勾配を求める関数


class TwoLayerNet:

    # コンストラクタ
    # 引数は(入力層のニューロン数, 隠れ層のニューロン数, 出力層のニューロン数)
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        # ガウス分布で input_size * hidden_size の配列生成
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 予測
    def predict(self, x):
        # 重み、バイアスの設定
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # y = (w1*入力値1 + b1) + w2*入力値2(1層目からの出力値) + b2
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1) # 活性化関数（シグモイド関数）を使用して、入力信号を出力信号へ変換
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2) # ソフトマックス関数を利用して、確率を出力

        return y

    # 損失関数(交差エントロピー)を求める
    def loss(self, x, t):
        # 予測値y
        y = self.predict(x)
        # 予測値yと、教師データtの損失関数の値を取得
        return cross_entropy_error(y, t)

    # 認識精度を求める
    def accuracy(self, x, t):
        # 予測値yハット
        y = self.predict(x)

        # 配列の中で最大の要素のインデックスを取得
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        # 認識精度の計算
        # 予測値 = 正解値の合計 / 全データ数
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 勾配を求める
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        # 勾配を保持
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 1層目の重みの勾配
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) # 1層目のバイアスの勾配
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 2層目の重みの勾配
        grads['b2'] = numerical_gradient(loss_W, self.params['b2']) # 2層目のバイアスの勾配

        return grads

    # 勾配を求める（高速版）
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads