import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None ## 다수의 RNN 계층을 리스트로 저장하는 용도.

        self.h, self.dh = None, None ## h는 forward() 메서드를 불렀을 때의 마지막 RNN 계층의 은닉 상태를 저장하고 dh는 backward()를 불렀을 때 하나 앞 블록의 은닉 상태의 기울기 저장
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape ## 미니배치 크기를 N, 입력 벡터의 차원 수를 D, xs는 T개 분량의 시계열 데이터를 하나로 모은 것.
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f') ## 출력값을 담을 그릇을 준비하고, 이어서 총 T회 반복되는 for 문 안에서 RNN 계층을 생성해 인스턴스 변수 layer에 추가.
        ## 그 사이에 RNN 계층이 각 시각 t의 은닉 상태 h를 계산하고, 이를 hs에 해당 인덱스의 값으로 설정.

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params) ## *는 리스트의 원소들을 추출해 메서드의 인수로 전달. 즉, self.params에 들어있는 Wx, Wh, b를 추출해 RNN 클래스의 __init__() 메서드에 전달.
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs