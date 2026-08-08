import numpy as np

class Kalman2D:
    def __init__(self, process_noise=0.01, measurement_noise=3.0):
        self.initialized = False
        self.state = np.zeros((4,1), dtype=np.float64)
        self.cov = np.eye(4, dtype=np.float64) * 1000.0
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)

    def initialize(self, x, y):
        self.state[:,0] = [x,y,0.0,0.0]
        self.cov = np.eye(4, dtype=np.float64)
        self.initialized = True

    def predict(self, dt=1.0):
        if not self.initialized:
            raise RuntimeError("Kalman filter not initialized")
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        Q = np.eye(4, dtype=np.float64) * self.q
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        return float(self.state[0,0]), float(self.state[1,0])

    def update(self, x, y):
        if not self.initialized:
            self.initialize(x,y)
            return float(x), float(y)
        z = np.array([[x],[y]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.r
        innovation = z - self.H @ self.state
        S = self.H @ self.cov @ self.H.T + R
        K = self.cov @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.cov = (np.eye(4) - K @ self.H) @ self.cov
        return float(self.state[0,0]), float(self.state[1,0])
