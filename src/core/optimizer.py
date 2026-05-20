# src/core/optimizer.py
import numpy as np

class AdamOptimizer:
    """Adam оптимизатор с градиентным клиппингом"""
    def __init__(self, params, lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=5.0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    
    def step(self, grads):
        # Gradient clipping
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if total_norm > self.clip_norm:
            scale = self.clip_norm / (total_norm + 1e-8)
            grads = [g * scale for g in grads]
        
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            p -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)