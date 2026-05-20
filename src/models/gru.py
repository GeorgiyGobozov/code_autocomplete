# src/models/gru.py
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.losses import sigmoid, softmax, cross_entropy_loss


class GRUCell:
    """GRU ячейка с BPTT"""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        k = 1.0 / np.sqrt(hidden_dim)
        
        self.W_r = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_z = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_h = np.random.uniform(-k, k, (hidden_dim, input_dim))
        
        self.U_r = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_z = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_h = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)
        
        self._collect_params()
    
    def _collect_params(self):
        self.params = [
            self.W_r, self.W_z, self.W_h,
            self.U_r, self.U_z, self.U_h,
            self.b_r, self.b_z, self.b_h
        ]
    
    def forward(self, x_t, h_prev):
        r_t = sigmoid(np.dot(x_t, self.W_r.T) + np.dot(h_prev, self.U_r.T) + self.b_r)
        z_t = sigmoid(np.dot(x_t, self.W_z.T) + np.dot(h_prev, self.U_z.T) + self.b_z)
        h_tilde = np.tanh(np.dot(x_t, self.W_h.T) + np.dot(r_t * h_prev, self.U_h.T) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'r_t': r_t,
            'z_t': z_t,
            'h_tilde': h_tilde,
            'h_t': h_t
        }
        
        return h_t, cache
    
    def backward(self, cache, dh_next):
        """
        Backward pass одного шага
        
        Args:
            cache: dict - сохранённые значения из forward
            dh_next: (B, H) - градиент от следующего шага по h
        
        Returns:
            dh_prev: (B, H) - градиент для предыдущего h
            dx_t: (B, d) - градиент для входа
            grads: dict - градиенты параметров
        """
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        r_t = cache['r_t']
        z_t = cache['z_t']
        h_tilde = cache['h_tilde']
        
        # Градиент для h_tilde
        dh_tilde = dh_next * z_t * (1 - h_tilde**2)
        
        # Градиент для z_t
        dz = dh_next * (h_tilde - h_prev) * z_t * (1 - z_t)
        
        # Градиент для r_t
        dr = np.dot(dh_tilde, self.U_h.T) * (1 - r_t**2) * h_prev
        
        # Градиенты для весов
        grads = {
            'dW_h': np.dot(dh_tilde.T, x_t),
            'dW_z': np.dot(dz.T, x_t),
            'dW_r': np.dot(dr.T, x_t),
            'dU_h': np.dot(dh_tilde.T, r_t * h_prev),
            'dU_z': np.dot(dz.T, h_prev),
            'dU_r': np.dot(dr.T, h_prev),
            'db_h': np.sum(dh_tilde, axis=0),
            'db_z': np.sum(dz, axis=0),
            'db_r': np.sum(dr, axis=0)
        }
        
        # Градиент для предыдущего скрытого состояния
        dh_prev = (dz * (1 - z_t) +
                   np.dot(dr, self.U_r) +
                   np.dot(dz, self.U_z) +
                   np.dot(dh_tilde, self.U_h) * r_t)
        
        # Градиент для входа
        dx_t = (np.dot(dr, self.W_r) +
                np.dot(dz, self.W_z) +
                np.dot(dh_tilde, self.W_h))
        
        return dh_prev, dx_t, grads


class GRUModel:
    """Полная GRU модель"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        k_emb = 1.0 / np.sqrt(embedding_dim)
        self.embedding = np.random.uniform(-k_emb, k_emb, (vocab_size, embedding_dim))
        
        self.gru_cell = GRUCell(embedding_dim, hidden_dim)
        
        k_out = 1.0 / np.sqrt(hidden_dim)
        self.W_y = np.random.uniform(-k_out, k_out, (vocab_size, hidden_dim))
        self.b_y = np.zeros(vocab_size)
        
        self._collect_params()
    
    def _collect_params(self):
        self.params = [self.embedding, self.W_y, self.b_y] + self.gru_cell.params
    
    def forward(self, x, return_sequences=True):
        B, T = x.shape
        h = np.zeros((B, self.hidden_dim))
        
        hs = []
        caches = []
        
        for t in range(T):
            x_t = self.embedding[x[:, t]]
            h, cache = self.gru_cell.forward(x_t, h)
            hs.append(h.copy())
            caches.append(cache)
        
        if return_sequences:
            hs_stack = np.stack(hs, axis=1)
            logits = np.dot(hs_stack, self.W_y.T) + self.b_y
            probs = softmax(logits, axis=-1)
            return probs, hs_stack, caches
        else:
            logits = np.dot(h, self.W_y.T) + self.b_y
            probs = softmax(logits, axis=-1)
            return probs, h, caches
    
    def compute_loss_and_grads(self, x, y):
        B, T = x.shape
        
        probs, hs, caches = self.forward(x, return_sequences=True)
        loss = cross_entropy_loss(probs, y)
        
        # Инициализация градиентов
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        d_embedding = np.zeros_like(self.embedding)
        
        # Градиенты для GRU
        dW_r = np.zeros_like(self.gru_cell.W_r)
        dW_z = np.zeros_like(self.gru_cell.W_z)
        dW_h = np.zeros_like(self.gru_cell.W_h)
        dU_r = np.zeros_like(self.gru_cell.U_r)
        dU_z = np.zeros_like(self.gru_cell.U_z)
        dU_h = np.zeros_like(self.gru_cell.U_h)
        db_r = np.zeros_like(self.gru_cell.b_r)
        db_z = np.zeros_like(self.gru_cell.b_z)
        db_h = np.zeros_like(self.gru_cell.b_h)
        
        dh_next = np.zeros((B, self.hidden_dim))
        
        for t in reversed(range(T)):
            # Градиент выходного слоя
            dlogits = probs[:, t]
            dlogits[np.arange(B), y[:, t]] -= 1
            dlogits /= B
            
            dW_y += np.dot(dlogits.T, hs[:, t])
            db_y += np.sum(dlogits, axis=0)
            
            dh = np.dot(dlogits, self.W_y) + dh_next
            
            cache = caches[t]
            dh_next, dx_t, grads = self.gru_cell.backward(cache, dh)
            
            # Накопление градиентов
            dW_r += grads['dW_r']
            dW_z += grads['dW_z']
            dW_h += grads['dW_h']
            dU_r += grads['dU_r']
            dU_z += grads['dU_z']
            dU_h += grads['dU_h']
            db_r += grads['db_r']
            db_z += grads['db_z']
            db_h += grads['db_h']
            
            # Градиент эмбеддинга
            for b in range(B):
                d_embedding[x[b, t]] += dx_t[b]
        
        grads = [
            d_embedding, dW_y, db_y,
            dW_r, dW_z, dW_h,
            dU_r, dU_z, dU_h,
            db_r, db_z, db_h
        ]
        
        return loss, grads
    
    def predict_next(self, prefix_ids):
        x = np.array([prefix_ids])
        probs, _, _ = self.forward(x, return_sequences=False)
        return probs[0]
    
    def save(self, filepath):
        np.savez(filepath,
                 embedding=self.embedding,
                 W_y=self.W_y,
                 b_y=self.b_y,
                 W_r=self.gru_cell.W_r,
                 W_z=self.gru_cell.W_z,
                 W_h=self.gru_cell.W_h,
                 U_r=self.gru_cell.U_r,
                 U_z=self.gru_cell.U_z,
                 U_h=self.gru_cell.U_h,
                 b_r=self.gru_cell.b_r,
                 b_z=self.gru_cell.b_z,
                 b_h=self.gru_cell.b_h,
                 vocab_size=self.vocab_size,
                 embedding_dim=self.embedding_dim,
                 hidden_dim=self.hidden_dim)
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.embedding = data['embedding']
        self.W_y = data['W_y']
        self.b_y = data['b_y']
        self.gru_cell.W_r = data['W_r']
        self.gru_cell.W_z = data['W_z']
        self.gru_cell.W_h = data['W_h']
        self.gru_cell.U_r = data['U_r']
        self.gru_cell.U_z = data['U_z']
        self.gru_cell.U_h = data['U_h']
        self.gru_cell.b_r = data['b_r']
        self.gru_cell.b_z = data['b_z']
        self.gru_cell.b_h = data['b_h']
        self._collect_params()