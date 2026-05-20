# src/models/lstm.py
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.losses import sigmoid, softmax, cross_entropy_loss


class LSTMCell:
    """LSTM ячейка с BPTT"""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        k = 1.0 / np.sqrt(hidden_dim)
        
        self.W_f = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_i = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_c = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_o = np.random.uniform(-k, k, (hidden_dim, input_dim))
        
        self.U_f = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_i = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_c = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_o = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        
        self._collect_params()
    
    def _collect_params(self):
        self.params = [
            self.W_f, self.W_i, self.W_c, self.W_o,
            self.U_f, self.U_i, self.U_c, self.U_o,
            self.b_f, self.b_i, self.b_c, self.b_o
        ]
    
    def forward(self, x_t, h_prev, c_prev):
        f_t = sigmoid(np.dot(x_t, self.W_f.T) + np.dot(h_prev, self.U_f.T) + self.b_f)
        i_t = sigmoid(np.dot(x_t, self.W_i.T) + np.dot(h_prev, self.U_i.T) + self.b_i)
        c_tilde = np.tanh(np.dot(x_t, self.W_c.T) + np.dot(h_prev, self.U_c.T) + self.b_c)
        o_t = sigmoid(np.dot(x_t, self.W_o.T) + np.dot(h_prev, self.U_o.T) + self.b_o)
        
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)
        
        cache = {
            'x_t': x_t, 'h_prev': h_prev, 'c_prev': c_prev,
            'f_t': f_t, 'i_t': i_t, 'c_tilde': c_tilde, 'o_t': o_t,
            'c_t': c_t, 'h_t': h_t
        }
        
        return h_t, c_t, cache
    
    def backward(self, cache, dh_next, dc_next):
        """
        Backward pass одного шага
        
        Args:
            cache: dict - сохранённые значения из forward
            dh_next: (B, H) - градиент от следующего шага по h
            dc_next: (B, H) - градиент от следующего шага по c
        
        Returns:
            dh_prev: (B, H) - градиент для предыдущего h
            dc_prev: (B, H) - градиент для предыдущего c
            dx_t: (B, d) - градиент для входа
            grads: dict - градиенты параметров
        """
        x_t = cache['x_t']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        f_t = cache['f_t']
        i_t = cache['i_t']
        c_tilde = cache['c_tilde']
        o_t = cache['o_t']
        c_t = cache['c_t']
        
        # Градиент для выходного ворота
        do = dh_next * np.tanh(c_t) * o_t * (1 - o_t)
        
        # Градиент для состояния ячейки
        dc = dh_next * o_t * (1 - np.tanh(c_t)**2) + dc_next
        
        # Градиенты для ворот
        df = dc * c_prev * f_t * (1 - f_t)
        di = dc * c_tilde * i_t * (1 - i_t)
        dc_tilde = dc * i_t * (1 - c_tilde**2)
        
        # Градиенты для весов
        grads = {
            'dW_f': np.dot(df.T, x_t),
            'dW_i': np.dot(di.T, x_t),
            'dW_c': np.dot(dc_tilde.T, x_t),
            'dW_o': np.dot(do.T, x_t),
            'dU_f': np.dot(df.T, h_prev),
            'dU_i': np.dot(di.T, h_prev),
            'dU_c': np.dot(dc_tilde.T, h_prev),
            'dU_o': np.dot(do.T, h_prev),
            'db_f': np.sum(df, axis=0),
            'db_i': np.sum(di, axis=0),
            'db_c': np.sum(dc_tilde, axis=0),
            'db_o': np.sum(do, axis=0)
        }
        
        # Градиент для предыдущего скрытого состояния
        dh_prev = (np.dot(df, self.U_f) +
                   np.dot(di, self.U_i) +
                   np.dot(dc_tilde, self.U_c) +
                   np.dot(do, self.U_o))
        
        # Градиент для предыдущего состояния ячейки
        dc_prev = dc * f_t
        
        # Градиент для входа
        dx_t = (np.dot(df, self.W_f) +
                np.dot(di, self.W_i) +
                np.dot(dc_tilde, self.W_c) +
                np.dot(do, self.W_o))
        
        return dh_prev, dc_prev, dx_t, grads


class LSTMModel:
    """Полная LSTM модель"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        k_emb = 1.0 / np.sqrt(embedding_dim)
        self.embedding = np.random.uniform(-k_emb, k_emb, (vocab_size, embedding_dim))
        
        self.lstm_cell = LSTMCell(embedding_dim, hidden_dim)
        
        k_out = 1.0 / np.sqrt(hidden_dim)
        self.W_y = np.random.uniform(-k_out, k_out, (vocab_size, hidden_dim))
        self.b_y = np.zeros(vocab_size)
        
        self._collect_params()
    
    def _collect_params(self):
        self.params = [self.embedding, self.W_y, self.b_y] + self.lstm_cell.params
    
    def forward(self, x, return_sequences=True):
        B, T = x.shape
        h = np.zeros((B, self.hidden_dim))
        c = np.zeros((B, self.hidden_dim))
        
        hs = []
        caches = []
        
        for t in range(T):
            x_t = self.embedding[x[:, t]]
            h, c, cache = self.lstm_cell.forward(x_t, h, c)
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
        
        # Градиенты для LSTM
        dW_f = np.zeros_like(self.lstm_cell.W_f)
        dW_i = np.zeros_like(self.lstm_cell.W_i)
        dW_c = np.zeros_like(self.lstm_cell.W_c)
        dW_o = np.zeros_like(self.lstm_cell.W_o)
        dU_f = np.zeros_like(self.lstm_cell.U_f)
        dU_i = np.zeros_like(self.lstm_cell.U_i)
        dU_c = np.zeros_like(self.lstm_cell.U_c)
        dU_o = np.zeros_like(self.lstm_cell.U_o)
        db_f = np.zeros_like(self.lstm_cell.b_f)
        db_i = np.zeros_like(self.lstm_cell.b_i)
        db_c = np.zeros_like(self.lstm_cell.b_c)
        db_o = np.zeros_like(self.lstm_cell.b_o)
        
        dh_next = np.zeros((B, self.hidden_dim))
        dc_next = np.zeros((B, self.hidden_dim))
        
        for t in reversed(range(T)):
            # Градиент выходного слоя
            dlogits = probs[:, t]
            dlogits[np.arange(B), y[:, t]] -= 1
            dlogits /= B
            
            dW_y += np.dot(dlogits.T, hs[:, t])
            db_y += np.sum(dlogits, axis=0)
            
            dh = np.dot(dlogits, self.W_y) + dh_next
            
            cache = caches[t]
            dh_next, dc_next, dx_t, grads = self.lstm_cell.backward(cache, dh, dc_next)
            
            # Накопление градиентов
            dW_f += grads['dW_f']
            dW_i += grads['dW_i']
            dW_c += grads['dW_c']
            dW_o += grads['dW_o']
            dU_f += grads['dU_f']
            dU_i += grads['dU_i']
            dU_c += grads['dU_c']
            dU_o += grads['dU_o']
            db_f += grads['db_f']
            db_i += grads['db_i']
            db_c += grads['db_c']
            db_o += grads['db_o']
            
            # Градиент эмбеддинга
            for b in range(B):
                d_embedding[x[b, t]] += dx_t[b]
        
        grads = [
            d_embedding, dW_y, db_y,
            dW_f, dW_i, dW_c, dW_o,
            dU_f, dU_i, dU_c, dU_o,
            db_f, db_i, db_c, db_o
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
                 W_f=self.lstm_cell.W_f,
                 W_i=self.lstm_cell.W_i,
                 W_c=self.lstm_cell.W_c,
                 W_o=self.lstm_cell.W_o,
                 U_f=self.lstm_cell.U_f,
                 U_i=self.lstm_cell.U_i,
                 U_c=self.lstm_cell.U_c,
                 U_o=self.lstm_cell.U_o,
                 b_f=self.lstm_cell.b_f,
                 b_i=self.lstm_cell.b_i,
                 b_c=self.lstm_cell.b_c,
                 b_o=self.lstm_cell.b_o,
                 vocab_size=self.vocab_size,
                 embedding_dim=self.embedding_dim,
                 hidden_dim=self.hidden_dim)
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.embedding = data['embedding']
        self.W_y = data['W_y']
        self.b_y = data['b_y']
        self.lstm_cell.W_f = data['W_f']
        self.lstm_cell.W_i = data['W_i']
        self.lstm_cell.W_c = data['W_c']
        self.lstm_cell.W_o = data['W_o']
        self.lstm_cell.U_f = data['U_f']
        self.lstm_cell.U_i = data['U_i']
        self.lstm_cell.U_c = data['U_c']
        self.lstm_cell.U_o = data['U_o']
        self.lstm_cell.b_f = data['b_f']
        self.lstm_cell.b_i = data['b_i']
        self.lstm_cell.b_c = data['b_c']
        self.lstm_cell.b_o = data['b_o']
        self._collect_params()