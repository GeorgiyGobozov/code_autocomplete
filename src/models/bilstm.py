# src/models/bilstm.py
import numpy as np
import sys
import os

# Добавляем путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.losses import sigmoid, softmax, cross_entropy_loss, perplexity, top_k_accuracy


class BiLSTMCell:
    """LSTM ячейка для BiLSTM (двунаправленная)"""
    def __init__(self, input_dim, hidden_dim, direction='forward'):
        """
        Args:
            input_dim: размерность входа
            hidden_dim: размер скрытого состояния
            direction: 'forward' или 'backward'
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.direction = direction
        
        # Инициализация весов
        k = 1.0 / np.sqrt(hidden_dim)
        
        # Веса для входов
        self.W_f = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_i = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_c = np.random.uniform(-k, k, (hidden_dim, input_dim))
        self.W_o = np.random.uniform(-k, k, (hidden_dim, input_dim))
        
        # Веса для скрытых состояний
        self.U_f = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_i = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_c = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        self.U_o = np.random.uniform(-k, k, (hidden_dim, hidden_dim))
        
        # Смещения
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
        """Forward pass одного шага LSTM"""
        # Вычисление ворот
        f_t = sigmoid(np.dot(x_t, self.W_f.T) + np.dot(h_prev, self.U_f.T) + self.b_f)
        i_t = sigmoid(np.dot(x_t, self.W_i.T) + np.dot(h_prev, self.U_i.T) + self.b_i)
        c_tilde = np.tanh(np.dot(x_t, self.W_c.T) + np.dot(h_prev, self.U_c.T) + self.b_c)
        o_t = sigmoid(np.dot(x_t, self.W_o.T) + np.dot(h_prev, self.U_o.T) + self.b_o)
        
        # Обновление состояния
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)
        
        cache = {
            'x_t': x_t, 'h_prev': h_prev, 'c_prev': c_prev,
            'f_t': f_t, 'i_t': i_t, 'c_tilde': c_tilde, 'o_t': o_t,
            'c_t': c_t, 'h_t': h_t
        }
        
        return h_t, c_t, cache
    
    def backward_step(self, cache, dh_next, dc_next):
        """Backward pass одного шага"""
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
        dU_o = np.dot(do.T, h_prev)
        dW_o = np.dot(do.T, x_t)
        db_o = np.sum(do, axis=0)
        
        dU_c = np.dot(dc_tilde.T, h_prev)
        dW_c = np.dot(dc_tilde.T, x_t)
        db_c = np.sum(dc_tilde, axis=0)
        
        dU_i = np.dot(di.T, h_prev)
        dW_i = np.dot(di.T, x_t)
        db_i = np.sum(di, axis=0)
        
        dU_f = np.dot(df.T, h_prev)
        dW_f = np.dot(df.T, x_t)
        db_f = np.sum(df, axis=0)
        
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
        
        grads = {
            'dW_f': dW_f, 'dW_i': dW_i, 'dW_c': dW_c, 'dW_o': dW_o,
            'dU_f': dU_f, 'dU_i': dU_i, 'dU_c': dU_c, 'dU_o': dU_o,
            'db_f': db_f, 'db_i': db_i, 'db_c': db_c, 'db_o': db_o
        }
        
        return dh_prev, dc_prev, dx_t, grads


class BiLSTMModel:
    """
    BiLSTM модель для Fill-in-the-Middle задачи.
    Предсказывает пропущенный символ в середине последовательности.
    """
    def __init__(self, base_vocab_size, embedding_dim, hidden_dim):
        """
        Args:
            base_vocab_size: int - исходный размер словаря (без <MASK>)
            embedding_dim: int - размерность эмбеддингов
            hidden_dim: int - размер скрытого состояния LSTM
        """
        self.base_vocab_size = base_vocab_size
        self.vocab_size = base_vocab_size + 1  # +1 для токена <MASK>
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Токен <MASK> будет иметь индекс base_vocab_size
        self.MASK_TOKEN_ID = base_vocab_size
        
        # Эмбеддинг слой
        k_emb = 1.0 / np.sqrt(embedding_dim)
        self.embedding = np.random.uniform(-k_emb, k_emb, (self.vocab_size, embedding_dim))
        
        # Прямой LSTM
        self.lstm_fwd = BiLSTMCell(embedding_dim, hidden_dim, direction='forward')
        
        # Обратный LSTM
        self.lstm_bwd = BiLSTMCell(embedding_dim, hidden_dim, direction='backward')
        
        # Выходной слой (2H -> vocab_size)
        k_out = 1.0 / np.sqrt(2 * hidden_dim)
        self.W_y = np.random.uniform(-k_out, k_out, (self.vocab_size, 2 * hidden_dim))
        self.b_y = np.zeros(self.vocab_size)
        
        self._collect_params()
    
    def _collect_params(self):
        """Сбор всех параметров модели"""
        self.params = [self.embedding, self.W_y, self.b_y] + \
                      self.lstm_fwd.params + self.lstm_bwd.params
    
    def forward_full_sequence(self, x):
        """
        Полный forward pass для всей последовательности
        
        Args:
            x: (B, T) - индексы токенов (могут содержать <MASK>)
        
        Returns:
            fwd_states: (B, T, H) - состояния прямого LSTM
            bwd_states: (B, T, H) - состояния обратного LSTM
            fwd_caches: list - кэши прямого LSTM
            bwd_caches: list - кэши обратного LSTM
        """
        B, T = x.shape
        
        # ========== Forward LSTM ==========
        h_fwd = np.zeros((B, self.hidden_dim))
        c_fwd = np.zeros((B, self.hidden_dim))
        fwd_states = []
        fwd_caches = []
        
        for t in range(T):
            x_t = self.embedding[x[:, t]]
            h_fwd, c_fwd, cache = self.lstm_fwd.forward(x_t, h_fwd, c_fwd)
            fwd_states.append(h_fwd.copy())
            fwd_caches.append(cache)
        
        fwd_states = np.stack(fwd_states, axis=1)  # (B, T, H)
        
        # ========== Backward LSTM ==========
        h_bwd = np.zeros((B, self.hidden_dim))
        c_bwd = np.zeros((B, self.hidden_dim))
        bwd_states = []
        bwd_caches = []
        
        for t in reversed(range(T)):
            x_t = self.embedding[x[:, t]]
            h_bwd, c_bwd, cache = self.lstm_bwd.forward(x_t, h_bwd, c_bwd)
            bwd_states.insert(0, h_bwd.copy())  # Вставляем в начало для правильного порядка
            bwd_caches.insert(0, cache)
        
        bwd_states = np.stack(bwd_states, axis=1)  # (B, T, H)
        
        return fwd_states, bwd_states, fwd_caches, bwd_caches
    
    def predict_at_position(self, x, mask_position):
        """
        Предсказание для маскированной позиции
        
        Args:
            x: (B, T) - входные индексы
            mask_position: int - позиция маскированного токена
        
        Returns:
            probs: (B, vocab_size) - вероятности для маскированной позиции
            combined: (B, 2H) - конкатенированные состояния
        """
        B, T = x.shape
        
        # Получаем состояния
        fwd_states, bwd_states, _, _ = self.forward_full_sequence(x)
        
        # Конкатенируем состояния на маскированной позиции
        combined = np.concatenate([
            fwd_states[:, mask_position],
            bwd_states[:, mask_position]
        ], axis=1)  # (B, 2H)
        
        # Выходной слой
        logits = np.dot(combined, self.W_y.T) + self.b_y
        probs = softmax(logits, axis=-1)
        
        return probs, combined, fwd_states, bwd_states
    
    def compute_loss_and_grads(self, x, y, mask_position):
        """
        Вычисление loss и градиентов
        
        Args:
            x: (B, T) - входные индексы (с <MASK> на позиции mask_position)
            y: (B,) - целевые токены (истинные символы)
            mask_position: int - позиция маскированного токена
        
        Returns:
            loss: float - значение функции потерь
            grads: list - градиенты для всех параметров
        """
        B, T = x.shape
        
        # Forward pass
        probs, combined, fwd_states, bwd_states = self.predict_at_position(x, mask_position)
        
        # Loss (только для маскированной позиции)
        loss = cross_entropy_loss(probs, y)
        
        # Инициализация градиентов
        dW_y = np.zeros_like(self.W_y)
        db_y = np.zeros_like(self.b_y)
        d_embedding = np.zeros_like(self.embedding)
        
        # Инициализация градиентов для forward LSTM
        dW_f_fwd = np.zeros_like(self.lstm_fwd.W_f)
        dW_i_fwd = np.zeros_like(self.lstm_fwd.W_i)
        dW_c_fwd = np.zeros_like(self.lstm_fwd.W_c)
        dW_o_fwd = np.zeros_like(self.lstm_fwd.W_o)
        dU_f_fwd = np.zeros_like(self.lstm_fwd.U_f)
        dU_i_fwd = np.zeros_like(self.lstm_fwd.U_i)
        dU_c_fwd = np.zeros_like(self.lstm_fwd.U_c)
        dU_o_fwd = np.zeros_like(self.lstm_fwd.U_o)
        db_f_fwd = np.zeros_like(self.lstm_fwd.b_f)
        db_i_fwd = np.zeros_like(self.lstm_fwd.b_i)
        db_c_fwd = np.zeros_like(self.lstm_fwd.b_c)
        db_o_fwd = np.zeros_like(self.lstm_fwd.b_o)
        
        # Инициализация градиентов для backward LSTM
        dW_f_bwd = np.zeros_like(self.lstm_bwd.W_f)
        dW_i_bwd = np.zeros_like(self.lstm_bwd.W_i)
        dW_c_bwd = np.zeros_like(self.lstm_bwd.W_c)
        dW_o_bwd = np.zeros_like(self.lstm_bwd.W_o)
        dU_f_bwd = np.zeros_like(self.lstm_bwd.U_f)
        dU_i_bwd = np.zeros_like(self.lstm_bwd.U_i)
        dU_c_bwd = np.zeros_like(self.lstm_bwd.U_c)
        dU_o_bwd = np.zeros_like(self.lstm_bwd.U_o)
        db_f_bwd = np.zeros_like(self.lstm_bwd.b_f)
        db_i_bwd = np.zeros_like(self.lstm_bwd.b_i)
        db_c_bwd = np.zeros_like(self.lstm_bwd.b_c)
        db_o_bwd = np.zeros_like(self.lstm_bwd.b_o)
        
        # Градиент выходного слоя
        dlogits = probs
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        
        # Градиенты для выходного слоя
        dW_y = np.dot(dlogits.T, combined)
        db_y = np.sum(dlogits, axis=0)
        
        # Градиент для конкатенированного состояния
        d_combined = np.dot(dlogits, self.W_y)  # (B, 2H)
        
        # Разделяем градиент на forward и backward части
        d_fwd = d_combined[:, :self.hidden_dim]   # (B, H)
        d_bwd = d_combined[:, self.hidden_dim:]   # (B, H)
        
        # ========== Backward pass для forward LSTM ==========
        # Нам нужны градиенты для позиции mask_position
        # Сохраняем состояния и кэши для backward
        _, _, fwd_caches, bwd_caches = self.forward_full_sequence(x)
        
        # Backward через forward LSTM до позиции mask_position
        dh_next_fwd = np.zeros((B, self.hidden_dim))
        dc_next_fwd = np.zeros((B, self.hidden_dim))
        
        for t in range(mask_position, -1, -1):
            cache = fwd_caches[t]
            
            if t == mask_position:
                dh_next_fwd += d_fwd
            
            dh_next_fwd, dc_next_fwd, dx_t_fwd, grads_fwd = self.lstm_fwd.backward_step(
                cache, dh_next_fwd, dc_next_fwd
            )
            
            # Накопление градиентов
            dW_f_fwd += grads_fwd['dW_f']
            dW_i_fwd += grads_fwd['dW_i']
            dW_c_fwd += grads_fwd['dW_c']
            dW_o_fwd += grads_fwd['dW_o']
            dU_f_fwd += grads_fwd['dU_f']
            dU_i_fwd += grads_fwd['dU_i']
            dU_c_fwd += grads_fwd['dU_c']
            dU_o_fwd += grads_fwd['dU_o']
            db_f_fwd += grads_fwd['db_f']
            db_i_fwd += grads_fwd['db_i']
            db_c_fwd += grads_fwd['db_c']
            db_o_fwd += grads_fwd['db_o']
            
            # Накопление градиентов эмбеддинга
            for b in range(B):
                d_embedding[x[b, t]] += dx_t_fwd[b]
        
        # ========== Backward pass для backward LSTM ==========
        dh_next_bwd = np.zeros((B, self.hidden_dim))
        dc_next_bwd = np.zeros((B, self.hidden_dim))
        
        for t in range(mask_position, T):
            cache = bwd_caches[t]
            
            if t == mask_position:
                dh_next_bwd += d_bwd
            
            dh_next_bwd, dc_next_bwd, dx_t_bwd, grads_bwd = self.lstm_bwd.backward_step(
                cache, dh_next_bwd, dc_next_bwd
            )
            
            # Накопление градиентов
            dW_f_bwd += grads_bwd['dW_f']
            dW_i_bwd += grads_bwd['dW_i']
            dW_c_bwd += grads_bwd['dW_c']
            dW_o_bwd += grads_bwd['dW_o']
            dU_f_bwd += grads_bwd['dU_f']
            dU_i_bwd += grads_bwd['dU_i']
            dU_c_bwd += grads_bwd['dU_c']
            dU_o_bwd += grads_bwd['dU_o']
            db_f_bwd += grads_bwd['db_f']
            db_i_bwd += grads_bwd['db_i']
            db_c_bwd += grads_bwd['db_c']
            db_o_bwd += grads_bwd['db_o']
            
            # Накопление градиентов эмбеддинга
            for b in range(B):
                d_embedding[x[b, t]] += dx_t_bwd[b]
        
        # Сбор всех градиентов
        grads = [
            d_embedding,           # embedding
            dW_y,                  # W_y
            db_y,                  # b_y
            # Forward LSTM
            dW_f_fwd, dW_i_fwd, dW_c_fwd, dW_o_fwd,
            dU_f_fwd, dU_i_fwd, dU_c_fwd, dU_o_fwd,
            db_f_fwd, db_i_fwd, db_c_fwd, db_o_fwd,
            # Backward LSTM
            dW_f_bwd, dW_i_bwd, dW_c_bwd, dW_o_bwd,
            dU_f_bwd, dU_i_bwd, dU_c_bwd, dU_o_bwd,
            db_f_bwd, db_i_bwd, db_c_bwd, db_o_bwd
        ]
        
        return loss, grads
    
    def fill_mask(self, context, mask_position, top_k=10):
        """
        Заполнение маскированного токена
        
        Args:
            context: list or np.ndarray - индексы контекста (с <MASK>)
            mask_position: int - позиция маски
            top_k: int - количество лучших предсказаний
        
        Returns:
            predictions: list of (char, prob) - топ-k предсказаний
        """
        x = np.array([context])
        probs, _, _, _ = self.predict_at_position(x, mask_position)
        probs = probs[0]  # (vocab_size,)
        
        # Получаем top-k (исключая сам <MASK> токен)
        indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in indices:
            if idx != self.MASK_TOKEN_ID:
                predictions.append((idx, probs[idx]))
        
        return predictions
    
    def save(self, filepath):
        """Сохранение модели"""
        np.savez(filepath,
                 embedding=self.embedding,
                 W_y=self.W_y,
                 b_y=self.b_y,
                 # Forward LSTM
                 W_f_fwd=self.lstm_fwd.W_f,
                 W_i_fwd=self.lstm_fwd.W_i,
                 W_c_fwd=self.lstm_fwd.W_c,
                 W_o_fwd=self.lstm_fwd.W_o,
                 U_f_fwd=self.lstm_fwd.U_f,
                 U_i_fwd=self.lstm_fwd.U_i,
                 U_c_fwd=self.lstm_fwd.U_c,
                 U_o_fwd=self.lstm_fwd.U_o,
                 b_f_fwd=self.lstm_fwd.b_f,
                 b_i_fwd=self.lstm_fwd.b_i,
                 b_c_fwd=self.lstm_fwd.b_c,
                 b_o_fwd=self.lstm_fwd.b_o,
                 # Backward LSTM
                 W_f_bwd=self.lstm_bwd.W_f,
                 W_i_bwd=self.lstm_bwd.W_i,
                 W_c_bwd=self.lstm_bwd.W_c,
                 W_o_bwd=self.lstm_bwd.W_o,
                 U_f_bwd=self.lstm_bwd.U_f,
                 U_i_bwd=self.lstm_bwd.U_i,
                 U_c_bwd=self.lstm_bwd.U_c,
                 U_o_bwd=self.lstm_bwd.U_o,
                 b_f_bwd=self.lstm_bwd.b_f,
                 b_i_bwd=self.lstm_bwd.b_i,
                 b_c_bwd=self.lstm_bwd.b_c,
                 b_o_bwd=self.lstm_bwd.b_o,
                 base_vocab_size=self.base_vocab_size,
                 vocab_size=self.vocab_size,
                 embedding_dim=self.embedding_dim,
                 hidden_dim=self.hidden_dim,
                 MASK_TOKEN_ID=self.MASK_TOKEN_ID)
    
    def load(self, filepath):
        """Загрузка модели"""
        data = np.load(filepath, allow_pickle=True)
        self.embedding = data['embedding']
        self.W_y = data['W_y']
        self.b_y = data['b_y']
        
        # Forward LSTM
        self.lstm_fwd.W_f = data['W_f_fwd']
        self.lstm_fwd.W_i = data['W_i_fwd']
        self.lstm_fwd.W_c = data['W_c_fwd']
        self.lstm_fwd.W_o = data['W_o_fwd']
        self.lstm_fwd.U_f = data['U_f_fwd']
        self.lstm_fwd.U_i = data['U_i_fwd']
        self.lstm_fwd.U_c = data['U_c_fwd']
        self.lstm_fwd.U_o = data['U_o_fwd']
        self.lstm_fwd.b_f = data['b_f_fwd']
        self.lstm_fwd.b_i = data['b_i_fwd']
        self.lstm_fwd.b_c = data['b_c_fwd']
        self.lstm_fwd.b_o = data['b_o_fwd']
        
        # Backward LSTM
        self.lstm_bwd.W_f = data['W_f_bwd']
        self.lstm_bwd.W_i = data['W_i_bwd']
        self.lstm_bwd.W_c = data['W_c_bwd']
        self.lstm_bwd.W_o = data['W_o_bwd']
        self.lstm_bwd.U_f = data['U_f_bwd']
        self.lstm_bwd.U_i = data['U_i_bwd']
        self.lstm_bwd.U_c = data['U_c_bwd']
        self.lstm_bwd.U_o = data['U_o_bwd']
        self.lstm_bwd.b_f = data['b_f_bwd']
        self.lstm_bwd.b_i = data['b_i_bwd']
        self.lstm_bwd.b_c = data['b_c_bwd']
        self.lstm_bwd.b_o = data['b_o_bwd']
        
        self.base_vocab_size = int(data['base_vocab_size'])
        self.vocab_size = int(data['vocab_size'])
        self.MASK_TOKEN_ID = int(data['MASK_TOKEN_ID'])
        self._collect_params()