# src/data/dataset.py
import numpy as np
import json
import os

class CodeDataset:
    """Датасет для автодополнения кода"""
    def __init__(self, data_path, seq_length=128, batch_size=128):
        npz = np.load(os.path.join(data_path, "dataset_char.npz"), allow_pickle=True)
        
        self.vocab_codepoints = npz["vocab"]
        self.vocab_size = len(self.vocab_codepoints)
        
        self.train = npz["train"].astype(np.int64)
        self.val = npz["val"].astype(np.int64)
        self.test = npz["test"].astype(np.int64)
        
        # Загрузка метаданных
        meta_path = os.path.join(data_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        print(f"Vocab size: {self.vocab_size}")
        print(f"Train tokens: {len(self.train)}")
        print(f"Val tokens: {len(self.val)}")
        print(f"Test tokens: {len(self.test)}")
    
    def get_batch(self, data, batch_idx):
        """Получение батча"""
        start = batch_idx * self.batch_size * self.seq_length
        end = start + self.batch_size * self.seq_length + 1
        
        if end >= len(data):
            return None, None
        
        batch_data = data[start:end]
        x = batch_data[:-1].reshape(self.batch_size, self.seq_length)
        y = batch_data[1:].reshape(self.batch_size, self.seq_length)
        
        return x, y
    
    def get_val_batches(self):
        """Получение валидационных батчей"""
        batches = []
        num_batches = (len(self.val) - 1) // (self.batch_size * self.seq_length)
        num_batches = min(num_batches, 20)  # eval batches: 20
        
        for i in range(num_batches):
            start = i * self.batch_size * self.seq_length
            end = start + self.batch_size * self.seq_length + 1
            if end >= len(self.val):
                break
            batch_data = self.val[start:end]
            x = batch_data[:-1].reshape(self.batch_size, self.seq_length)
            y = batch_data[1:].reshape(self.batch_size, self.seq_length)
            batches.append((x, y))
        
        return batches
    
    def id_to_char(self, i):
        """Конвертация индекса в символ"""
        if i == 0 or i >= len(self.vocab_codepoints):
            return '?'
        return chr(int(self.vocab_codepoints[i]))
    
    def decode_sequence(self, ids):
        """Декодирование последовательности"""
        return ''.join(self.id_to_char(i) for i in ids if i != 0)