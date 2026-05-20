# src/train.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Добавляем путь для импортов
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.core.optimizer import AdamOptimizer
from src.core.losses import cross_entropy_loss, perplexity, top_k_accuracy
from src.data.dataset import CodeDataset

# Импортируем модели (создайте файлы lstm.py, gru.py с кодом из предыдущего ответа)
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel


class Trainer:
    """Тренер для LSTM и GRU моделей"""
    def __init__(self, model, dataset, config, model_name="Model"):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.model_name = model_name
        
        self.optimizer = AdamOptimizer(
            model.params,
            lr=config['learning_rate'],
            beta1=config.get('beta1', 0.9),
            beta2=config.get('beta2', 0.999),
            eps=config.get('eps', 1e-8),
            clip_norm=config.get('clip_norm', 5.0)
        )
        
        self.max_epochs = config['max_epochs']
        self.patience = config['patience']
        self.steps_per_epoch = config['steps_per_epoch']
        
        self.train_losses = []
        self.val_losses = []
        self.val_top1 = []
        self.val_top5 = []
        self.val_top10 = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def evaluate(self, batches):
        """Оценка модели на валидации"""
        total_loss = 0
        all_probs = []
        all_targets = []
        
        for x, y in batches:
            probs, _, _ = self.model.forward(x, return_sequences=True)
            loss = cross_entropy_loss(probs, y)
            total_loss += loss
            
            probs_flat = probs.reshape(-1, self.model.vocab_size)
            targets_flat = y.reshape(-1)
            all_probs.append(probs_flat)
            all_targets.append(targets_flat)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        avg_loss = total_loss / len(batches)
        ppl = perplexity(avg_loss)
        top1 = top_k_accuracy(all_probs, all_targets, 1)
        top5 = top_k_accuracy(all_probs, all_targets, 5)
        top10 = top_k_accuracy(all_probs, all_targets, 10)
        
        return ppl, top1, top5, top10, avg_loss
    
    def train_epoch(self):
        """Одна эпоха обучения"""
        total_loss = 0
        
        for batch_idx in range(self.steps_per_epoch):
            x, y = self.dataset.get_batch(self.dataset.train, batch_idx)
            if x is None:
                break
            
            loss, grads = self.model.compute_loss_and_grads(x, y)
            self.optimizer.step(grads)
            total_loss += loss
        
        return total_loss / self.steps_per_epoch
    
    def train(self, verbose=True):
        """Полный цикл обучения"""
        val_batches = self.dataset.get_val_batches()
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch()
            train_ppl = perplexity(train_loss)
            self.train_losses.append(train_ppl)
            
            # Validation
            val_ppl, top1, top5, top10, val_loss = self.evaluate(val_batches)
            self.val_losses.append(val_ppl)
            self.val_top1.append(top1)
            self.val_top5.append(top5)
            self.val_top10.append(top10)
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.max_epochs} | "
                      f"Train PPL: {train_ppl:.4f} | Val PPL: {val_ppl:.4f} | "
                      f"Top-1: {top1:.4f} | Top-5: {top5:.4f} | Top-10: {top10:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                # Сохраняем лучшую модель
                self.best_model_params = [p.copy() for p in self.model.params]
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Восстанавливаем лучшую модель
        for p, best_p in zip(self.model.params, self.best_model_params):
            p[:] = best_p
        
        return self.best_epoch
    
    def plot_training_curves(self, save_path=None):
        """Построение графиков обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Perplexity
        axes[0].plot(self.train_losses, label='Train Perplexity')
        axes[0].plot(self.val_losses, label='Val Perplexity')
        axes[0].axvline(x=self.best_epoch, color='r', linestyle='--', 
                       label=f'Best epoch ({self.best_epoch+1})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Perplexity')
        axes[0].set_title(f'{self.model_name} - Perplexity')
        axes[0].legend()
        axes[0].grid(True)
        
        # Top-k accuracy
        axes[1].plot(self.val_top1, label='Top-1')
        axes[1].plot(self.val_top5, label='Top-5')
        axes[1].plot(self.val_top10, label='Top-10')
        axes[1].axvline(x=self.best_epoch, color='r', linestyle='--',
                       label=f'Best epoch ({self.best_epoch+1})')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{self.model_name} - Top-k Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
        # В src/train.py, найдите метод evaluate_test и замените на этот:

    def evaluate_test(self, test_data):
        """Оценка на тестовом множестве"""
        num_batches = len(test_data) // (self.dataset.batch_size * self.dataset.seq_length)
        num_batches = min(num_batches, 100)
        all_probs = []
        all_targets = []
    
        for batch_idx in range(num_batches):
            x, y = self.dataset.get_batch(test_data, batch_idx)
            if x is None:
                break
            probs, _, _ = self.model.forward(x, return_sequences=True)
            all_probs.append(probs.reshape(-1, self.model.vocab_size))
            all_targets.append(y.reshape(-1))
    
        if len(all_probs) == 0:
            return 100.0, 0.0, 0.0, 0.0
    
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    
        loss = cross_entropy_loss(all_probs, all_targets)
        ppl = perplexity(loss)
        top1 = top_k_accuracy(all_probs, all_targets, 1)
        top5 = top_k_accuracy(all_probs, all_targets, 5)
        top10 = top_k_accuracy(all_probs, all_targets, 10)
    
        return ppl, top1, top5, top10
    
    def generate_completions(self, prefixes, top_k=10):
        """Генерация продолжений для префиксов"""
        results = []
        
        for prefix in prefixes:
            # Конвертируем префикс в индексы
            prefix_ids = []
            for ch in prefix:
                found = False
                for i, cp in enumerate(self.dataset.vocab_codepoints):
                    if chr(cp) == ch:
                        prefix_ids.append(i)
                        found = True
                        break
                if not found:
                    prefix_ids.append(0)  # <UNK>
            
            # Forward pass для префикса
            x = np.array([prefix_ids])
            probs, _, _ = self.model.forward(x, return_sequences=False)
            
            # Получаем top-k для следующего символа
            top_indices = np.argsort(probs[0])[-top_k:][::-1]
            top_probs = probs[0][top_indices]
            
            # Конвертируем обратно в символы
            completions = []
            for idx, prob in zip(top_indices, top_probs):
                char = self.dataset.id_to_char(idx)
                completions.append((char, prob))
            
            results.append({
                'prefix': prefix,
                'completions': completions
            })
        
        return results