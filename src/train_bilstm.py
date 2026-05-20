# src/train_bilstm.py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from src.core.optimizer import AdamOptimizer
from src.core.losses import cross_entropy_loss, perplexity, top_k_accuracy
from src.models.bilstm import BiLSTMModel


class BiLSTMTrainer:
    """Тренер для BiLSTM модели (Fill-in-the-Middle)"""
    def __init__(self, model, dataset, config, model_name="BiLSTM"):
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
        
        self.seq_length = config['seq_length']
        self.batch_size = config['batch_size']
        self.max_epochs = config['max_epochs']
        self.patience = config['patience']
        self.steps_per_epoch = config['steps_per_epoch']
        
        # Позиция маски - середина последовательности
        self.mask_position = self.seq_length // 2
        
        self.train_losses = []
        self.val_losses = []
        self.val_top1 = []
        self.val_top5 = []
        self.val_top10 = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def create_masked_batch(self, data, batch_idx):
        """
        Создание батча с маскированными токенами
        
        Args:
            data: np.ndarray - данные
            batch_idx: int - номер батча
        
        Returns:
            x: (B, T) - входные данные с <MASK>
            y: (B,) - целевые токены
            mask_position: int - позиция маски
        """
        start = batch_idx * self.batch_size * self.seq_length
        end = start + self.batch_size * self.seq_length
        
        if end >= len(data):
            return None, None, None
        
        # Берём последовательность
        batch = data[start:end].reshape(self.batch_size, self.seq_length)
        
        # Копируем для входных данных
        x = batch.copy()
        
        # Целевые токены - символы на позиции маски
        y = x[:, self.mask_position].copy()
        
        # Заменяем на <MASK> токен
        x[:, self.mask_position] = self.model.MASK_TOKEN_ID
        
        return x, y, self.mask_position
    
    def get_val_batches(self, val_data):
        """Получение валидационных батчей"""
        batches = []
        num_batches = (len(val_data) - self.seq_length) // (self.batch_size * self.seq_length)
        num_batches = min(num_batches, self.config['eval_batches'])
        
        for i in range(num_batches):
            start = i * self.batch_size * self.seq_length
            end = start + self.batch_size * self.seq_length
            if end >= len(val_data):
                break
            
            batch = val_data[start:end].reshape(self.batch_size, self.seq_length)
            x = batch.copy()
            y = x[:, self.mask_position].copy()
            x[:, self.mask_position] = self.model.MASK_TOKEN_ID
            
            batches.append((x, y, self.mask_position))
        
        return batches
    
    def evaluate(self, batches):
        """Оценка модели на валидации"""
        total_loss = 0
        all_probs = []
        all_targets = []
        
        for x, y, mask_pos in batches:
            probs, _, _, _ = self.model.predict_at_position(x, mask_pos)
            loss = cross_entropy_loss(probs, y)
            total_loss += loss
            
            all_probs.append(probs)
            all_targets.append(y)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        avg_loss = total_loss / len(batches)
        ppl = perplexity(avg_loss)
        top1 = top_k_accuracy(all_probs, all_targets, 1)
        top5 = top_k_accuracy(all_probs, all_targets, 5)
        top10 = top_k_accuracy(all_probs, all_targets, 10)
        
        return ppl, top1, top5, top10, avg_loss
    
    def train_epoch(self, train_data):
        """Одна эпоха обучения"""
        total_loss = 0
        
        for batch_idx in range(self.steps_per_epoch):
            x, y, mask_pos = self.create_masked_batch(train_data, batch_idx)
            if x is None:
                break
            
            loss, grads = self.model.compute_loss_and_grads(x, y, mask_pos)
            self.optimizer.step(grads)
            total_loss += loss
        
        return total_loss / self.steps_per_epoch
    
    def train(self, train_data, val_data, verbose=True):
        """Полный цикл обучения"""
        val_batches = self.get_val_batches(val_data)
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(train_data)
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
        axes[0].set_title(f'{self.model_name} - Perplexity (Fill-in-the-Middle)')
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
    
    def evaluate_test(self, test_data):
        """Оценка на тестовом множестве"""
        num_batches = len(test_data) // (self.batch_size * self.seq_length)
        num_batches = min(num_batches, 100)
        all_probs = []
        all_targets = []
        
        for batch_idx in range(num_batches):
            x, y, mask_pos = self.create_masked_batch(test_data, batch_idx)
            if x is None:
                break
            probs, _, _, _ = self.model.predict_at_position(x, mask_pos)
            all_probs.append(probs)
            all_targets.append(y)
        
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        loss = cross_entropy_loss(all_probs, all_targets)
        ppl = perplexity(loss)
        top1 = top_k_accuracy(all_probs, all_targets, 1)
        top5 = top_k_accuracy(all_probs, all_targets, 5)
        top10 = top_k_accuracy(all_probs, all_targets, 10)
        
        return ppl, top1, top5, top10
    
    def demo_fill_mask(self, examples, top_k=10):
        """
        Демонстрация восстановления маскированных токенов
        
        Args:
            examples: list of tuples (context_tokens, mask_position, true_token)
            top_k: int - количество предсказаний
        """
        print("\n" + "=" * 60)
        print("ДЕМОНСТРАЦИЯ BiLSTM (Fill-in-the-Middle)")
        print("=" * 60)
        
        for i, (context_tokens, mask_pos, true_token) in enumerate(examples):
            # Получаем предсказания
            predictions = self.model.fill_mask(context_tokens, mask_pos, top_k)
            
            # Декодируем контекст для отображения
            context_str = ''.join(self.dataset.id_to_char(t) for t in context_tokens)
            # Заменяем маскированный токен на <MASK>
            masked_str = context_str[:mask_pos] + "<MASK>" + context_str[mask_pos+1:]
            true_char = self.dataset.id_to_char(true_token)
            
            print(f"\nПример {i+1}:")
            print(f"Контекст: {masked_str}")
            print(f"Истинный символ: '{true_char}'")
            print("Top-10 предсказаний:")
            
            for j, (idx, prob) in enumerate(predictions[:top_k]):
                char = self.dataset.id_to_char(idx)
                print(f"  {j+1:2d}. '{char}' (p = {prob:.4f})")