# src/core/losses.py
import numpy as np

def softmax(x, axis=-1):
    """Стабильный softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss"""
    eps = 1e-8
    if y_pred.ndim == 3:
        B, T, V = y_pred.shape
        y_pred_flat = y_pred.reshape(-1, V)
        y_true_flat = y_true.reshape(-1)
        log_probs = -np.log(y_pred_flat[np.arange(len(y_true_flat)), y_true_flat] + eps)
        return np.mean(log_probs)
    else:
        log_probs = -np.log(y_pred[np.arange(len(y_true)), y_true] + eps)
        return np.mean(log_probs)

def perplexity(loss):
    """Perplexity из loss"""
    return np.exp(loss)

def top_k_accuracy(y_pred, y_true, k):
    """Top-k accuracy"""
    if y_pred.ndim == 3:
        B, T, V = y_pred.shape
        y_pred = y_pred.reshape(-1, V)
        y_true = y_true.reshape(-1)
    
    top_k_pred = np.argsort(y_pred, axis=1)[:, -k:]
    correct = np.any(top_k_pred == y_true.reshape(-1, 1), axis=1)
    return np.mean(correct)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100)))