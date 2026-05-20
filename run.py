#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from src.data.dataset import CodeDataset
from src.train import Trainer
from src.train_bilstm import BiLSTMTrainer
from src.models import LSTMModel, GRUModel, BiLSTMModel


def create_directories():
    """Создание необходимых директорий"""
    dirs = [
        RESULTS_DIR,
        MODELS_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        os.path.join(RESULTS_DIR, 'logs')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def print_table(results):
    """Вывод таблицы результатов"""
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # Заголовок
    header = f"{'Метрика':<20} {'LSTM (val)':<15} {'GRU (val)':<15} " \
             f"{'LSTM (test)':<15} {'GRU (test)':<15}"
    if 'bilstm' in results:
        header += f" {'BiLSTM (test)':<15}"
    print(header)
    print("-" * (80 + (15 if 'bilstm' in results else 0)))
    
    metrics = [
        ('Perplexity', 'ppl'),
        ('Top-1 Accuracy', 'top1'),
        ('Top-5 Accuracy', 'top5'),
        ('Top-10 Accuracy', 'top10')
    ]
    
    for metric_name, key in metrics:
        row = f"{metric_name:<20}"
        
        if 'lstm' in results:
            row += f"{results['lstm']['val'][key]:<15.4f} "
            row += f"{results['lstm']['test'][key]:<15.4f} "
        else:
            row += f"{'N/A':<15} {'N/A':<15} "
        
        if 'gru' in results:
            row += f"{results['gru']['val'][key]:<15.4f} "
            row += f"{results['gru']['test'][key]:<15.4f} "
        else:
            row += f"{'N/A':<15} {'N/A':<15} "
        
        if 'bilstm' in results:
            row += f"{results['bilstm']['test'][key]:<15.4f}"
        
        print(row)


def print_completions(completions):
    """Вывод результатов автодополнения"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ АВТОДОПОЛНЕНИЯ (LSTM)")
    print("=" * 60)
    
    for result in completions:
        print(f"\nПрефикс: '{result['prefix']}'")
        print("Top-10 продолжений:")
        for i, (char, prob) in enumerate(result['completions']):
            if char == '\n':
                display_char = '\\n'
            elif char == '\t':
                display_char = '\\t'
            elif char == '\r':
                display_char = '\\r'
            else:
                display_char = char
            print(f"  {i+1:2d}. '{display_char}' (p = {prob:.4f})")


def save_results(results, language, variant):
    """Сохранение результатов в файл"""
    report_path = os.path.join(REPORTS_DIR, f"{language}_{variant}_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №3\n")
        f.write(f"Язык: {language}\n")
        f.write(f"Вариант: {variant}\n")
        f.write(f"Дата: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ГИПЕРПАРАМЕТРЫ:\n")
        f.write(f"  seq_length: {SEQUENCE_LENGTH}\n")
        f.write(f"  batch_size: {BATCH_SIZE}\n")
        f.write(f"  embedding_dim: {EMBEDDING_DIM}\n")
        f.write(f"  hidden_dim: {HIDDEN_DIM}\n")
        f.write(f"  learning_rate: {LEARNING_RATE}\n")
        f.write(f"  max_epochs: {MAX_EPOCHS}\n")
        f.write(f"  patience: {PATIENCE}\n")
        f.write(f"  gradient_clip: {CLIP_NORM}\n\n")
        
        if 'lstm' in results:
            f.write("LSTM РЕЗУЛЬТАТЫ:\n")
            f.write(f"  Лучшая эпоха: {results['lstm']['best_epoch'] + 1}\n")
            f.write(f"  Val Perplexity: {results['lstm']['val']['ppl']:.4f}\n")
            f.write(f"  Test Perplexity: {results['lstm']['test']['ppl']:.4f}\n")
            f.write(f"  Test Top-1: {results['lstm']['test']['top1']:.4f}\n")
            f.write(f"  Test Top-5: {results['lstm']['test']['top5']:.4f}\n")
            f.write(f"  Test Top-10: {results['lstm']['test']['top10']:.4f}\n\n")
        
        if 'gru' in results:
            f.write("GRU РЕЗУЛЬТАТЫ:\n")
            f.write(f"  Лучшая эпоха: {results['gru']['best_epoch'] + 1}\n")
            f.write(f"  Val Perplexity: {results['gru']['val']['ppl']:.4f}\n")
            f.write(f"  Test Perplexity: {results['gru']['test']['ppl']:.4f}\n")
            f.write(f"  Test Top-1: {results['gru']['test']['top1']:.4f}\n")
            f.write(f"  Test Top-5: {results['gru']['test']['top5']:.4f}\n")
            f.write(f"  Test Top-10: {results['gru']['test']['top10']:.4f}\n\n")
        
        if 'bilstm' in results:
            f.write("BiLSTM РЕЗУЛЬТАТЫ (Fill-in-the-Middle):\n")
            f.write(f"  Лучшая эпоха: {results['bilstm']['best_epoch'] + 1}\n")
            f.write(f"  Test Perplexity: {results['bilstm']['test']['ppl']:.4f}\n")
            f.write(f"  Test Top-1: {results['bilstm']['test']['top1']:.4f}\n")
            f.write(f"  Test Top-5: {results['bilstm']['test']['top5']:.4f}\n")
            f.write(f"  Test Top-10: {results['bilstm']['test']['top10']:.4f}\n\n")
    
    print(f"\nОтчёт сохранён в: {report_path}")


def run_lstm_gru(dataset, language, variant, show_plots=True):
    """Запуск LSTM и GRU моделей"""
    results = {}
    
    # LSTM
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ LSTM")
    print("=" * 40)
    
    lstm_model = LSTMModel(dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    lstm_trainer = Trainer(lstm_model, dataset, LSTM_CONFIG, model_name="LSTM")
    lstm_best_epoch = lstm_trainer.train()
    
    lstm_plot_path = os.path.join(PLOTS_DIR, f"{language}_{variant}_lstm.png")
    lstm_trainer.plot_training_curves(save_path=lstm_plot_path if show_plots else None)
    
    lstm_model_path = os.path.join(MODELS_DIR, f"{language}_{variant}_lstm.npz")
    lstm_model.save(lstm_model_path)
    
    lstm_test_ppl, lstm_test_top1, lstm_test_top5, lstm_test_top10 = lstm_trainer.evaluate_test(dataset.test)
    
    results['lstm'] = {
        'model': lstm_model,
        'trainer': lstm_trainer,
        'best_epoch': lstm_best_epoch,
        'val': {
            'ppl': lstm_trainer.best_val_loss,
            'top1': lstm_trainer.val_top1[lstm_best_epoch],
            'top5': lstm_trainer.val_top5[lstm_best_epoch],
            'top10': lstm_trainer.val_top10[lstm_best_epoch]
        },
        'test': {
            'ppl': lstm_test_ppl,
            'top1': lstm_test_top1,
            'top5': lstm_test_top5,
            'top10': lstm_test_top10
        }
    }
    
    # GRU
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ GRU")
    print("=" * 40)
    
    gru_model = GRUModel(dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    gru_trainer = Trainer(gru_model, dataset, GRU_CONFIG, model_name="GRU")
    gru_best_epoch = gru_trainer.train()
    
    gru_plot_path = os.path.join(PLOTS_DIR, f"{language}_{variant}_gru.png")
    gru_trainer.plot_training_curves(save_path=gru_plot_path if show_plots else None)
    
    gru_model_path = os.path.join(MODELS_DIR, f"{language}_{variant}_gru.npz")
    gru_model.save(gru_model_path)
    
    gru_test_ppl, gru_test_top1, gru_test_top5, gru_test_top10 = gru_trainer.evaluate_test(dataset.test)
    
    results['gru'] = {
        'model': gru_model,
        'trainer': gru_trainer,
        'best_epoch': gru_best_epoch,
        'val': {
            'ppl': gru_trainer.best_val_loss,
            'top1': gru_trainer.val_top1[gru_best_epoch],
            'top5': gru_trainer.val_top5[gru_best_epoch],
            'top10': gru_trainer.val_top10[gru_best_epoch]
        },
        'test': {
            'ppl': gru_test_ppl,
            'top1': gru_test_top1,
            'top5': gru_test_top5,
            'top10': gru_test_top10
        }
    }
    
    # Генерация автодополнений
    prefixes = get_demo_prefixes(language)
    completions = lstm_trainer.generate_completions(prefixes, top_k=10)
    print_completions(completions)
    
    return results, completions


def run_bilstm_only(dataset, language, variant, show_plots=True):
    """Запуск только BiLSTM модели"""
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ BiLSTM (Fill-in-the-Middle)")
    print("=" * 40)
    
    bilstm_model = BiLSTMModel(
        base_vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=BILSTM_CONFIG['hidden_dim']
    )
    
    bilstm_trainer = BiLSTMTrainer(bilstm_model, dataset, BILSTM_CONFIG, model_name="BiLSTM")
    best_epoch = bilstm_trainer.train(dataset.train, dataset.val)
    
    bilstm_plot_path = os.path.join(PLOTS_DIR, f"{language}_{variant}_bilstm.png")
    bilstm_trainer.plot_training_curves(save_path=bilstm_plot_path if show_plots else None)
    
    bilstm_model_path = os.path.join(MODELS_DIR, f"{language}_{variant}_bilstm.npz")
    bilstm_model.save(bilstm_model_path)
    
    test_ppl, test_top1, test_top5, test_top10 = bilstm_trainer.evaluate_test(dataset.test)
    
    # Демонстрация Fill-in-the-Middle
    demo_examples = []
    for _ in range(10):
        start = np.random.randint(0, len(dataset.test) - BILSTM_CONFIG['seq_length'])
        seq = dataset.test[start:start + BILSTM_CONFIG['seq_length']]
        mask_pos = BILSTM_CONFIG['seq_length'] // 2
        true_token = seq[mask_pos]
        context = seq.copy()
        context[mask_pos] = bilstm_model.MASK_TOKEN_ID
        demo_examples.append((context.tolist(), mask_pos, true_token))
    
    bilstm_trainer.demo_fill_mask(demo_examples, top_k=10)
    
    return {
        'model': bilstm_model,
        'trainer': bilstm_trainer,
        'best_epoch': best_epoch,
        'test': {
            'ppl': test_ppl,
            'top1': test_top1,
            'top5': test_top5,
            'top10': test_top10
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Запуск обучения для автодополнения кода')
    parser.add_argument('--language', type=str, default=LANGUAGE,
                       help=f'Язык программирования. По умолчанию: {LANGUAGE}')
    parser.add_argument('--variant', type=str, default=VARIANT,
                       help=f'Номер варианта. По умолчанию: {VARIANT}')
    parser.add_argument('--bilstm', action='store_true',
                       help='Запустить BiLSTM дополнительно к LSTM и GRU')
    parser.add_argument('--only-bilstm', action='store_true',
                       help='Запустить ТОЛЬКО BiLSTM (без LSTM/GRU)')
    parser.add_argument('--data-dir', type=str, default=DATASETS_DIR,
                       help=f'Путь к датасетам. По умолчанию: {DATASETS_DIR}')
    parser.add_argument('--no-plots', action='store_true',
                       help='Не показывать графики')
    
    args = parser.parse_args()
    
    # Формирование пути к данным
    data_path = os.path.join(args.data_dir, args.language, args.variant, "data_char")
    
    if not os.path.exists(data_path):
        print(f"Ошибка: Директория {data_path} не найдена!")
        sys.exit(1)
    
    # Создание директорий
    create_directories()
    
    # Загрузка данных
    print("\n" + "=" * 60)
    print(f"Запуск эксперимента: {args.language}/{args.variant}")
    print("=" * 60)
    
    print("\nЗагрузка датасета...")
    dataset = CodeDataset(
        data_path,
        seq_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    results = {}
    show_plots = not args.no_plots
    
    # ========== ОСНОВНАЯ ЛОГИКА ЗАПУСКА ==========
    if args.only_bilstm:
        # Только BiLSTM
        print("\n>>> Режим: ТОЛЬКО BiLSTM")
        results['bilstm'] = run_bilstm_only(dataset, args.language, args.variant, show_plots)
    
    elif args.bilstm:
        # LSTM + GRU + BiLSTM (ВСЕ МОДЕЛИ)
        print("\n>>> Режим: ВСЕ МОДЕЛИ (LSTM + GRU + BiLSTM)")
        lstm_gru_results, _ = run_lstm_gru(dataset, args.language, args.variant, show_plots)
        results.update(lstm_gru_results)
        results['bilstm'] = run_bilstm_only(dataset, args.language, args.variant, show_plots)
    
    else:
        # Только LSTM и GRU (по умолчанию)
        print("\n>>> Режим: ТОЛЬКО LSTM + GRU (без BiLSTM)")
        lstm_gru_results, _ = run_lstm_gru(dataset, args.language, args.variant, show_plots)
        results.update(lstm_gru_results)
    
    # Вывод таблицы результатов
    print_table(results)
    
    # Сохранение результатов
    save_results(results, args.language, args.variant)
    
    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН!")
    print("=" * 60)
    print(f"  Режим: ", end='')
    if args.only_bilstm:
        print("Только BiLSTM")
    elif args.bilstm:
        print("LSTM + GRU + BiLSTM")
    else:
        print("LSTM + GRU")
    
    if 'lstm' in results:
        print(f"  LSTM лучшая эпоха: {results['lstm']['best_epoch'] + 1}")
        print(f"  LSTM Test Perplexity: {results['lstm']['test']['ppl']:.4f}")
    if 'gru' in results:
        print(f"  GRU лучшая эпоха: {results['gru']['best_epoch'] + 1}")
        print(f"  GRU Test Perplexity: {results['gru']['test']['ppl']:.4f}")
    if 'bilstm' in results:
        print(f"  BiLSTM лучшая эпоха: {results['bilstm']['best_epoch'] + 1}")
        print(f"  BiLSTM Test Perplexity: {results['bilstm']['test']['ppl']:.4f}")
    
    print(f"\n  Графики: {PLOTS_DIR}")
    print(f"  Модели: {MODELS_DIR}")
    print(f"  Отчёт: {REPORTS_DIR}")


if __name__ == "__main__":
    main()