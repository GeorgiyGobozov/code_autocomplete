# 📚 Code Autocomplete: LSTM, GRU & BiLSTM from Scratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Реализация LSTM, GRU и BiLSTM на чистом NumPy для задачи автодополнения кода**

</div>

---

## 📋 Оглавление

- [Описание проекта](#-описание-проекта)
- [Архитектура](#-архитектура)
- [Структура проекта](#-структура-проекта)
- [Установка](#-установка)
- [Запуск](#-запуск)
- [Результаты](#-результаты)
- [Конфигурация](#-конфигурация)
- [Демонстрация](#-демонстрация)
- [Часто задаваемые вопросы](#-часто-задаваемые-вопросы)

---

## 🎯 Описание проекта

Данный проект представляет собой **полную реализацию с нуля** рекуррентных нейронных сетей для задачи автодополнения кода на символьном уровне.

### Основные возможности:

- ✅ **LSTM** (Long Short-Term Memory) с полным BPTT
- ✅ **GRU** (Gated Recurrent Unit) с полным BPTT
- ✅ **BiLSTM** (Bidirectional LSTM) для Fill-in-the-Middle задачи
- ✅ Adam оптимизатор с gradient clipping
- ✅ Визуализация обучения (perplexity, top-k accuracy)
- ✅ Сохранение/загрузка моделей
- ✅ Генерация автодополнений для произвольных префиксов

### Особенности реализации:

- 🔥 **Только NumPy** - без PyTorch/TensorFlow
- 🔥 **Ручной BPTT** - без autograd
- 🔥 **Полный контроль** - над всеми аспектами обучения

---

## 🏗 Архитектура

### LSTM (Long Short-Term Memory)

```
f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)     # forget gate
i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)     # input gate
C̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)  # candidate cell
o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)     # output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t          # cell state
h_t = o_t ⊙ tanh(C_t)                    # hidden state
```

### GRU (Gated Recurrent Unit)

```
r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)     # reset gate
z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)     # update gate
h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### BiLSTM (Bidirectional LSTM)

- Прямой LSTM: слева направо
- Обратный LSTM: справа налево
- Конкатенация состояний на позиции маски
- Задача: Fill-in-the-Middle (восстановление пропущенного символа)

---

## 📁 Структура проекта

```
code_autocomplete/
│
├── src/
│   ├── models/
│   │   ├── lstm.py           # LSTM модель с BPTT
│   │   ├── gru.py            # GRU модель с BPTT
│   │   └── bilstm.py         # BiLSTM для Fill-in-the-Middle
│   │
│   ├── core/
│   │   ├── optimizer.py      # Adam + gradient clipping
│   │   ├── losses.py         # Функции потерь и метрики
│   │   └── utils.py          # Вспомогательные функции
│   │
│   ├── data/
│   │   └── dataset.py        # Загрузка данных
│   │
│   ├── train.py              # Тренер для LSTM/GRU
│   └── train_bilstm.py       # Тренер для BiLSTM
│
├── datasets/                  # Директория с датасетами
│   └── code_autocomplete/
│       ├── python/
│       │   └── variant_04/
│       │       └── data_char/
│       │           ├── dataset_char.npz
│       │           └── meta.json
│       ├── cpp/
│       └── java/
│
├── results/                   # Результаты экспериментов
│   ├── models/               # Сохранённые модели (.npz)
│   ├── plots/                # Графики обучения (.png)
│   ├── reports/              # Отчёты (.txt)
│   └── logs/                 # Логи обучения
│
├── config.py                  # Конфигурация параметров
├── run.py                     # Главный скрипт запуска
├── requirements.txt           # Зависимости
└── README.md                  # Документация
```

---

## 🚀 Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/code-autocomplete.git
cd code-autocomplete
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Подготовка данных

Поместите файлы датасета в соответствующую директорию:

```
datasets/code_autocomplete/<language>/<variant>/data_char/
├── dataset_char.npz
└── meta.json
```

---

## 🎮 Запуск

### Базовый запуск (LSTM + GRU)

```bash
python run.py --language python --variant variant_04
```

### Запуск с BiLSTM (все 3 модели)

```bash
python run.py --language python --variant variant_04 --bilstm
```

### Только BiLSTM

```bash
python run.py --language python --variant variant_04 --only-bilstm
```

### Другие языки

```bash
# C++
python run.py --language cpp --variant variant_04 --bilstm

# Java
python run.py --language java --variant variant_04 --bilstm
```

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--language` | Язык (python/cpp/java) | python |
| `--variant` | Номер варианта | variant_04 |
| `--bilstm` | Запустить BiLSTM + LSTM/GRU | False |
| `--only-bilstm` | Только BiLSTM | False |
| `--data-dir` | Путь к датасетам | datasets/code_autocomplete |
| `--no-plots` | Не показывать графики | False |

---

## 📊 Результаты (variant_04, Python)

### Сравнительная таблица

| Метрика | LSTM (val) | GRU (val) | LSTM (test) | GRU (test) | BiLSTM (test) |
|---------|------------|-----------|-------------|------------|---------------|
| **Perplexity** | **1.73** | 5.80 | **2.64** | 14.79 | 7.04 |
| **Top-1** | **0.54** | 0.54 | 0.35 | 0.33 | **0.48** |
| **Top-5** | **0.79** | 0.79 | 0.58 | 0.58 | **0.68** |
| **Top-10** | **0.88** | 0.88 | 0.73 | 0.73 | **0.80** |

### Ключевые выводы

- 🏆 **LSTM** - лучшая модель для автодополнения (perplexity 2.64)
- ⚡ **GRU** - быстрая сходимость, но требует early stopping
- 🔄 **BiLSTM** - эффективен для Fill-in-the-Middle (80% top-10)

### Графики обучения

После запуска графики сохраняются в `results/plots/`:

- `python_variant04_lstm.png` - обучение LSTM
- `python_variant04_gru.png` - обучение GRU
- `python_variant04_bilstm.png` - обучение BiLSTM

---

## ⚙️ Конфигурация

### Основные гиперпараметры (`config.py`)

```python
# Размеры
SEQUENCE_LENGTH = 128      # Длина окна
BATCH_SIZE = 128           # Размер батча
EMBEDDING_DIM = 24         # Размерность эмбеддингов
HIDDEN_DIM = 96            # Размер скрытого состояния

# Оптимизация
LEARNING_RATE = 2e-3       # Learning rate
CLIP_NORM = 5.0            # Gradient clipping

# Обучение
MAX_EPOCHS = 20            # Максимум эпох
PATIENCE = 6               # Early stopping patience

# BiLSTM
BILSTM_HIDDEN_DIM = 64     # Меньше (2H = 128)
BILSTM_BATCH_SIZE = 64     # Меньше из-за памяти
```

### Изменение параметров

Отредактируйте `config.py` или используйте переменные окружения:

```bash
set LANGUAGE=python
set VARIANT=variant_04
set BATCH_SIZE=64
python run.py --bilstm
```

---

## 🎨 Демонстрация

### Автодополнение (LSTM)

```python
# Пример вывода
Префикс: 'def '
Top-10 продолжений:
   1. 't' (p = 0.1958)  # def test, def tokenize
   2. 's' (p = 0.0862)  # def self, def save
   3. 'i' (p = 0.0818)  # def init, def int
   4. 'o' (p = 0.0667)  # def object, def open
   5. '=' (p = 0.0538)  # def __init__=
   ...
```

### Fill-in-the-Middle (BiLSTM)

```python
# Пример вывода
Контекст: 'i<MASK>_h_features'
Истинный символ: 'n'
Top-10 предсказаний:
   1. 'n' (p = 0.3238)  # in_h_features ✅
   2. 't' (p = 0.1042)  # it_h_features
   3. 'm' (p = 0.0930)  # im_h_features
   ...
```

---

## ❓ Часто задаваемые вопросы

### Вопрос: Почему GRU показывает нестабильное обучение?

**Ответ:** GRU имеет меньше параметров и вентилей, что делает его более склонным к переобучению на больших датасетах. Рекомендуется использовать early stopping (patience=3-5) и меньший learning rate.

### Вопрос: Как увеличить скорость обучения?

**Ответ:** 
- Уменьшите `SEQUENCE_LENGTH` до 64
- Уменьшите `BATCH_SIZE` (но это может ухудшить качество)
- Используйте меньше эпох (`MAX_EPOCHS=10`)

### Вопрос: BiLSTM требует больше памяти?

**Ответ:** Да, BiLSTM использует два LSTM (прямой и обратный). Рекомендуемые параметры:
- `HIDDEN_DIM=64` (вместо 96)
- `BATCH_SIZE=64` (вместо 128)

### Вопрос: Как использовать обученную модель?

```python
from src.models import LSTMModel
from src.data.dataset import CodeDataset

# Загрузка модели
model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
model.load('results/models/python_variant04_lstm.npz')

# Предсказание
prefix_ids = [dataset.char_to_id(c) for c in "def "]
probs = model.predict_next(prefix_ids)
```

### Вопрос: Как добавить свой датасет?

1. Подготовьте файлы в формате `.npz`:
   - `vocab` - массив Unicode codepoint
   - `train`, `val`, `test` - 1D массивы индексов
2. Создайте `meta.json` с метаданными
3. Поместите в `datasets/code_autocomplete/<lang>/<variant>/data_char/`

---

## 📈 Производительность

### Время обучения (CPU Intel i7, 16GB RAM)

| Модель | Эпоха | Всего (20 эпох) |
|--------|-------|-----------------|
| LSTM | ~90 сек | ~30 мин |
| GRU | ~85 сек | ~28 мин |
| BiLSTM | ~60 сек | ~20 мин |

### Потребление памяти

| Модель | Batch=128 | Batch=64 |
|--------|-----------|----------|
| LSTM | ~2.5 GB | ~1.5 GB |
| GRU | ~2.0 GB | ~1.2 GB |
| BiLSTM | ~3.5 GB | ~2.0 GB |

---
