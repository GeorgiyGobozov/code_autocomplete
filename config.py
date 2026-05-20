# config.py
import os

# ============================================
# ПУТИ
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets", "code_autocomplete")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# Создание директорий
for dir_path in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# ГИПЕРПАРАМЕТРЫ
# ============================================

# Общие параметры
SEQUENCE_LENGTH = 128     # T - длина окна
BATCH_SIZE = 128          # B - размер батча
EMBEDDING_DIM = 24        # d - размерность эмбеддингов
HIDDEN_DIM = 96           # H - размер скрытого состояния

# Параметры Adam
LEARNING_RATE = 2e-3
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# Gradient clipping
CLIP_NORM = 5.0

# Обучение
MAX_EPOCHS = 20
PATIENCE = 6              # Early stopping patience
STEPS_PER_EPOCH = 120     # Количество батчей на эпоху
EVAL_BATCHES = 20         # Количество батчей для валидации

# LSTM конфиг
LSTM_CONFIG = {
    'seq_length': SEQUENCE_LENGTH,
    'batch_size': BATCH_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'learning_rate': LEARNING_RATE,
    'beta1': BETA1,
    'beta2': BETA2,
    'eps': EPS,
    'clip_norm': CLIP_NORM,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'steps_per_epoch': STEPS_PER_EPOCH,
    'eval_batches': EVAL_BATCHES
}

# GRU конфиг
GRU_CONFIG = {
    'seq_length': SEQUENCE_LENGTH,
    'batch_size': BATCH_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'learning_rate': LEARNING_RATE,
    'beta1': BETA1,
    'beta2': BETA2,
    'eps': EPS,
    'clip_norm': CLIP_NORM,
    'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE,
    'steps_per_epoch': STEPS_PER_EPOCH,
    'eval_batches': EVAL_BATCHES
}

# BiLSTM конфиг (для дополнительного задания)
BILSTM_CONFIG = {
    'seq_length': 128,       # T - длина окна
    'batch_size': 64,        # B - меньше из-за большего количества параметров
    'embedding_dim': 24,     # d - размерность эмбеддингов
    'hidden_dim': 64,        # H - меньше, т.к. после конкатенации будет 128
    'learning_rate': 2e-3,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-8,
    'clip_norm': 5.0,
    'max_epochs': 20,
    'patience': 6,
    'steps_per_epoch': 120,
    'eval_batches': 20
}

# ============================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================

# Язык и вариант (ИЗМЕНИТЬ ПОД СВОЙ ВАРИАНТ!)
LANGUAGE = "python"        # "python", "cpp" или "java"
VARIANT = "variant_01"     # Номер варианта (variant_01, variant_02, ...)

# Префиксы для демонстрации автодополнения
DEMO_PREFIXES = [
    "def ",
    "if ",
    "for i in ",
    "while ",
    "class ",
    "return ",
    "import ",
    "print(",
    "self.",
    "try:"
]

# Для других языков можно добавить специфические префиксы
LANGUAGE_PREFIXES = {
    "python": DEMO_PREFIXES,
    "cpp": [
        "int ",
        "void ",
        "if ",
        "for ",
        "while ",
        "return ",
        "#include ",
        "class ",
        "public:",
        "template<"
    ],
    "java": [
        "public ",
        "private ",
        "class ",
        "if ",
        "for ",
        "while ",
        "return ",
        "System.",
        "import ",
        "@Override"
    ]
}

# Функция для получения полного пути к данным
def get_data_path(language=None, variant=None):
    if language is None:
        language = LANGUAGE
    if variant is None:
        variant = VARIANT
    return os.path.join(DATASETS_DIR, language, variant, "data_char")

# Функция для получения префиксов под язык
def get_demo_prefixes(language=None):
    if language is None:
        language = LANGUAGE
    return LANGUAGE_PREFIXES.get(language, DEMO_PREFIXES)

