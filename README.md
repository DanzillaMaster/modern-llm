# Modern LLM

Домашние задания курса по современным языковым моделям (LLM, СПбГУ, 2026). Каждое задание лежит в своей папке и запускается независимо: свой `pyproject.toml`, свой `uv.lock`, свой README с разбором.

## Задания

| Папка | Задание | Что сделано |
|---|---|---|
| [`homework-01/`](homework-01/) | Llama 3 с нуля | архитектура Llama 3 на PyTorch (RMSNorm, RoPE, GQA, SwiGLU) и цикл обучения на Tiny Shakespeare |
| [`homework-02/`](homework-02/) | FSDP | обучение Llama из ДЗ1 через PyTorch FSDP2 с шардированием по блокам трансформера |

## Запуск

Зависимостями управляет [uv](https://docs.astral.sh/uv/). Из папки нужного задания:

```bash
uv sync
uv run train.py
```

Подробности по каждому заданию лежат в его README.

## Структура

```
modern-llm/
├── homework-01/
│   ├── model.py         модель
│   ├── train.py         данные, обучение, логирование, генерация
│   ├── config.json      параметры модели и обучения
│   ├── data/            датасет (скачивается автоматически)
│   └── outputs/         графики, логи TensorBoard, чекпоинт, сэмплы
└── homework-02/
    ├── llama.zip        model.py и train.py из ДЗ1 без изменений
    ├── train_fsdp.py    FSDP-обучение (запуск через torchrun)
    ├── config.json      параметры модели и обучения
    └── outputs/         графики, логи TensorBoard, чекпоинт, сэмплы
```

Датасеты и чекпоинты в репозиторий не попадают (см. `.gitignore`), логи TensorBoard лежат вместе с графиками.
