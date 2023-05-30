# Multi-vs-single-task-classifiers

Install `pip install -r requirements.txt -q`

# Single task classifier 
Запуск с использованием конфига yaml с помощью hydra

`python transformer_glue.py`
Проект можно посмотреть на [w&b](https://wandb.ai/gusevski/Multi-vs-single-task-classifiers?workspace=user-gusevski)

На основе Config можно настроить следующие параметры:
- выбрать одну из предложенных задач "sst2", "mrpc", "cola"
- выбрать модель
- зафиксировать seed
- определить количество эпох
- настроить устройство для запуска
- log_file - csv с метриками
- логировать в w&b

На выходе 2 папке с логами wandb и outputs, куда сохраняются лог hydra и чекпоинты моделей

# Multi task classifier


