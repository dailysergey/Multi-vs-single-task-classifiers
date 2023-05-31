# Multi-vs-single-task-classifiers

Install `pip install -r requirements.txt -q`

# Single task classifier 
Запуск с использованием конфига yaml с помощью hydra:

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

На выходе 2 папке с логами wandb и outputs, куда сохраняется лог hydra и чекпоинты моделей

Таблица ниже показывает полученное accuracy на тестовой выборке в процентах, усреднённое по трём запускам.

|model|cola|mrpc|sst2|
|---|---|---|---|
|**roberta-base**|82\.74|88\.48|93\.27|

# Multi task classifier

Пойдем на paper with code и посмотрим, что пишут про Multi Task Learning, какие подходы:
- [7 Apr 2022, A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods](https://paperswithcode.com/paper/a-survey-of-multi-task-learning-in-natural) 
 * * MTL enables shared representations to include features from all tasks, thus
improving the consistency of task-specific decoding in each sub-task. Furthermore, the co-existence
of features from different objectives naturally performs feature crosses, which enables the model to
learn more complex features.
 * * joint training описывают для задач классификации
- [ACL 2019 BAM! Born-Again Multi-Task Networks for Natural Language Understanding](https://paperswithcode.com/paper/bam-born-again-multi-task-networks-for)
- [ACL 2019 Multi-Task Deep Neural Networks for Natural Language Understanding](https://paperswithcode.com/paper/multi-task-deep-neural-networks-for-natural)
Возьмем за основу реализацию данной статьи из [репо](https://github.com/ABaldrati/MT-BERT)