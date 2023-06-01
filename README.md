# Multi-vs-single-task-classifiers

Обучить классификатор в режиме Multi-task Learning любым методом на следующих задачах GLUE - Cola, SST-2, MRPC. Сравнить с наивным подходом к обучению на все задачи сразу и с отдельной моделью под каждую из задач.

Проект можно посмотреть на [w&b](https://wandb.ai/gusevski/Multi-vs-single-task-classifiers?workspace=user-gusevski)

Install `pip install -r requirements.txt -q`

# Single task classifier 

Запуск:

`python transformer_glue.py`

Пример запуска с параметрами hydra config:

`python transformer_glue.py model="bert-base-uncased" device="cuda:1" TRAINING_ARGS.num_train_epochs=4 TRAINING_ARGS.seed=41`


На основе Config можно настроить следующие параметры:
- выбрать одну из предложенных задач "sst2", "mrpc", "cola"
- выбрать модель
- зафиксировать seed
- определить количество эпох
- настроить устройство для запуска
- log_file - csv с метриками
- логировать в w&b

На выходе 2 папке с логами wandb и outputs, куда сохраняется лог hydra и чекпоинты моделей

Таблица ниже показывает полученное accuracy на тестовой выборке в процентах, усреднённое по трём запускам c разными seed-ами.

|model|cola|mrpc|sst2|
|---|---|---|---|
|**roberta-base**|82\.74|88\.48|93\.27|
|**bert-base-uncased**|81\.14|84\.31|92\.31|

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

* * Адатируем реализацию [MT_BERT](https://github.com/ABaldrati/MT-BERT), актуализировав библиотеки,дополним hydra, w&b, сохраним checkpoint-ы

Запуск:

`python mt_transformer_glue.py`

Пример запуска с параметрами hydra config:

`python mt_transformer_glue.py seed=40 epochs=3`


Таблица ниже показывает полученное accuracy на тестовой выборке в процентах, усреднённое по трём запускам c разными seed-ами.


|model|cola_mrpc_sst2 accuracy|cola_mrpc_sst2 f1-score|
|---|---|---|
|**mutlitask-bert-base-uncased**|82\.75|88\.37|