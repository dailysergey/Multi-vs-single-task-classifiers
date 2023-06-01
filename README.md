# Multi-vs-single-task-classifiers GLUE - Cola, SST-2, MRPC

Обучить классификатор в режиме Multi-task Learning любым методом на следующих задачах GLUE - Cola, SST-2, MRPC

Проведенные эксперименты можно посмотреть на [w&b](https://wandb.ai/gusevski/Multi-vs-single-task-classifiers?workspace=user-gusevski).

Хорошая практика состоит в том, чтобы разделять модель и все функции в папках "model" и "utils". Однако в этом репозитории я организовал код логически в разных папках для удобства запуска из одного файла.

Install `pip install -r requirements.txt -q`

# Single task classifier 

Запуск:

`python transformer_glue.py`

Пример запуска с параметрами hydra config:

`python transformer_glue.py model="bert-base-uncased" device="cuda:1" TRAINING_ARGS.num_train_epochs=4 TRAINING_ARGS.seed=41`


На основе Config можно настроить следующие гиперпараметры:
- выбрать одну из предложенных задач "sst2", "mrpc", "cola"
- выбрать модель
- зафиксировать seed
- определить количество эпох
- настроить устройство для запуска
- log_file - csv с метриками
- логировать в w&b

Результаты будут сохраняться в двух папких: wandb и outputs, которые содержат логи w&b и hydra соответственно, а также чекпоинты моделей.

Таблица ниже показывает достигнутую точность, усредненную по трем запускам с разными seed-ами на тестовой выборке в процентах.

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

 
Адаптируем реализацию [MT_BERT](https://github.com/ABaldrati/MT-BERT), актуализировав библиотеки, дополним hydra, w&b, сохраним checkpoint-ы.
В [MT_BERT](https://github.com/ABaldrati/MT-BERT) воспроизвели модель на основе статьи [ACL 2019 Multi-Task Deep Neural Networks for Natural Language Understanding](https://paperswithcode.com/paper/multi-task-deep-neural-networks-for-natural). Данная модель (MT-DNN) для изучения представлений в различных задачах NLU. MT-DNN расширяет модель, предложенную в статье [Representation learning using multi-task deep neural networks for semantic classification and information retrieval](https://aclanthology.org/N15-1092), путем включения BERT.

Запуск:

`python mt_transformer_glue.py`

Пример запуска с параметрами hydra config:

`python mt_transformer_glue.py seed=40 epochs=3`


Таблица ниже показывает достгнутое accuracy f1-score на тестовой выборке в процентах, усреднённое по трём запускам c разными seed-ами. Метрики удобно смотреть с помощью табличного вида [w&b](https://wandb.ai/gusevski/Multi-vs-single-task-classifiers/table?workspace=user-gusevski)


|model|cola_mrpc_sst2 accuracy|cola_mrpc_sst2 f1-score|
|---|---|---|
|**mutlitask-bert-base-uncased**|82\.75|88\.37|


# Выводы

Анализируя качество моделей, обученных с применением multitask learning для задач понимания естественного языка (NLU) и моделей, обученных с применением single task learning, можно сделать следующие выводы:

1. Преимущество multitask learning: Модели, обученные с использованием multitask learning, имеют потенциал для достижения лучшего обобщения и общей производительности. При обучении модели на нескольких задачах одновременно, она может совместно использовать общие знания и структуры, что может привести к улучшению ее способности к пониманию естественного языка.
2. Улучшение обобщения: Multitask learning может помочь модели лучше обобщать знания, полученные из обучающих данных, на новые примеры или задачи, которые модель не видела во время обучения. Общие признаки и знания, полученные из решения различных задач, могут способствовать более эффективному и точному пониманию новых текстовых данных.
3. Потенциальные ограничения: Однако в случае multitask learning возможны и некоторые ограничения. Например, сложность и разнообразие задач могут сказаться на производительности модели. Если задачи сильно отличаются друг от друга или имеют разные требования к обучающим данным, это может затруднить обучение модели. Некоторые задачи могут доминировать в обучении, что может привести к ухудшению производительности на других задачах.
4. Подход single task learning: Модели, обученные с использованием single task learning, могут быть специализированы для конкретных задач и обеспечивать высокое качество в рамках своей узкой области. Они могут быть предпочтительными, если у вас есть одна основная задача, которую нужно решить без сильной связи с другими задачами.

В целом, выбор между multitask learning и single task learning зависит от конкретных требований и контекста вашей задачи. Multitask learning может быть полезным для улучшения обобщения и производительности модели на нескольких задачах NLU, тогда как single task learning может быть предпочтительным, когда требуется специализация на конкретной задаче без учета других.