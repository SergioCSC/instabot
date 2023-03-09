# instabot
Определение, какие из заданных аккаунтов инстаграма являются ботами, какие бизнес аккаунтами, и какие аккаунтами пользователей.

Использованы сторонние библиотеки:

- https://fasttext.cc для определения языка постов и комментариев. Лицензия Creative Commons Attribution-Share-Alike License 3.0.

- https://github.com/cjhutto/vaderSentiment (часть nltk) для определения эмоциональной окраски постов и комментариев. Лицензия MIT License

&nbsp;
## Инструкция

0. Общая информация
1. Установка
2. Парсинг аккаунтов с инстаграма
3. Подготовка данных для нейросети в виде json
4. Деление на бот/не бот/бизнес с помощью обученной модели
5. Либо обучение модели на размеченных аккаунтах
6. Известные проблемы



&nbsp;
## Общая информация


Все промежуточные данные попадают в папку *data* кроме одного исключения: при подготовке данных для обучения json складываются в папку *learning_datasets*.

В папке *data/samples* есть подпапки с примерами исходных данных и данных промежуточных этапов.

В файле *config.py* настройки. Например,

```python
SLOW_MODE = 0
``` 
-- быстрый режим обучения (~5 минут на размеченных аккаунтах из *learning_datasets.zip*)
```python
SLOW_MODE = 1
```
-- средний режим обучения (с определением тональности постов на русском и английском, ~30 минут)
```python
SLOW_MODE = 2
``` 
-- медленный режим обучения (~6 часов, используется библиотека *huggingface*, с определением тональности постов на 10 основных языках)


Остальные важные настройки снабжены комментариями в самом файле.
Также есть настройки парсинга (*parser_im/parse.py*) в файле *parser_im/parser_cfg.py*



&nbsp;
## Установка


Что надо иметь:

* Linux или Windows. Проверял на на Windows 10 и на Ubuntu 20.04 (внутри WSL).
* python 3.9. Не 3.10, т.к. либа определения языка текста fasttext не устанавливается на 3.10
* минимум 4 гб свободного места на диске, 4 гб оперативной памяти
* git
* авторизация на сервере
* API ключ на сайте *parser.im*. Его нужно положить в переменную KEY в файл *parser_im/parser_cfg.py* 
* (желательно) архив *learning_datasets.zip* -- набор json с чуть менее 2000 размеченных аккаунтов и постов самых разных ботов, не ботов и бизнес. Либо просто список размеченных названий аккаунтов, как в файле *data/samples/input_for_parse/accounts_marked.txt* и готовность ждать, пока *parser.im* напарсит достаточное для обучения (желательно >1000) количество аккаунтов и их постов из этого списка.


Установка в Linux:

```console
mkdirs /create/path/to/project/
cd /create/path/to/project/
git clone ssh://instal@84.252.139.114/home/instal/instabot
cd instabot
git checkout master
python -m venv env  # be sure it's python 3.9, not python 3.10
source env/bin/activate  
python -m pip install -r requirements.txt
```


Установка в Windows (из-под PowerShell):

```console
mkdirs create\path\to\project\
cd create\path\to\project\
git clone ssh://instal@84.252.139.114/home/instal/instabot
cd instabot
git checkout master
python -m venv env  # be sure it's python 3.9, not python 3.10

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\env\Scripts\activate.ps1
python -m pip install -r requirements.txt
```



&nbsp;
## Парсинг аккаунтов с инстаграма

	Используемые данные:
	    - сайт *parser.im*
	    - файл списка аккаунтов или размеченного списка аккаунтов

	Выходные данные:
            - data/accounts.txt
            - data/posts.txt


Cписок аккаунтов или, в случае обучения нейросети, размеченный список аккаунтов подаются на вход *parse.py*, он использует сайт *parser.im* для скачивания с инстаграма признаков и постов по каждому аккаунту (на выходе файлы *data/accounts.txt* и *data/posts.txt*). *parse.py* может работать много часов, постоянно опрашивая *parser.im* и почти никогда не скачивает всё, так уж работает *parser.im*. Поэтому можно прервать в любой момент *parse.py* и довольствоваться тем, что насобиралось.

Пример команды:        
```console
python parser_im/parse.py data/samples/input_for_parse/accounts_non_marked.txt
```

Либо (для обучения):   
```console
python parser_im/parse.py data/samples/input_for_parse/accounts_marked.txt
```


&nbsp;
## Подготовка данных для нейросети в виде json

    Используемые данные:
        - data/accounts.txt
        - data/posts.txt
        - (в случае обучения) исходный размеченный файл списка аккаунтов

    Выходные данные:
        - data/accounts_with_posts.json (в случае инференса)
        - learning_datasets/accounts_with_posts.json (в случае обучения)


    (Если parse.py так и не создал файлы data/accounts.txt и data/posts.txt из-за пробем с parser.im, можно скопировать их в data/ из data/samples/output_of_parse_and_input_for_prepare)


    Инференс. Полученные c инстаграма признаки и посты в виде файлов data/accounts.txt и data/posts.txt объединяются по аккаунтам в файл data/accounts_with_posts.json, пригодный для нейросети. В нём у всех аккаунтов ключ bot будет установлен в -1 (то есть, неразмеченный аккаунт).

    Пример команды: 

    python parser_im/prepare.py


    3б: Обучение. Если нам нужно создать размеченный json (в целях обучения нейросети), то нужно дополнительно указать исходный файл (т.е. тот, который подавался на вход parse.py) списка аккаунтов, он должен быть размеченным (см. пример data/samples/input_for_parse/accounts_marked.txt). В этом случае json будет помещён не в папку data, а в папку learning_datasets. Также, в случае обучения, если такой файл уже существует -- будет создан новый с названием типа accounts_with_posts_1.json

    Пример команды: 

python parser_im/prepare.py data/samples/input_for_parse/accounts_marked.txt



&nbsp;
## Инференс (деление на бот/не бот/бизнес с помощью обученной модели)

	Используемые данные:
	    - data/accounts_with_posts.json
	    - model/features.pickle
	    - model/depended_features.pickle
	    - model/model.pt
	    - папка third_party_models

	Выходные данные:
            - папка third_party_models
	    - model/test_dataframe.h5
	    - результат инференса (бот/не бот/бизнес) в консоли


Пример команды: 
```console
python inference.py
```

Эта команда запускает скрипт (*collect.py*) сбора и очистки признаков из *data/accounts_with_posts.json*, используя только признаки, которые использовались при обучении (из файла *model/features.pickle*). Использование некоторых признаков зависит не только от конкретного аккаунта, но и от других в обучающей выборке -- данные о том, как их использовать сохранены в *model/depended_features.pickle*.

При первом запуске в папку *third_party_models* скачиваются сторонние модели (~1 Гб) для распознавания языка текста, тональности и т.д. 

Очищенные признаки сохраняются в тестовый датафрейм *model/test_dataframe.h5*. Затем запускается собственно инференс, он использует этот тестовый датафрейм *model/test_dataframe.h5*, а также модель *model/model.pt*  На выходе получаются разметка модели (бот/не бот/бизнес) по оценке модели в колонке **dectected_bot**:

0 -- не бот (человек)
1 -- бот
2 -- бизнес аккаунт

В колонке **bot** везде стоит -1, это значит, что исходные данные не размечены.


&nbsp;
## Обучение модели на размеченных аккаунтах

Если же нам надо обучить модель на новых данных (включая или нет данные прошлых парсингов инстаграма -- модель будет обучаться на всех json из папки *learning_datasets*), то четыре шага:

- смотрим папку *learning_datasets* и добавляем/удаляем туда json файлы, на которых мы хотим проводить обучение. Если мы туда сами ничего не добавляли, то там лежит только файл *accounts_with_posts.json* (или несколько таких с названиями типа *accounts_with_posts_1.json*, если парсинг размеченных аккаунтов выполнялся несколько раз). Рекомендуется распаковать в эту папку архив *learning_datasets.zip*, он содержит более 1000 размеченных аккаунтов и постов самых разных ботов, не ботов и бизнес.

&nbsp;
- сбор данных в датафрейм, очистка:

        Используемые данные:
            - все json из папки learning_datasets
            - папка third_party_models

        Выходные данные:
            - папка third_party_models
            - model/features.pickle
            - model/depended_features.pickle
            - model/train_dataframe.h5
            - model/test_dataframe.h5


    Пример команды: 
    ```console
    python collect.py
    ```

    Эта команда запускает сбор и очистку признаков из всех json файлов в папке *learning_datasets*. Найденные и пригодные к использованию признаки сохраняются в *model/features.pickle*, как использовать некоторые признаки, использование которых зависит от других аккаунтов в обучающей выборке, записывается в *model/depended_features.pickle*. 

    При первом запуске в папку *third_party_models* скачиваются сторонние модели (~1 Гб) для распознавания языка текста, тональности и т.д.

    Очищенные признаки сохраняются в обучающий датафрейм *model/train_dataframe.h5*, часть данных сохраняется в тестовый датафрейм *model/test_dataframe.h5* -- он недоступен на этапе обучения.

&nbsp;
- обучение:

        Используемые данные:
            - model/train_dataframe.h5

        Выходные данные:
            - model/model.pt

    Пример команды без рисования графика:
    ```console
    python learning.py
    ```
    Пример команды с графиком:
    ```console
    python plot.py
    ```

    Проводится обучение на данных из обучающего датасета *model/train_dataframe.h5*, с использованием библиотеки torch и параметров из *config.py*. Полученная модель сохраняется в *model/model.pt*. Если запускаете под ОС с графической оболочкой, можно вместо *learning.py* запустить *plot.py* -- тогда в конце будет выведен график процента правильных ответов на обучающем и валидационном множествах аккаунтов в зависимости от текущей эпохи обучения.

&nbsp;
- оценка качества обучения на тестовом датасете: 

    Входные данные:

        - model/model.pt
        - model/test_dataframe.h5

    Выходные данные (в консоли):

        - процент правильных ответов accuracy
        - таблица с оценкой качества обучения. 
        
        Если данных в тестовом датафрейме оказалось слишком мало (присутствуют размеченные аккаунты не всех типов из бот/не бот/бизнес), то таблица не выводится.

        - результат инференса (бот/не бот/бизнес)

    Пример команды:
    ```console

        python inference.py -l  # или python inference.py --after-learning
    ```

    Пример вывода в консоли:


    ```console

    accuracy: 0.93896484375

                precision    recall  f1-score   support

    human          0.89      0.86      0.87        97
    bots           0.99      0.99      0.99       144
    business       0.92      0.94      0.93       152

    saved_pk               saved_username  bot  detected_bot

    49263250197              vfs_dieerste    2             2
    2197455134                alfiyatorty    2             0
    1662267159                juvidesigns    2             2
    52367178006                  iubzn101    1             1
    50012820036             rozaanikina18    1             1

    ```

    **accuracy** -- процент правильных ответов

    **human** -- для класса 'не бот'

    **bots**  -- для класса 'бот'
    
    **business** -- для класса 'бизнес аккаунты'

    **precision** -- точность
    
    **recall** -- полнота
    
    **f1-score** -- гармоническое среднее точности и полноты
    
    **support** -- количество аккаунтов данного класса среди всех размеченных

    исходная разметка (бот/не бот/бизнес) в колонке **bot**, по оценке модели -- в колонке **dectected_bot**:

    0 -- не бот (человек)
    1 -- бот
    2 -- бизнес аккаунт



&nbsp;
## Известные проблемы



   - Иногда при нарушении порядка запуска скриптов происходит нарушение консистентности данных: количество признаков модели из файла *model/model.pt* не соответствует количеству признаков в файле *model/test_dataframe.h5*, откуда inference.py берёт данные аккаунтов для вывода результата бот/не бот/бизнес. Выглядит это в виде исключения RuntimeError при исполнении *inference.py*:


            RuntimeError: Error(s) in loading state_dict for InstaNet: size mismatch for fc1.weight: copying a param with shape torch.Size([5, 146]) from checkpoint, the shape in current model is torch.Size([5, 143]).
    
        В этом случае нужно повторить обучение.

&nbsp;
- Иногда при установке библиотек:

    ```console
    python -m pip install -r requirements.txt
    ```

    Возникает ошибка, содержащая строку:

    ```console
    Running setup.py install for fasttext did not run successfully.
    ```

    Это происходит, если используется python, отличный от версии 3.9. Нужно установить эту версию питона и (необязательно) установить её в качестве дефолтной. Например, под Linux:

    ```console
    apt install python3.9
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 42
    update-alternatives --config python3
    ```

    и выбрать нужную версию в меню.


&nbsp;
- При запуске 

   ```console
    python.exe plot.py 
   ```
    
    или 

   ```console
    python learning.py 
   ```

    возникает ошибка

    ```console
    OSError: ``../instabot/model/train_dataframe.h5`` does not exist
    ```

    Это означает, что предварительно не был запущен сбор данных: 

   ```console
    python collect.py
   ```

&nbsp;
- При запуске

    ```console
    python collect.py
    ```

    возникает ошибка:

    ```console
        ValueError: No objects to concatenate
    ```

    Это означает, что перед запуском команды не были собраны размеченные аккаунты для обучения (папка *learning_datasets* не содержит ни одного json). Решение: распаковать *learning_datasets.zip* в папку *learning_datasets*, либо выполнить шаги 2 и 3 (случай обучения)


- Иногда при запуске

    ```console
        python.exe inference.py -l
    ```

    таблица оценки качества обучения не выводится, а колонка bot, вместо того, чтобы содержать размеченные значения для аккаунтов, содержит -1. Это происходит, если нарушен порядок запуска скриптов, то есть между запуском 

    ```console
        python collect.py
    ``` 

    и 
    
    ```console
        python inference.py -l 
    ```

    был вызван 

    ```console
        python inference.py  # (без параметров) 
    ```

    и в тестовый датафрейм *model/test_dataframe.h5* попали аккаунты из неразмеченного *data/accounts_with_posts*.json, вместо аккаунтов из размеченных json в папке *learning_datasets*. Решение: пройти этап обучения ещё раз.
