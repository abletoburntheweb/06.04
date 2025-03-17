import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import os
import re
import requests
# функция для загрузки документа по ссылке из гугл драйв
def load_csv_from_google_drive(url: str) -> pd.DataFrame:
    # Извлекаем идентификатор файла из URL
    match_ = re.search('/d/([a-zA-Z0-9-_]+)', url)
    if match_ is None:
        raise ValueError('Invalid Google Drive URL')
    file_id = match_.group(1)
    # Собираем ссылку для скачивания CSV-файла
    download_url = f'https://drive.google.com/uc?id={file_id}'
    # Читаем CSV-файл в DataFrame
    df = pd.read_csv(download_url)
    return df


new_mays_df = load_csv_from_google_drive(
    'https://drive.google.com/file/d/1XthB1LqCYcBFk_-trZ3WbpIWwvB4fC5-/view?usp=drive_link')

new_mays_df.head()
new_mays_df.info()
# посчитаем коэффициент корреляции для всего датафрейма
# округлим значение до сотых
# получается корреляционная матрица
corr_matrix = new_mays_df.corr().round(2)
corr_matrix
# для наглядности построим тепловую карту
import seaborn as sns
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
plt.show()
# отберем признаки с наиболее высокой корреляцией с целевой переменной
# и поместим их в переменную X
X = new_mays_df[['Po', 'U', 'N', 'VV', 'Pa']]
y = new_mays_df['W1']
# импортируем необходимый модуль
from sklearn.model_selection import train_test_split

# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости результата
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)
# импортируем стохастический градиентный спуск из модуля sklearn.linear_model
from sklearn.linear_model import SGDClassifier

# создадим объект этого класса и запишем его в переменную model
model = SGDClassifier(alpha=0.001, random_state = 42)

# обучим нашу модель
model.fit(X_train, y_train)

# выполним предсказание класса на тестовой выборке
y_pred = model.predict(X_test)
# построим матрицу ошибок
from sklearn.metrics import confusion_matrix

# передадим ей тестовые и прогнозные значения
model_matrix = confusion_matrix(y_test, y_pred)

# для удобства создадим датафрейм
model_matrix_df = pd.DataFrame(model_matrix)
model_matrix_df
from sklearn.metrics import accuracy_score
model_accuracy = accuracy_score(y_test, y_pred)
round(model_accuracy, 2)