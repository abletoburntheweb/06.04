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
    'https://drive.google.com/file/d/1Cwj_J2lW-_AVudzg3c16RWHkicrcqD3D/view?usp=drive_link')

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
### Задача регресии. Предскажем значения столбца ff10
# отберем признаки с наиболее высокой корреляцией с целевой переменной
# и поместим их в переменную X
X = new_mays_df[['Ff', 'ff3', 'N', 'U']]
y = new_mays_df['ff10']
from sklearn.model_selection import train_test_split

# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)
# размерность обучающей выборки
print(X_train.shape, y_train.shape)

# размерность тестовой выборки
print(X_test.shape, y_test.shape)
# из набора линейных моделей библиотеки sklearn импортируем линейную регрессию
from sklearn.linear_model import LinearRegression
# создадим объект этого класса и запишем в переменную model
model = LinearRegression()

# обучим нашу модель
model.fit(X_train, y_train)
# на основе нескольких независимых переменных (Х) предскажем скоромть порыва ветра (y)
y_pred = model.predict(X_test)

# выведем первые пять значений с помощью диапазона индексов
print(y_pred[:5])
# импортируем модуль метрик
from sklearn import metrics

# выведем корень среднеквадратической ошибки
# сравним тестовые и прогнозные значения цен на жилье
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# посмотрим на еще одну метрику, R2
print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))
### Задача регресии. Предскажем значения столбца T
# отберем признаки с наиболее высокой корреляцией с целевой переменной
# и поместим их в переменную X
X = new_mays_df[['U', 'Po','Td', 'Pa']]
y = new_mays_df['T']
from sklearn.model_selection import train_test_split

# разобьем данные на обучающую и тестовую выборку
# размер тестовой выборки составит 30%
# также зададим точку отсчета для воспроизводимости
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.4,
                                                    random_state = 42)
# из набора линейных моделей библиотеки sklearn импортируем линейную регрессию
from sklearn.linear_model import LinearRegression
# создадим объект этого класса и запишем в переменную model
model = LinearRegression()

# обучим нашу модель
model.fit(X_train, y_train)
# на основе нескольких независимых переменных (Х) предскажем цену на жилье (y)
y_pred = model.predict(X_test)

# выведем первые пять значений с помощью диапазона индексов
print(y_pred[:5])
print(y_test[:5])
# импортируем модуль метрик
from sklearn import metrics
# выведем корень среднеквадратической ошибки
# сравним тестовые и прогнозные значения цен на жилье
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# посмотрим на еще одну метрику, R2
print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))
