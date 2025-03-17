# Импортируем модуль random для выполнения операций с генерацией случайных чисел и прочими случайными операциями.
import random

# Импортируем библиотеку numpy (сокращение от "Numerical Python") для эффективной работы с массивами и выполнения математических операций на них.
import numpy as np

# Импортируем библиотеку pandas для работы с данными в виде таблиц и датафреймов, предоставляя удобные инструменты для анализа и манипуляций данными.
import pandas as pd

# Импортируем библиотеку matplotlib.pyplot для создания графиков и визуализации данных, предоставляя множество функций для построения различных видов графиков.
import matplotlib.pyplot as plt

# Импортируем модуль warnings для управления предупреждениями во время выполнения программы.
# В этом фрагменте кода устанавливается игнорирование предупреждений типа FutureWarning, что может быть полезным для скрытия определенных сообщений о будущих изменениях в библиотеках.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Product:
    """
    Класс Product представляет товар с указанным именем и количеством (по умолчанию None).
    """

    def __init__(self, name, count=None):
        self.name = name
        if count is not None:
            self.count = count
        else:
            min_rand = random.randint(250, 500)
            delt_rand = 100
            self.count = np.random.normal(min_rand, delt_rand, 12)
            # подробнее: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
            # и тут: https://miptstats.github.io/courses/python/07_random.html

            # округление
            self.count = np.round(self.count, 0)

    def __str__(self):
        """
        Возвращает строковое представление товара, включая его имя и количество за каждый из 12 месяцев.
        """
        return f'{self.name}: {[i for i in self.count]}'

    def __repr__(self):
        """
        Возвращает строковое представление товара, включая его имя и количество за каждый из 12 месяцев.
        """
        return f'{self.name}: {[i for i in self.count]}'

    def to_dict(self):
        """
        Преобразует товар в словарь, где ключ - имя товара, а значение - список количества за каждый из 12 месяцев.
        """
        return {
            self.name: self.count
        }

    def sum(self):
        """
        Возвращает общее количество проданного товара за 12 месяцев.
        """
        return sum(self.count)

    def avg(self):
        """
        Возвращает среднее количество проданного товара за 12 месяцев.
        """
        return round(sum(self.count) / len(self.count), 0)

    def msd(self):
        """
        Возвращает среднеквадратичное отклонение (СКО) количества товара за 12 месяцев.
        """
        avg_value = self.avg()
        upper_value = sum([(v - avg_value) ** 2 for v in self.count])
        msd_square = upper_value / (len(self.count) - 1)
        return round(msd_square ** 0.5, 0)

    # Создаем список товаров.
    products = [
        Product(name='Футболки'),
        Product(name='Джинсы'),
        Product(name='Платья'),
        Product(name='Пальто'),
        Product(name='Шорты'),
        Product(name='Юбки'),
        Product(name='Рубашки'),
        Product(name='Свитера'),
        Product(name='Брюки'),
        Product(name='Жакеты')
    ]
    # print(products)
    products

    def convert_list_products_to_dict(p_list: list):
        """
        Конвертирует лист продуктов в словарь для визуализации.
        """
        result = {}
        for p in p_list:
            result[p.name] = p.count
        return result

    df = pd.DataFrame(convert_list_products_to_dict(products))
    # print(df)
    df

    # Покажем количество товаров на графике, отображая данные для каждого товара.
    for product in products:
        plt.plot([i for i in range(12)], product.count, label=product.name)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

    # Вычислим средние значения для каждого товара из списка и сохраним их в список p0.
    p0 = [product.avg() for product in products]

    # Создадим список строк, в которых каждая строка содержит имя продукта и его среднее значение,
    # используя списки p0 и products в соответствии с итерацией через zip().
    # print([f'{p.name}: {p_avg}' for p_avg, p in zip(p0, products)])
    [f'{p.name}: {p_avg}' for p_avg, p in zip(p0, products)]

    # Вычислим значения СКО (среднеквадратичного отклонения) для каждого товара в списке products.
    msd_products = [product.msd() for product in products]

    # Создадим список строк, в которых каждая строка содержит имя товара и его СКО значение,
    # используя списки msd_products и products в соответствии с итерацией через zip().
    # print([f'{product.name}: {msd_value}' for msd_value, product in zip(msd_products, products)])
    [f'{product.name}: {msd_value}' for msd_value, product in zip(msd_products, products)]

    # Сгенерируем предсказанные значения, добавляя к средним значениям p0 случайный шум
    # с нормальным распределением. Это позволит смоделировать случайную изменчивость данных.
    predict_values = np.round(p0 + np.random.normal(0, msd_products, len(msd_products)), 1)
    # print(predict_values)
    predict_values

    # Обновляем значения 'count' для продуктов в соответствии с предсказанными значениями 'predict_values'.
    for product, predict_value in zip(products, predict_values):
        product.count = np.append(product.count, predict_value)
    # print(products)
    products
    # Создаем DataFrame (таблицу) на основе словаря, полученного из списка продуктов 'products'
    # с помощью функции 'convert_list_products_to_dict'.
    df = pd.DataFrame(convert_list_products_to_dict(products))
    # print(df)
    df

    for product in products:
        plt.plot([i for i in range(13)], product.count, label=product.name)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()

    # Преобразуем списки p0 и msd_products в массивы NumPy для выполнения операций с массивами.
    p0_np = np.array(p0)
    msd_np = np.array(msd_products)

    # Создаем DataFrame 'products_df' на основе словаря, полученного из списка продуктов 'products'.
    products_df = pd.DataFrame(convert_list_products_to_dict(products))

    # Вычисляем 'condition_one', сравнивая каждое значение в 'products_df' с условием.
    # Условие считается истинным, если разница между значением 'products_df' и 'p0_np' меньше 2-х раз 'msd_np'.
    condition_one = products_df - p0_np < (2 * msd_np)
    condition_one

    # Вызываем метод '.all()' для DataFrame 'condition_one', чтобы проверить,
    # выполняется ли условие ('True') для всех элементов в DataFrame.
    condition_one.all()