import xml.etree.ElementTree as ET
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    sofas_data = []
    for sofa in root.findall('sofa'):
        model = sofa.find('model').text
        brand = sofa.find('brand').text
        price = int(sofa.find('price').text)
        sofas_data.append({"model": model, "brand": brand, "price": price})
    return sofas_data


def generate_sales_data(sofas_data):
    for sofa in sofas_data:
        sales = [random.randint(10, 100) for _ in range(12)]
        sofa["sales"] = sales
    return sofas_data

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def predict_future_sales(sales_data):
    future_months = 1
    predictions = []

    for data in sales_data:
        X = np.array(range(1, 13)).reshape(-1, 1)
        y = np.array(data['sales'])

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.array([[13]])
        future_sales = model.predict(future_X)

        data['predicted_sales'] = future_sales.tolist()
        predictions.append(data)

    return predictions

def visualize_data(sales_data):
    for data in sales_data:
        plt.plot(range(1, 13), data['sales'], label=data['model'])

    plt.xlabel('Месяцы')
    plt.ylabel('Продажи')
    plt.title('Динамика продаж моделей диванов')
    plt.legend()
    plt.show()

def display_table_with_pandas(data):
    rows = []
    for sofa in data:
        row = {
            "Модель": sofa["model"],
            "Бренд": sofa["brand"],
            "Цена": sofa["price"],
            **{f"Месяц {i + 1}": sale for i, sale in enumerate(sofa["sales"])},
            "Прогноз (13-й месяц)": sofa["predicted_sales"][0]  # Только один прогноз
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    display(df)


def main():
    xml_file = 'sofas.xml'
    json_file = 'sofas_sales.json'

    sofas_data = read_xml(xml_file)
    sales_data = generate_sales_data(sofas_data)
    write_json(json_file, sales_data)

    visualize_data(sales_data)
    predicted_data = predict_future_sales(sales_data)
    display_table_with_pandas(predicted_data)
if __name__ == "__main__":
    main()