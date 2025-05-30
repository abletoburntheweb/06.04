import xml.etree.ElementTree as ET
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error  
import numpy as np
from IPython.display import display

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
        data['predicted_sales'] = future_sales[0]
        predictions.append(data)
    return predictions

def visualize_data(sales_data):
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(sales_data):
        plt.plot(range(1, 13), data['sales'], label=data['model'], color=colors[i % len(colors)])
        plt.scatter(13, data['predicted_sales'], color=colors[i % len(colors)], label=f"{data['model']} (прогноз)")
    plt.xlabel('Месяцы')
    plt.ylabel('Продажи')
    plt.title('Динамика продаж моделей диванов')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sales_forecast.png')
    plt.show()

def display_table(data):
    rows = []
    for sofa in data:
        row = {
            "Цена": sofa["price"],
            **{f"Месяц_{i + 1}": sale for i, sale in enumerate(sofa["sales"])},
            "Прогноз": sofa["predicted_sales"]
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    display(df)

def evaluate_forecast(sales_data):
    actual_sales = [sofa['sales'][11] for sofa in sales_data]
    predicted_sales = [sofa['predicted_sales'] for sofa in sales_data]
    mse = mean_squared_error(actual_sales, predicted_sales)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_sales, predicted_sales)
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

def main():
    xml_file = 'sofas.xml'
    json_file = 'sofas_sales.json'
    sofas_data = read_xml(xml_file)
    sales_data = generate_sales_data(sofas_data)
    write_json(json_file, sales_data)
    predicted_data = predict_future_sales(sales_data)
    visualize_data(predicted_data)
    print("   Отображение таблицы данных   ")
    display_table(predicted_data)
    print("    Оценка точности прогноза   ")
    evaluate_forecast(predicted_data)

if __name__ == "__main__":
    main()