import xml.etree.ElementTree as ET
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
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


def visualize_data(sales_data):
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    plt.figure(figsize=(10, 6))
    for i, data in enumerate(sales_data):
        plt.plot(range(1, 13), data['sales'], label=data['model'], color=colors[i % len(colors)])
        plt.scatter(13, data['forecast'], color=colors[i % len(colors)], label=f"{data['model']} (прогноз)")
    plt.xlabel('Месяцы')
    plt.ylabel('Продажи')
    plt.title('Динамика продаж моделей диванов')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('sales_forecast.png')

    plt.show()


def display_table(data):
    df = pd.DataFrame(data)
    sales_columns = pd.DataFrame(df["sales"].to_list(), columns=[f"Месяц_{i + 1}" for i in range(12)])
    df = df.drop(columns=["sales"]).join(sales_columns)
    df['Прогноз_13_месяц'] = df['forecast']
    display(df)


def forecast_sales(sales_data):
    for sofa in sales_data:
        last_three_months = sofa['sales'][-3:]
        sofa['forecast'] = sum(last_three_months) / len(last_three_months)
    return sales_data


def apply_expert_rules(sales_data):
    for sofa in sales_data:
        december_sales = sofa['sales'][11]
        average_sales = sum(sofa['sales']) / len(sofa['sales'])

        if december_sales > average_sales:
            sofa['forecast'] *= 1.1

        average_price = sum([s['price'] for s in sales_data]) / len(sales_data)
        if sofa['price'] > average_price:
            sofa['forecast'] *= 0.95

    return sales_data


def calculate_correlation_matrix(sales_data):
    sales_df = pd.DataFrame([sofa['sales'] for sofa in sales_data])
    corr_matrix = sales_df.corr().round(2)
    return corr_matrix


def plot_heatmap(corr_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Корреляционная матрица продаж")
    plt.tight_layout()

    plt.savefig('heatmap.png')

    plt.show()


def evaluate_forecast(sales_data):
    actual_sales = [sofa['sales'][11] for sofa in sales_data]
    predicted_sales = [sofa['forecast'] for sofa in sales_data]

    mse = mean_squared_error(actual_sales, predicted_sales)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_sales, predicted_sales)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')


def analyze_seasonality(sales_data):
    sales_df = pd.DataFrame([sofa['sales'] for sofa in sales_data])
    monthly_avg_sales = sales_df.mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 13), monthly_avg_sales, marker='o', linestyle='-', color='b')
    plt.xlabel('Месяцы')
    plt.ylabel('Средние продажи')
    plt.title('Сезонность продаж')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('seasonality.png')

    plt.show()


def main():
    xml_file = 'sofas.xml'
    json_file = 'sofas_sales.json'

    sofas_data = read_xml(xml_file)

    sales_data = generate_sales_data(sofas_data)

    sales_data = forecast_sales(sales_data)

    sales_data = apply_expert_rules(sales_data)

    write_json(json_file, sales_data)

    corr_matrix = calculate_correlation_matrix(sales_data)
    plot_heatmap(corr_matrix)

    visualize_data(sales_data)

    print("   Отображение таблицы данных   ")
    display_table(sales_data)

    print("    Оценка точности прогноза   ")
    evaluate_forecast(sales_data)

    print("  ")
    analyze_seasonality(sales_data)

if __name__ == "__main__":
    main()