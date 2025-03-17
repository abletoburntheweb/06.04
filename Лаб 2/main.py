import xml.etree.ElementTree as ET
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
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
    for data in sales_data:
        plt.plot(range(1, 13), data['sales'], label=data['model'])

    plt.xlabel('Месяцы')
    plt.ylabel('Продажи')
    plt.title('Динамика продаж моделей диванов')
    plt.legend()
    plt.show()


def display_table(data):
    df = pd.DataFrame(data)

    sales_columns = pd.DataFrame(df["sales"].to_list(), columns=[f"Месяц_{i + 1}" for i in range(12)])
    df = df.drop(columns=["sales"]).join(sales_columns)

    display(df)


def main():
    xml_file = 'sofas.xml'
    json_file = 'sofas_sales.json'

    sofas_data = read_xml(xml_file)
    sales_data = generate_sales_data(sofas_data)
    write_json(json_file, sales_data)
    visualize_data(sales_data)
    display_table(sales_data)


if __name__ == "__main__":
    main()
