import xml.etree.ElementTree as ET
import random
import json
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt


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
    root = tk.Tk()
    root.title("Таблица продаж диванов")

    columns = ["Модель", "Бренд", "Цена"] + [f"Месяц_{i + 1}" for i in range(12)]

    tree = ttk.Treeview(root, columns=columns, show='headings')
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor='center')

    for sofa in data:
        row = [sofa["model"], sofa["brand"], sofa["price"]] + sofa["sales"]
        tree.insert("", tk.END, values=row)

    tree.pack(expand=True, fill='both')
    root.mainloop()


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
