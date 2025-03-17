import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from IPython.display import display


def load_data(file_path):
    print("Загрузка данных о продажах")
    df = pd.read_csv(file_path)
    print(f"Данные загружены. Размерность: {df.shape}")
    return df


def preprocess_data(df):

    df = df.dropna()
    print(f"После удаления пропущенных значений: {df.shape}")

    df['SalesGrowth'] = (df['Sales_Month_12'] > df['Sales_Month_11']).astype(int)

    features = ['price', 'Sales_Month_9', 'Sales_Month_10', 'Sales_Month_11']
    X = df[features]
    y = df['SalesGrowth']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Размер обучающей выборки: {X_train.shape[0]}, тестовой: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Данные масштабированы.")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("Обучение модели логистической регрессии с SGD")
    model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Обучение завершено.")
    return model


def evaluate_model(model, X_test, y_test):
    print("Оценка точности модели")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1, target_names=["Продажи упали", "Продажи выросли"])

    print(f'Точность модели: {accuracy:.4f}')
    print('Матрица ошибок:')
    print(cm)

    print('Отчет по классификации:')
    report_lines = report.split("\n")
    for line in report_lines:
        line = line.replace("precision", "Точность")
        line = line.replace("recall", "Полнота")
        line = line.replace("f1-score", "F1-мера")
        line = line.replace("support", "Поддержка")
        line = line.replace("accuracy", "Общая точность")
        line = line.replace("macro avg", "Среднее по классам")
        line = line.replace("weighted avg", "Средневзвешенное среднее")
        print(line)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Продажи упали', 'Продажи выросли'],
                yticklabels=['Продажи упали', 'Продажи выросли'])
    plt.xlabel('Предсказание')
    plt.ylabel('Реальность')
    plt.title('Матрица ошибок')
    plt.show()


def main():
    file_path = 'sales_data.csv'
    df = load_data(file_path)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
