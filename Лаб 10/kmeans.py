from sklearn.datasets import make_blobs
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import pandas as pd

# Генерация данных
X, y = make_blobs(n_samples=100, n_features=2, random_state=68, cluster_std=2, centers=9)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Генерация данных")
plt.show()


# Функция расчета среднего внутрикластерного расстояния
def calculate_avg_intra_cluster_distance(X, labels, metric='euclidean'):
    total_distance = 0
    num_clusters = len(np.unique(labels))
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 1:
            total_distance += np.sum(pairwise_distances(cluster_points, metric=metric))
    return total_distance / (len(X) - num_clusters)


# Функция расчета среднего межкластерного расстояния
def calculate_avg_inter_cluster_distance(X, labels, metric='euclidean'):
    cluster_centers = np.array([np.mean(X[labels == cluster_id], axis=0) for cluster_id in np.unique(labels)])
    return np.sum(pairwise_distances(cluster_centers, metric=metric))


# Иерархическая кластеризация: построение дендрограмм
methods = ['single', 'complete', 'ward']
titles = ["Ближайший сосед", "Дальний сосед", "Метод Уорда"]

for method, title in zip(methods, titles):
    mergings = linkage(X, method=method)
    plt.figure(figsize=(10, 6))
    dendrogram(mergings)
    plt.title(f"Дендрограмма ({title})")
    plt.show()

# Выбор оптимальной дендрограммы (метод Уорда)
mergings_best = linkage(X, method='ward')
threshold = 10
T_hierarchy = fcluster(mergings_best, threshold, 'distance')

# Определение количества кластеров из дендрограммы
num_clusters_hierarchy = len(np.unique(T_hierarchy))

# Отображение разбиения на кластеры и центроиды
plt.scatter(X[:, 0], X[:, 1], c=T_hierarchy, cmap='plasma')
centroids_hierarchy = np.array([X[T_hierarchy == i].mean(axis=0) for i in range(1, num_clusters_hierarchy + 1)])
plt.scatter(centroids_hierarchy[:, 0], centroids_hierarchy[:, 1], c='red', marker='x', s=200, label='Центроиды')
plt.title(f"Иерархическая кластеризация с {num_clusters_hierarchy} кластерами")
plt.legend()
plt.show()

# Расчет метрик для иерархической кластеризации
avg_intra_distance_hierarchy = calculate_avg_intra_cluster_distance(X, T_hierarchy)
avg_inter_distance_hierarchy = calculate_avg_inter_cluster_distance(X, T_hierarchy)

# Метод локтя для k-средних
inertia_values = []
avg_intra_distances = []
avg_inter_distances = []

for k in range(1, 11):  # k от 1 до 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    inertia_values.append(kmeans.inertia_)
    avg_intra_distances.append(calculate_avg_intra_cluster_distance(X, labels))
    avg_inter_distances.append(calculate_avg_inter_cluster_distance(X, labels))

# Построение графика метода локтя
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title("Метод локтя: Определение оптимального числа кластеров")
plt.xlabel("Количество кластеров")
plt.ylabel("Инерция (Сумма квадратов расстояний до центроида)")
plt.show()

# Выбор оптимального числа кластеров
optimal_k = 9

# Кластеризация методом KMeans
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(X)
predictions_kmeans = kmeans_optimal.predict(X)

# Отображение разбиения на кластеры и центроиды
plt.scatter(X[:, 0], X[:, 1], c=predictions_kmeans, cmap='plasma')
plt.scatter(kmeans_optimal.cluster_centers_[:, 0], kmeans_optimal.cluster_centers_[:, 1], c='red', marker='x', s=200,
            label='Центроиды')
plt.title(f"KMeans с {optimal_k} кластерами")
plt.legend()
plt.show()

# Расчет метрик для метода k-средних
avg_intra_distance_kmeans = calculate_avg_intra_cluster_distance(X, predictions_kmeans)
avg_inter_distance_kmeans = calculate_avg_inter_cluster_distance(X, predictions_kmeans)

# Создание сравнительной таблицы
columns = pd.MultiIndex.from_product([['Иерархический метод', 'Метод k-средних'],
                                      ['Количество кластеров',
                                       'Сумма квадратов расстояний до центроида',
                                       'Сумма средних внутрикластерных расстояний',
                                       'Сумма межкластерных расстояний']])
df = pd.DataFrame(columns=columns)

# Добавление значений для иерархического метода
df['Иерархический метод', 'Количество кластеров'] = [num_clusters_hierarchy]
df['Иерархический метод', 'Сумма квадратов расстояний до центроида'] = [mergings_best[-1, 2]]
df['Иерархический метод', 'Сумма средних внутрикластерных расстояний'] = [avg_intra_distance_hierarchy]
df['Иерархический метод', 'Сумма межкластерных расстояний'] = [avg_inter_distance_hierarchy]

# Добавление значений для метода k-средних
df['Метод k-средних', 'Количество кластеров'] = [optimal_k]
df['Метод k-средних', 'Сумма квадратов расстояний до центроида'] = [inertia_values[optimal_k - 1]]
df['Метод k-средних', 'Сумма средних внутрикластерных расстояний'] = [avg_intra_distance_kmeans]
df['Метод k-средних', 'Сумма межкластерных расстояний'] = [avg_inter_distance_kmeans]

# Вывод таблицы
print(df)

# Построение графиков для сравнения методов
plt.figure(figsize=(12, 6))

# Графики для иерархической кластеризации
plt.plot(range(1, 11), avg_intra_distances, marker='o', label='Внутрикластерные расстояния (Иерархический метод)')
plt.plot(range(1, 11), avg_inter_distances, marker='o', label='Межкластерные расстояния (Иерархический метод)')

# Графики для метода k-средних
plt.plot(range(1, 10), [inertia_values[i] for i in range(9)], marker='o', linestyle='dashed',
         label='Сумма квадратов расстояний до центроида (Метод k-средних)')
plt.plot(range(1, 10), avg_intra_distances[:9], marker='o', linestyle='dashed',
         label='Сумма средних внутрикластерных расстояний (Метод k-средних)')
plt.plot(range(1, 10), avg_inter_distances[:9], marker='o', linestyle='dashed',
         label='Сумма межкластерных расстояний (Метод k-средних)')

plt.title("Сравнение методов кластеризации")
plt.xlabel("Количество кластеров")
plt.ylabel("Расстояния")
plt.legend()
plt.show()