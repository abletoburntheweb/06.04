from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
import math
X, y = make_classification(n_samples=60,
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           n_clusters_per_class=1,
                           n_classes=2,
                           random_state=9,
                           class_sep=2)
plt.scatter(X[:, 0], X[:, 1])
def update_cluster_centers(X, c):
    centers = np.zeros((2, 2))
    for i in range(1, 3):
        ix = np.where(c == i)
        centers[i - 1, :] = np.mean(X[ix, :], axis=1)
    return centers
mergings = linkage(X, method='ward')
T = fcluster(mergings, 2, criterion='maxclust')
clusters = update_cluster_centers(X, T)
clusters
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.scatter(clusters[:, 0], clusters[:, 1], c='blue')
class SOM:
    def __init__(self, n, c):
        """
        n - количестов атрибутов
        C - количество кластеров
        """
        self.n = n
        self.c = c
        self.a = [0 for _ in range(n)]

    def calculate_a(self, i):
        """
        Вычисление значение шага относительного текущего выбора
        """
        return (50 - i) / 100

    def winner(self, weights, sample):
        """
        Вычисляем выигравший нейрон (вектор) по Евклидову расстоянию
        """
        d0 = 0
        d1 = 0
        for i in range(len(sample)):
            d0 += math.pow((sample[i] - weights[0][i]), 2)
            d1 += math.pow((sample[i] - weights[1][i]), 2)

        if d0 > d1:
            return 0
        else:
            return 1

    def update(self, weights, sample, j):
        """
        Обновляем значение для выигравшего нейрона
        """
        for i in range(len(weights)):
            weights[j][i] = weights[j][i] + self.calculate_a(self.a[j]) * (sample[i] - weights[j][i])

        print(f'\nШаг для {j} кластера = {self.calculate_a(self.a[j])}')
        self.a[j] += 1
        print(f'Веса после обновления:')
        print(weights)

        return weights

    # Обучающая выборка (m, n)
    # m - объем выборки
    # n - количество атрибутов в записи
    np.random.shuffle(X)
    T = X
    m, n = len(T), len(T[0])

    # Обучающие веса (n, C)
    # n - количество атрибутов в записи
    # C - количество кластеров
    C = 2

    weights = np.random.normal(100, 10, size=(n, C)) / 100
    weights
    som = SOM(n, C)
    # som
    for i in range(m):
        sample = T[i]
        J = som.winner(weights, sample)
        weights = som.update(weights, sample, J)
        s = X[0]
        J = som.winner(weights, s)

        print(f"Элемент принадлежит к {J} кластеру, на самом деле к {y[0]} кластеру")
        print("Обученные веса: ")
        print(weights)
        predicted = np.array([som.winner(weights, s) for s in X])
        predicted
        y == predicted
        from sklearn.metrics import accuracy_score

        print(f'Точность кластеризации: {accuracy_score(y, predicted) * 100}%')
