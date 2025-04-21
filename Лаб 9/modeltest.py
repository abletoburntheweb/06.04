import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_2d_separator(classifier, X, title=""):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.show()


X, y = make_moons(noise=0.3, random_state=23)

print('Координаты точек:')
print(X[:10])
print('Метки класса:')
print(y[:10])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Сгенерированные данные")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=23)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")
plt.title("Обучающая выборка")
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
plt.title("Тестовая выборка")
plt.show()


# === 3. Функция для тестирования модели ===
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    print(f"\n{model_name}")
    print("Предсказания модели:", prediction)
    print("Истинные значения:", y_test)
    print("Матрица ошибок:\n", confusion_matrix(y_test, prediction))
    print("Оценка точности:", accuracy_score(y_test, prediction))

    plot_2d_separator(model, X, title=f"{model_name}")



# k-NN (1, 3, 5, 9)
for k in [1, 3, 5, 9]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    evaluate_model(knn, f"k-NN (k={k})")

# Random Forest (5, 10, 15, 20, 50)
for n in [5, 10, 15, 20, 50]:
    rf = RandomForestClassifier(n_estimators=n, random_state=23)
    evaluate_model(rf, f"Random Forest (n={n})")

# Наивный Байес
nb = GaussianNB()
evaluate_model(nb, "Наивный Байес")
