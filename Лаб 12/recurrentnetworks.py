import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Загружаем текстовый файл и приводим его к нижнему регистру
text = open("sample_data/text.txt", "r", encoding="utf-8").read().lower()

# Создаем список уникальных символов и два словаря для их индексации
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Устанавливаем длину последовательности и шаг для создания обучающих данных
maxlen = 120  # Можно выбрать значение между 100 и 120
step = 10
sequences = []
next_chars = []

# Создаем последовательности и следующие символы для обучения
for i in range(0, len(text) - maxlen, step):
    sequences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# Инициализация массивов для входных данных (X) и меток (y)
X = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

# Заполняем X и y данными: X будет представлять собой закодированные последовательности, y - следующие символы
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Создаем модель с четырьмя слоями LSTM и полносвязным слоем на выходе
model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars)), return_sequences=True))  # Первый слой LSTM
model.add(LSTM(256, return_sequences=True))  # Второй слой LSTM
model.add(LSTM(128, return_sequences=True))  # Третий слой LSTM
model.add(LSTM(64))  # Четвертый слой LSTM
model.add(Dense(len(chars), activation='softmax'))  # Выходной слой

# Компилируем модель с категориальной кросс-энтропией и оптимизатором rmsprop
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Обучаем модель
model.fit(X, y, batch_size=512, epochs=100)

# Функция для генерации текста на основе начального текста
def generate_text(seed_text, length=400):
    generated = seed_text
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        seed_text = seed_text[1:] + next_char
        generated += next_char
    return generated

# Выбираем случайный стартовый индекс для генерации и выводим сгенерированный текст
start_index = np.random.randint(0, len(text) - maxlen - 1)
seed_text = text[start_index: start_index + maxlen]
print(generate_text(seed_text))
