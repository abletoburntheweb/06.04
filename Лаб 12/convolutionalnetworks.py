import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

model.save('sample_data/my_model.h5')

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
model = load_model('sample_data/my_model.h5')

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
def load_and_preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


img_paths = [
    'sample_data/image.jpg',
    'sample_data/image2.jpg',
    'sample_data/image3.jpg'
]

plt.figure(figsize=(10, 5))

for i, img_path in enumerate(img_paths):
    img_array = load_and_preprocess_img(img_path)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    plt.subplot(1, 3, i + 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicted: {class_names[predicted_class[0]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()