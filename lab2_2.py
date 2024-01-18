import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Загрузка датасета Iris
df = pd.read_csv('/home/and/python_poned/lab_2_AI/data.csv')
X = df.iloc[:, :4]
y = df.iloc[:, 4].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Преобразование меток классов в one-hot encoding
num_classes = 3
y_one_hot = np.eye(num_classes)[y]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class Perceptron:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            print("Epoch:", epoch, "/", epochs)
            for i in range(X.shape[0]):
                input_data = X[i, :].reshape(1, -1)
                target = y[i, :].reshape(1, -1)

               # Прямой проход (Forward pass)
                output = self.predict(input_data)

               # Обратный проход (обновление весов и смещения)
                self.weights += learning_rate * np.dot(input_data.T, (target - output))
                self.bias += learning_rate * (target - output)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Создаем и обучаем перцептрон
input_size = X_train.shape[1]
output_size = num_classes
perceptron = Perceptron(input_size, output_size)
perceptron.train(X_train, y_train)

# Предсказываем классы для тестовых данных
predictions = perceptron.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Оцениваем точность модели
accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_classes)
print(f'Accuracy: {accuracy}')