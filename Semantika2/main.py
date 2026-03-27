import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


class NeuralNetwork:
    def __init__(self, layers, activation='relu', dropout_rate=0.0, task_type='classification'):
        self.layers = layers
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        self.weights = []
        self.biases = []
        self.history = {'loss': [], 'val_loss': []}
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов (He/Xavier)"""
        for i in range(len(self.layers) - 1):
            if self.activation_name == 'relu':
                scale = np.sqrt(2.0 / self.layers[i])
            else:
                scale = np.sqrt(1.0 / self.layers[i])

            w = np.random.randn(self.layers[i], self.layers[i + 1]) * scale
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_derivative(x):
        s = NeuralNetwork._sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_derivative(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _get_activation(self, name):
        activations = {
            'relu': (self._relu, self._relu_derivative),
            'sigmoid': (self._sigmoid, self._sigmoid_derivative),
            'tanh': (self._tanh, self._tanh_derivative)
        }
        return activations.get(name, (self._relu, self._relu_derivative))

    def _forward(self, X, training=False):
        """Прямое распространение"""
        cache = {'activations': [X], 'pre_activations': []}
        current_input = X

        for i in range(len(self.weights) - 1):
            z = current_input @ self.weights[i] + self.biases[i]
            cache['pre_activations'].append(z)

            activation_func, _ = self._get_activation(self.activation_name)
            a = activation_func(z)

            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * mask
                cache[f'dropout_mask_{i}'] = mask

            cache['activations'].append(a)
            current_input = a

        z_out = current_input @ self.weights[-1] + self.biases[-1]
        cache['pre_activations'].append(z_out)

        if self.task_type == 'classification':
            if self.layers[-1] == 1:
                output = self._sigmoid(z_out)
            else:
                output = self._softmax(z_out)
        else:
            output = z_out

        cache['activations'].append(output)
        return output, cache

    def _compute_loss(self, y_true, y_pred, eps=1e-15):
        """Вычисление функции потерь"""
        m = y_true.shape[0]

        if self.task_type == 'classification':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            if self.layers[-1] == 1:
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            else:
                if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                    y_true_onehot = np.zeros((m, self.layers[-1]))
                    y_true_onehot[np.arange(m), y_true.astype(int).flatten()] = 1
                    y_true = y_true_onehot
                loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        else:
            loss = np.mean((y_true - y_pred) ** 2)

        return loss

    def _compute_loss_derivative(self, y_true, y_pred, eps=1e-15):
        """Производная функции потерь"""
        if self.task_type == 'classification':
            y_pred = np.clip(y_pred, eps, 1 - eps)
            if self.layers[-1] == 1:
                return (y_pred - y_true) / y_true.shape[0]
            else:
                if len(y_true.shape) == 1 or y_true.shape[1] == 1:
                    m = len(y_true)
                    y_true_flat = np.array(y_true).flatten().astype(int)
                    y_true_flat = np.clip(y_true_flat, 0, self.layers[-1] - 1)
                    y_true_onehot = np.zeros((m, self.layers[-1]))
                    y_true_onehot[np.arange(m), y_true_flat] = 1
                    y_true = y_true_onehot
                return (y_pred - y_true) / y_true.shape[0]
        else:
            return 2 * (y_pred - y_true) / y_true.shape[0]

    def _backward(self, y_true, y_pred, cache):
        """Обратное распространение"""
        gradients = {'dw': [], 'db': []}
        dz = self._compute_loss_derivative(y_true, y_pred)

        for i in reversed(range(len(self.weights))):
            a_prev = cache['activations'][i]
            dw = a_prev.T @ dz
            db = np.sum(dz, axis=0, keepdims=True)

            gradients['dw'].insert(0, dw)
            gradients['db'].insert(0, db)

            if i > 0:
                dz = dz @ self.weights[i].T
                if self.dropout_rate > 0 and f'dropout_mask_{i - 1}' in cache:
                    dz = dz * cache[f'dropout_mask_{i - 1}']

                z_prev = cache['pre_activations'][i - 1]
                _, activation_deriv = self._get_activation(self.activation_name)
                dz = dz * activation_deriv(z_prev)

        return gradients

    def _update_weights(self, gradients, learning_rate):
        """Обновление весов"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients['dw'][i]
            self.biases[i] -= learning_rate * gradients['db'][i]

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.01,
            validation_data=None, early_stopping=False, patience=10, verbose=True):
        """Обучение сети"""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = max(1, n_samples // batch_size)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                if start_idx >= end_idx:
                    continue

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred, cache = self._forward(X_batch, training=True)
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss

                gradients = self._backward(y_batch, y_pred, cache)
                self._update_weights(gradients, learning_rate)

            avg_loss = epoch_loss / max(1, n_batches)
            self.history['loss'].append(avg_loss)

            if validation_data is not None:
                X_val, y_val = validation_data
                if len(y_val.shape) == 1:
                    y_val = y_val.reshape(-1, 1)

                y_val_pred, _ = self._forward(X_val, training=False)
                val_loss = self._compute_loss(y_val, y_val_pred)
                self.history['val_loss'].append(val_loss)

                if early_stopping:
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}"
                if validation_data is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        return self

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        if self.task_type != 'classification':
            raise ValueError("predict_proba доступен только для классификации")
        output, _ = self._forward(X, training=False)
        return output

    def predict(self, X):
        """Предсказание классов/значений"""
        output, _ = self._forward(X, training=False)

        if self.task_type == 'classification':
            if self.layers[-1] == 1:
                return (output >= 0.5).astype(int).flatten()
            else:
                return np.argmax(output, axis=1)
        else:
            return output.flatten()

    def score(self, X, y, metric='f1'):
        """Оценка качества"""
        y_pred = self.predict(X)

        if self.task_type == 'classification':
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            y = y.flatten()

            if metric == 'f1':
                return f1_score(y, y_pred, average='weighted', zero_division=0)
            elif metric == 'accuracy':
                return np.mean(y == y_pred)
        else:
            if metric == 'mae':
                return mean_absolute_error(y, y_pred)
            elif metric == 'rmse':
                return np.sqrt(mean_squared_error(y, y_pred))
        return None
