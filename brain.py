import numpy as np


class SettingsBrain:
    def __init__(self, settings):
        self.input_size = settings["input_size"]
        self.action_size = settings["action_size"]
        self.hidden_layers = settings["hidden_layers"]
        self.hidden_size = settings["hidden_size"]
        self.skip_connections = settings["skip_connections"]
        self.mutation_rate = settings["mutation_rate"]


class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

    def forward(self, inputs):
        x = np.dot(inputs, self.weights) + self.biases
        return np.maximum(x, 0)


class Mutator:
    def __init__(self, settings):

        mutation_rate = settings.mutation_rate
        self.distribution = np.array([0, 0.01, 0.02, 0.03, 0.05])
        self.probabilities = np.exp(
            mutation_rate * np.arange(len(self.distribution)))
        self.probabilities /= np.sum(self.probabilities)

    def mutate(self, brain):
        for layer in brain.model:
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    if np.random.choice(self.distribution, p=self.probabilities) != 0:
                        layer.weights[i][j] += np.random.choice(
                            self.distribution, p=self.probabilities)
            for i in range(layer.biases.shape[0]):
                if np.random.choice(self.distribution, p=self.probabilities) != 0:
                    layer.biases[i] += np.random.choice(
                        self.distribution, p=self.probabilities)


class Brain:
    def __init__(self, settings):
        self.settings = settings
        self.model = self._build_model() if not self.settings.skip_connections \
            else self._build_model_skip()
        self.mutator = Mutator(settings)
        self.id = np.random.randint(0, 1000000)

    def _build_model_skip(self):
        model = []
        cumulative_inputs = self.settings.input_size

        for size in self.settings.hidden_size:
            model.append(Layer(cumulative_inputs, size))
            cumulative_inputs += size

        model.append(Layer(cumulative_inputs, self.settings.action_size))

        return model

    def _build_model(self):
        model = []

        previous_size = self.settings.input_size
        for size in self.settings.hidden_size:
            model.append(Layer(previous_size, size))
            previous_size = size

        model.append(Layer(previous_size,
                     self.settings.action_size))

        return model

    def _predict_skip(self, inputs):
        cumulative_inputs = [inputs]
        for layer in self.model:
            combined_inputs = np.concatenate(cumulative_inputs)
            cumulative_inputs.append(layer.forward(combined_inputs))
        return np.argmax(cumulative_inputs[-1])

    def _predict(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return np.argmax(x)

    def predict(self, inputs):
        return self._predict_skip(inputs) if self.settings.skip_connections \
            else self._predict(inputs)

    def mutate(self):
        self.mutator.mutate(self)
        self.id += 1

    def kill(self):
        self.model = self._build_model() if not self.settings.skip_connections \
            else self._build_model_skip()

    def replicate(self, target):
        for i, layer in enumerate(self.model):
            layer.weights = target.model[i].weights.copy()
            layer.biases = target.model[i].biases.copy()
        target.id = self.id

    def save(self, path):
        np.save(path, self.model)

    def load(self, path):
        self.model = np.load(path, allow_pickle=True)
