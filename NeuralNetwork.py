import numpy as np


class NeuralNetwork:
    border_error = 10000
    min_alpha = 0.00001

    def __init__(self, error, *args):
        self.property_checker(error, *args)
        self.__error = error
        self.__weight_matrices = [np.random.rand(rows, columns) for rows, columns in zip(args, args[1:])]
        self.__layers = []
        self.__current_learning_error = 0

    @property
    def error(self):
        return self.__error

    @property
    def weight_matrices(self):
        return self.__weight_matrices[:]

    @weight_matrices.setter
    def weight_matrices(self, matrices):
        if not isinstance(matrices, list) or\
                any(not isinstance(matrix, (list, np.ndarray)) for matrix in matrices) or\
                any(not isinstance(row, (list, np.ndarray)) for matrix in matrices for row in matrix):
            raise ValueError("Array of matrices expected, illegal argument")
        matrices_shapes = [matrix.shape[:] for matrix in matrices]
        if any(union[0][1] != union[1][0] for union in zip(matrices_shapes, matrices_shapes[1:])):
            raise ValueError("Neurons on different layers doesn't match, matrices are illegal")
        self.__weight_matrices = matrices[:]

    @property
    def layers(self):
        return self.__layers[:]

    @property
    def curr_error(self):
        return self.__current_learning_error

    def get_session_error(self):
        return self.__layers[-1] - self.__layers[0]

    def get_numeric_session_error(self):
        sqrt_session_error = self.get_session_error().ravel() ** 2
        return np.sum(sqrt_session_error) ** 2

    def get_learning_error(self):
        learning_error = self.__current_learning_error
        self.__current_learning_error = 0
        return learning_error

    @staticmethod
    def get_adaptive_alpha(input_data):
        divisor = np.sum(input_data.dot(input_data.transpose()))
        return 1 / divisor if divisor != 0 else NeuralNetwork.min_alpha

    def process(self, initial_layer, ending_layer, input_data):
        self.__layers.clear()
        self.__layers.append(input_data.reshape(1, len(input_data)))
        for weights in self.__weight_matrices[initial_layer: ending_layer]:
            layer_output = self.__layers[-1].dot(weights)
            self.__layers.append(layer_output)
        return self.__layers[-1].ravel()

    def get_new_weights(self, old_weights, input_data, next_weights=None):
        const_multi = (input_data.transpose() * self.get_adaptive_alpha(input_data)).dot(self.get_session_error())
        if next_weights is not None:
            const_multi = const_multi.dot(next_weights.transpose())
        return old_weights - const_multi

    def learning(self, input_data):
        self.process(0, len(self.__weight_matrices), input_data)
        number_of_matrices = len(self.__weight_matrices)
        for index, old_weights in enumerate(self.__weight_matrices):
            input_data = self.__layers[index]
            if index != number_of_matrices - 1:
                new_weights = self.get_new_weights(old_weights, input_data, self.__weight_matrices[index + 1])
            else:
                new_weights = self.get_new_weights(old_weights, input_data, None)
            self.__weight_matrices[index] = new_weights
        self.__current_learning_error += self.get_numeric_session_error()

    @classmethod
    def property_checker(cls, *args):
        if not isinstance(args[1], int) or args[1] <= 0 or args[1] > cls.border_error:
            raise ValueError(f"Border learning error meant to be integer value in range {cls.border_error}")
        if any(not isinstance(layer_value, int) or layer_value <= 0 for layer_value in args[2::]):
            raise ValueError(f"Layer neuron amounts meant to be positive integer values")


if __name__ == "__main__":
    pass
