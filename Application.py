from ImageManager import ImageManager
from NeuralNetwork import NeuralNetwork
import numpy
import json
import os


class Application:
    @staticmethod
    def learning():
        image_filename = "ImageFiles/" + input("Enter filename of the image we'll be working with: ")
        block_rows = int(input("Enter amount of one block row pixels: "))
        block_columns = int(input("Enter amount of one block column pixels: "))
        image_rows = int(input("Enter amount of image row pixels: "))
        image_columns = int(input("Enter amount of image column pixels: "))
        rgb_channels = int(input("Enter amount of image channels: "))
        second_layer = int(input("Enter amount of second layer neurons: "))
        border_error = int(input("Enter border learning error: "))
        first_third_layer = block_rows * block_columns * rgb_channels
        image_manager = ImageManager(rgb_channels, block_rows, block_columns, image_rows, image_columns)
        neural_network = NeuralNetwork(border_error, first_third_layer, second_layer, first_third_layer)
        image_data = image_manager.image_to_network_input(image_filename)
        learning_sign = True
        print("///...Learning with entered image")
        while learning_sign:
            for data in image_data:
                neural_network.learning(data)
            learning_sign = neural_network.get_learning_error() > neural_network.error
        print("Learning is done...///")
        matrices = neural_network.weight_matrices
        matrices_data = {"layers": [matrices[0].shape[0],
                                    matrices[0].shape[1],
                                    matrices[1].shape[1]],
                         "data": [matrix.tolist() for matrix in matrices]}
        with open("WeightMatrices/WeightMatrices.json", "w") as file:
            json.dump(matrices_data, file, indent=5)
        print("Matrices are written into file, ready to work with them in the future")

    @staticmethod
    def zip():
        if os.path.getsize("WeightMatrices/WeightMatrices.json") == 0:
            print("Matrix file is empty, there is no data to convert the image. Do learning with this Image previously")
            return
        image_filename = "ImageFiles/" + input("Enter filename of the image we want to zip: ")
        data_filename = "CachedImageFiles/" + input("Enter filename of the data we want to get: ")
        block_rows = int(input("Enter amount of one block row pixels: "))
        block_columns = int(input("Enter amount of one block column pixels: "))
        image_rows = int(input("Enter amount of image row pixels: "))
        image_columns = int(input("Enter amount of image column pixels: "))
        rgb_channels = int(input("Enter amount of image channels: "))
        with open("WeightMatrices/WeightMatrices.json", "r") as file:
            matrix_data = json.load(file)
        expected_first_layer_neurons = matrix_data["layers"][0]
        got_first_layer_neurons = block_rows * block_columns * rgb_channels
        if expected_first_layer_neurons != got_first_layer_neurons:
            print("There is wrong matrix data to convert your image, do learning previously")
            return
        image_manager = ImageManager(rgb_channels, block_rows, block_columns, image_rows, image_columns)
        neural_network = NeuralNetwork(100, 1, 1, 1)
        neural_network.weight_matrices = [numpy.array(matrix) for matrix in matrix_data["data"]]
        image_data = image_manager.image_to_network_input(image_filename)
        network_output = [neural_network.process(0, 1, data).tolist() for data in image_data]
        with open(data_filename, "w") as file:
            json.dump(network_output, file)
        print("Data successfully written into the file, ready to read and convert it back")

    @staticmethod
    def unzip():
        if os.path.getsize("WeightMatrices/WeightMatrices.json") == 0:
            print("Matrix file is empty, there is no data to convert the image. Do learning with this 2nd layer amount")
            return
        data_filename = "CachedImageFiles/" + input("Enter filename of the data we want to get: ")
        with open(data_filename, "r") as file:
            network_input = json.load(file)
        with open("WeightMatrices/WeightMatrices.json", "r") as file:
            matrix_data = json.load(file)
        if len(network_input[0]) != matrix_data["layers"][1]:
            print("There is wrong matrix data to convert your image. Do learning with this 2nd layer amount")
        network_input = [numpy.array(data) for data in network_input]
        block_rows = int(input("Enter amount of one block row pixels: "))
        block_columns = int(input("Enter amount of one block column pixels: "))
        image_rows = int(input("Enter amount of image row pixels: "))
        image_columns = int(input("Enter amount of image column pixels: "))
        rgb_channels = int(input("Enter amount of image channels: "))
        new_filename = "ImageFiles/" + input("Enter filename of new image: ")
        image_manager = ImageManager(rgb_channels, block_rows, block_columns, image_rows, image_columns)
        neural_network = NeuralNetwork(100, 1, 1, 1)
        neural_network.weight_matrices = [numpy.array(matrix) for matrix in matrix_data["data"]]
        network_output = [neural_network.process(1, 2, data) for data in network_input]
        image_manager.network_output_to_image(network_output, new_filename)
        print("Image successfully unzipped")

    @classmethod
    def process(cls):
        choice = input("1)Learning :: 2)Zip :: 3)Unzip : ")
        if choice == "1":
            cls.learning()
        elif choice == "2":
            cls.zip()
        elif choice == "3":
            cls.unzip()


if __name__ == "__main__":
    pass
