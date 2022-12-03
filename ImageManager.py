import matplotlib.pyplot as pypl
import numpy as np


class ImageManager:
    def __init__(self, *args):
        ImageManager.property_checker(*args)
        self.__rgb_channels, self.__block_rows, self.__block_columns, self.__image_rows, self.__image_columns = args

    @property
    def rgb_channels(self):
        return self.__rgb_channels

    @property
    def block_rows(self):
        return self.__block_rows

    @property
    def block_columns(self):
        return self.__block_columns

    @property
    def image_rows(self):
        return self.__image_rows

    @property
    def image_columns(self):
        return self.__image_columns

    def image_to_network_input(self, image_filename):
        image_data = pypl.imread(image_filename)
        self.image_size_checker(image_data, self.image_rows, self.image_columns)
        row_divided_data = np.vsplit(image_data, self.image_rows / self.block_rows)
        network_input = []
        for exact_row in row_divided_data:
            divided_row = np.hsplit(exact_row, self.image_columns / self.block_columns)
            network_input.extend([block.ravel() for block in divided_row])
        return np.array(network_input)

    @staticmethod
    def stack_row_blocks(block_index, block_step, image_blocks):
        row_slice = image_blocks[block_index: block_index + block_step]
        return np.hstack(row_slice)

    def network_output_to_image(self, network_output, image_filename):
        self.network_output_checker(network_output, self.__block_rows, self.__block_columns, self.__rgb_channels)
        for i in range(len(network_output)):
            for j in range(len(network_output[i])):
                item = network_output[i][j]
                if item < 0 or item > 1:
                    network_output[i][j] = float(round(item))
        image_blocks = [block.reshape((self.block_rows,
                                       self.block_columns,
                                       self.rgb_channels)) for block in network_output]
        image_data = []
        block_step = int(self.image_columns / self.block_columns)
        for block_index in range(0, len(image_blocks), block_step):
            image_data.extend(self.stack_row_blocks(block_index, block_step, image_blocks))
        pypl.imsave(image_filename, np.array(image_data))

    @staticmethod
    def network_output_checker(network_output, expected_rows, expected_columns, rgb_channels):
        expected_block_elements = expected_rows * expected_columns * rgb_channels
        if any(len(data) != expected_block_elements for data in network_output):
            raise ValueError("Amount of block rows, columns and rgb channels doesn't match with entered image_data")

    @staticmethod
    def image_size_checker(image_data, expected_rows, expected_columns):
        if image_data.shape[0] != expected_rows or image_data.shape[1] != expected_columns:
            raise ValueError(f"Image size error, convert denied")

    @staticmethod
    def property_checker(*args):
        if len(args) != 5:
            raise ValueError(f"Five values expected")
        if any(not isinstance(item, int) or item <= 0 for item in args):
            raise ValueError(f"Positive int type value expected for Image property")
        if not (args[3] % args[1] == 0 and args[4] % args[2] == 0):
            raise ValueError(f"Statement: block_value * int = side isn't working")


if __name__ == "__main__":
    pass
