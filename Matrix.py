from random import random


class Matrix:
    def __init__(self, rows, columns, is_random=True):
        Matrix.property_checker(rows, columns)
        self.__rows = rows
        self.__columns = columns
        self.__data = [[random() if is_random else 0 for _ in range(columns)] for __ in range(rows)]

    @property
    def rows(self):
        return self.__rows

    @property
    def columns(self):
        return self.__columns

    @property
    def data(self):
        return self.__data.copy()

    @data.setter
    def data(self, new_data):
        if not isinstance(new_data, list) or any(not isinstance(row, list) for row in new_data):
            raise ValueError("List of lists expected as data setter parameter")
        if len(new_data) != self.__rows or any(len(row) != self.__columns for row in new_data):
            raise ValueError("Data rows and columns amount is illegal")
        if any(not isinstance(item, (int, float)) for row in new_data for item in row):
            raise ValueError("Data rows and columns information contains not numeric values")
        self.__data = new_data

    def set_information(self, row, column, information):
        if any(not isinstance(item, (int, float)) for item in [row, column, information]):
            raise ValueError("Function gets only numeric information")
        if row not in range(self.__rows) or column not in range(self.__columns):
            raise ValueError("Data row and column parameters are out of range")
        self.__data[int(row)][int(column)] = information

    def matrix_difference(self, operand):
        if type(operand) != Matrix or self.__columns != operand.columns or self.__rows != operand.rows:
            raise NameError("Wrong parameter summary!")
        result = Matrix(self.__rows, self.__columns, False)
        for i in range(self.__rows):
            for j in range(self.__columns):
                result.data[i][j] = self.__data[i][j] - operand.data[i][j]
        return result

    def matrix_multiplication(self, operand):
        if type(operand) != Matrix or self.__columns != operand.rows:
            raise NameError("Wrong parameter multiplied!")
        result = Matrix(self.__rows, operand.columns, False)
        for i in range(self.__rows):
            for j in range(operand.columns):
                for k in range(self.__columns):
                    result.data[i][j] += self.__data[i][k] * operand.data[k][j]
        return result

    def number_multiplication(self, number):
        if type(number) != int:
            raise NameError("Not number multiplied!")
        result = Matrix(self.__rows, self.__columns, False)
        for i in range(self.__rows):
            for j in range(self.__columns):
                result.data[i][j] = self.__data[i][j] * number
        return result

    def transposition(self):
        result = Matrix(self.__columns, self.__rows, False)
        for i in range(self.__rows):
            for j in range(self.__columns):
                result.data[j][i] = self.__data[i][j]
        return result

    def __str__(self):
        output = ""
        for _ in self.__data:
            output += str(_) + "\n"
        return output

    @staticmethod
    def property_checker(*args):
        if any(not isinstance(item, int) for item in args):
            raise ValueError("Matrix constructor: Int parameters for rows and columns expected")


if __name__ == "__main__":
    pass
