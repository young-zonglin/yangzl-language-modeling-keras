import sys


def get_matrix_memory_size(matrix):
    """
    calculate memory size of a matrix.
    :param matrix: 2d numpy array
    :return: object_size(in GB), values_size(in GB), item_size(in byte)
    """
    object_size = bytes_to_gb(sys.getsizeof(matrix))
    values_size = bytes_to_gb(matrix.nbytes)
    item_size = matrix.itemsize
    return object_size, values_size, item_size


def get_array_memory_size(array_shape, item_size):
    """
    calculate memory size of array using shape and item size
    :param array_shape: shape attribute of numpy array
    :param item_size: element size of array in byte
    :return: array memory size in GB
    """
    element_number = 1
    for tmp in array_shape:
        element_number *= tmp
    return bytes_to_gb(element_number * item_size)


def bytes_to_gb(bytes_number):
    return bytes_number / (1024**3)
