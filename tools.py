import sys
import os
import parameters
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


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


def get_filenames_under_path(path):
    """
    get filename seq under path.
    :param path: string
    :return: filename seq
    """
    filenames = list()
    for filename in os.listdir(path):
        filename = os.path.join(path, filename)
        if os.path.isdir(filename):
            continue
        filenames.append(filename)
    return filenames


def process_format_to_model_input(input_output_pairs, vocab_size, max_length):
    """
    处理输入输出对的格式，使得符合模型的输入要求
    :param input_output_pairs: [[12, 25], [12, 25, 11], ..., [12, 25, 11, ..., 23]]
    :param vocab_size: 语料库词汇表的规模
    :param max_length: 语料库按停顿符切成序列后，最长序列（分词之后）的长度
    :return: (X, y), X: 2d numpy array, y shape: (input_output_pairs length, vocab_size+1)
    """
    y_shape = len(input_output_pairs), vocab_size + 1
    y_memory_size = get_array_memory_size(y_shape, item_size=8)
    print('One-hot编码后的输出占用内存大小为：', y_memory_size, 'GB')
    if y_memory_size > parameters.Y_MEMORY_SIZE_THRESHOLD_GB:
        print('内存占用超过', parameters.Y_MEMORY_SIZE_THRESHOLD_GB, 'GB')
        sys.exit(0)

    # pad input sequences
    # lists of list => 2d numpy array
    input_output_pairs = pad_sequences(input_output_pairs, maxlen=max_length, padding='pre')
    print('Input-output pairs:\n', input_output_pairs)

    # split into input and output
    X, y = input_output_pairs[:, :-1], input_output_pairs[:, -1]
    # sequence prediction is a problem of multi-class classification
    # one-hot encode output, word index => one-hot vector
    y = to_categorical(y, num_classes=vocab_size + 1)
    return X, y
