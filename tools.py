import sys
import os
import parameters
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


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
    # print('One-hot编码后的输出占用内存大小为：', y_memory_size, 'GB')
    if y_memory_size > parameters.Y_MEMORY_SIZE_THRESHOLD_GB:
        print('内存占用超过', parameters.Y_MEMORY_SIZE_THRESHOLD_GB, 'GB')
        sys.exit(0)

    # pad input sequences
    # lists of list => 2d numpy array
    input_output_pairs = pad_sequences(input_output_pairs, maxlen=max_length, padding='pre')
    # print('Input-output pairs:\n', input_output_pairs)

    # split into input and output
    X, y = input_output_pairs[:, :-1], input_output_pairs[:, -1]
    # sequence prediction is a problem of multi-class classification
    # one-hot encode output, word index => one-hot vector
    y = to_categorical(y, num_classes=vocab_size + 1)
    return X, y


def generate_text_from_corpus(path):
    """
    生成器函数，返回一个迭代器
    :param path:
    :return: 迭代器，可以遍历path下所有文件的内容
    """
    filenames = get_filenames_under_path(path)
    for filename in filenames:
        with open(filename, 'r', encoding=parameters.OPEN_FILE_ENCODING) as file:
            yield file.read()


def fit_tokenizer(path):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(generate_text_from_corpus(path))
    return tokenizer


def generate_input_output_pair_from_corpus(path, tokenizer):
    for text in generate_text_from_corpus(path):
        for line in text.split('\n'):
            encoded = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(encoded)):
                input_output_pair = encoded[: i + 1]
                yield input_output_pair


def generate_batch_samples_from_corpus(path, tokenizer, vocab_size, max_length):
    while True:
        batch_samples_count = 0
        input_output_pairs = list()
        for input_output_pair in generate_input_output_pair_from_corpus(path, tokenizer):
            if batch_samples_count < parameters.BATCH_SAMPLES_NUMBER:
                input_output_pairs.append(input_output_pair)
                batch_samples_count += 1
            else:
                X, y = process_format_to_model_input(input_output_pairs, vocab_size, max_length)
                yield X, y
                input_output_pairs = list()
                batch_samples_count = 0


# if __name__ == '__main__':
#     for file_content in generate_text_from_corpus('E:\自然语言处理数据集\搜狐新闻数据(SogouCS)_segment'):
#         print('111')
