import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
import tools
import parameters
import network_conf


# 使用整个语料库做为训练数据而不只是某个文本 => done
# 使用序列标注的方法减少词汇表的大小 => done
# fix seed of generator of numpy random number for reproducibility
np.random.seed(7)


class LanguageModel:
    def __init__(self):
        self.vocab_size = 0
        self.max_length = 0
        self.model = None
        self.tokenizer = None
        self.X = None
        self.y = None

    def load_data(self, train_data_path):
        # ['content of file1', ..., 'content of file2']
        raw_corpus_data = []
        filenames = tools.get_filenames_under_path(train_data_path)
        for filename in filenames:
            with open(filename, 'r', encoding='gbk') as train_file:
                raw_corpus_data.append(train_file.read())

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(raw_corpus_data)
        self.vocab_size = len(self.tokenizer.word_index)
        print('Vocabulary size: %d' % self.vocab_size)

        # 使用LSTM编码任意长度的序列
        input_output_pairs = list()
        for text in raw_corpus_data:
            for line in text.split('\n'):
                encoded = self.tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(encoded)):
                    input_output_pair = encoded[: i+1]
                    input_output_pairs.append(input_output_pair)
        print('Total number of input-output pair: {}'.format(len(input_output_pairs)))

        y_shape = len(input_output_pairs), self.vocab_size + 1
        y_memory_size = tools.get_array_memory_size(y_shape, item_size=8)
        print('One-hot编码后的输出占用内存大小为：', y_memory_size, 'GB')
        if y_memory_size > parameters.Y_MEMORY_SIZE_THRESHOLD_GB:
            print('内存占用超过', parameters.Y_MEMORY_SIZE_THRESHOLD_GB, 'GB')
            sys.exit(0)

        # pad input sequences
        self.max_length = max([len(input_output_pair) for input_output_pair in input_output_pairs])
        # lists of list => 2d numpy array
        input_output_pairs = pad_sequences(input_output_pairs, maxlen=self.max_length, padding='pre')
        print('Input-output pairs:\n', input_output_pairs)
        print('Max input-output pair length: {}'.format(self.max_length))

        # split into input and output
        self.X, self.y = input_output_pairs[:, :-1], input_output_pairs[:, -1]
        # sequence prediction is a problem of multi-class classification
        # one-hot encode output, word index => one-hot vector
        self.y = to_categorical(self.y, num_classes=self.vocab_size + 1)

    def load_model(self):
        # define model
        model = Sequential()
        # input shape: (batch_size/samples, seq_length/time_step) =>
        # output shape: (batch_size/samples, time_step, output_dim/features/word vector dim)
        # the output shape of Embedding layer fit LSTM layer
        # TODO 训练词向量（CBOW和skip-gram）
        model.add(Embedding(input_dim=self.vocab_size + 1,
                            output_dim=network_conf.embedding_output_dim,
                            input_length=self.max_length - 1))
        # TODO 阅读RNN和LSTM原始论文，再看一遍相应博客
        model.add(LSTM(units=network_conf.lstm_layer_unit))
        # TODO 继续阅读dropout原始论文
        model.add(Dropout(rate=network_conf.dropout_layer_rate,
                          seed=network_conf.dropout_layer_seed))
        # softmax output layer
        model.add(Dense(self.vocab_size + 1))
        model.add(Activation('softmax'))
        print(model.summary())
        self.model = model

    def compile_model(self):
        # config process of optimization
        # TODO 学习损失函数
        # TODO 学习梯度下降（SGD, Adam, RMSprop等）
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit_model(self):
        # train network
        history = self.model.fit(self.X, self.y, epochs=500, verbose=2, batch_size=1)
        print('history:')
        print(history.history)

    def evaluate_model(self):
        # evaluate model
        # TODO K-fold交叉验证
        # TODO 学习分类模型评价指标
        scores = self.model.evaluate(self.X, self.y, batch_size=32)
        print("\n==================================\n性能评估：")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def predict(self):
        input_seq_index = np.random.randint(len(self.X))
        input_seq = self.X[input_seq_index]
        prediction = self.model.predict(input_seq, verbose=2)
        print('Model out vector:\n', prediction)
        y_index = np.argmax(prediction)
        print('Predict word index:', y_index)
        out_word = ''
        for word, index in self.tokenizer.word_index.items():
            if index == y_index:
                out_word = word
                break
        print('Input output pair:', input_seq, '->', out_word)

    # generate a sequence from a language model
    def generate_seq(self, seed_text, n_words):
        in_text = seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            # pre-pad sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=self.max_length-1, padding='pre')
            # predict probabilities for each word
            y_index = self.model.predict_classes(encoded, verbose=0)
            # map predicted word index to word
            out_word = ''
            for word, index in self.tokenizer.word_index.items():
                if index == y_index:
                    out_word = word
                    break
            # append to input
            in_text += ' ' + out_word
        return in_text.replace(' ', '')

    def save_model(self):
        pass

    def fit_model_generator(self):

        pass

    def evaluate_model_generator(self):
        pass

    def predict_with_generator(self):
        pass
