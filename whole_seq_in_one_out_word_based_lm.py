import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
import tools
import network_conf
import parameters


# 使用整个语料库做为训练数据而不只是某个文本 => done
# 使用序列标注的方法减少词汇表的大小 => done
# TODO 得到可复现的结果
# fix seed of generator of numpy random number for reproducibility
np.random.seed(7)


class LanguageModel:
    def __init__(self):
        self.vocab_size = network_conf.VOCAB_SIZE
        self.max_length = network_conf.MAX_LENGTH
        self.train_data_path = None
        self.val_data_path = None
        self.test_data_path = None
        self.model = None
        self.template_model = None
        self.tokenizer = None
        self.X = None
        self.y = None

    # 装载全量数据
    def load_data(self, train_data_path):
        self.tokenizer = tools.fit_tokenizer(train_data_path)
        self.vocab_size = len(self.tokenizer.word_index)
        print('Vocabulary size: %d' % self.vocab_size)

        input_output_pairs = list()
        for input_output_pair in tools.generate_input_output_pair_from_corpus(train_data_path,
                                                                              self.tokenizer):
            input_output_pairs.append(input_output_pair)
        print('Total number of input-output pair: {}'.format(len(input_output_pairs)))

        self.max_length = max([len(input_output_pair) for input_output_pair in input_output_pairs])
        print('Max input-output pair length: {}'.format(self.max_length))

        self.X, self.y = tools.process_format_to_model_input(input_output_pairs,
                                                             self.vocab_size,
                                                             self.max_length)

    def define_model(self):
        if parameters.DISTRIBUTED_MULTI_GPU_MODE:
            # 在cpu上建立模型
            with tf.device('/cpu:0'):
                # TODO 模型参数的网格搜索
                # define model
                template_model = Sequential()
                # input shape: (batch_size/samples, seq_length/time_step) =>
                # output shape: (batch_size/samples, time_step, output_dim/features/word vector dim)
                # the output shape of Embedding layer fit LSTM layer
                # TODO 训练词向量（CBOW和skip-gram）
                template_model.add(Embedding(input_dim=self.vocab_size + 1,
                                             output_dim=network_conf.EMBEDDING_OUTPUT_DIM,
                                             input_length=self.max_length - 1))
                # TODO 阅读RNN和LSTM原始论文，再看一遍相应博客
                template_model.add(LSTM(units=network_conf.LSTM_LAYER_UNIT))
                # TODO 继续阅读dropout原始论文
                template_model.add(Dropout(rate=network_conf.DROPOUT_LAYER_RATE,
                                           seed=network_conf.DROPOUT_LAYER_SEED))
                # softmax output layer
                template_model.add(Dense(self.vocab_size + 1))
                template_model.add(Activation('softmax'))
                self.template_model = template_model
            # 多卡并行训练 => done
            model = multi_gpu_model(template_model, gpus=parameters.GPU_NUMBER)
        else:
            model = Sequential()
            model.add(Embedding(input_dim=self.vocab_size + 1,
                                output_dim=network_conf.EMBEDDING_OUTPUT_DIM,
                                input_length=self.max_length - 1))
            model.add(LSTM(units=network_conf.LSTM_LAYER_UNIT))
            model.add(Dropout(rate=network_conf.DROPOUT_LAYER_RATE,
                              seed=network_conf.DROPOUT_LAYER_SEED))
            model.add(Dense(self.vocab_size + 1))
            model.add(Activation('softmax'))
            self.template_model = model
        print('\n############### Model summary ##################')
        print(model.summary())
        self.model = model

    def compile_model(self):
        # config process of optimization
        # TODO 学习损失函数
        # TODO 学习梯度下降（SGD, Adam, RMSprop等）
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    # 使用全量训练数据进行训练
    def fit_model(self):
        early_stopping = EarlyStopping(monitor='acc',
                                       patience=5, min_delta=0.0001,
                                       verbose=1, mode='max')
        # train network
        history = self.model.fit(self.X, self.y, epochs=500, batch_size=1,
                                 verbose=1, callbacks=[early_stopping], shuffle=True)
        print('\n========================== history ===========================')
        acc = history.history.get('acc')
        loss = history.history['loss']
        print('train data acc:', acc)
        print('train data loss', loss)
        print('\n======================= acc & loss ============================')
        for i in range(len(acc)):
            print('epoch {0:<4} | acc: {1:6.3f}% | loss: {2:<10.5f}'.format(i+1, acc[i]*100, loss[i]))
        plt_x = [x+1 for x in range(len(acc))]
        plt_acc = plt_x, acc
        plt_loss = plt_x, loss
        tools.plot_figure('acc & loss', plt_acc, plt_loss)

    # 使用全量测试数据评估模型
    def evaluate_model(self):
        # evaluate model
        # TODO K-fold交叉验证
        # TODO 学习分类模型评价指标
        scores = self.model.evaluate(self.X, self.y, batch_size=32)
        print("\n================= 性能评估 ====================")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    # 适用于装载全量数据的情况下
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

    # 均适用
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

    # TODO 每轮迭代保存模型
    def save_model(self, file_path):
        self.template_model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def prepare_for_generator(self, train_data_path, val_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.tokenizer = tools.fit_tokenizer(self.train_data_path)
        self.vocab_size = len(self.tokenizer.word_index)
        print('Vocabulary size: %d' % self.vocab_size)
        if parameters.TRAIN_N_GRAM:
            self.max_length = network_conf.N_GRAM
            print('Train', self.max_length, 'gram model.')
        else:
            self.max_length = max([len(input_output_pair) for input_output_pair in
                                   tools.generate_input_output_pair_from_corpus(
                                       self.train_data_path,
                                       self.tokenizer)])
            print('Max input-output pair length: {}'.format(self.max_length))

    # 处理超过内存的数据集
    def fit_model_with_generator(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001,
                                       verbose=1, mode='min')
        # 训练集、验证集和测试集的比例为7:2:1 => done
        # TODO steps应自适应于BATCH_SAMPLES_NUMBER
        history = self.model.fit_generator(tools.generate_batch_samples_from_corpus(self.train_data_path,
                                                                                    self.tokenizer,
                                                                                    self.vocab_size,
                                                                                    self.max_length),
                                           validation_data=tools.generate_batch_samples_from_corpus(self.val_data_path,
                                                                                                    self.tokenizer,
                                                                                                    self.vocab_size,
                                                                                                    self.max_length),
                                           validation_steps=20000,
                                           steps_per_epoch=80000, epochs=1000, verbose=1,
                                           callbacks=[early_stopping])
        print('\n========================== history ===========================')
        acc = history.history.get('acc')
        loss = history.history['loss']
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']
        print('train data acc:', acc)
        print('train data loss', loss)
        print('val data acc', val_acc)
        print('val data loss', val_loss)
        print('\n======================= acc & loss & val_acc & val_loss ============================')
        for i in range(len(acc)):
            print('epoch {0:<4} | acc: {1:6.3f}% | loss: {2:<10.5f} |'
                  ' val_acc: {3:6.3f}% | val_loss: {4:<10.5f}'.format(i + 1,
                                                                      acc[i] * 100, loss[i],
                                                                      val_acc[i]*100, val_loss[i]))
        # 训练完毕后，将每轮迭代的acc、loss、val_acc、val_loss以画图的形式进行展示 => done
        plt_x = [x + 1 for x in range(len(acc))]
        plt_acc = plt_x, acc
        plt_loss = plt_x, loss
        plt_val_acc = plt_x, val_acc
        plt_val_loss = plt_x, val_loss
        tools.plot_figure('acc & loss & val_acc & val_loss',
                          plt_acc, plt_loss, plt_val_acc, plt_val_loss)

    def evaluate_model_with_generator(self):
        scores = self.model.evaluate_generator(generator=tools.generate_batch_samples_from_corpus(self.test_data_path,
                                                                                                  self.tokenizer,
                                                                                                  self.vocab_size,
                                                                                                  self.max_length),
                                               steps=10000)
        print("\n==================================\n性能评估：")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def predict_with_generator(self):
        pass
