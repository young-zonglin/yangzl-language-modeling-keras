import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model

import network_conf
import parameters
import tools

# TensorFlow显存管理
# 相关链接：http://www.friskit.me/2017/02/01/keras-tensorflow-vram-video-memory-configproto/
# http://blog.csdn.net/cq361106306/article/details/52950081
# http://blog.csdn.net/leibaojiangjun1/article/details/53671257
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 最多只能占用xx%的总显存
config.gpu_options.allow_growth = True  # 按需申请显存
set_session(tf.Session(config=config))  # 主动创建一个使用指定配置的Session，替换自动创建的参数均为默认值的Session

# 使用整个语料库做为训练数据而不只是某个文本 => done
# 使用序列标注的方法减少词汇表的大小 => done
# TODO 得到可复现的结果
# fix seed of generator of numpy random number for reproducibility
np.random.seed(7)


class LanguageModel:
    def __init__(self):
        # 词汇表的大小，one hot编码word index，one-hot向量/数组长度为vocab_size+1
        self.vocab_size = None
        self.max_length = None  # 最长序列的长度，序列包含输入和输出
        self.train_data_path = None
        self.val_data_path = None
        self.test_data_path = None
        self.model = None  # 配置model的loss和优化器，然后fit和evaluate它，并用model进行预测
        self.template_model = None  # 模板model用于保存网络结构和参数
        self.tokenizer = None  # 通过在corpus上fit，得到dict，用于word=>index
        self.X = None  # 装载全量数据，得到的输入矩阵
        self.y = None  # 装载全量数据，得到输出矩阵
        # 训练集/验证集/测试集的样本数
        self.train_samples_num = 0
        self.val_samples_num = 0
        self.test_samples_num = 0

    # 装载全量数据
    def load_data(self, train_data_path):
        self.tokenizer = tools.fit_tokenizer(train_data_path)
        self.vocab_size = len(self.tokenizer.word_index)
        print('Vocabulary size: %d' % self.vocab_size)  # 对vocab_size进行格式化输出

        input_output_pairs = list()
        for input_output_pair in tools.generate_input_output_pair_from_corpus(train_data_path,
                                                                              self.tokenizer):
            input_output_pairs.append(input_output_pair)
        print('Total number of input-output pair: {}'.format(len(input_output_pairs)))

        self.max_length = max([len(input_output_pair) for input_output_pair in input_output_pairs])
        print('Max input-output pair length: {}'.format(self.max_length))  # 对max_length进行格式化输出

        self.X, self.y = tools.process_format_to_model_input(input_output_pairs,
                                                             self.vocab_size,
                                                             self.max_length)

    def define_model(self):
        if parameters.DISTRIBUTED_MULTI_GPU_MODE:  # 数据并行，多卡并行训练model
            # 在cpu上建立模板模型
            with tf.device('/cpu:0'):
                # TODO 超参的网格搜索
                # seq_model.add(layer)
                template_model = Sequential()
                # input shape: (batch_size/samples, seq_length/time_steps/input_length)
                # output shape: (batch_size/samples, time_steps, output_dim/features/word vector dim)
                # the output shape of Embedding layer fit LSTM layer
                # TODO 训练词向量（CBOW和skip-gram）
                template_model.add(Embedding(input_dim=self.vocab_size + 1,  # 输入是word index，one hot编码它
                                             output_dim=network_conf.WORD_EMBEDDING_DIM,  # Embedding层输出词向量
                                             input_length=self.max_length - 1))
                # TODO 阅读RNN和LSTM原始论文，再看一遍相应博客
                # LSTM层可以编码任意长度的序列，输出为序列的特征向量
                # 即序列在特征空间的位置/坐标，从多个维度刻画该序列
                # 神经网络层输出向量
                template_model.add(LSTM(units=network_conf.SEQ_FEATURE_VECTOR_DIM))
                # TODO 继续阅读dropout原始论文
                # Dropout层可以阻断工作信号的正向传播过程和误差信号的反向传播过程
                template_model.add(Dropout(rate=network_conf.DROPOUT_RATE,  # drop一定比率上一层单元的输出
                                           seed=network_conf.DROPOUT_LAYER_SEED))  # 固定随机数种子，为了结果的可复现
                # softmax output layer
                # add全连接层，输入可以自动推断，需要指定输出shape
                # 输出矩阵的shape为(samples, one-hot vector dim)
                # 所以，model的输出向量维度应与one-hot vector一样
                template_model.add(Dense(self.vocab_size + 1))
                # add激活层，语言模型，序列预测，是一个多分类问题，所以使用softmax激活函数
                # 使得model的输出具有概率意义，归一化为给定前N-1个词，下一个词是各个词的条件概率分布
                template_model.add(Activation('softmax'))
                self.template_model = template_model
            # 多卡并行训练模型，数据并行，参数服务器 => done
            # 关于数据并行，见我的印象笔记“思考：多机分布式并行训练（应用）模型”
            model = multi_gpu_model(template_model, gpus=parameters.GPU_NUMBER)
        else:  # 单卡训练模型，或者在CPU上，基于多线程并行训练模型
            model = Sequential()
            model.add(Embedding(input_dim=self.vocab_size + 1,
                                output_dim=network_conf.WORD_EMBEDDING_DIM,
                                input_length=self.max_length - 1))
            model.add(LSTM(units=network_conf.SEQ_FEATURE_VECTOR_DIM))
            model.add(Dropout(rate=network_conf.DROPOUT_RATE,
                              seed=network_conf.DROPOUT_LAYER_SEED))
            model.add(Dense(self.vocab_size + 1))
            model.add(Activation('softmax'))
            self.template_model = model
        print('\n############### Template Model Summary ##################')
        print(self.template_model.summary())
        print('\n############### Model Summary ##################')
        print(model.summary())
        self.model = model

    def compile_model(self):
        # config process of optimization
        # TODO 学习损失函数
        # TODO 学习梯度下降（SGD, Adam, RMSprop等）
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])  # metrics用于训练和测试阶段，这里指定'accuracy'，训练时，每轮迭代都会输出模型的准确率

    # 使用全量训练数据进行训练
    def fit_model(self):
        # 提前结束训练，避免过拟合，加快训练速度
        # 如果连续若干轮迭代，模型都未能变得更好，就提前结束训练
        # 连续patience轮迭代，准确率增加量都没有超过阈值min_delta，或者没有增加，则结束训练
        early_stopping = EarlyStopping(monitor='acc',
                                       patience=5, min_delta=0.0001,
                                       verbose=1, mode='max')
        # train network
        # 逼近隐藏函数 => loss最小，优化问题 => 梯度下降，近似求解
        history = self.model.fit(self.X, self.y, epochs=500, batch_size=1,
                                 verbose=1, callbacks=[early_stopping], shuffle=True)
        print('\n========================== history ===========================')
        acc = history.history.get('acc')  # fit函数返回History对象，它有一个history字典
        loss = history.history['loss']  # acc和loss都是列表
        print('train data acc:', acc)   # 训练数据的loss和acc，验证集的val_loss和val_acc，见印象笔记“2018.1.1-2018.1.22”
        print('train data loss:', loss)
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
        # 训练、验证或者评估模型，均是基于批，一次把一个批的数据送到GPU里（如果直接把全量数据传给GPU，可能显存会OOM）
        scores = self.model.evaluate(self.X, self.y, batch_size=32)
        print("\n================= 性能评估 ====================")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    # 适用于装载全量数据的情况下
    # model fit和evaluate完毕，应用它
    def predict(self):
        # numpy的随机数详见ScikitLearnStudy项目的NumPyRandomStudy.py
        input_seq_index = np.random.randint(len(self.X))  # 从[0, total input-output pair number)中产生一个随机整数
        input_seq = self.X[input_seq_index]
        # 输入一个样本（1d numpy数组），则返回一个预测向量（1d numpy数组）
        # 输入多个样本，则返回预测矩阵，均为2d numpy数组
        prediction = self.model.predict(input_seq, verbose=2)
        print('Model out vector:\n', prediction)
        # 沿着指定轴，返回最大值的索引
        y_index = np.argmax(prediction)  # 得到下一个词的index，进一步index => word，得到下一个词
        print('Predict word index:', y_index)
        out_word = ''
        # for k, v in dict.items()
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
        for _ in range(n_words):  # '_'是占位符，我只是想循环这么多次而已
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            # pre-pad sequences to a fixed length
            encoded = pad_sequences([encoded], maxlen=self.max_length-1, padding='pre')
            # predict probabilities for each word
            # 输入一个样本，则直接返回它的label
            # 输入多个样本，则返回class label 1d numpy array
            y_index = self.model.predict_classes(encoded, verbose=0)  # 预测下一个词的index
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
        # model用于预测
        self.model = load_model(file_path)

    def prepare_for_generator(self, train_data_path, val_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.tokenizer = tools.fit_tokenizer(self.train_data_path)
        self.vocab_size = len(self.tokenizer.word_index)
        print('Vocabulary size: %d' % self.vocab_size)
        for _ in tools.generate_input_output_pair_from_corpus(self.train_data_path, self.tokenizer):
            self.train_samples_num += 1
        for _ in tools.generate_input_output_pair_from_corpus(self.val_data_path, self.tokenizer):
            self.val_samples_num += 1
        for _ in tools.generate_input_output_pair_from_corpus(self.test_data_path, self.tokenizer):
            self.test_samples_num += 1
        print('Train data samples num: %d' % self.train_samples_num)
        print('Val data samples num: %d' % self.val_samples_num)
        print('Test data samples num: %d' % self.test_samples_num)

        # 使用LSTM网络构建N-Gram模型，则模型输入为前N-1个词的index，输出为下一个词的index
        # 训练数据、验证集、测试数据和未见样本，它们的输入部分都相同，都是前N-1个词的index
        # 使用全量数据训练模型（即一次性将全部数据加载到内存里），这种模式不支持构建N-Gram模型
        # 基于生成器训练模型，这种模式支持构建N-Gram模型和编码任意长度序列，事实上，它俩都是N-Gram模型
        # max_length-1即为时间步，时间步意味着LSTM层要编码多长的序列，要循环编码多少次
        # 只需控制max_length即可控制网络结构和输入矩阵的shape
        # max_length取3即构建三元语法，给定前两个词，预测下一个词
        if parameters.TRAIN_N_GRAM:
            self.max_length = network_conf.N_GRAM
            print('Train', self.max_length, 'gram model.')
        else:
            # 最长序列的长度，input-output pair是一个列表，更多见印象笔记“关于input-output pair，2018-2-11 17:26”
            content_len = [(input_output_pair, len(input_output_pair)) for input_output_pair in
                           tools.generate_input_output_pair_from_corpus(
                               self.train_data_path,
                               self.tokenizer)]
            print('Total number of input-output pair: {}'.format(len(content_len)))
            max_length = -1
            max_index = -1
            for i in range(len(content_len)):
                length = content_len[i][1]
                if length > max_length:
                    max_length = length
                    max_index = i
            self.max_length = max_length
            print('Max input-output pair length: {}'.format(self.max_length))
            print('Max input-output pair:\n{}'.format(content_len[max_index][0]))
            print('Raw content of max input-output pair:')
            for word_index in content_len[max_index][0]:
                for word, index in self.tokenizer.word_index.items():
                    if index == word_index:
                        print(word, end=' ')
                        break

    # 处理超过内存的数据集
    # 模型的训练使用的是min-batch梯度下降，本来就是batch by batch
    # 通过一个生成器，内存里只需维持一个批的数据，内存友好
    def fit_model_with_generator(self):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001,
                                       verbose=1, mode='min')
        # 训练集、验证集和测试集的比例为7:2:1 => done
        # steps应自适应于BATCH_SAMPLES_NUMBER => done
        batch_samples_number = parameters.BATCH_SAMPLES_NUMBER
        history = self.model.fit_generator(tools.generate_batch_samples_from_corpus(self.train_data_path,
                                                                                    self.tokenizer,
                                                                                    self.vocab_size,
                                                                                    self.max_length),
                                           validation_data=tools.generate_batch_samples_from_corpus(self.val_data_path,
                                                                                                    self.tokenizer,
                                                                                                    self.vocab_size,
                                                                                                    self.max_length),
                                           validation_steps=self.val_samples_num / batch_samples_number,
                                           steps_per_epoch=self.train_samples_num / batch_samples_number,
                                           epochs=1000, verbose=1,
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

    # 通过生成器来生成测试数据，本来测试数据也是batch by batch地被处理
    # 一次传一个批的测试数据给GPU，每个批可以计算一下指标，求平均则可以得到模型在这个测试集上的性能表现
    def evaluate_model_with_generator(self):
        batch_samples_number = parameters.BATCH_SAMPLES_NUMBER
        scores = self.model.evaluate_generator(generator=tools.generate_batch_samples_from_corpus(self.test_data_path,
                                                                                                  self.tokenizer,
                                                                                                  self.vocab_size,
                                                                                                  self.max_length),
                                               steps=self.test_samples_num/batch_samples_number)
        print("\n==================================\n性能评估：")
        print("%s: %.4f" % (self.model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    def predict_with_generator(self):
        pass  # 占位符，可以先不实现这个方法
