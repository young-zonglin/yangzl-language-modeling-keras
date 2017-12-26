import numpy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
import tools
import sys

# TODO 使用整个语料库做为训练数据而不只是某个文本
# TODO 使用序列标注的方法减少词汇表的大小
# fix seed of generator of random number for reproducibility
numpy.random.seed(7)
train_text_url = 'E:\自然语言处理数据集\搜狐新闻数据(SogouCS)_segment\\news.sohunews.010801.txt.utf-8.xml.train.seq.clean.gbk.segment'
# ['content of file1', ..., 'content of file2']
raw_corpus_data = []
with open(train_text_url, 'r', encoding='gbk') as train_file:
    raw_corpus_data.append(train_file.read())
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_corpus_data)
vocab_size = len(tokenizer.word_index)
print('Vocabulary size: %d' % vocab_size)
# 使用LSTM编码任意长度的序列
input_output_pairs = list()
for text in raw_corpus_data:
    for line in text.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(encoded)):
            input_output_pair = encoded[:i+1]
            input_output_pairs.append(input_output_pair)
print('Total number of input-output pair: {}'.format(len(input_output_pairs)))
y_shape = len(input_output_pairs), vocab_size+1
y_memory_size = tools.get_array_memory_size(y_shape, 4)
print('one-hot编码后的输出占用内存大小为：', y_memory_size, 'GB')
if y_memory_size > 2:
    print('内存占用超过2GB')
    sys.exit(0)
# pad input sequences
max_length = max([len(input_output_pair) for input_output_pair in input_output_pairs])
# lists of list => 2d numpy array
input_output_pairs = pad_sequences(input_output_pairs, maxlen=max_length, padding='pre')
print('input-output pair:\n', input_output_pairs)
print('Max input-output pair length: {}'.format(max_length))
# split into input and output
X, y = input_output_pairs[:, :-1], input_output_pairs[:, -1]
# sequence prediction is problem of a multi-class classification
# one-hot encode output, word index => one-hot vector
y = to_categorical(y, num_classes=vocab_size+1)
# define model
model = Sequential()
# (batch_size, seq_length/time_step) => (batch_size, seq_length/time_step, output_dim)
# the output shape of Embedding layer fit LSTM layer
model.add(Embedding(input_dim=vocab_size+1, output_dim=64, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dropout(0.5, seed=7))
# softmax output layer
model.add(Dense(vocab_size+1, activation='softmax'))
print(model.summary())
# config process of optimization
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train network
model.fit(X, y, epochs=500, verbose=2, batch_size=1)
# evaluate model
model.evaluate(X, y)
