import numpy
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout

# fix random seed for reproducibility
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
input_output_pairs = list()
for text in raw_corpus_data:
    for line in text.split('\n'):
        encoded = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(encoded)):
            input_output_pair = encoded[:i+1]
            input_output_pairs.append(input_output_pair)
print('Total number of input-output pair: {}'.format(len(input_output_pairs)))
# pad input sequences
max_length = max([len(input_output_pair) for input_output_pair in input_output_pairs])
# a list of list => 2d numpy array
input_output_pairs = pad_sequences(input_output_pairs, maxlen=max_length, padding='pre')
print('input-output pair:\n', input_output_pairs)
print('Max input-output pair length: {}'.format(max_length))
# TODO 计算一个矩阵占用内存的大小
# split into input and output
X, y = input_output_pairs[:, :-1], input_output_pairs[:, -1]
# sequence prediction is problem of a multi-class classification
# one-hot encode output, word index => one-hot vector
y = to_categorical(y, num_classes=vocab_size+1)
# define model
model = Sequential()
# (batch_size, seq_length) => (batch_size, seq_length, output_dim)
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
