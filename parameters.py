import os

DISTRIBUTED_MULTI_GPU_MODE = False
FULL_DATA_MODE = True
TRAIN_N_GRAM = True
Y_MEMORY_SIZE_THRESHOLD_GB = 2

# 学习os模块 => done
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
SMALL_DATA_PATH = os.path.join(PROJECT_ROOT, 'small_data')
FIGURE_PATH = os.path.join(PROJECT_ROOT, 'figure')

GPU_NUMBER = 2

# TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train_data')
# VAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'val_data')
# TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test_data')
#
OPEN_FILE_ENCODING = 'gbk'
SAVE_FILE_ENCODING = 'utf-8'
#
# BATCH_SAMPLES_NUMBER = 256  # 64 128 256

# 使用人民日报标注语料训练基于字的LM
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train-people-daily')
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'val-people-daily')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test-people-daily')

# OPEN_FILE_ENCODING = 'utf-8'
# SAVE_FILE_ENCODING = 'utf-8'

BATCH_SAMPLES_NUMBER = 256  # 64 128 256

if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(TRAIN_DATA_PATH)
    print(__file__)
    print(os.path.realpath(__file__))
    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.realpath(__file__)))
