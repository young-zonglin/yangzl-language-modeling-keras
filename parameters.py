import os


FULL_DATA_MODE = False
TRAIN_N_GRAM = False
Y_MEMORY_SIZE_THRESHOLD_GB = 2

# 学习os模块 => done
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train_data')
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'val_data')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test_data')
SMALL_DATA_PATH = os.path.join(PROJECT_ROOT, 'small_data')

OPEN_FILE_ENCODING = 'gbk'
SAVE_FILE_ENCODING = 'utf-8'

GPU_NUMBER = 2
BATCH_SAMPLES_NUMBER = 64


if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(TRAIN_DATA_PATH)
    print(__file__)
    print(os.path.realpath(__file__))
    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.realpath(__file__)))
