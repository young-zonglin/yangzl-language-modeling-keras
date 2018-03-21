import os

DISTRIBUTED_MULTI_GPU_MODE = True
FULL_DATA_MODE = False
TRAIN_N_GRAM = True
Y_MEMORY_SIZE_THRESHOLD_GB = 2

# 学习os模块 => done
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train_data')
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'val_data')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'test_data')
SMALL_DATA_PATH = os.path.join(PROJECT_ROOT, 'small_data')
FIGURE_PATH = os.path.join(PROJECT_ROOT, 'figure')

OPEN_FILE_ENCODING = 'gbk'
SAVE_FILE_ENCODING = 'utf-8'

GPU_NUMBER = 2
BATCH_SAMPLES_NUMBER = 256  # 64 128 256
TRAIN_EPOCH_SAMPLES = 80000 * 64
VAL_SAMPLES = 20000 * 64


if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(TRAIN_DATA_PATH)
    print(__file__)
    print(os.path.realpath(__file__))
    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.realpath(__file__)))
