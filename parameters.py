import os


Y_MEMORY_SIZE_THRESHOLD_GB = 2
# TODO 学习os模块
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data')


if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(TRAIN_DATA_PATH)
