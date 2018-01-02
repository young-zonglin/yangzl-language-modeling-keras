import os


Y_MEMORY_SIZE_THRESHOLD_GB = 2
# 学习os模块 => done
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'data')


if __name__ == '__main__':
    print(PROJECT_ROOT)
    print(TRAIN_DATA_PATH)
    print(__file__)
    print(os.path.realpath(__file__))
    print(os.path.dirname(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.realpath(__file__)))
