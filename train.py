import parameters
from language_model import LanguageModel


def main():
    if parameters.FULL_DATA_MODE:
        lm = LanguageModel()
        lm.load_data(train_data_path=parameters.SMALL_DATA_PATH)
        lm.define_model()
        lm.compile_model()
        lm.fit_model()
        lm.evaluate_model()
        print(lm.generate_seq('女', 5))
    else:
        lm = LanguageModel()
        lm.prepare_for_generator(train_data_path=parameters.TRAIN_DATA_PATH,
                                 val_data_path=parameters.VAL_DATA_PATH,
                                 test_data_path=parameters.TEST_DATA_PATH)
        lm.define_model()
        lm.compile_model()
        lm.fit_model_with_generator()
        lm.evaluate_model_with_generator()


if __name__ == '__main__':
    main()
