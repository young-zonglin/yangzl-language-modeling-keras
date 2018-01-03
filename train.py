from whole_seq_in_one_out_word_based_lm import LanguageModel
import parameters


def main():
    if parameters.FULL_DATA_MODE:
        lm = LanguageModel()
        lm.load_data(train_data_path=parameters.TRAIN_DATA_PATH)
        lm.define_model()
        lm.compile_model()
        lm.fit_model()
        lm.evaluate_model()
    else:
        lm = LanguageModel()
        lm.prepare_for_generator(train_data_path=parameters.TRAIN_DATA_PATH)
        lm.define_model()
        lm.compile_model()
        lm.fit_model_with_generator()
        lm.evaluate_model_with_generator()


if __name__ == '__main__':
    main()
