from whole_seq_in_one_out_word_based_lm import LanguageModel
import parameters


def main():
    lm = LanguageModel()
    lm.load_data(train_data_path=parameters.TRAIN_DATA_PATH)
    lm.load_model()
    lm.compile_model()
    lm.fit_model()
    lm.evaluate_model()


if __name__ == '__main__':
    main()
