import splitfolders


def split_data(input_path):
    splitfolders.ratio(input_path, output="../../data",
                       seed=1337, ratio=(.6, .2, .2), group_prefix=None, move=True)


if __name__ == "__main__":
    split_data("../../data/sign_language_dataset")
