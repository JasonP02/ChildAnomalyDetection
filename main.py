# main

from test import test_model
from train import train_standard_model
from param_tune import tune_hyperparameters
from data_processing import process_videos_in_folder, YOLO_model, get_adjacency_matrix, preprocess_training, preprocess_testing
from config import Config


def main(tune=False, preprocess=False, test=False, train=True):
    #
    #
    #
    #
    #

    if tune:
        tune_hyperparameters()
    elif preprocess and train:
        train_standard_model(preprocess=True)
    elif not preprocess and train:
        train_standard_model(preprocess=False)
    elif preprocess and test:
        test_model(preprocess=True)
    elif not preprocess and test:
        test_model(preprocess=False)
    elif preprocess and not train and not test and not tune:
        config = Config()
        process_videos_in_folder(config.training_videos, YOLO_model(), config.training_graphs, get_adjacency_matrix())
        process_videos_in_folder(config.testing_videos, YOLO_model(), config.training_graphs, get_adjacency_matrix())
        preprocess_training()
        preprocess_testing()



if __name__ == "__main__":
    main()  # Set to True for hyperparameter tuning, False for standard training
