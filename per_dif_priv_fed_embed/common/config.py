from __future__ import annotations
import inspect


class Config:
    LOG_FOLDER = '/home/moshe/GIT/per_dif_priv_fed_embed/log'
    HDF_FILES_DIR = '../putemg-downloader/Data-HDF5'
    FEATURES_DATAFRAMES_DIR = 'features_dataframes'
    DATASET = 'cifar10'
    DATASETS_ROOT_PATH = '/home/moshe/data/datasets/cifar'
    SAVED_MODELS_DIR = '/home/moshe/saved_models/cifar/convnet_with_centroids'
    SAVED_FIGURES_DIR = '/home/moshe/saved_figures/centroids'
    # DEVICE = 'cpu'
    USE_BATCHNORM = False
    USE_DROPOUT = False
    DEVICE = 'cuda'
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    NUM_WORKERS = 2
    OPTIMIZER_TYPE = 'sgd'
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3
    MOMENTUM = 0.9
    UNIFIED_DATA_DIR = '../data/unified_dataframes'
    TENSORS_DATA_DIR = '../data/tensors_datasets'
    WINDOWED_DATA_DIR = '../data/windowed_tensors_datasets'
    NUM_CLASSES = 7
    WINDOW_SIZE = 260
    EVAL_EVERY = 1
    TEST_AT_END = True
    NUM_CLIENT_AGG = 5
    NUM_INTERNAL_EPOCHS = 1
    WRITE_TO_WANDB = False
    # DP
    ADD_DP_NOISE = False
    LOT_SIZE_IN_BATCHES = 5
    DP_C = 5
    DP_EPSILON = 1.0
    DP_DELTA = 0.001
    DP_SIGMA = 0.01 * 3.776479532659047  # sqrt(2 * log(1.25 / DP_DELTA))

    @staticmethod
    def to_dict() -> dict:
        members = inspect.getmembers(Config, lambda a: not (inspect.isroutine(a)))
        members = [(k, v) for (k, v) in members if not k.startswith('_')]
        return dict(members)
