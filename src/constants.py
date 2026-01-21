### file/dir paths

DATA_DIR = "data"
TRAIN_TSV = DATA_DIR + "/train.tsv"
TRAIN_DIR = DATA_DIR + "/train"

### train image transforms and augmentations

IM_HEIGHT = 200
IM_WIDTH = 800
MAX_LABEL_LENGTH = 40 # based on the dataset

### training params

VALIDATION_SIZE = 0.2
LOG_DIR = 'logs/ocr'