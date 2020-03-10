from os import path

BASE_PATH = "datasets/fer2013"

INPUT_PATH = path.sep.join([BASE_PATH, "fer2013/fer2013.csv"])

NUM_CLASSES = 7

TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5   = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5  = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

BATCH_SIZE = 128

OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])
