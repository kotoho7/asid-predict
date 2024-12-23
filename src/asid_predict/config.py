"""
モデルの設定など定数たち
"""

# Model version
VERSION = 12.1

# Model parameters
INPUT_DIMS = (
    6  # [magnitude, depth, hypocenter_lat, hypocenter_lon, station_lat, station_lon]
)
DENSE_UNITS = 1000
DROPOUT_RATE = 0.1
HIDDEN_LAYERS = 7
OUTPUT_DIMS = 1  # 予測震度

# Training parameters
BATCH_SIZE = 128
EPOCHS = 40
VALIDATION_SPLIT = 0.1

# Path
TRAIN_DATA = "data/train_data.json"

SAVE_PATH = "out"
