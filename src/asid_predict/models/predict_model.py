"""
予測モデルの定義
"""

import os

import numpy as np
import keras

from asid_predict.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    INPUT_DIMS,
    OUTPUT_DIMS,
    SAVE_PATH,
    VERSION,
)
from asid_predict.data_processing.train_record_generator import TrainingRecordGenerator
from asid_predict.dataclass import EarthquakeRecord
from .generate_model_input import generate_training_and_test_data

__all__ = ["PredictModel"]


def _build_model() -> keras.Model:
    """モデル構造を作成"""
    model = keras.Sequential(
        [
            keras.Input(shape=(INPUT_DIMS,), name="input"),
            keras.layers.Dense(DENSE_UNITS),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dropout(DROPOUT_RATE),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dense(DENSE_UNITS, activation="sigmoid"),
            keras.layers.Dense(OUTPUT_DIMS, name="output"),
        ]
    )

    return model


class PredictModel:
    def __init__(self):
        self.model = _build_model()
        self.compile_model()

    def initialize_dataset_for_training(
        self,
        earthquakes: list[EarthquakeRecord],
        train_data_generator: TrainingRecordGenerator,
        test_ratio: float = 0.1,
    ) -> list[list[EarthquakeRecord]]:
        """学習用データセットを初期化"""
        (train_input, train_output), test_data, train_records_earthquakes = (
            generate_training_and_test_data(
                earthquakes, train_data_generator, test_ratio
            )
        )

        self.x_train = train_input
        self.y_train = train_output
        self.test_data = test_data

        return train_records_earthquakes

    def compile_model(self):
        """モデルをコンパイル"""
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def execute_training(
        self,
        x_train: np.ndarray = None,
        y_train: np.ndarray = None,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> keras.callbacks.History:
        """学習を実行"""
        return self.model.fit(
            x_train if x_train else self.x_train,
            y_train if y_train else self.y_train,
            epochs=epochs,
            batch_size=batch_size,
        )

    def evaluate(self, x_test: np.ndarray = None, y_test: np.ndarray = None):
        """精度の確認"""
        if x_test is None or y_test is None:
            x_test, y_test = self.test_data
        self.model.evaluate(x_test, y_test)

    def _generate_filename(self, file_extension: str) -> str:
        """ファイル名を生成"""

        # モデルの情報を取得
        n_layers = len(self.model.layers)
        history = self.model.history.history if hasattr(self.model, "history") else None
        epochs = len(history["loss"]) if history else 0
        batch_size = (
            self.model.optimizer.iterations.numpy() // epochs if epochs > 0 else 0
        )
        loss = history["loss"][-1] if history else 0

        # ファイル名を生成
        return f"asid_{VERSION}_n{n_layers}_e{epochs}_b{batch_size}_l{loss:.5f}.{file_extension}"

    def save(self, save_dir: str):
        """モデルを保存"""
        filename = self._generate_filename("keras")
        filepath = os.path.join(save_dir, filename)
        self.model.save(filepath)

    def save_weight(self, save_dir: str = SAVE_PATH):
        """モデルの重みを保存"""
        filename = self._generate_filename("weights.h5")
        filepath = os.path.join(save_dir, filename)
        self.model.save_weights(filepath)

    def load(self, filepath: str):
        """保存されたモデルを読み込む"""
        self.model = keras.models.load_model(filepath)

    def load_weight(self, filepath: str):
        """保存されたモデルの重みを読み込む"""
        self.model.load_weights(filepath)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """予測"""
        return self.model.predict(x)
