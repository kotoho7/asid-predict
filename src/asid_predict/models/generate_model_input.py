from typing import Callable
import numpy as np

from tqdm.auto import tqdm

from asid_predict.dataclass import EarthquakeRecord, TrainingRecord

from .normalization import normalize_input, normalize_output


def generate_training_and_test_data(
    earthquakes: list[EarthquakeRecord],
    train_records_from_earthquake: Callable[[EarthquakeRecord], list[TrainingRecord]],
    test_ratio: float = 0.1,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    list[list[TrainingRecord]],
]:
    """学習用・テスト用データを生成"""

    # 学習用データ
    train_records_all: list[TrainingRecord] = []

    # ほとんどプロット用だけのデータ
    train_records_earthquakes: list[list[TrainingRecord]] = []

    # 地震レコードから学習用データ作成
    for earthquake in tqdm(earthquakes, total=len(earthquakes)):
        train_records = train_records_from_earthquake(earthquake)
        train_records_earthquakes.append(train_records)
        train_records_all.extend(train_records)

    # ランダム振り分け
    np.random.shuffle(train_records_all)

    num_test = int(len(train_records_all) * test_ratio)
    num_train = len(train_records_all) - num_test

    train_data = train_records_all[:num_train]
    test_data = train_records_all[num_train:]

    train_input, train_output = _normalize_data(train_data)
    test_input, test_output = _normalize_data(test_data)

    return (
        (np.array(train_input), np.array(train_output)),
        (np.array(test_input), np.array(test_output)),
        train_records_earthquakes,
    )


def _normalize_data(data: list[TrainingRecord]) -> tuple[np.ndarray, np.ndarray]:
    """入力・教師データを作成"""

    normalized_input = [normalize_input(d) for d in data]
    normalized_output = [normalize_output(d.amplification_factor) for d in data]

    return np.array(normalized_input), np.array(normalized_output)
