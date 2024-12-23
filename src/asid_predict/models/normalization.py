"""
学習データの正規化など
"""

from asid_predict.dataclass import TrainingRecord

__all__ = [
    "normalize_input",
    "normalize_output",
    "reverse_normalize_input",
    "reverse_normalize_output",
]


def _normalize_range(value: float, start: float, end: float) -> float:
    """指定した範囲内で正規化"""
    return (value - start) / (end - start)


def _reverse_normalize_range(value: float, start: float, end: float) -> float:
    """指定した範囲内で逆正規化"""
    return (value * (end - start)) + start


def normalize_input(record: TrainingRecord) -> list[float]:
    """入力データの正規化"""
    return [
        _normalize_range(record.magnitude, 2.0, 9.0),  # マグニチュード 2.0 ~ 9.0
        _normalize_range(record.depth, 0, 800),  # 深さ 0 ~ 800km
        _normalize_range(record.hypocenter_lat, 20.0, 50.0),  # 震源緯度 20° ~ 50°
        _normalize_range(record.hypocenter_lon, 120.0, 150.0),  # 震源経度 120° ~ 150°
        _normalize_range(record.station_lat, 20.0, 50.0),  # 観測点緯度 20° ~ 50°
        _normalize_range(record.station_lon, 120.0, 150.0),  # 観測点経度 120° ~ 150°
    ]


def normalize_output(amplification_factor: float) -> list[float]:
    """出力データの正規化"""
    return [max(min(amplification_factor, 1), 0)]  # 0 ~ 1


def reverse_normalize_input(input_val: list[float]) -> TrainingRecord:
    """入力データの逆正規化"""
    return TrainingRecord(
        _reverse_normalize_range(input_val[0], 2.0, 9.0),  # マグニチュード 2.0 ~ 9.0
        _reverse_normalize_range(input_val[1], 0, 800),  # 深さ 0 ~ 800km
        _reverse_normalize_range(input_val[2], 20.0, 50.0),  # 震源緯度 20° ~ 50°
        _reverse_normalize_range(input_val[3], 120.0, 150.0),  # 震源経度 120° ~ 150°
        _reverse_normalize_range(input_val[4], 20.0, 50.0),  # 観測点緯度 20° ~ 50°
        _reverse_normalize_range(input_val[5], 120.0, 150.0),  # 観測点経度 120° ~ 150°
        None,
        None,
    )


def reverse_normalize_output(input_val: float) -> float:
    """出力データの逆正規化"""
    return input_val[0]  # 0 ~ 1は変わらないのでそのまま返す
