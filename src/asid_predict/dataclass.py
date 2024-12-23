"""
JSONデータ,学習用データ,予測用データなどのデータクラスを定義するモジュール
"""

from pydantic.dataclasses import dataclass


@dataclass
class StationRecord:
    """観測点データ"""

    name: str
    lat: float
    lon: float
    arv400: float
    intensity: float


@dataclass
class EarthquakeRecord:
    """震源・観測点データ"""

    lon: float
    lat: float
    magnitude: float
    depth: float
    name: str
    stations: list[StationRecord]


@dataclass
class TrainingRecord:
    """学習用データ"""

    # モデルの入力になる変数
    magnitude: float
    depth: float
    hypocenter_lat: float
    hypocenter_lon: float
    station_lat: float
    station_lon: float

    # モデルの出力になる変数
    pgv400: float | None
    amplification_factor: float | None  # 補完すると出てくる 最終的にこれ使う


@dataclass
class Earthquake:
    lat: float
    lon: float
    depth: float
    magnitude: float


@dataclass
class ObservationPoint:
    lat: float
    lon: float
    arv400: float


@dataclass
class RegionalObservationPoint:
    name: str
    lat: float
    lon: float
    arv400: float
    region: str
