"""
地震関連の距離減衰式や変換関数など
"""

import math

from asid_predict.dataclass import EarthquakeRecord, TrainingRecord
from .geo import calc_distance

__all__ = [
    "is_pacific_plate_area",
    "calculate_pgv400",
    "calculate_intensity",
    "convert_intensity_to_pgv",
    "convert_pgv_to_intensity",
    "calc_amplification_factor_from_pgv400",
    "calc_pgv400_from_amplification_factor",
]


def is_pacific_plate_area(lon: float, lat: float) -> bool:
    """太平洋プレートの範囲内かどうかを判定"""
    return lat > (-2.2278 * lon) + 329.1898


def calculate_pgv400(distance: float, magnitude: float, depth: float) -> float:
    """距離減衰式で工学的基盤Vs=400m/sでの最大速度を計算[pgv400]"""
    mw = magnitude - 0.171
    min_distance = math.sqrt(distance**2 + depth**2)
    x = max(3, min_distance - (10 ** ((0.5 * mw) - 1.85)))

    pgv600 = 10 ** (
        (0.58 * mw)
        + (0.0038 * depth)
        - 1.29
        - math.log10(x + (0.0028 * (10 ** (0.5 * mw))))
        - (0.002 * x)
    )
    # ((600/400)^0.66) ≒ 1.31
    pgv400 = pgv600 * 1.31

    return pgv400


def calculate_intensity(
    distance: float, mjma: float, depth: float, arv400: float = None
) -> float:
    """距離減衰式で震度を計算"""
    pgv400 = calculate_pgv400(distance, mjma, depth)
    return convert_pgv_to_intensity(pgv400 if arv400 == None else pgv400 * arv400)


def convert_intensity_to_pgv(intensity: float) -> float:
    """震度からPGVを計算"""
    return 10 ** ((intensity - 2.54) / 1.82)


def convert_pgv_to_intensity(pgv: float) -> float:
    """PGVから震度を計算"""
    if pgv > 0:
        return 2.54 + (1.82 * math.log10(pgv))
    else:
        return -99


def calc_amplification_factor_from_pgv400(
    earthuake: EarthquakeRecord, record: TrainingRecord
) -> float:
    """pgv400からamplification_factorを計算"""
    # 計算値
    distance = calc_distance(
        earthuake.lat,
        earthuake.lon,
        record.station_lat,
        record.station_lon,
    )
    calc_pgv400 = calculate_pgv400(
        distance,
        earthuake.magnitude,
        earthuake.depth,
    )

    # 倍率
    amplification_factor_orig = record.pgv400 / calc_pgv400

    return (amplification_factor_orig / 20) ** (
        1 / 4
    )  # PGV指数関数で補間辛そうだから軽減するための変換


def calc_pgv400_from_amplification_factor(
    earthuake: EarthquakeRecord, record: TrainingRecord
) -> float:
    """amplification_factorからpgv400を計算"""
    distance = calc_distance(
        earthuake.lat,
        earthuake.lon,
        record.station_lat,
        record.station_lon,
    )
    calc_pgv400 = calculate_pgv400(
        distance,
        earthuake.magnitude,
        earthuake.depth,
    )

    # 倍率
    amplification_factor_orig = record.amplification_factor**4 * 20

    return amplification_factor_orig * calc_pgv400
