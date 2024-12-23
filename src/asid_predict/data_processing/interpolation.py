"""
補間の実装
"""

import numpy as np
from pykrige.ok import OrdinaryKriging

from asid_predict.dataclass import TrainingRecord
from asid_predict.utils import calc_distance, calculate_pgv400


def interpolate_train_records(
    records: list[TrainingRecord],
    predict_points: list,
) -> list[TrainingRecord]:
    if len(records) < 3:
        # 3点未満なら補間しない
        return []

    x = np.array([[r.station_lon, r.station_lat] for r in records])
    y = np.array([[r.amplification_factor] for r in records])

    predict_x = np.array([[p["lon"], p["lat"]] for p in predict_points])

    try:
        # nuggetパラメータを追加して数値的安定性を向上
        ok = OrdinaryKriging(
            x[:, 0],
            x[:, 1],
            y,
            variogram_model="spherical",
        )
        zvalues, sigmasq = ok.execute("points", predict_x[:, 0], predict_x[:, 1])
    except Exception as e:
        print(f"補間エラー: {str(e)}")
        print(f"入力データ数: {len(records)}")
        raise

    # 結果をTrainingRecordに変換
    reference_record = records[0]
    return [
        _create_interpolated_record(
            reference_record=reference_record,
            predict_point=predict_x[i],
            interpolated_value=zvalues[i],
        )
        for i in range(len(predict_x))
    ]


def _create_interpolated_record(
    reference_record: TrainingRecord,
    predict_point: np.ndarray,
    interpolated_value: float,
) -> TrainingRecord:
    """補間された地点のTrainingRecordを作成"""
    station_lon, station_lat = predict_point[0], predict_point[1]

    # 震源と観測点の距離を計算
    distance = calc_distance(
        reference_record.hypocenter_lat,
        reference_record.hypocenter_lon,
        station_lat,
        station_lon,
    )

    # 距離減衰式でPGV400を計算し、補間された倍率を適用
    base_pgv400 = calculate_pgv400(
        distance=distance,
        magnitude=reference_record.magnitude,
        depth=reference_record.depth,
    )
    interpolated_pgv400 = base_pgv400 * interpolated_value

    return TrainingRecord(
        magnitude=reference_record.magnitude,
        depth=reference_record.depth,
        hypocenter_lat=reference_record.hypocenter_lat,
        hypocenter_lon=reference_record.hypocenter_lon,
        station_lat=station_lat,
        station_lon=station_lon,
        pgv400=interpolated_pgv400,
        amplification_factor=interpolated_value,
    )
