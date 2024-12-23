"""
モデルを使った震度予測
"""

import numpy as np

from asid_predict.utils import (
    calc_pgv400_from_amplification_factor,
    convert_pgv_to_intensity,
)

from asid_predict.dataclass import (
    TrainingRecord,
    Earthquake,
    ObservationPoint,
    RegionalObservationPoint,
)

from asid_predict.models import PredictModel, normalize_input
from asid_predict.models import reverse_normalize_output


def predict_intensities(
    model: PredictModel,
    targets: list[ObservationPoint],
    eq: Earthquake,
) -> list[float]:
    """個別地点の震度予測"""

    # モデルに入力する値に変換
    data: list[TrainingRecord] = [
        TrainingRecord(
            eq.magnitude,
            eq.depth,
            eq.lat,
            eq.lon,
            p.lat,
            p.lon,
            None,
            None,
        )
        for p in targets
    ]

    # 予測実行
    x = np.array([normalize_input(d) for d in data])
    y = model.predict(x)

    # 計測震度に変換
    intensities: list[float] = []
    for i in range(len(data)):
        record = data[i]
        record.amplification_factor = reverse_normalize_output(y[i])
        record.pgv400 = calc_pgv400_from_amplification_factor(eq, record)

        arv400 = targets[i].arv400
        pgv = record.pgv400 * arv400
        intensity = convert_pgv_to_intensity(pgv)

        intensities.append(intensity)

    return intensities


def predict_intensities_area(
    model: PredictModel,
    targets: list[RegionalObservationPoint],
    eq: Earthquake,
) -> list[dict]:
    """細分区域ごとの震度予測"""

    points = [ObservationPoint(t.lat, t.lon, t.arv400) for t in targets]
    result = predict_intensities(model, points, eq)

    regions_dict = {}

    for i in range(len(targets)):
        intensity = result[i]
        target = targets[i]

        regions_dict[target.region] = max(
            regions_dict.get(target.region, intensity), intensity
        )

    regions = sorted(
        [
            {"code": region, "maxInt": intensity}
            for region, intensity in regions_dict.items()
        ],
        key=lambda x: x["maxInt"],
        reverse=True,
    )

    return regions
