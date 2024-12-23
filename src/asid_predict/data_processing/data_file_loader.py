"""
ファイルから観測データの読み込み
"""

import importlib.resources as resources
import json

from asid_predict.config import TRAIN_DATA
from asid_predict.dataclass import EarthquakeRecord
from asid_predict.utils import is_pacific_plate_area

__all__ = ["DataFileLoader"]


class DataFileLoader:
    def __init__(self, eq_data_json_path: str = TRAIN_DATA):
        # 地震データ
        with open(eq_data_json_path if eq_data_json_path else TRAIN_DATA, "r") as f:
            earthquake_dict = json.load(f)
            self.earthquakes = [EarthquakeRecord(**d) for d in earthquake_dict]

        # 予測点データ
        with resources.open_text("asid_predict.data", "predict_points.json") as f:
            self.predict_points = json.load(f)

        # 揺れないほうの海岸点データ
        with resources.open_text("asid_predict.data", "coast_points.json") as f:
            self.coast_points = json.load(f)

    def _is_target_earthquake(
        self,
        earthquake: EarthquakeRecord,
        target_is_pasific_plate: bool,
        min_depth: float = 120,
        min_lon: float = 120,
        max_lon: float = 150,
        min_lat: float = 20,
        max_lat: float = 50,
    ) -> bool:
        """学習対象の地震かどうかを判定"""

        # どっちのプレート
        is_pacific = is_pacific_plate_area(earthquake.lon, earthquake.lat)
        target_plate = is_pacific if target_is_pasific_plate else not is_pacific

        # 震源位置
        area = (
            min_lon < earthquake.lon
            and earthquake.lon < max_lon
            and min_lat < earthquake.lat
            and earthquake.lat < max_lat
        )

        # 深さ
        depth = earthquake.depth > min_depth

        return target_plate and area and depth

    def get_filtered_earthquakes(
        self, target_is_pasific_plate: bool, min_depth: float = 120
    ):
        """学習対象の地震のみを取得"""
        filtered_earthquake_records: list[EarthquakeRecord] = filter(
            lambda earthquake: self._is_target_earthquake(
                earthquake,
                target_is_pasific_plate,
                min_depth,
            ),
            self.earthquakes,
        )
        return list(filtered_earthquake_records)
