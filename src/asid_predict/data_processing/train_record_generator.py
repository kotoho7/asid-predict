"""
観測データの補間や水増で学習データTrainingRecordを生成するクラス
"""

import random

from asid_predict.dataclass import EarthquakeRecord, TrainingRecord
from asid_predict.utils import (
    calc_amplification_factor_from_pgv400,
    calculate_intensity,
    convert_intensity_to_pgv,
    convert_pgv_to_intensity,
    calc_distance,
)
from .interpolation import interpolate_train_records

# 補間多すぎるとつらいから割合を決める

INTERPOLATE_RATE = 0.3
INTERPOLATE_RATE_FAR = 0.3
KYORI_GENSUI_RATE = 0.02


class TrainingRecordGenerator:
    def __init__(self, predict_points: dict, coast_points: dict):
        self.predict_points = predict_points
        self.coast_points = coast_points

    def from_earthquake(self, earthuake: EarthquakeRecord) -> list[TrainingRecord]:
        """学習用データ作成"""

        # 各種データの作成
        records_raw = self._create_raw_records(earthuake)
        records_dup = self._create_duplicate_records(earthuake, records_raw)
        records_coast = self._create_coast_records(earthuake, records_raw)
        records_simple_i = self._create_instant_records(earthuake, records_raw)
        records_interpolate = self._create_interpolate_records(
            earthuake, records_raw, records_coast, records_simple_i
        )

        return (
            # 実測データ
            records_raw
            # 実測の複製水増ししデータ
            + records_dup
            # 補間データ
            + random.sample(
                records_interpolate,
                min(len(records_interpolate), int(len(records_raw + records_dup) * 20)),
            )
            # 揺れない場所データもちょっと入れよう
            + random.sample(
                records_coast,
                int(min(len(records_coast) * 0.1, len(records_raw + records_dup) * 10)),
            )
            # 簡易補間データもちょっとだけ入れよう
            + random.sample(
                records_simple_i,
                int(
                    min(
                        len(records_simple_i) * 0.05,
                        len(records_raw + records_dup) * 10,
                    )
                ),
            )
        )

    def _create_raw_records(self, earthuake: EarthquakeRecord) -> list[TrainingRecord]:
        """実測データの作成"""
        records_raw: list[TrainingRecord] = []
        sorted_stations = sorted(
            earthuake.stations, key=lambda x: x.intensity, reverse=True
        )

        for stataion in sorted_stations:
            PGVobs = convert_intensity_to_pgv(stataion.intensity)
            PGVobs400 = PGVobs * (1 / stataion.arv400)

            record = TrainingRecord(
                magnitude=earthuake.magnitude,
                depth=earthuake.depth,
                hypocenter_lat=earthuake.lat,
                hypocenter_lon=earthuake.lon,
                station_lat=stataion.lat,
                station_lon=stataion.lon,
                pgv400=PGVobs400,
                amplification_factor=None,
            )

            # 周囲30km以内に自分より2倍以上PGV400が高い観測点があれば除外
            if self._can_add_station(records_raw, record, 30, 3):
                continue

            records_raw.append(record)

        self._calc_amplification_factor(earthuake, records_raw)
        return records_raw

    def _create_coast_records(
        self, earthuake: EarthquakeRecord, records_raw: list[TrainingRecord]
    ) -> list[TrainingRecord]:
        """揺れない場所データ(補間用)の作成"""
        records_coast = self._gen_instant_interpolate_points(
            earthuake, records_raw, self.coast_points, self._coast_pick_rate
        )
        self._calc_amplification_factor(earthuake, records_coast)
        return records_coast

    def _create_instant_records(
        self, earthuake: EarthquakeRecord, records_raw: list[TrainingRecord]
    ) -> list[TrainingRecord]:
        """ちょっと水増しデータ(補間用)の作成"""
        records_instant: list[TrainingRecord] = []
        for record in self._gen_instant_interpolate_points(
            earthuake, records_raw, self.predict_points, self._instant_pick_rate
        ):
            if self._can_add_station(records_raw, record, 80, 0):
                continue

            records_instant.append(record)
        self._calc_amplification_factor(earthuake, records_instant)
        return records_instant

    def _create_interpolate_records(
        self,
        earthuake: EarthquakeRecord,
        records_raw: list[TrainingRecord],
        records_coast: list[TrainingRecord],
        records_instant: list[TrainingRecord],
    ) -> list[TrainingRecord]:
        """補間データの作成"""
        max_distance = self._calc_max_distance(earthuake)
        records_interpolate: list[TrainingRecord] = []
        random_predict_points = random.sample(
            self.predict_points, int(len(self.predict_points) * 0.1)
        )

        for record in interpolate_train_records(
            records_raw + records_coast + records_instant,
            random_predict_points,
        ):
            # そもそも減衰式の範囲外なら除外
            distance = calc_distance(
                earthuake.lat,
                earthuake.lon,
                record.station_lat,
                record.station_lon,
            )
            if distance > max_distance:
                continue

            # 周囲100km以内に自分より3倍以上PGV400が高い観測点があれば除外
            if self._can_add_station(records_raw, record, 100, 3):
                continue

            records_interpolate.append(record)

        return records_interpolate

    def _create_duplicate_records(
        self, earthquake: EarthquakeRecord, records_raw: list[TrainingRecord]
    ) -> list[TrainingRecord]:
        """生データの水増しデータを作成"""
        records_dup = []

        for record in records_raw:
            if record.pgv400 <= convert_intensity_to_pgv(0.0):
                continue

            # pgv400の値に基づいて水増し回数を決定（最大300回）
            dup_count = int(record.pgv400**0.6 * 10)

            for _ in range(dup_count):
                # 緯度経度を±0.1°の範囲でランダムに変更
                new_lat = record.station_lat + random.uniform(-0.1, 0.1)
                new_lon = record.station_lon + random.uniform(-0.1, 0.1)

                # pgv400とamplification_factorに0.9-1.1の乱数をかける
                amplification_factor = random.uniform(0.9, 1.1)
                new_pgv400 = record.pgv400 * amplification_factor
                new_amplification_factor = (
                    record.amplification_factor * amplification_factor
                )

                # 新しいレコードを作成
                new_record = TrainingRecord(
                    magnitude=record.magnitude,
                    depth=record.depth,
                    hypocenter_lat=record.hypocenter_lat,
                    hypocenter_lon=record.hypocenter_lon,
                    station_lat=new_lat,
                    station_lon=new_lon,
                    pgv400=new_pgv400,
                    amplification_factor=new_amplification_factor,
                )
                records_dup.append(new_record)

        return records_dup

    def _can_add_station(
        self,
        existing: list[TrainingRecord],
        add: TrainingRecord,
        neighbor_range: float = 20,
        neighbor_threshold: float = 2,
    ) -> bool:
        """周辺に自分よりn倍以上PGV400が高い観測点があるか判定"""
        for record in existing:
            d = calc_distance(
                add.station_lat, add.station_lon, record.station_lat, record.station_lon
            )
            if d < neighbor_range and record.pgv400 * neighbor_threshold > add.pgv400:
                return True

        return False

    def _calc_amplification_factor(
        self, earthuake: EarthquakeRecord, records: list[TrainingRecord]
    ):
        """TrainingRecordリストのamplification_factorを計算して入れる"""

        for i in range(len(records)):
            records[i].amplification_factor = calc_amplification_factor_from_pgv400(
                earthuake,
                records[i],
            )

    def _gen_instant_interpolate_points(
        self,
        earthuake: EarthquakeRecord,
        records_raw: list[TrainingRecord],
        predict_points: dict,
        joken,
    ) -> list[TrainingRecord]:
        """一番近い観測点から簡易的に補間"""

        records_interpolate: list[TrainingRecord] = []
        for point in predict_points:

            # 一定確率で除外
            if random.random() > INTERPOLATE_RATE:
                continue

            LAT = point["lat"]
            LON = point["lon"]

            # 最も近い震度データがある点を求める
            nearest_distance = 999999
            nearest_record = None
            for record_raw in records_raw:
                distance = calc_distance(
                    LAT, LON, record_raw.station_lat, record_raw.station_lon
                )
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_record = record_raw

            # 距離の条件を満たす場合に計算
            if nearest_record is None or not joken(LAT, LON, nearest_record):
                continue

            for_calc_distance = calc_distance(
                LAT,
                LON,
                nearest_record.station_lat,
                LON + (nearest_record.station_lon - LON) * 2.5,
            )

            # 最も近いPGV400からの距離減衰
            PGV400 = convert_intensity_to_pgv(
                convert_pgv_to_intensity(nearest_record.pgv400)
                - KYORI_GENSUI_RATE * for_calc_distance
            )

            records_interpolate.append(
                TrainingRecord(
                    magnitude=earthuake.magnitude,
                    depth=earthuake.depth,
                    hypocenter_lat=earthuake.lat,
                    hypocenter_lon=earthuake.lon,
                    station_lat=LAT,
                    station_lon=LON,
                    pgv400=PGV400,
                    amplification_factor=None,
                )
            )

        return records_interpolate

    def _coast_pick_rate(self, lat: float, lon: float, record: TrainingRecord) -> bool:
        distance = calc_distance(lat, lon, record.station_lat, record.station_lon)
        return (
            80 < distance and distance < 300
        ) or random.random() < INTERPOLATE_RATE_FAR

    def _instant_pick_rate(
        self, lat: float, lon: float, record: TrainingRecord
    ) -> bool:
        # 西に経度2度分遠ければ揺れないでしょう
        if lon < record.station_lon - 2:
            return random.random() < INTERPOLATE_RATE_FAR

        distance = calc_distance(lat, lon, record.station_lat, record.station_lon)
        return (
            30 < distance and distance < 100 and random.random() < INTERPOLATE_RATE_FAR
        )

    def _calc_max_distance(self, earthuake: EarthquakeRecord) -> float:
        """学習データの最大距離を計算(従来法震度-3以上)"""
        # 範囲制限計算(100kmごとに計算)
        for distance in range(500, 2500, 100):
            intensity = calculate_intensity(
                distance, earthuake.magnitude, earthuake.depth
            )
            if intensity < -3:
                return distance

        return 2500
