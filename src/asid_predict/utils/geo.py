"""
距離計算など
"""

import math

__all__ = ["calc_distance"]

# 地球の赤道半径[km]
EQUATORIAL_RADIUS = 6378.137
# 地球の極半径[km]
POLAR_RADIUS = 6356.752


def _degrees_to_radians(degrees: float) -> float:
    """
    度をラジアンに変換
    """
    return math.radians(degrees)


def calc_distance(
    latitude1: float, longitude1: float, latitude2: float, longitude2: float
) -> float:
    """
    2点間の距離を計算(測地線航海算法)

    :param latitude1: 1地点目の緯度
    :param longitude1: 1地点目の経度
    :param latitude2: 2地点目の緯度
    :param longitude2: 2地点目の経度
    :return: 2点間の距離[km]
    """
    # 同じ値の場合距離は0
    if latitude1 == latitude2 and longitude1 == longitude2:
        return 0.0

    # ラジアン変換
    lat_rad1 = _degrees_to_radians(latitude1)
    lon_rad1 = _degrees_to_radians(longitude1)
    lat_rad2 = _degrees_to_radians(latitude2)
    lon_rad2 = _degrees_to_radians(longitude2)

    # 化成緯度
    reduced_lat1 = math.atan((POLAR_RADIUS / EQUATORIAL_RADIUS) * math.tan(lat_rad1))
    reduced_lat2 = math.atan((POLAR_RADIUS / EQUATORIAL_RADIUS) * math.tan(lat_rad2))

    # 球面上の距離
    spherical_distance = math.acos(
        math.sin(reduced_lat1) * math.sin(reduced_lat2)
        + math.cos(reduced_lat1)
        * math.cos(reduced_lat2)
        * math.cos(lon_rad1 - lon_rad2)
    )

    # 0除算回避
    if spherical_distance == 0.0:
        return 0.0

    # 扁平率
    flattening = (EQUATORIAL_RADIUS - POLAR_RADIUS) / EQUATORIAL_RADIUS

    # 距離補正量
    correction = (
        flattening
        / 8.0
        * (
            (math.sin(spherical_distance) - spherical_distance)
            * (math.sin(reduced_lat1) + math.sin(reduced_lat2)) ** 2
            / math.cos(spherical_distance / 2.0) ** 2
            - (math.sin(spherical_distance) + spherical_distance)
            * (math.sin(reduced_lat1) - math.sin(reduced_lat2)) ** 2
            / math.sin(spherical_distance / 2.0) ** 2
        )
    )

    # 距離[km]
    return EQUATORIAL_RADIUS * (spherical_distance + correction)
