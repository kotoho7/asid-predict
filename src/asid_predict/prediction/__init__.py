"""
学習済みモデルを用いて予測を行うためのモジュール
"""

from .predictor import predict_intensities, predict_intensities_area

__all__ = ["predict_intensities", "predict_intensities_area"]
