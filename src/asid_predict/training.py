"""
学習を行ってモデルを保存する1連の流れを行う関数
"""

from asid_predict.data_processing.data_file_loader import DataFileLoader
from asid_predict.data_processing.train_record_generator import TrainingRecordGenerator
from asid_predict.models.predict_model import PredictModel


def execute_training_process(
    train_json_path: str = None,
    target_is_pasific_plate: bool = True,
    min_depth: float = 120,
    epochs: int = 30,
    batch_size: int = 64,
    save_path: str = None,
) -> PredictModel:
    """学習"""

    # 地震データ, 予測点データの読み込み
    print("1/5 地震データと予測点データの読み込み")
    data_loader = DataFileLoader(train_json_path)
    train_earthquakes = data_loader.get_filtered_earthquakes(
        target_is_pasific_plate, min_depth
    )

    # 学習用データ生成用クラス
    training_data_generator = TrainingRecordGenerator(
        data_loader.predict_points, data_loader.coast_points
    )

    # 学習モデルの初期化
    model = PredictModel()

    # 地震データと学習用データ生成クラスから学習用データを初期化
    print("2/5 学習用データの作成")
    model.initialize_dataset_for_training(
        earthquakes=train_earthquakes,
        train_data_generator=training_data_generator.from_earthquake,
    )

    # 学習を実行
    print("3/5 モデルの学習")
    model.execute_training(
        epochs=epochs,
        batch_size=batch_size,
    )

    # 精度の確認
    print("4/5 モデルの評価")
    model.evaluate()

    # 学習済みモデルを保存
    print("5/5 学習済みモデルの保存")
    model.save_weight(save_path)

    return model
