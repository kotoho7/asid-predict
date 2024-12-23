# ASID-Predict

> [!NOTE]
> これは [防災アプリ開発 Advent Calendar 2024](https://adventar.org/calendars/9939) 24日目の記事 [深発地震でも予想震度を計算したい！](https://note.com/kotoho7/n/n910578ba15bb) で作成したパッケージです。

深発地震に対応した震度予測モデルの作成用プロジェクトです。Google Colaboratoryで使用するため、Pythonパッケージにしています。

## 概要

深発地震に対応した震度予測モデルを作成します。

このパッケージでは以下の内容を行います。

- 地震データの読み込み
- 学習用データの作成
- モデルの学習・保存・読み込み
- 学習済みモデルによる震度予測

このモデルは、深発地震で実際に観測された最大速度(PGV)について、距離減衰式で求めた最大速度からの増幅率を学習します。

詳しくは [note記事](https://note.com/kotoho7/n/n910578ba15bb) にあります。

## Google Colaboratory で実行

[sample.ipynb (Colab)](https://colab.research.google.com/github/kotoho7/asid-predict/blob/main/notebooks/sample.ipynb) で簡易的な学習と震度予測が実行できます。  
※Colabはブラウザ上で動作します

## インストール方法

```bash
pip install git+https://github.com/kotoho7/asid-predict
```

## 使い方

### 学習に使うJSONファイルの用意

以下の形式のJSONファイルを作成し、ファイル名 `data/train_data.json` で保存

```json
{
  "lon": 140.065,
  "lat": 28.862,
  "magnitude": 7.3,
  "depth": 430.0,
  "name": "小笠原諸島西方沖",
  "stations": [
    {
      "name": "CHB001",
      "lat": 35.9571,
      "lon": 139.8731,
      "arv400": 1.6408,
      "intensity": 2.19
    },
    {
      "name": "CHB002",
      "lat": 35.7871,
      "lon": 139.9024,
      "arv400": 1.4195,
      "intensity": 1.71
    }
  ]
}
```

### 学習

簡易的には、`execute_training_process()` 関数を実行することで、地震データの読み込みからモデルの学習,学習済みモデルの保存まで実行されます。

```python
from asid_predict import execute_training_process

# モデルの学習
model = execute_training_process()
```

また、学習済みモデルを使い、`predict_intensities()` 関数で震度予測ができます。

```python
from asid_predict.prediction import predict_intensities, predict_intensities_area

# 震度予測
intensities = predict_intensities(model, stations, earthquake)

# 震度予測(細分区域ごと)
regions = predict_intensities_area(model, stations_with_region_code, earthquake)
```

詳しくは [sample.ipynb](./notebooks/sample.ipynb) に実際に動くコードがあります。
