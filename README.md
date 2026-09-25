# Fisher-LIME

多クラス分類器の局所出力を少数の「クラス間関係を表す軸」に圧縮し、
その軸をLIMEで説明できるかを検証する研究用リポジトリです。

現在の最初の実験は、提案手法を作る前提として、次の仮説を検証します。

> 多クラス分類器の確率出力は、入力空間全体では高次元でも、個々の説明対象の近傍では低次元である。

## 最小実験

合成10クラスデータでMLP分類器を学習し、予測marginが低・中・高の点を選びます。
各点の周囲に摂動を生成して確率行列を作り、重み付きPCAにより以下を測定します。

- 95%および99%の局所出力変動を説明する次元数
- 局所出力の変動量（出力が一定なだけのケースを識別するため）
- 各圧縮次元での確率復元RMSE
- 復元後の予測クラス一致率
- 近傍内に現れたハード予測クラス数

実行環境にはPython 3.10以上と、`numpy`、`pandas`、`scikit-learn`、
`matplotlib`が必要です。

```bash
python3 experiments/run_local_dimension.py
```

短時間の動作確認は次で実行できます。

```bash
python3 experiments/run_local_dimension.py \
  --targets-per-margin 3 \
  --perturbations 100 \
  --radii 0.15 0.4
```

結果は `result/local_dimension/` に生成されます。

- `local_metrics.csv`: 説明対象・近傍幅ごとの測定値
- `reconstruction_curves.csv`: 圧縮次元ごとの復元性能
- `summary.csv`: margin・近傍幅ごとの集計
- `effective_dimension.png`: 95%有効次元の比較図
- `variation_energy.png`: 局所出力変動量の比較図

計算部分のテストは次で実行します。

```bash
python3 -m unittest discover -s tests -v
```

## 大域・局所有効次元の直接比較

合成5・10・20クラスデータとMLP・Random Forestを使い、評価データ全体の
大域有効次元と、各説明対象の最近傍集合における局所有効次元を比較します。
同じ点数を評価データ全体からランダム抽出した対照も設け、少標本による見かけの
次元低下と局所性による次元低下を区別します。

```bash
python experiments/global_local_dimension_study.py
```

結果は `result/global_local_dimension/` に生成されます。

- `all_evaluations.csv`: 説明対象・近傍割合ごとの測定値
- `model_metadata.csv`: BB精度と大域有効次元
- `summary.csv`: margin別の局所・ランダム比較とbootstrap信頼区間
- `compact_summary.csv`: クラス数・BB・近傍割合ごとの集計
- `local_global_dimension_curve.png`: 局所次元と大域次元の比
- `local_vs_random_dimension.png`: 同数の最近傍集合とランダム集合の比較

## 20クラスでの圧縮率・忠実性トレードオフ

20クラス合成データについて、PCA出力軸数を1、2、3、5、8、10、15、19と変え、
通常LIMEとPCA-LIMEを独立した評価摂動上で比較します。BBはロジスティック回帰、
MLP、RBF-SVM、単一決定木、決定木Bagging、Random Forest、勾配ブースティングです。
PCAだけの確率復元誤差と、入力から圧縮軸を予測する局所代理モデルまで含めた忠実性を
分けて記録します。

```bash
python experiments/pca_fidelity_tradeoff_study.py
```

結果は `result/pca_fidelity_tradeoff/` に生成されます。

- `all_evaluations.csv`: 各説明対象・軸数の全評価値
- `dimension_summary.csv`: 固定軸数ごとの忠実性とbootstrap信頼区間
- `adaptive_95_summary.csv`: 95%有効次元を自動選択した結果
- `fidelity_tradeoff.png`: 通常LIMEに対するRMSE増加曲線
- `model_comparison.csv`: BB・軸数別の主要指標
- `model_comparison.png`: 5・8・10軸でのBB横断比較

## 同一近傍での一貫評価（局所次元と忠実性の接続）

同じBB・同じ説明対象・同じ近傍分布で、局所出力次元、競合クラス数、通常LIMEの
係数行列の有効ランク、圧縮のみ・通常LIME・PCA-LIMEのheld-out忠実性を対象ごとに
並べて記録します。近傍はisotropic Gaussian（従来）と、標準化学習データの共分散に
沿ったGaussian（データの線形従属を保つ）を選べます。

```bash
python3 experiments/unified_local_evaluation.py
```

結果は `result/unified_local_evaluation/` に生成されます。

- `neighborhoods.csv`: 近傍ごとの次元・競合クラス数・係数行列ランク・通常LIME忠実性
- `dimension_evaluations.csv`: 近傍×軸数（固定軸、出力q95、線形q95）ごとの評価
- `mechanism_summary.csv`: 「競合クラス数−1」と出力次元・線形次元の比較
- `fidelity_summary.csv`: 忠実性、対応のあるR²損失（seed→対象の階層bootstrap）、
  絶対忠実性と追加損失の合格率。`stratum`で出力ほぼ一定の近傍を分離
- `fidelity_by_seed.csv`: 同じ集計のseed別結果
- `fidelity_by_dimension.png`: 軸数ごとのheld-out R²

## PCA圧縮LIMEと通常LIMEの比較

学習用とは別の摂動点を使い、通常の多出力局所線形回帰と、PCAで出力を
圧縮してから局所線形回帰する方法を比較します。

```bash
python3 experiments/compare_pca_lime.py
```

比較は元のクラス確率空間で行い、圧縮のみの復元誤差、代理モデルを含む誤差、
ハード予測一致率、説明係数数を記録します。結果は `result/pca_lime/` に生成されます。

## Fisher圧縮との比較

PCAの95%有効次元に合わせ、argmax予測を群とするhard Fisherと、BBのクラス確率を
所属度とするsoft Fisherを比較します。

```bash
python3 experiments/compare_fisher_lime.py
```

hard Fisherは局所近傍に1種類の予測クラスしかない場合には定義できないため、
忠実性に加えて手法の利用可能率も記録します。
正則化感度実験に基づき、比較実験のFisher散布行列の正則化係数は既定で1.0です。

## 一般化実験

合成10クラス、Iris、Wine、Digitsに対し、MLPとRandom Forestを3 seedで学習し、
通常LIME、PCA-LIME、hard Fisher-LIME、soft Fisher-LIMEを比較します。

```bash
python3 experiments/generalization_study.py
```

結果は `result/generalization/` に生成されます。全体集計では通常LIMEに対する
RMSE差のbootstrap 95%信頼区間も報告します。

## 安定性実験

同じ説明対象で摂動サンプリングを反復し、復元後の説明係数のcosine類似度、
上位入力特徴のJaccard係数、圧縮部分空間の類似度を比較します。

```bash
python3 experiments/stability_study.py
```

PCA・Fisher軸については、1軸が実効的に何クラスを含むかと、絶対負荷量のうち
上位2クラスが占める割合も記録します。

`paired_summary.csv`は全手法が全反復で利用可能だった近傍だけで比較した結果、
`paired_differences.csv`は同じ近傍での通常LIMEとの差です。上位特徴数が入力特徴数
以上の場合（Irisなど）はJaccardが必ず1になるため、NaNとして報告します。
hard Fisherが要求より少ない軸しか返さなかった割合は`dimension_shortfall_rate`です。

Fisher散布行列の正則化感度は次で確認できます。

```bash
python3 experiments/fisher_regularization_study.py
```
