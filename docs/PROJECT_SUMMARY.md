# Project Summary

## 目的

多クラス分類ブラックボックス（BB）の局所確率出力を少数の軸へ圧縮し、
クラス間関係として説明するFisher-LIMEの可能性を検証する。

通常のLIMEはクラスごとの確率を個別に局所回帰するため、クラス数が多いと
説明項目が増え、確率がどのクラス（群）からどのクラス（群）へ移ったかを
読み取りにくい。本プロジェクトでは入力特徴の解釈可能性を維持したまま、
BBの出力側を局所的に圧縮する。

## 現在の研究仮説

1. 多クラスBBの確率出力は、説明対象の局所近傍では最大の `n - 1` より低次元である。
2. 必要な局所次元はクラス数、BB、予測margin、近傍の広さに依存する。
3. 圧縮軸を説明するLIMEは、通常LIMEと同程度の忠実性・安定性を、より少ない
   説明項目で達成できる可能性がある。

## 構成

- `fisher_lime/local_dimension.py`: 重み付きPCAと局所出力次元の測定
- `fisher_lime/fisher_projection.py`: hard/soft Fisher射影と確率空間への復元
- `fisher_lime/surrogate.py`: 重み付き多出力Ridge代理モデルと忠実性指標
- `fisher_lime/diagnostics.py`: 競合クラス数、LIME係数行列の有効ランク、
  isotropic／データ共分散Gaussian近傍の生成とデータspan外成分の測定
- `experiments/unified_local_evaluation.py`: 同一BB・同一対象・同一近傍分布で、
  局所次元・競合クラス数・係数行列ランク・圧縮のみ／通常LIME／PCA-LIMEの
  held-out忠実性を対象ごとに記録する主実験
- `experiments/run_local_dimension.py`: 合成データによる局所低次元性の初期実験
- `experiments/global_local_dimension_study.py`: 評価データ上の大域・局所次元と
  同数ランダム対照の比較（補助実験。最近傍集合はLIME近傍より広く、次元は近傍内の
  予測クラス数とほぼ連動する）
- `experiments/compare_pca_lime.py`: 通常LIMEとPCA-LIMEの比較
- `experiments/pca_fidelity_tradeoff_study.py`: 20クラス・7種類のBBでの固定PCA軸数と
  忠実性・説明量のトレードオフ
- `experiments/compare_fisher_lime.py`: PCA、hard Fisher、soft Fisherの比較
- `experiments/generalization_study.py`: データ・BB・seedを広げた忠実性比較
- `experiments/stability_study.py`: 摂動反復による説明安定性比較
- `experiments/fisher_regularization_study.py`: Fisher散布行列の正則化感度
- `tests/test_local_dimension.py`: 数値計算の単体テスト
- `reports/pre_xai_dimension_report/`: XAI代理モデル学習前の局所出力次元実験を説明する
  Data appレポートのソース
- `reports/pre_xai_dimension_report.html`: 上記レポートを単体で閲覧できるHTML
- `references/`: 関連論文
- `result/`: 再生成可能な実験結果（Git管理外）

## 重要な設計判断

- BBへの入力は圧縮せず、BBが返す多クラス確率ベクトルを圧縮対象とする。
- 局所低次元性そのものをFisherの性能から切り離すため、復元誤差を最小化する
  PCAを基準手法とする。
- hard FisherはBBのargmaxを群ラベルとする。soft FisherはBBの各クラス確率を
  所属度としてクラス平均と散布行列を計算する。
- PCAとFisherの比較次元は、学習近傍で95%の出力分散を説明するPCA次元にそろえる。
- Fisherの局所within-scatterは悪条件になりやすいため、感度実験に基づき、
  比較実験では正則化係数1.0を既定値とする。
- 低変動な出力を「低次元」と過大評価しないため、有効次元と出力変動量を併記する。
- 局所低次元性の検証では、同数のランダム集合を対照とし、標本数による次元低下と
  近傍選択による次元低下を区別する。
- 忠実性は圧縮空間ではなく、復元した元のクラス確率空間で評価する。
- 安定性では軸の符号・回転不定性を避けるため、復元後の入力→クラス効果行列と
  射影行列を比較する。
- 全特徴を使う多出力Ridgeは目的変数について線形なので、PCA-LIMEの予測は
  「通常LIMEの予測を出力PCA部分空間へ射影したもの」と厳密に一致する。したがって
  圧縮後の説明は通常LIME係数行列の低ランク近似であり、忠実性に効くのは出力点群の
  PCA次元だけでなく係数行列の有効ランク（`linear_q95`）である。
- 局所低次元性と説明の忠実性は、同じ近傍分布で対象ごとに対応づけて評価する
  （`unified_local_evaluation.py`）。PCA学習用と評価用の摂動は分ける。
- 忠実性は「圧縮のみのR²」「通常LIMEの絶対R²」「圧縮による追加損失」を分けて
  報告する。RMSE増加率（%）は通常LIMEの誤差が分母なので、BB間比較には使わない。
- 出力がほぼ一定の近傍（総分散1e-4未満）は圧縮が自明に成功するため別集計する。
- 同じseedの行は同じBBを共有するため、信頼区間はseed→対象の階層bootstrapで求め、
  seed別結果も出す。3 seedでは探索的な区間として扱う。
- 手法間の安定性比較は、全手法が利用可能だった近傍に限定した対応のある比較を主とする。

## 実行方法

```bash
python3 experiments/run_local_dimension.py
python3 experiments/unified_local_evaluation.py
python experiments/global_local_dimension_study.py
python3 experiments/compare_pca_lime.py
python experiments/pca_fidelity_tradeoff_study.py
python3 experiments/compare_fisher_lime.py
python3 experiments/generalization_study.py
python3 experiments/stability_study.py
python3 experiments/fisher_regularization_study.py
python3 -m unittest discover -s tests -v
```

XAI学習前の実験レポートは`reports/pre_xai_dimension_report.html`をブラウザで開く。
レポートの再ビルドにはCodex Data Analyticsプラグインの`data-app.mjs build`および
`export-offline`を使用する。

必要なライブラリは `numpy`、`pandas`、`scikit-learn`、`matplotlib`。

## 既知の制約

- 人による解釈性評価は未実装。軸のクラス負荷集中度は代理指標に留まる。
- 現在の比較は密なRidge回帰であり、LIMEらしい疎な特徴選択は未実装。
- 実データへのGaussian摂動がデータ多様体上にある保証はない。合成データでは40特徴中
  16特徴が線形従属（標準化データのrank 24）で、isotropic摂動の二乗ノルムの約40%が
  データspan外に出る。データ共分散Gaussian近傍で線形の範囲では解消できるが、
  非線形な多様体への対応は未実装。
- 局所出力の低次元性と「理解しやすい説明」の間はまだつながっていない。係数数の削減は
  測っているが、人にとっての理解しやすさは未評価。
- 通常LIME自体の絶対忠実性が低い条件（RF、半径0.4など）が多く、圧縮による追加損失が
  小さくても忠実な説明とは言えない。
- 95%出力分散で選んだ次元は学習近傍での値であり、held-out摂動で95%を保つ保証はない
  （20クラスRFでは約3割の近傍で下回る）。
- soft Fisherは出力が数値的に一定な近傍では判別軸を定義できない。
- PCAもFisherも、軸上で正負に分かれるクラス群が意味的にまとまる保証はない。
- 必要な出力軸数はBB依存であり、特にRandom Forestと決定木Baggingでは他のBBより
  多くの軸が必要だった。RFの高いq95は、予測クラスが1つでも多数クラスの票が小さく
  揺れることによる部分が大きく、q95は変動の大きさを無視する点に注意する。単一決定木は通常LIME自体の局所忠実性が低い。
