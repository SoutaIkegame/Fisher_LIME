# PCA・直交回転・Sparse PCA 比較実験計画

## 1. 目的

局所摂動に対するブラックボックスのクラス確率

\[
P=[p_1,\ldots,p_N]^\top\in\mathbb{R}^{N\times K}
\]

を少数軸へ圧縮し、入力特徴からその軸を説明する場合に、次の3方式を比較する。

1. 局所重み付きPCA
2. 局所重み付きPCAの直交Varimax回転
3. 局所重み付きSparse PCA

比較したい問いは次の3点である。

- 同じ軸数で、元のクラス確率と予測クラスをどこまで保存できるか。
- 各軸を、少数のクラスと少数の入力特徴で説明できるか。
- 摂動標本を作り直したときにも、同じ軸と説明が得られるか。

## 2. 重要な理論上の対照

PCA基底を \(V_q\in\mathbb{R}^{K\times q}\)、直交行列を
\(R\in\mathbb{R}^{q\times q}\) とする。回転後の基底とスコアは

\[
V_q^{\mathrm{rot}}=V_qR,\qquad
U^{\mathrm{rot}}=(P-\mu)V_qR=UR
\]

であるため、復元値は

\[
U^{\mathrm{rot}}(V_q^{\mathrm{rot}})^\top
=URR^\top V_q^\top
=UV_q^\top
\]

となる。したがってPCAとその直交回転は、圧縮だけの復元誤差が理論上同一である。

現在使用している多出力Ridgeも、全軸へ同じ正則化係数を適用する限り直交回転に
対して不変である。PCAと回転PCAの最終確率予測が実質的に異なる場合は、手法差では
なく、実装、軸ごとの処理、または数値誤差を疑う。

このため、PCA対回転PCAの主比較は忠実性ではなく、軸の読みやすさと軸単位の
安定性とする。Sparse PCAは部分空間自体が変わるため、忠実性とのトレードオフを
含めて比較する。

## 3. 共通データ生成

### 3.1 主実験

既存の `pca_fidelity_tradeoff_study.py` と条件をそろえる。

- クラス数: 20
- 入力特徴数: 40
- データseed: 11, 23, 37
- BB: logistic regression、MLP、RBF-SVM、decision tree、bagged trees、
  random forest、gradient boosting
- 説明対象: 予測marginの low・medium・high から各6点
- 摂動半径: 0.15、0.4
- 局所重み: 現行のLIMEカーネル

1条件は `BB × seed × 対象点 × 半径` とする。同じ条件内では、全手法に完全に同じ
摂動点、BB確率、局所重みを渡す。

### 3.2 局所データの3分割

各条件について独立に次の3集合を生成する。

- fit: 600点。圧縮器と代理モデルの学習に使用する。
- validation: 600点。Sparse PCAの疎性と方式別軸数の選択に使用する。
- test: 1000点。最終指標だけに使用する。

選択後は、選ばれた軸数と疎性設定を固定して fit と validation を結合し、圧縮器と
代理モデルを再学習してtestを評価する。testは軸数、正則化、表示軸の選択に使わない。

### 3.3 外的妥当性

主実験の設定を固定した後、既存コードで利用できる `digits` に対し、MLPとrandom
forest、3 seedで追試する。小クラスの `iris` と `wine` は動作確認には使えるが、軸を
疎にする利点が小さいため主張の根拠にはしない。

## 4. 比較手法

### 4.1 局所重み付きPCA

局所重みを正規化した値を \(\alpha_i=w_i/\sum_jw_j\) とし、

\[
\mu=\sum_i\alpha_ip_i,\qquad
C=\sum_i\alpha_i(p_i-\mu)(p_i-\mu)^\top
\]

を用いる。上位固有ベクトルを \(V_q\) とし、

\[
U=(P-\mathbf{1}\mu^\top)V_q
\]

を代理モデルの目的変数にする。

### 4.2 PCAの直交Varimax回転

PCAと同じ \(q\) 次元部分空間内で、クラス負荷が少数クラスへ集中するよう直交
Varimax回転を求める。回転行列はfit集合だけで決める。

回転対象はクラス方向 \(V_q\) とする。回転後も列を正規直交基底に保ち、符号は各軸の
絶対値最大のクラス負荷が正になるよう固定する。軸の順序は回転後スコアの重み付き
分散の降順に並べ直す。

### 4.3 局所重み付きSparse PCA

確率単体の接空間を保つため、各方向 \(d_j\) に

\[
\mathbf{1}^\top d_j=0,\qquad \lVert d_j\rVert_2=1
\]

を課したSparse PCAを主方式とする。目的は概念的に

\[
\min_{Z,D}
\sum_i w_i\lVert p_i-\mu-Dz_i\rVert_2^2
+\lambda\lVert D\rVert_1
\]

である。非凸最適化なのでPCA基底で初期化し、固定seedで複数回初期化してfit目的値が
最小の解を使う。

比較実装として一般的な制約なしSparse PCAも残し、確率和のずれを測る。制約なし方式を
採用候補にする場合は、単体への後処理射影を行う前後の両方を報告する。

疎性係数 \(\lambda\) の絶対値はデータ尺度に依存する。そのため固定値だけを比較せず、
平均有効クラス数がおおむね 2、3、5、8、12、20 となる解を探索し、validation上の
忠実性とのPareto曲線を作る。

## 5. 軸数を公平にそろえる方法

単一の規則だけでは「同じ説明量」と「同じ忠実性」を同時にそろえられないため、
次の2つを分けて報告する。

### 5.1 主解析: 同じ軸数

fit集合のPCA固有値から

\[
q_{\mathrm{var95}}=
\min\left\{q:
\frac{\sum_{j=1}^q\lambda_j}{\sum_{j=1}^{K-1}\lambda_j}\geq0.95
\right\}
\]

を求め、3方式すべてに同じ \(q_{\mathrm{var95}}\) を与える。これにより、軸数を固定して
表現方法だけを比較できる。

併せて \(q\in\{1,2,3,5,8,10,15,K-1\}\) の固定軸数でも全方式を評価し、結論が
95%閾値だけに依存しないことを確認する。

Sparse PCAの代表点は、validationの正規化復元損失が同じ \(q\) のPCAより0.02以上
悪化しない候補のうち、有効クラス数が最小のものとする。該当候補がない場合は
「同じ軸数では忠実性制約を満たさない」と記録し、最小損失の候補を参考値として残す。

### 5.2 副解析: 同じ復元忠実性

各方式についてvalidationで

\[
R^2_{\mathrm{oracle}}
=1-
\frac{\sum_iw_i\lVert p_i-\widetilde p_i\rVert_2^2}
{\sum_iw_i\lVert p_i-\mu_{\mathrm{fit}}\rVert_2^2}
\]

が0.95以上となる最小軸数を \(q_{\mathrm{rec95}}\) とする。PCAと回転PCAでは一致する
はずである。Sparse PCAでは、各 \(q\) についてvalidation忠実性が最大の疎性候補を
用いて最小軸数を決める。

`q_var95` はfit集合でのPCA累積寄与率、`q_rec95` は未知摂動での方式別復元率であり、
出力列名と本文で混同しない。

## 6. 評価指標

### 6.1 圧縮だけの忠実性

- `oracle_output_r2`: 上式のtest集合での値
- `oracle_rmse`: 元確率空間の重み付きRMSE
- `oracle_argmax_agreement`: BBとの予測クラス一致率
- `top2_margin_mae`: 上位2クラス確率差の絶対誤差
- `simplex_sum_error`: \(|\sum_k\widetilde p_{ik}-1|\) の重み付き平均
- `negative_probability_mass`: 負の復元値の総量

### 6.2 XAI代理モデルを含む忠実性

各方式のスコアを、現在と同じ重み付き多出力Ridgeで入力特徴から予測する。

- `surrogate_output_r2`
- `surrogate_rmse`
- `surrogate_argmax_agreement`
- ordinary multiclass LIMEに対するRMSE差とargmax一致率差
- `compression_sse` と `total_sse`。圧縮損失と代理モデルを含む総損失を分ける。

軸スコアのRMSEは軸の尺度に依存するため、方式間の主指標にしない。

### 6.3 軸の読みやすさ

軸 \(j\) の絶対クラス負荷を

\[
a_{kj}=\frac{|d_{kj}|}{\sum_l|d_{lj}|}
\]

と正規化し、次を求める。

- `effective_classes`: \(\exp(-\sum_ka_{kj}\log a_{kj})\)
- `top2_class_mass`: 最大2クラスが持つ負荷質量
- `active_classes_5pct`: \(a_{kj}\geq0.05\) のクラス数
- 正側・負側それぞれの上位3クラス
- 軸ごとの入力係数についても同じentropyとtop-k質量

表示に必要な要素数として

\[
B_{0.05}=\sum_j
\left(
\#\{k:a_{kj}\geq0.05\}
+\#\{l:c_{lj}\geq0.05\}
\right)
\]

を記録する。ただし閾値依存を避けるため、`effective_classes` と
`top2_class_mass` を主指標にする。

### 6.4 安定性

各対象・半径について独立な摂動生成を10回繰り返す。

- 選ばれた `q_var95` と `q_rec95` の分布
- 射影部分空間の類似度
- Hungarian法で軸を対応づけ、符号を合わせた後の軸cosine
- 上位クラス集合のJaccard係数
- 上位入力特徴集合のJaccard係数
- 復元後の入力→クラス効果行列のcosine

軸対応の安定性は固定 \(q\) で測る。方式ごとの自動選択次元が異なる比較には、軸対応
ではなく部分空間類似度と最終効果行列を使う。

## 7. 集計と統計比較

摂動点を独立な観測単位として扱わず、説明対象1点を単位とする。同じ対象・半径の
3方式を対応づけた差として集計する。

- BB、margin群、半径ごとの中央値、四分位範囲、平均
- seedと説明対象を階層的に再標本化する95% bootstrap信頼区間
- 主比較は `rotation - PCA` と `Sparse PCA - PCA`
- 忠実性と読みやすさの両方を散布図にし、単一の総合点へ早期に集約しない

仮説検定を行う場合の主要評価項目は、`surrogate_output_r2`、
`surrogate_argmax_agreement`、`effective_classes` の3つに限定し、残りは診断指標とする。

## 8. 事前に確認する不変条件

本実験の前に小さな数値テストで次を確認する。

1. PCAと回転PCAのoracle復元値が許容誤差内で一致する。
2. 同じRidge正則化なら、PCAと回転PCAの最終確率予測が一致する。
3. 全PCA方向は確率単体の接空間にあり、各方向のクラス成分和がほぼ0である。
4. 制約付きSparse PCAも方向の成分和がほぼ0である。
5. fit、validation、test間に摂動点の共有がない。
6. 全方式が同じ摂動点、確率、重み、軸数を受け取る。

## 9. 段階的な実行

### Phase A: 実装・調整用

MLP、random forest、gradient boosting、seed 11、各margin群3対象で動かす。
Sparse PCAの最適化収束、疎性探索範囲、実行時間、単体制約を確認する。この段階の
test結果で採否を決めず、設定を固定する。

### Phase B: 確証用主実験

7 BB、3 seed、各margin群6対象、2半径で実行する。Phase A後に固定した探索範囲と
閾値を変えない。

### Phase C: 実データ追試

`digits` のMLPとrandom forestで同じ評価を行う。実データでの摂動がデータ多様体上に
ある保証はないため、結果にはその制約を明記する。

## 10. 採用判断

- 回転PCAは、PCAと忠実性が一致することを確認した上で、`effective_classes` または
  表示要素数を20%以上減らし、軸cosineの平均を0.05以上悪化させなければ表示方式の
  第一候補とする。
- Sparse PCAは、PCAに対する `surrogate_output_r2` の低下が0.02以内、argmax一致率の
  低下が1 percentage point以内で、表示要素数を25%以上減らせる条件が複数のBBと
  seedで再現した場合に採用候補とする。
- 条件を満たさない場合は、重み付きPCAを圧縮器として維持し、回転または疎化は表示層
  だけの処理として再検討する。

これらの閾値は理論値ではなく、今回の比較を曖昧にしないための暫定的な実務基準である。
最終的には人による理解度評価を追加して妥当性を確認する。

## 11. 保存する成果物

- 条件・反復単位の生データCSV
- BB・margin・半径・方式別の集計CSV
- 固定軸数での忠実性―読みやすさPareto図
- `q_var95` と `q_rec95` の比較図
- 軸安定性図
- 代表対象について、各軸の正負クラスと上位入力特徴を並べた表
- 実行条件、seed、依存ライブラリ版を含むJSONメタデータ

結果ファイルは既存方針どおり `result/` 以下へ出力し、再生成可能なコードと実験設計を
Git管理対象にする。
