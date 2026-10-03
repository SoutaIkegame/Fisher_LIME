# 「仮定1の確かめ方」説明動画のプロンプト（Grok向け）

スライド15・16枚目（仮定1の確かめ方）を補う短い動画を、場面ごとに生成するためのプロンプト。

## 依頼内容から変えた点

- 平面の領域を2つ（赤・青）から3つ（クラス2・3・9）にした。確率が高いのが2・3・9の3クラスなので、説明する点のまわりに3つの領域が接している方が話が合う
- 星が黄色なので、クラス9の領域は緑にした（2 = 赤、3 = 青、9 = 緑）
- 動画生成AIは画面内の文字や数字を正確に描けないことが多い。数字の入った場面は、背景の動きだけ生成して、数字は編集ソフトで後から重ねる方が確実
- Grokの動画は1本が短い（数秒〜10秒程度）ので、場面ごとに分けて生成し、つなげる
- 依頼の範囲は場面1〜4。場面5〜7は、引き算と集計、軸の数え方まで見せたい場合の追加案

## 共通のスタイル指定（各プロンプトの先頭に貼る）

```
Clean 2D educational motion graphics, flat vector style, white background, no 3D, no camera shake, smooth easing, minimal and calm. Consistent visual elements across all scenes: a square 2D plane divided into three colored regions that meet at one point near the center — top-left region soft red (class "2"), top-right region soft blue (class "3"), bottom region soft green (class "9"); thin white boundary lines between regions. A golden-yellow five-pointed star sits slightly inside the red region, very close to the point where the three regions meet. Small data dots (red dots in the red region, blue dots in the blue region, green dots in the green region) are scattered across the plane; all dots and the star have the same size. 16:9.
```

## 場面1　平面と説明する点（約5秒）

```
[共通のスタイル指定]
The plane fades in. Colored data dots pop in region by region: red dots in the red region, blue dots in the blue region, green dots in the green region. Finally the golden-yellow star appears near the meeting point of the three regions with a gentle pulse. Small labels "2", "3", "9" appear inside each region. Hold the final frame for one second.
```

## 場面2　摂動データを撒く（約5秒）

```
[共通のスタイル指定]
Start from the plane with the colored dots and the yellow star. A dashed circle appears around the star. About 40 small white dots with a thin dark outline spray outward from the star and settle inside the dashed circle, like seeds being scattered. Some white dots land in the red region, some in the blue region, some in the green region. Hold the final frame.
```

## 場面3・4　1点ずつAIで確率を出す（約8秒、必要なら2本に分ける）

```
[共通のスタイル指定]
Start from the plane with the star and the white dots inside the dashed circle. On the right side of the frame there is a dark rounded box labeled "AI". One white dot near the red side glows, a thin arrow goes from it to the AI box, and a horizontal row of ten small cells appears below the box, labeled 0 to 9 above the cells. In the cells, only cells 2, 3 and 9 are clearly filled (cell 2 the fullest), the others are almost empty. The row fades out. Then another white dot near the green side glows and the same thing happens, this time cell 9 is fuller. Repeat for a third dot near the blue side, where cell 3 is fuller. Each previous row disappears before the next one appears.
```

場面3・4で表示する10クラスの確率（手書き数字の実データの例、小数第2位まで）。数字は編集で重ねる。

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 説明する点（星） | 0.01 | 0.03 | 0.39 | 0.39 | 0.01 | 0.00 | 0.00 | 0.04 | 0.00 | 0.12 |
| 摂動データ1（赤寄り） | 0.00 | 0.03 | 0.54 | 0.29 | 0.01 | 0.00 | 0.00 | 0.05 | 0.00 | 0.06 |
| 摂動データ2（緑寄り） | 0.01 | 0.04 | 0.34 | 0.37 | 0.01 | 0.00 | 0.00 | 0.04 | 0.00 | 0.19 |
| 摂動データ3（青寄り） | 0.01 | 0.03 | 0.35 | 0.41 | 0.01 | 0.00 | 0.00 | 0.03 | 0.00 | 0.15 |

## 追加案　場面5　元の点の確率を引いて「ずれ」にする（約6秒）

```
[共通のスタイル指定]
On the right side, two rows of ten cells are stacked: the top row belongs to one white dot, the bottom row to the yellow star. A minus sign appears between them, and a third row appears below showing the difference: cells 0, 1, 4, 5, 6, 7, 8 become empty, while cells 2, 3 and 9 show small bars going up or down from a center line (cell 2 up, cells 3 and 9 down). The difference row is highlighted with a soft orange outline.
```

表示する値の例（摂動データ1 − 説明する点）。2が+0.15、3が−0.10、9が−0.06、そのほかは0.00付近。

## 追加案　場面6　クラスごとにずれを集計して「動くクラス」を数える（約6秒）

```
[共通のスタイル指定]
On the right side, a bar chart with ten bars labeled 0 to 9 starts empty. Many difference rows fly in one after another and drop into the chart, each making bars 2, 3 and 9 grow while the other bars stay almost flat. At the end, bars 2, 3 and 9 are tall and orange, the rest are tiny and gray. A bracket groups bars 2, 3 and 9.
```

編集で重ねる文字の例。「2・3・9でずれ全体の97%」「動くクラス 3個」。

## 追加案　場面7　ずれを点として並べ、必要な軸を数える（約8秒）

```
[共通のスタイル指定]
The plane fades away. A new empty square with a light grid appears. Hundreds of small orange dots appear one by one around the center, forming an elongated oval cloud tilted slightly. A long thick arrow is drawn along the longest direction of the cloud (labeled "axis 1"), then a shorter arrow perpendicular to it (labeled "axis 2"). A third very short arrow flickers and disappears, showing it is not needed.
```

編集で重ねる文字の例。「軸1でずれ全体の69%」「軸2まで99%」「必要な軸 2本」。

## 動画で伝えたい流れ（ナレーション案）

1. 説明したい点（星）のまわりに、摂動データ（白点）を600点撒く
2. 1点ずつAIに入れて、10クラスの確率を出す。どの点でも、高いのは2・3・9だけ
3. 各点の確率から、星の確率を引いて「ずれ」にする。引くのは、動かないクラスを数えないため
4. クラスごとにずれを集計すると、2・3・9で97%。だから動くクラスは3個
5. ずれを点として並べると、細長い雲になる。1本目の軸で69%、2本目までで99%を表せるので、必要な軸は2本
