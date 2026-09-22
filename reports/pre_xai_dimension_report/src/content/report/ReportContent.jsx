import React from "react";

import {
  DataComponent, DataTable, EvidenceChart, MetricCard, ReportSection,
  RichNarrative, useDataApp,
} from "../../data-app-public.jsx";

const radiusSpec = {
  type: "line", x: "radius", y: "mean_effective_dimension_95", series: "margin_group",
  xLabel: "摂動半径", yLabel: "95%有効次元 q95", valueDecimals: 2, stackable: false,
};

const localitySpec = {
  type: "line", x: "neighborhood_percent", y: "mean_ratio", series: "class_label",
  xLabel: "近傍に含める評価データ (%)", yLabel: "局所有効次元 / 大域有効次元",
  valueDecimals: 2, stackable: false,
};

const comparisonSpec = {
  type: "bar", x: "configuration", y: "local_q95", fields: ["local_q95", "random_q95"],
  xLabel: "クラス数・BB", yLabel: "95%有効次元 q95", valueDecimals: 1, stackable: false,
};

const mean = (values) => values.reduce((sum, value) => sum + Number(value || 0), 0) / Math.max(values.length, 1);
const fmt = (value, digits = 2) => Number(value).toFixed(digits);

function Narrative({ id, queryIds, children, className = "" }) {
  const { reviewedRows, visible } = useDataApp();
  if (!visible(id)) return null;
  const ids = queryIds || [];
  const sourceRowsByQuery = Object.fromEntries(ids.map((queryId) => [queryId, reviewedRows(queryId)]));
  return <ReportSection id={id} title={id} queryId={ids[0]} queryIds={ids}
    sourceRowsByQuery={sourceRowsByQuery} showHeading={false} className={className}>
    {children}
  </ReportSection>;
}

export function ReportContent() {
  const { appTitle, setAppTitle, canEdit, mode, reviewedRows, visible } = useDataApp();
  const radiusRows = reviewedRows("radius_summary");
  const localityRows = reviewedRows("locality_curve");
  const detailRows = reviewedRows("margin_detail");

  const localityChartRows = [];
  for (const classes of [5, 10, 20]) {
    for (const fraction of [0.02, 0.05, 0.1, 0.25, 1]) {
      const rows = localityRows.filter((row) => Number(row.classes) === classes && Number(row.neighborhood_fraction) === fraction);
      localityChartRows.push({
        classes,
        class_label: `${classes}クラス`,
        neighborhood_percent: fraction * 100,
        mean_ratio: mean(rows.map((row) => row.local_global_ratio)),
      });
    }
  }

  const comparisonRows = localityRows.filter((row) => Number(row.neighborhood_fraction) === 0.02).map((row) => ({
    ...row,
    configuration: `K=${row.classes} · ${String(row.black_box).toUpperCase()}`,
  }));
  const localRatios = comparisonRows.map((row) => Number(row.local_global_ratio));
  const significant = detailRows.filter((row) => Number(row.neighborhood_fraction) < 1 && Number(row.difference_ci_high) < 0);
  const testedComparisons = detailRows.filter((row) => Number(row.neighborhood_fraction) < 1);
  const k20 = comparisonRows.filter((row) => Number(row.classes) === 20);
  const lowMarginRows = detailRows.filter((row) => row.margin_group === "low" && Number(row.neighborhood_fraction) === 0.02);
  const globalTableRows = [5, 10, 20].map((classes) => {
    const rows = localityRows.filter((row) => Number(row.classes) === classes);
    return { classes, theoretical_max: classes - 1, observed_global_q95: mean(rows.map((row) => row.global_q95)) };
  });

  return <article className="report-content" aria-label="XAI学習前の局所出力次元レポート">
    <header className="report-hero">
      <div className="report-kicker">PRE-XAI EXPERIMENT REPORT · 2026-09-22</div>
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        aria-label={canEdit && mode === "edit" ? "Edit report heading" : undefined}
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}
        onKeyDown={canEdit && mode === "edit" ? (event) => {
          if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
        } : undefined}>{appTitle}</h1>
      <RichNarrative id="report:description" className="report-deck" label="Edit report introduction"
        value="多クラス分類器の確率出力は、入力点の近くに限定すると少ない軸で表せるのか。**XAIの代理モデルを学習させる前**の実験だけを整理した。結論は、今回の合成データでは局所有効次元が大域の約41〜58%まで下がった一方、20クラスではなお約9〜10次元が必要だった、というものだった。" />
      <div className="scope-badge">対象範囲：近傍生成 → BB予測確率 → 次元の測定</div>
    </header>

    <section className="pipeline" aria-label="このレポートの対象範囲">
      <div className="pipeline-step"><span>1</span><strong>評価点・近傍</strong><small>x と近傍点 x′</small></div>
      <div className="pipeline-arrow">→</div>
      <div className="pipeline-step accent"><span>2</span><strong>ブラックボックス</strong><small>BB(x′)</small></div>
      <div className="pipeline-arrow">→</div>
      <div className="pipeline-step"><span>3</span><strong>確率出力</strong><small>P = [p₁ … pK]</small></div>
      <div className="pipeline-arrow">→</div>
      <div className="pipeline-step"><span>4</span><strong>PCAで測定</strong><small>q95 / q99</small></div>
      <div className="pipeline-stop">ここまで</div>
      <div className="pipeline-step muted"><span>5</span><strong>XAIを学習</strong><small>本レポートの対象外</small></div>
    </section>

    <Narrative id="report-summary" queryIds={["locality_curve", "margin_detail"]} className="report-summary">
      <RichNarrative id="report-summary:body" className="report-summary-lead" label="Edit summary"
        value={`## 先に分かったこと\n\n**局所化すると、BBの確率出力に必要な次元は一貫して小さくなった。** 評価データの最近傍2%を使ったとき、q95は大域の **${fmt(Math.min(...localRatios) * 100, 0)}〜${fmt(Math.max(...localRatios) * 100, 0)}%** だった。同じ点数をランダム抽出した対照よりも低く、単にサンプル数を減らした結果ではない。\n\nただし「局所なら常に2〜3次元」とまでは言えない。20クラスでは最近傍2%でもq95が **${fmt(Math.min(...k20.map((row) => row.local_q95)), 1)}〜${fmt(Math.max(...k20.map((row) => row.local_q95)), 1)}** 必要だった。主張できるのは、まず **大域より相対的に低次元になる** こと。`} />
    </Narrative>

    <div className="report-facts" aria-label="主要な結果">
      {visible("metric-local-ratio") && <MetricCard id="metric-local-ratio" title="最近傍2%の局所 / 大域"
        queryId="locality_curve" sourceRows={localityRows} value={`${fmt(Math.min(...localRatios) * 100, 0)}–${fmt(Math.max(...localRatios) * 100, 0)}%`}
        description="K=5, 10, 20 × MLP, RF の6条件。q95の比。" />}
      {visible("metric-controls") && <MetricCard id="metric-controls" title="ランダム対照より低い"
        queryId="margin_detail" sourceRows={detailRows} value={`${significant.length}/${testedComparisons.length}`}
        description="近傍率2〜25%の全条件で、局所−ランダム差の95% CI上限が0未満。" />}
      {visible("metric-k20") && <MetricCard id="metric-k20" title="20クラス・最近傍2%"
        queryId="locality_curve" sourceRows={localityRows} value={`${fmt(Math.min(...k20.map((row) => row.local_q95)), 1)}–${fmt(Math.max(...k20.map((row) => row.local_q95)), 1)}`}
        description="q95。圧縮はされるが、絶対的にはまだ約10次元。" />}
    </div>

    <section className="report-section">
      <Narrative id="question-and-measure" queryIds={["locality_curve", "model_metadata"]}>
        <RichNarrative id="question-and-measure:body" value="## 問いと測り方\n\n調べた問いは **「Kクラス分類器の出力確率ベクトルは、大域では高次元でも、ある入力点の近くでは低次元になるか」**。各近傍点をBBへ通し、得られた確率ベクトルの集合にPCAを当てた。\n\n**q95** は、出力確率の変動の95%を説明するのに必要な最小の主成分数。値が小さいほど、クラス間の確率変化を少数の軸で要約できる。K個の確率は合計1なので、理論上の最大次元は K−1。ただし実際の有効次元は、その範囲でBBの出力がどのように動くかから決まる。\n\nここでPCAは **次元を測る道具** としてのみ使用した。圧縮後の値を説明するLIME等の代理モデルは、まだ学習していない。" />
      </Narrative>
      <DataComponent id="global-dimension-table" title="大域q95と理論上限" queryId="locality_curve"
        kind="table" sourceRows={localityRows} displayRows={globalTableRows}
        description="大域q95は全held-out評価データのBB確率出力から測定。">
        <DataTable rows={globalTableRows} searchable={false} compactNumbers={false} columns={[
          { field: "classes", label: "クラス数 K" },
          { field: "theoretical_max", label: "理論上限 K−1" },
          { field: "observed_global_q95", label: "観測された大域 q95", decimals: 1 },
        ]} />
      </DataComponent>
    </section>

    <section className="report-section experiment-block">
      <Narrative id="experiment-one" queryIds={["radius_summary"]}>
        <RichNarrative id="experiment-one:body" value="## 実験1：ガウス摂動の半径を変える\n\n10クラス・入力20次元の合成分類データでMLPを学習した。予測マージンが low / medium / high の評価点を各30点選び、各点の周囲に1,000個のガウス摂動を生成。標準化入力空間で摂動半径を **0.15、0.4、0.8** と変え、BBの確率出力に重み付きPCAを適用した。LIMEのカーネル重みは使ったが、LIMEの線形代理モデルは学習していない。\n\n半径を広げるにつれ、q95はおおむね **約1.4〜1.6 → 約2.1〜2.2 → 約3.9〜4.5** と増えた。近くでは少数のクラス関係だけが動き、遠くまで見るほど別の変化方向が加わる、という仮説に合う。" />
      </Narrative>
      <EvidenceChart id="radius-chart" queryId="radius_summary" title="摂動半径を広げると有効次元が増える"
        description="線は予測マージン群別の平均q95。各点で1,000摂動、各群30評価点。"
        spec={radiusSpec} rows={radiusRows} sourceRows={radiusRows} height={310} />
      <div className="finding-note">
        <strong>注意：</strong> high-margin・半径0.15では変動エネルギーが0.00012、観測クラス数も平均1.0だった。q95が小さくても「出力がほぼ動かなかっただけ」の場合がある。そのため次の実験では、実データ近傍とランダム対照を直接比較した。
      </div>
    </section>

    <section className="report-section experiment-block">
      <Narrative id="experiment-two" queryIds={["locality_curve", "margin_detail", "model_metadata"]}>
        <RichNarrative id="experiment-two:body" value="## 実験2：実データ上で局所と大域を直接比べる\n\n入力40次元、K = 5 / 10 / 20の合成データを各9,000件生成し、5,400件でBBを学習、3,600件を評価に使用した。BBは **MLPとRandom Forest**、乱数seedは3通り。各条件で予測マージン low / medium / high の点を10点ずつ選んだ。\n\n各評価点について、標準化入力空間で最近傍の **2%、5%、10%、25%、100%** を局所集合とした。大域はheld-out評価データ全体（100%）。さらに、局所集合と同じ件数を評価データからランダム抽出する対照を5回作った。これにより「近いから低次元」なのか、「点数が少ないから低次元」なのかを分けて調べた。" />
      </Narrative>
      <EvidenceChart id="locality-curve" queryId="locality_curve" title="近傍を広げると局所有効次元は大域へ近づく"
        description="MLPとRFの平均。1.0は大域と同じq95を表す。"
        spec={localitySpec} rows={localityChartRows} sourceRows={localityRows} height={330} />
      <EvidenceChart id="local-random-chart" queryId="locality_curve" title="最近傍2%は同数のランダム抽出より低次元"
        description="各値は3 seeds × 3 margin groupsの平均。局所性そのものの効果を確認する対照。"
        spec={comparisonSpec} rows={comparisonRows} sourceRows={localityRows} height={340} />
      <DataComponent id="local-random-table" title="最近傍2%の数値" queryId="locality_curve"
        kind="table" sourceRows={localityRows} displayRows={comparisonRows}
        description="q95と大域比。">
        <DataTable rows={comparisonRows} searchable={false} compactNumbers={false} columns={[
          { field: "classes", label: "K" },
          { field: "black_box", label: "BB" },
          { field: "global_q95", label: "大域 q95", decimals: 1 },
          { field: "local_q95", label: "局所 q95", decimals: 2 },
          { field: "random_q95", label: "ランダム q95", decimals: 2 },
          { field: "local_global_ratio", label: "局所 / 大域", presentation: "percent", decimals: 1 },
        ]} />
      </DataComponent>
    </section>

    <section className="report-section">
      <Narrative id="not-constant" queryIds={["margin_detail"]}>
        <RichNarrative id="not-constant:body" value="## 「出力が動かないから低次元」だけではない\n\n境界に近い low-margin 点の最近傍2%でも、局所変動エネルギーは大域の **45〜77%** 残っていた。それでもq95は大域より小さい。したがって、少なくともこの条件では、低次元性をすべて「予測が一定だったため」と説明することはできない。\n\nまた、近傍率2〜25%について集計した **72条件すべて** で、局所q95と同数ランダムq95の差のbootstrap 95%信頼区間上限が0未満だった。局所集合の次元低下は、サンプル数だけでは説明しにくい。" />
      </Narrative>
      <DataComponent id="low-margin-table" title="境界付近（low margin）・最近傍2%" queryId="margin_detail"
        kind="table" sourceRows={detailRows} displayRows={lowMarginRows}
        description="変動を保ちながら次元が下がっているかを見る。">
        <DataTable rows={lowMarginRows} searchable={false} compactNumbers={false} columns={[
          { field: "classes", label: "K" },
          { field: "black_box", label: "BB" },
          { field: "mean_local_q95", label: "局所 q95", decimals: 2 },
          { field: "mean_local_global_q95_ratio", label: "q95 局所 / 大域", presentation: "percent", decimals: 1 },
          { field: "mean_local_global_variation_ratio", label: "変動量 局所 / 大域", presentation: "percent", decimals: 1 },
          { field: "mean_local_hard_class_count", label: "局所の観測クラス数", decimals: 2 },
        ]} />
      </DataComponent>
    </section>

    <section className="conclusion-grid">
      <Narrative id="conclusions" queryIds={["radius_summary", "locality_curve", "margin_detail"]}>
        <RichNarrative id="conclusions:body" value="## 現時点で言えること\n\n1. **局所出力空間は大域より低次元になった。** K=5〜20、MLP/RF、3 seeds、3マージン群で同じ方向だった。\n2. **近傍を広げると次元は増えた。** 局所から大域へ連続的に近づく。\n3. **クラス数が多いほど絶対次元は残る。** 20クラスでは局所2%でも約9〜10次元。\n4. **小標本だけが原因ではない。** 同数ランダム抽出は局所集合より高次元だった。\n5. **PCAは現状、測定器として有効。** これだけでは、圧縮後の説明が読みやすいか、忠実か、安定かは分からない。" />
      </Narrative>
      <Narrative id="limitations" queryIds={["model_metadata"]}>
        <RichNarrative id="limitations:body" value="## 限界と次の検証\n\n- データは合成データのみ。実データで再現する必要がある。\n- 距離は標準化40次元空間のユークリッド距離。表形式データの意味的な近さとは限らない。\n- q95は線形PCAの指標であり、曲がった低次元構造は直接測れない。\n- 局所対大域の直接比較で使ったBBはMLPとRandom Forestのみ。\n- 95%という閾値に依存するため、q99とparticipation ratioも補助的に見る必要がある。\n\n次段階は、この低次元軸を説明対象 U としてXAI代理モデルに学習させ、通常のクラス別LIMEと **忠実性・安定性・説明量** を比較することになる。" />
      </Narrative>
    </section>

    <Narrative id="reproduction" queryIds={["radius_summary", "locality_curve", "model_metadata"]} className="report-disclosure">
      <RichNarrative id="reproduction:body" value="## 再現情報\n\n**実験1:** `python experiments/run_local_dimension.py`  \n**実験2:** `python experiments/global_local_dimension_study.py`\n\n主な入力結果は `result/local_dimension/summary.csv`、`result/global_local_dimension/compact_summary.csv`、`result/global_local_dimension/summary.csv`、`result/global_local_dimension/model_metadata.csv`。本HTMLの図表はこれらの保存済みCSVから作成した。BBのtest accuracyは学習成否の確認用メタデータであり、このレポートのXAI性能指標ではない。" />
    </Narrative>
  </article>;
}
