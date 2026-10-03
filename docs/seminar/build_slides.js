// Build the 10/05 seminar deck from docs/seminar/2026-10-05_slides.md.
//
//   NODE_PATH=<dir with pptxgenjs>/node_modules node docs/seminar/build_slides.js \
//     docs/seminar/slide_data.json docs/seminar/2026-10-05_seminar.pptx <pptx skill>/scripts/apply_theme.js
//
// slide_data.json (written by make_slide_data.py) holds the real digits image and the
// per-class deviation shares of one digits explanation point.
// Numbers on result slides come from result/topclass_vs_pca (real data) and
// result/unified_local_evaluation (synthetic, appendix D), run on 2026-10-03.

const fs = require("fs");
const pptxgen = require("pptxgenjs");

const [dataPath, outPath, applyThemePath] = process.argv.slice(2);
const DATA = JSON.parse(fs.readFileSync(dataPath, "utf8"));
const SEMDIR = require("path").dirname(dataPath);
const REGION_FIGURE = require("path").join(SEMDIR, "region_figure.png");
const TAB = JSON.parse(fs.readFileSync(require("path").join(SEMDIR, "tabular_coef.json"), "utf8"));
const tabImg = (n) => require("path").join(SEMDIR, `tabular_${n}.png`);
const { applyTheme } = require(applyThemePath);

const THEME = {
  name: "Fisher-LIME Seminar",
  headFontFace: "Yu Gothic",
  bodyFontFace: "Yu Gothic",
  colors: {
    dk1: "1D2433", lt1: "FFFFFF", dk2: "4A5568", lt2: "EEF1F5",
    accent1: "D9532B", accent2: "2F78A8", accent3: "E3A21A",
    accent4: "3A9D7A", accent5: "9AA3B2", accent6: "6B4FA0",
    hlink: "2F78A8", folHlink: "6B4FA0",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "多クラス分類器の局所説明を「クラス間の確率の移動」としてまとめる";
const C = pres.SchemeColor;
const W = 13.333;

// Class colors used across the deck: 0 = blue, 6 = vermilion, 9 = amber.
const CLS = { 0: C.accent2, 6: C.accent1, 9: C.accent3 };

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 7.5, fill: { color: C.text1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.8, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.7, h: 0.7, fontSize: 20, color: C.accent5, align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 7.5, fill: { color: C.background1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 30, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "多クラスLIMEの出力側の次元削減　2026-10-05", options: { x: 0.6, y: 7.02, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING",
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 7.5, fill: { color: C.text1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.8, fontSize: 30, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.1, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- helpers ----------
let section = "";
function newSlide(master, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function startSection(title) { section = title; pres.addSection({ title }); }

function text(s, t, o) { s.addText(t, { isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top", ...o }); }
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 } });
}
function badge(s, x, y, label, fill, size) {
  const d = size || 0.5;
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.text1 }, line: { color: fill || C.text1 } });
  s.addText(String(label), { isTextBox: true, x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontSize: Math.round(d * 32), bold: true, color: C.background1 });
}
// Class chip: the deck's motif (a rounded square holding a class label).
function chip(s, x, y, label, fill, size) {
  const d = size || 0.42;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: d, h: d, rectRadius: 0.06, fill: { color: fill }, line: { color: fill } });
  s.addText(String(label), { isTextBox: true, x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontSize: Math.round(d * 34), bold: true, color: C.background1 });
}
function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const sub = typeof it === "object" && it.sub;
    const t = sub ? it.text : it;
    return { text: t, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0, breakLine: i < items.length - 1, fontSize: sub ? (o && o.subSize) || 14 : (o && o.fontSize) || 16, color: sub ? C.text2 : C.text1, paraSpaceAfter: 6 } };
  });
  s.addText(runs, { isTextBox: true, margin: 0, valign: "top", ...o, fontSize: undefined });
}
// Grid of cells; value in [0,1] mapped to transparency of the given color.
function grid(s, x, y, cell, matrix, colorOf, gap) {
  const g = gap === undefined ? 0.02 : gap;
  matrix.forEach((row, r) => row.forEach((v, c) => {
    const [color, strength] = colorOf(v, r, c);
    s.addShape(pres.shapes.RECTANGLE, { x: x + c * cell, y: y + r * cell, w: cell - g, h: cell - g, fill: { color, transparency: Math.round(100 - 100 * Math.max(0, Math.min(1, strength))) }, line: { color: C.background1, width: 0.5 } });
  }));
}
function arrow(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color: fill || C.accent5 }, line: { color: fill || C.accent5 } });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12, dataLabelFontSize: 11, dataLabelColor: HEX.dk1, titleFontSize: 13, titleColor: HEX.dk1 };
function chartFrame() { return { valGridLine: { color: "DDE2EA", size: 0.75 }, catGridLine: { style: "none" }, valAxisLineShow: false }; }
function tableRows(header, rows, opts) {
  const o = opts || {};
  const head = header.map((h) => ({ text: h, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: "left" } }));
  const body = rows.map((r, i) => r.map((v) => (typeof v === "object" ? v : { text: String(v), options: { fill: { color: i % 2 ? C.background1 : C.background2 } } })));
  return [head, ...body];
}
function table(s, header, rows, o) {
  s.addTable(tableRows(header, rows), { fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.75, color: "FFFFFF" }, valign: "middle", margin: [3, 6, 3, 6], ...o });
}

// Digit image (8x8, values 0..1) from data.json.
const DIGIT = DATA.digit_image;
// Illustrative axis-1 heatmap built on the digit's strokes: left stem -> toward 6 (red), top-right -> toward 0 (blue).
function axis1Color(v, r, c) {
  if (v < 0.15) return [C.background2, 1];
  if (c <= 3 && r >= 1) return [C.accent1, 0.25 + 0.75 * v];
  if (r <= 2 && c >= 4) return [C.accent2, 0.3 + 0.7 * v];
  return [C.accent1, 0.2 * v];
}
function axis2Color(v, r, c) {
  if (v < 0.15) return [C.background2, 1];
  if (r >= 5 && c >= 4) return [C.accent3, 0.3 + 0.7 * v];
  return [C.accent5, 0.15 * v];
}


// Agenda slide shown at the start of each section; the current section is highlighted.
const AGENDA = [
  ["導入", "今日の問い"],
  ["問題と発想", "LIMEのおさらい、多クラスLIMEの問題、クラス側をまとめる発想、目標"],
  ["手法と仮定", "手順、上位2クラスとの違い、成り立つための仮定"],
  ["実験", "仮定の確かめ方、実験の設定、忠実さの測り方、結果"],
  ["課題と今後", "課題、今後の計画、まとめ"],
];
function agenda(active) {
  const next = AGENDA[active][0];
  const s = newSlide("CONTENT", "目次", active === 0 ? "最初に全体の流れです。" : `次は「${next}」です。`);
  AGENDA.forEach(([name, desc], i) => {
    const y = 1.45 + i * 1.05;
    const on = i === active;
    if (on) card(s, 0.6, y - 0.12, 12.1, 0.95, "FBE6DF");
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: y + 0.05, w: 0.6, h: 0.6, fill: { color: on ? C.accent1 : C.accent5, transparency: on ? 0 : 55 }, line: { color: on ? C.accent1 : C.accent5, transparency: on ? 0 : 55 } });
    s.addText(String(i + 1), { isTextBox: true, x: 0.9, y: y + 0.05, w: 0.6, h: 0.6, margin: 0, align: "center", valign: "middle", fontSize: 20, bold: true, color: C.background1 });
    s.addText(name, { isTextBox: true, x: 1.8, y, w: 3.6, h: 0.7, margin: 0, valign: "middle", fontSize: 24, bold: true, color: C.text1, transparency: on ? 0 : 70 });
    s.addText(desc, { isTextBox: true, x: 5.4, y, w: 7.1, h: 0.7, margin: 0, valign: "middle", fontSize: 16, color: on ? C.text2 : C.text1, transparency: on ? 0 : 75 });
  });
  return s;
}


// Superpixels on the 8x8 digit: A = left stem, B = upper right, C = lower right.
function regionOf(r, c) { return c <= 2 ? "A" : r <= 3 ? "B" : "C"; }
const REGION_TINT = { A: C.accent6, B: C.accent4, C: C.accent3 };
function digitWithRegions(s, x, y, cell, hidden, mode, coef) {
  DIGIT.forEach((row, r) => row.forEach((v, c) => {
    const reg = regionOf(r, c);
    const cx = x + c * cell, cy = y + r * cell;
    let fill, strength;
    if (mode === "heat") {
      const k = coef[reg];
      fill = k >= 0 ? C.accent1 : C.accent2;
      strength = 0.15 + 0.75 * Math.min(1, Math.abs(k) / 0.3);
    } else if (hidden.includes(reg)) {
      fill = C.accent5; strength = 0.85;
    } else if (mode === "regions") {
      fill = REGION_TINT[reg]; strength = 0.22;
    } else {
      fill = C.background2; strength = 1;
    }
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: cell, h: cell, fill: { color: fill, transparency: Math.round(100 - 100 * strength) }, line: { color: C.background1, width: 0.25 } });
    if (!hidden.includes(reg) && v > 0.05) {
      const m = cell * 0.18;
      s.addShape(pres.shapes.RECTANGLE, { x: cx + m, y: cy + m, w: cell - 2 * m, h: cell - 2 * m, fill: { color: C.text1, transparency: Math.round(100 - 100 * v) }, line: { color: C.text1, transparency: 100 } });
    }
  }));
  if (mode === "regions" || mode === "heat") {
    [["A", 1, 3.5], ["B", 5, 1.5], ["C", 5, 5.6]].forEach(([l, cc, rr]) => {
      s.addShape(pres.shapes.OVAL, { x: x + cc * cell, y: y + rr * cell, w: cell * 1.1, h: cell * 1.1, fill: { color: C.background1 }, line: { color: C.text1, width: 1 } });
      s.addText(l, { isTextBox: true, x: x + cc * cell, y: y + rr * cell, w: cell * 1.1, h: cell * 1.1, margin: 0, align: "center", valign: "middle", fontSize: Math.max(9, Math.round(cell * 40)), bold: true, color: C.text1 });
    });
  }
}

// ---------- slides ----------
startSection("導入");

{
  const s = newSlide("TITLE", null, "今回は、研究テーマを初めて紹介します。多クラス分類器をLIMEで説明するときの問題と、それに対する考え方、ここまでの実験結果を話します。手法はまだ完成していないので、前提となる仮定が成り立つかを確かめた結果が中心です。");
  s.addText("多クラス分類器の局所説明を\n「クラス間の確率の移動」としてまとめる", { placeholder: "title" });
  s.addText("LIMEの出力側を次元削減する手法の検討", { placeholder: "body" });
  text(s, "発表者名　／　2026-10-05", { x: 0.8, y: 5.1, w: 8, h: 0.4, fontSize: 16, color: C.background1 });
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].forEach((k, i) => chip(s, 0.8 + i * 0.55, 6.2, k, CLS[k] || C.text2, 0.42));
}

agenda(0);

{
  const s = newSlide("CONTENT", "今日の問い", "この発表では、数字認識の例を最初から最後まで使います。このAIは6と答えましたが、0の確率も0.4あり、迷っています。知りたいのは、画像のどこが0ではなく6という判断につながったかです。この問いに答えるのが、今日の目標です。\n\n（左の画像は手書き数字データの実際の画像。右の確率は説明のための例。）");
  text(s, "手書き数字を0〜9に分類するAIが、ある画像に次の確率を出した", { x: 0.6, y: 1.3, w: 12, h: 0.4, fontSize: 18, color: C.text2 });
  grid(s, 0.9, 2.0, 0.42, DIGIT, (v) => [C.text1, v], 0.03);
  text(s, "入力画像（手書き数字、8×8画素）", { x: 0.9, y: 5.45, w: 3.6, h: 0.3, fontSize: 12, color: C.text2 });
  const probs = [0.4, 0.007, 0.007, 0.007, 0.007, 0.007, 0.5, 0.008, 0.007, 0.05];
  s.addChart(pres.charts.BAR, [{ name: "確率", labels: probs.map((_, k) => String(k)), values: probs }], {
    x: 5.0, y: 1.85, w: 7.7, h: 3.7, barDir: "col", barGapWidthPct: 40,
    chartColors: probs.map((_, k) => (k === 6 ? HEX.accent1 : k === 0 ? HEX.accent2 : k === 9 ? HEX.accent3 : HEX.accent5)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMaxVal: 0.6, valAxisMinVal: 0, valAxisLabelFormatCode: "0.0",
    showLegend: false, showTitle: true, title: "AIの出力（各数字の確率、例）", catAxisTitle: "数字", ...chartText, ...chartFrame(),
  });
  card(s, 0.6, 5.95, 12.1, 0.85, C.background2);
  s.addText([
    { text: "予測は6。では、なぜ", options: { color: C.text1 } },
    { text: "0", options: { color: C.accent2, bold: true } },
    { text: "ではなく", options: { color: C.text1 } },
    { text: "6", options: { color: C.accent1, bold: true } },
    { text: "なのか", options: { color: C.text1 } },
  ], { isTextBox: true, x: 0.9, y: 5.95, w: 11.5, h: 0.85, margin: 0, fontSize: 26, bold: true, valign: "middle" });
}

startSection("問題と発想");
agenda(1);

{
  const s = newSlide("CONTENT", "LIMEのおさらい（1）摂動画像を作る", "まず、ふつうのLIMEが画像をどう説明するかを確認します。LIMEは、画像を意味のある小さな領域、スーパーピクセルに分けます。実際は数十個に分けますが、ここでは説明のために、左の縦線A、右上B、右下Cの3つにします。次に、領域をランダムに隠した画像を何枚も作り、それぞれをAIに入れて、説明したいクラスの確率を記録します。ここでは6の確率です。隠したときに確率が下がる領域は、6という判断に効いていることになります。");
  const steps = [
    ["画像をスーパーピクセルに分ける", "実際は数十個。ここではA・B・Cの3つ"],
    ["領域をランダムに隠した画像を作る", "隠した領域は灰色"],
    ["AIに入れて6の確率を記録する", "どの領域を残したか（1）、隠したか（0）と一緒に"],
  ];
  steps.forEach(([t, d], i) => {
    const y = 1.45 + i * 1.0;
    badge(s, 0.6, y, i + 1, C.text1, 0.45);
    text(s, t, { x: 1.2, y: y - 0.03, w: 4.4, h: 0.45, fontSize: 16, bold: true });
    text(s, d, { x: 1.2, y: y + 0.42, w: 4.4, h: 0.4, fontSize: 12, color: C.text2 });
  });
  digitWithRegions(s, 1.4, 4.45, 0.27, [], "regions");
  text(s, "元画像とスーパーピクセル", { x: 0.9, y: 6.67, w: 3.2, h: 0.3, fontSize: 11, color: C.text2, align: "center" });
  const pert = [[[], "元画像", "1", "1", "1", "0.50"], [["A"], "Aを隠す", "0", "1", "1", "0.20"], [["B"], "Bを隠す", "1", "0", "1", "0.80"], [["C"], "Cを隠す", "1", "1", "0", "0.40"]];
  pert.forEach(([hid, name], i) => {
    const x = 5.9 + i * 1.72;
    digitWithRegions(s, x, 1.5, 0.18, hid, "plain");
    text(s, name, { x: x - 0.1, y: 3.0, w: 1.65, h: 0.3, fontSize: 12, align: "center", color: C.text2 });
  });
  const hd = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: "center" } });
  const cl = (t, i, o) => ({ text: t, options: { align: "center", fill: { color: i % 2 ? C.background1 : C.background2 }, ...o } });
  s.addTable([
    [hd("摂動画像"), hd("A"), hd("B"), hd("C"), hd("6の確率")],
    ...pert.map((r, i) => [cl(r[1], i), cl(r[2], i), cl(r[3], i), cl(r[4], i), cl(r[5], i, { bold: true, color: C.accent1 })]),
  ], { x: 5.9, y: 3.55, w: 6.8, colW: [2.0, 1.0, 1.0, 1.0, 1.8], rowH: 0.48, fontSize: 15, color: C.text1, border: { type: "solid", pt: 1, color: "FFFFFF" }, valign: "middle" });
  text(s, "1 = 残した、0 = 隠した（数値は説明のための例）", { x: 5.9, y: 6.05, w: 6.8, h: 0.3, fontSize: 11, color: C.text2 });
}

{
  const s = newSlide("CONTENT", "LIMEのおさらい（2）線形回帰してヒートマップにする", "次に、記録した表を使って、6の確率を領域の有無で線形回帰します。係数は、その領域があると6の確率がどれだけ上がるかを表します。この例では、Aの係数が+0.30、Bが−0.30、Cが+0.10です。係数を領域の色にしたものがLIMEのヒートマップで、赤い領域は6の確率を上げ、青い領域は下げます。つまり「左の縦線があるから6、右上があると6らしさが下がる」と読めます。実際のLIMEでは、元画像に近い摂動画像ほど重みを大きくして回帰します。");
  card(s, 0.6, 1.4, 6.3, 2.2, C.background2);
  text(s, "線形回帰の結果", { x: 0.85, y: 1.55, w: 5.8, h: 0.35, fontSize: 13, color: C.text2 });
  s.addText([
    { text: "6の確率 ≈ 0.40＋" }, { text: "0.30A", options: { bold: true, color: C.accent1 } },
    { text: "−" }, { text: "0.30B", options: { bold: true, color: C.accent2 } }, { text: "＋0.10C" },
  ], { isTextBox: true, x: 0.85, y: 1.95, w: 5.95, h: 0.6, margin: 0, fontSize: 19, color: C.text1 });
  text(s, "係数 ＝ その領域があると、6の確率がどれだけ上がるか", { x: 0.85, y: 2.75, w: 5.9, h: 0.6, fontSize: 14 });
  const legend = [[C.accent1, "赤", "あると6の確率が上がる領域"], [C.accent2, "青", "あると6の確率が下がる領域"]];
  legend.forEach(([col, n, d], i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.85, y: 4.0 + i * 0.7, w: 0.45, h: 0.45, rectRadius: 0.06, fill: { color: col }, line: { color: col } });
    text(s, d, { x: 1.45, y: 4.0 + i * 0.7, w: 5.2, h: 0.45, fontSize: 16, valign: "middle" });
  });
  text(s, "読み方　左の縦線（A）があるから6。右上（B）があると6らしさが下がる", { x: 0.85, y: 5.55, w: 6.0, h: 0.8, fontSize: 15, bold: true, color: C.accent1 });
  arrow(s, 7.15, 3.6, 0.5, 0.45, C.accent1);
  digitWithRegions(s, 8.1, 1.55, 0.52, [], "heat", { A: 0.30, B: -0.30, C: 0.10 });
  text(s, "6のヒートマップ（LIMEの出力）", { x: 8.1, y: 5.8, w: 4.2, h: 0.35, fontSize: 13, color: C.text2, align: "center" });
}

{
  const s = newSlide("CONTENT", "多クラス分類でのLIMEのやり方", "いま見た手順で説明できるのは、6の確率という出力1本だけです。多クラスでは出力が10本あるので、同じ摂動画像を使って、クラスごとに10回回帰します。結果として、クラスごとのヒートマップが10枚できます。公式の実装では説明するクラスを指定でき、ふつうは予測されたクラス、この例では6だけを見ます。");
  const ys = 1.55, h = 2.1;
  const boxes = [
    ["摂動画像", "領域の隠し方を変えた画像を多数作る"],
    ["AI", "それぞれを分類する"],
    ["10クラスの確率", "1枚ごとに10個の数"],
    ["10個の線形モデル", "クラスごとに別々に回帰"],
    ["ヒートマップ10枚", "クラスごとの説明"],
  ];
  const bw = 2.05, gap = 0.46;
  boxes.forEach(([t, d], i) => {
    const x = 0.6 + i * (bw + gap);
    card(s, x, ys, bw, h, i === 4 ? "FBE6DF" : C.background2);
    text(s, t, { x: x + 0.15, y: ys + 0.15, w: bw - 0.3, h: 0.45, fontSize: 14, bold: true });
    text(s, d, { x: x + 0.15, y: ys + 1.45, w: bw - 0.3, h: 0.6, fontSize: 12, color: C.text2 });
    if (i < 4) arrow(s, x + bw + 0.08, ys + h / 2 - 0.15, 0.3, 0.3);
  });
  // tiny visuals inside boxes
  const m4 = [[1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 0]];
  [0, 1, 2].forEach((k) => grid(s, 0.8 + k * 0.6, ys + 0.75, 0.12, m4, (v) => [C.text2, v ? (k === 1 ? 0.15 : 0.7) : (k === 1 ? 0.7 : 0.15)], 0.01));
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6 + (bw + gap) + 0.55, y: ys + 0.7, w: 0.95, h: 0.6, rectRadius: 0.08, fill: { color: C.text1 }, line: { color: C.text1 } });
  text(s, "AI", { x: 0.6 + (bw + gap) + 0.55, y: ys + 0.7, w: 0.95, h: 0.6, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle" });
  const pv = [0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05, 0.1];
  pv.forEach((v, k) => s.addShape(pres.shapes.RECTANGLE, { x: 0.6 + 2 * (bw + gap) + 0.2 + k * 0.16, y: ys + 1.3 - v * 1.1, w: 0.12, h: v * 1.1, fill: { color: k === 6 ? C.accent1 : k === 0 ? C.accent2 : C.accent5 }, line: { color: C.background2 } }));
  text(s, "p₀ ≈ β₀ᵀz\np₁ ≈ β₁ᵀz\n…\np₉ ≈ β₉ᵀz", { x: 0.6 + 3 * (bw + gap) + 0.2, y: ys + 0.62, w: 1.7, h: 0.8, fontSize: 12, color: C.text1 });
  for (let k = 0; k < 10; k++) grid(s, 0.6 + 4 * (bw + gap) + 0.15 + (k % 5) * 0.37, ys + 0.6 + Math.floor(k / 5) * 0.4, 0.085, m4, (v) => [k === 6 ? C.accent1 : C.accent5, v ? 0.7 : 0.15], 0.005);

  bullets(s, [
    "前の2枚の手順で説明できるのは、1つのクラス（6の確率）だけ",
    "多クラスでは、同じ摂動画像を使って、クラスごとに別々に回帰する",
    { sub: true, text: "0の確率、1の確率、…、9の確率をそれぞれ線形回帰する" },
    "結果として、クラスごとのヒートマップが10枚できる",
  ], { x: 0.6, y: 4.0, w: 7.6, h: 2.8 });
  card(s, 8.6, 4.15, 4.1, 1.6, C.background2);
  text(s, "クラスごとの線形モデル", { x: 8.85, y: 4.3, w: 3.6, h: 0.35, fontSize: 13, color: C.text2 });
  text(s, "pₖ(z) ≈ βₖ₀ + βₖᵀ z", { x: 8.85, y: 4.75, w: 3.6, h: 0.5, fontSize: 22, bold: true });
  text(s, "k = 0, 1, …, 9", { x: 8.85, y: 5.3, w: 3.6, h: 0.35, fontSize: 13, color: C.text2 });
}

{
  const s = newSlide("CONTENT", "問題点", "実際に使うときは、10枚のヒートマップをすべて見ることはなく、確率の高い6と0だけを見ると思います。ただ、比べるクラスは人が確率を見て決めたものです。そのため、一緒に動いている9を見落としたり、確率は高いのに近傍ではほとんど動かないクラスを選んだりすることがあります。もう1つの問題はクラス間の関係です。今日の問いは「なぜ0ではなく6か」で、これは6と0の関係についての問いです。ところが通常のLIMEはクラスごとに説明を返すので、関係は人が2枚のマップを見比べて組み立てるしかありません。");
  const cw = 5.85, ch = 5.3, y = 1.4;
  [0, 1].forEach((i) => card(s, 0.6 + i * (cw + 0.4), y, cw, ch));
  badge(s, 0.85, y + 0.25, 1, C.accent1);
  text(s, "比べるクラスを、人が決めている", { x: 1.5, y: y + 0.25, w: cw - 1.1, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
  bullets(s, [
    "実際には10枚すべては見ず、確率の高い6と0だけを見ることが多い",
    "選び方は人任せなので、一緒に動いている9は見落とす",
    "確率は高くても、近傍ではほとんど動かないクラスを選んでしまうこともある",
  ], { x: 0.9, y: y + 1.0, w: cw - 0.6, h: 2.6 });
  [[0, 0.4, true], [6, 0.5, true], [9, 0.05, false]].forEach(([k, p, seen], i) => {
    chip(s, 1.2 + i * 1.6, y + 3.85, k, seen ? CLS[k] : C.accent5, 0.6);
    text(s, seen ? "見る" : "見落とす", { x: 0.95 + i * 1.6, y: y + 4.55, w: 1.1, h: 0.35, fontSize: 13, align: "center", color: seen ? C.text1 : C.accent1, bold: !seen });
  });

  const x2 = 0.6 + cw + 0.4;
  badge(s, x2 + 0.25, y + 0.25, 2, C.accent1);
  text(s, "クラス間の関係が見えない", { x: x2 + 0.9, y: y + 0.25, w: cw - 1.1, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
  bullets(s, [
    "確率の和は1なので、6が上がれば別のどこかが下がる",
    "知りたいのは「0から6へ確率が移った理由」",
    "通常のLIMEでは、6のマップと0のマップを人が見比べて差を読むしかない",
  ], { x: x2 + 0.3, y: y + 1.0, w: cw - 0.6, h: 2.6 });
  const hm6 = DIGIT.map((row, r) => row.map((v, c) => (v > 0.15 && c <= 3 ? v : v * 0.3)));
  const hm0 = DIGIT.map((row, r) => row.map((v, c) => (v > 0.15 && (r <= 2 || c >= 4) ? v : v * 0.3)));
  grid(s, x2 + 0.5, y + 3.55, 0.15, hm6, (v) => [C.accent1, v], 0.01);
  text(s, "6のマップ", { x: x2 + 0.4, y: y + 4.8, w: 1.4, h: 0.3, fontSize: 12, align: "center", color: C.text2 });
  text(s, "−", { x: x2 + 1.8, y: y + 3.8, w: 0.6, h: 0.7, fontSize: 36, bold: true, align: "center" });
  grid(s, x2 + 2.45, y + 3.55, 0.15, hm0, (v) => [C.accent2, v], 0.01);
  text(s, "0のマップ", { x: x2 + 2.35, y: y + 4.8, w: 1.4, h: 0.3, fontSize: 12, align: "center", color: C.text2 });
  text(s, "＝ ？\n人が頭の中で\n引き算する", { x: x2 + 3.9, y: y + 3.6, w: 1.8, h: 1.2, fontSize: 14, bold: true, color: C.accent1 });
}

{
  const s = newSlide("CONTENT", "説明をまとめる2つの方向", "説明を読みやすくするには、この表を小さくすればよいわけです。行をまとめる方向は、画像LIMEのスーパーピクセルがまさにそうで、特徴のグループ化などの手法もあります。この研究は列、つまりクラスの方をまとめます。特徴は元のままなので、どの領域が効いたかという読み方は変わりません。");
  text(s, "多クラスLIMEの説明は「特徴 × クラス」の表になっている", { x: 0.6, y: 1.3, w: 12, h: 0.4, fontSize: 18, color: C.text2 });
  const gx = 1.8, gy = 3.25, cwid = 0.85, chh = 0.55;
  const cols = ["クラス0", "クラス1", "…", "クラス9"], rows = ["領域1", "領域2", "領域3", "…"];
  cols.forEach((c, j) => { s.addShape(pres.shapes.RECTANGLE, { x: gx + j * cwid, y: gy - chh, w: cwid - 0.04, h: chh - 0.04, fill: { color: C.text1 }, line: { color: C.text1 } }); text(s, c, { x: gx + j * cwid, y: gy - chh, w: cwid - 0.04, h: chh - 0.04, fontSize: 12, color: C.background1, align: "center", valign: "middle", bold: true }); });
  rows.forEach((r, i) => {
    text(s, r, { x: gx - 0.95, y: gy + i * chh, w: 0.85, h: chh - 0.04, fontSize: 12, align: "right", valign: "middle", color: C.text2 });
    cols.forEach((_, j) => s.addShape(pres.shapes.RECTANGLE, { x: gx + j * cwid, y: gy + i * chh, w: cwid - 0.04, h: chh - 0.04, fill: { color: C.background2 }, line: { color: C.background2 } }));
  });
  // column-direction arrow (this study)
  s.addShape(pres.shapes.LEFT_RIGHT_ARROW, { x: gx, y: gy - chh - 0.6, w: 4 * cwid - 0.04, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  // row-direction arrow (existing)
  s.addShape(pres.shapes.UP_DOWN_ARROW, { x: gx + 4 * cwid + 0.2, y: gy, w: 0.42, h: 4 * chh - 0.04, fill: { color: C.accent5 }, line: { color: C.accent5 } });

  card(s, 7.0, 1.95, 5.7, 2.15, "FBE6DF");
  text(s, "列（クラス）をまとめる　この研究", { x: 7.25, y: 2.1, w: 5.3, h: 0.45, fontSize: 18, bold: true, color: C.accent1 });
  text(s, "例　クラス0とクラス6を「0から6への移動」にまとめる", { x: 7.25, y: 2.65, w: 5.3, h: 0.7, fontSize: 15 });
  text(s, "特徴は元のままなので、どの領域が効いたかという読み方は変わらない", { x: 7.25, y: 3.3, w: 5.3, h: 0.7, fontSize: 13, color: C.text2 });
  card(s, 7.0, 4.35, 5.7, 1.95, C.background2);
  text(s, "行（特徴）をまとめる　既存の手法が多い", { x: 7.25, y: 4.5, w: 5.3, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  text(s, "例　ピクセルをスーパーピクセルにまとめる、科目を「理系科目」にまとめる", { x: 7.25, y: 5.05, w: 5.3, h: 1.0, fontSize: 15 });
}

{
  const s = newSlide("CONTENT", "発想", "入力空間は、10クラスそれぞれの領域に分かれています。ただ、説明したい点のまわりだけを見ると、そこに入ってくる領域は0・6・9の3つくらいです。残りの7クラスの領域は遠くにあるので、摂動しても確率はほぼ0のまま動きません。それなら、確率の動きは少数の方向で表せるはずです。方向ごとに説明すれば、量も減り、クラス間の関係も直接読める、というのが出発点です。\n\n（図は説明のためのイメージ。実際の入力空間はもっと高次元。）");
  s.addImage({ path: REGION_FIGURE, x: 0.6, y: 1.4, w: 8.3, h: 4.32 });
  text(s, "（説明のためのイメージ）", { x: 0.6, y: 5.8, w: 8.3, h: 0.3, fontSize: 11, color: C.text2 });
  card(s, 9.2, 1.4, 3.5, 5.35);
  bullets(s, [
    "入力空間は、10クラスの領域に分かれている",
    "説明したい点のまわりに入ってくるのは、0・6・9の3つくらい",
    "残りの7クラスは、摂動してもほぼ0のまま",
    "それなら、確率の動きは少数の方向で表せるはず",
  ], { x: 9.4, y: 1.6, w: 3.15, h: 3.6, fontSize: 15 });
  text(s, "方向の例", { x: 9.4, y: 5.25, w: 3.1, h: 0.3, fontSize: 12, color: C.text2 });
  chip(s, 9.4, 5.6, 0, C.accent2, 0.45); text(s, "↓", { x: 9.9, y: 5.6, w: 0.35, h: 0.45, fontSize: 20, bold: true, valign: "middle", color: C.accent2 });
  chip(s, 10.35, 5.6, 6, C.accent1, 0.45); text(s, "↑", { x: 10.85, y: 5.6, w: 0.35, h: 0.45, fontSize: 20, bold: true, valign: "middle", color: C.accent1 });
  text(s, "0が下がり、6が上がる", { x: 9.4, y: 6.15, w: 3.2, h: 0.4, fontSize: 14 });
}

{
  const s = newSlide("CONTENT", "目標とする説明", "目指す出力は、この表のような形です。軸1を見れば「主に0と6で迷っていて、縦線があるから6になった」と読めます。軸ごとに、出力の動きのどれだけを占めるかも出るので、「9の関与は小さい」ことも分かります。条件は、通常のLIMEと比べて、AIの振る舞いを近似する精度をほとんど落とさないことです。");
  card(s, 0.6, 1.35, 3.6, 4.3);
  text(s, "通常のLIME", { x: 0.85, y: 1.5, w: 3.1, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  text(s, "クラスごとのヒートマップ10枚", { x: 0.85, y: 1.95, w: 3.1, h: 0.4, fontSize: 13, color: C.text2 });
  const small = DIGIT.map((row) => row.map((v) => v));
  for (let k = 0; k < 10; k++) {
    const x = 0.95 + (k % 3) * 1.05, y = 2.5 + Math.floor(k / 3) * 0.75;
    if (k === 9) { grid(s, 0.95 + 1.05, 2.5 + 3 * 0.75, 0.075, small, (v) => [C.accent5, v * 0.6], 0.004); continue; }
    grid(s, x, y, 0.075, small, (v) => [C.accent5, v * 0.6], 0.004);
  }
  arrow(s, 4.35, 3.3, 0.5, 0.45, C.accent1);
  card(s, 5.0, 1.35, 7.7, 4.3, "FBE6DF");
  text(s, "この研究　軸ごとのヒートマップ1〜2枚", { x: 5.25, y: 1.5, w: 7.2, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  // axis 1
  grid(s, 5.35, 2.15, 0.27, DIGIT, axis1Color, 0.02);
  chip(s, 7.75, 2.2, 0, C.accent2, 0.45); text(s, "→", { x: 8.25, y: 2.2, w: 0.4, h: 0.45, fontSize: 20, bold: true, valign: "middle" }); chip(s, 8.65, 2.2, 6, C.accent1, 0.45);
  text(s, "軸1　出力の動きの90%", { x: 7.75, y: 2.8, w: 4.7, h: 0.4, fontSize: 16, bold: true });
  text(s, "左の縦線（赤）が6寄り、上の閉じた弧（青）が0寄り", { x: 7.75, y: 3.2, w: 4.7, h: 0.7, fontSize: 13, color: C.text2 });
  // axis 2
  grid(s, 7.75 + 0.0, 4.1, 0.14, DIGIT, axis2Color, 0.01);
  text(s, "軸2　0・6 → 9　出力の動きの8%\n9らしく見せる領域（黄）", { x: 9.05, y: 4.1, w: 3.5, h: 1.0, fontSize: 13, color: C.text2 });
  text(s, "（数値とヒートマップは説明のための例）", { x: 5.35, y: 5.25, w: 7, h: 0.3, fontSize: 11, color: C.text2 });
  card(s, 0.6, 5.9, 12.1, 0.85, C.text1);
  s.addText([
    { text: "目標　", options: { color: C.accent3, bold: true } },
    { text: "LIMEに劣らない忠実さを保ったまま、解釈しやすさを上げる", options: { color: C.background1, bold: true } },
  ], { isTextBox: true, x: 0.9, y: 5.9, w: 11.6, h: 0.85, margin: 0, fontSize: 22, valign: "middle" });
}

startSection("手法と仮定");
agenda(2);

{
  const s = newSlide("CONTENT", "手法", "手順の違いは、回帰する目的変数だけです。通常のLIMEは「6の確率」を回帰しますが、この手法は「軸1のスコア」を回帰します。軸1が「6が上がり0が下がる」方向なら、スコアは6寄りか0寄りかを表す1つの数です。表の例では、Aを隠すと0寄りに、Bを隠すと6寄りに動きます。回帰で得られる係数は、Aが正、Bが負です。これをヒートマップにすれば、「Aがあるから6、Bがあるから0の可能性も残った」と1枚で読めます。軸はPCAで近傍の確率の動きから決めるので、どのクラスの対比を見るかを人が選ぶ必要はありません。");
  const steps = [
    "摂動画像を作り、AIの確率（10個の数）を得る",
    "確率の点の集まりにPCAをかけ、動きの大きい方向（軸）を求める",
    "各摂動画像の確率を、軸の上の1つの数（スコア）に変える",
    "スコアを、領域の有無で線形回帰する（通常のLIMEと同じ）",
    "回帰係数をヒートマップにする",
  ];
  steps.forEach((t, i) => {
    badge(s, 0.6, 1.45 + i * 1.02, i + 1, i === 1 || i === 2 ? C.accent1 : C.text1, 0.45);
    text(s, t, { x: 1.2, y: 1.42 + i * 1.02, w: 3.8, h: 0.9, fontSize: 15 });
  });
  const hdr = ["摂動画像", "A", "B", "C", "0の確率", "6の確率", "スコア（6 − 0）"];
  const rows = [["元画像", 1, 1, 1, "0.40", "0.50", "+0.10"], ["Aを隠す", 0, 1, 1, "0.70", "0.20", "−0.50"], ["Bを隠す", 1, 0, 1, "0.10", "0.80", "+0.70"], ["Cを隠す", 1, 1, 0, "0.45", "0.40", "−0.05"]];
  table(s, hdr, rows.map((r, i) => r.map((v, j) => ({ text: String(v), options: { fill: { color: i % 2 ? C.background1 : C.background2 }, bold: j === 6, color: j === 6 ? C.accent1 : C.text1, align: j === 0 ? "left" : "center" } }))), { x: 5.4, y: 1.45, w: 7.3, colW: [1.25, 0.6, 0.6, 0.6, 1.2, 1.2, 1.85], rowH: 0.45, fontSize: 14 });
  card(s, 5.4, 4.0, 7.3, 2.75);
  text(s, "回帰の結果", { x: 5.65, y: 4.15, w: 3, h: 0.35, fontSize: 13, color: C.text2 });
  s.addText([
    { text: "スコア ≈ −0.05 ＋ ", options: {} },
    { text: "0.60A", options: { color: C.accent1, bold: true } },
    { text: " − ", options: {} },
    { text: "0.60B", options: { color: C.accent2, bold: true } },
    { text: " ＋ 0.15C", options: {} },
  ], { isTextBox: true, x: 5.65, y: 4.55, w: 6.9, h: 0.5, margin: 0, fontSize: 20, color: C.text1 });
  [["A", C.accent1, 0.95, "6寄り"], ["B", C.accent2, 0.95, "0寄り"], ["C", C.accent1, 0.3, "やや6寄り"]].forEach(([l, col, st, d], i) => {
    const x = 5.75 + i * 2.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.3, w: 0.9, h: 0.9, rectRadius: 0.08, fill: { color: col, transparency: Math.round(100 - st * 100) }, line: { color: col } });
    text(s, l, { x, y: 5.3, w: 0.9, h: 0.9, fontSize: 22, bold: true, align: "center", valign: "middle", color: st > 0.5 ? C.background1 : C.text1 });
    text(s, `領域${l}\n${d}`, { x: x + 1.0, y: 5.4, w: 1.1, h: 0.75, fontSize: 13, color: C.text2 });
  });
}

{
  const s = newSlide("CONTENT", "上位2クラスのLIMEとの違い", "当然、上位2クラスの差を説明すればよいのでは、という疑問が出ると思います。競合が2クラスだけなら、この手法も同じ結果になります。違いが出るのは、3クラス以上が同時に競合しているときと、確率は高いけれど近傍ではほとんど動かないクラスがあるときです。この手法は確率の動きから軸を決めるので、そうした場合も扱えます。");
  card(s, 0.6, 1.4, 12.1, 1.0, C.background2);
  s.addText([
    { text: "確率が高い2クラスの差（", options: {} },
    { text: "6", options: { color: C.accent1, bold: true } },
    { text: " − ", options: {} },
    { text: "0", options: { color: C.accent2, bold: true } },
    { text: "）をLIMEで説明すればよいのでは？", options: {} },
  ], { isTextBox: true, x: 0.9, y: 1.4, w: 11.6, h: 1.0, margin: 0, fontSize: 22, bold: true, valign: "middle", color: C.text1 });
  const cell = (t, o) => ({ text: t, options: { fill: { color: C.background1 }, ...o } });
  s.addTable([
    [cell("状況", { bold: true, color: C.background1, fill: { color: C.text1 } }), cell("上位2クラスの差", { bold: true, color: C.background1, fill: { color: C.text2 } }), cell("この手法", { bold: true, color: C.background1, fill: { color: C.accent1 } })],
    [cell("0と6だけが競合", { bold: true, fill: { color: C.background2 } }), cell("同じ結果"), cell("同じ結果")],
    [cell("0・6・9が競合", { bold: true, fill: { color: C.background2 } }), cell("9を取りこぼす"), cell("複数の軸で表せる", { color: C.accent1, bold: true })],
    [cell("どのクラスを比べるか", { bold: true, fill: { color: C.background2 } }), cell("確率の高い順に人が決める"), cell("近傍での確率の動きから決まる", { color: C.accent1, bold: true })],
  ], { x: 0.6, y: 2.8, w: 12.1, colW: [3.4, 4.35, 4.35], rowH: 0.8, fontSize: 18, color: C.text1, border: { type: "solid", pt: 1, color: "DDE2EA" }, valign: "middle", margin: [4, 10, 4, 10] });
}

{
  const s = newSlide("CONTENT", "成り立つための仮定", "この手法が働くには、2つの仮定が必要です。1つ目は、近傍で出力が動く方向が少ないことです。競合するクラスが少なければ、この仮定は必ず成り立ちます。確率の和が1なので、2クラスだけが動くなら、片方が上がった分だけもう片方が下がり、動く方向は1つです。ただし逆は言えず、多くのクラスがまとまって動く場合にも方向は少なくなります。2つ目は、通常のLIMEと同じ「局所的に線形で近似できる」という仮定です。");
  card(s, 0.6, 1.35, 12.1, 4.15);
  badge(s, 0.85, 1.55, 1, C.accent1, 0.45);
  text(s, "仮定1　局所的には、出力が少数の方向にしか動かない", { x: 1.45, y: 1.52, w: 10.9, h: 0.5, fontSize: 19, bold: true, valign: "middle" });
  bullets(s, [
    "典型的には、近傍で競合するクラスが少ないとき",
    { sub: true, text: "0と6だけが動くなら、6が上がった分だけ0が下がるので1方向" },
    { sub: true, text: "一般に、動くクラスがm個なら、方向は多くてもm−1個" },
    "多くのクラスがまとまって動く場合も、方向は少なくなる",
  ], { x: 0.9, y: 2.2, w: 5.4, h: 3.1, subSize: 14 });
  // scatter: 2 classes on a line vs 3 classes spread
  const rnd = (i) => Math.sin(i * 12.9898) * 43758.5453 % 1;
  const xs = [], y1 = [], y2 = [];
  for (let i = 0; i < 40; i++) { const t = 0.3 * Math.abs(rnd(i + 1)) - 0.15; const u = 0.08 * rnd(i + 99); xs.push(+(0.4 - t).toFixed(3)); y1.push(+(0.5 + t).toFixed(3)); y2.push(+(0.5 + t - u).toFixed(3)); }
  const scatterOpts = (title, color) => ({ lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 6, chartColors: [color], valAxisMinVal: 0.2, valAxisMaxVal: 0.8, catAxisMinVal: 0.1, catAxisMaxVal: 0.7, showLegend: false, showTitle: true, title, showValAxisTitle: true, valAxisTitle: "6の確率", showCatAxisTitle: true, catAxisTitle: "0の確率", valAxisTitleFontSize: 11, catAxisTitleFontSize: 11, valAxisTitleColor: HEX.dk2, catAxisTitleColor: HEX.dk2, valAxisLabelFormatCode: "0.0", catAxisLabelFormatCode: "0.0", ...chartText, ...chartFrame() });
  s.addChart(pres.charts.SCATTER, [{ name: "0", values: xs }, { name: "6", values: y1 }], { x: 6.5, y: 2.05, w: 3.0, h: 3.3, ...scatterOpts("0と6だけが動く → 直線上", HEX.accent1) });
  s.addChart(pres.charts.SCATTER, [{ name: "0", values: xs }, { name: "6", values: y2 }], { x: 9.6, y: 2.05, w: 3.0, h: 3.3, ...scatterOpts("9も動く → 平面に広がる", HEX.accent3) });
  card(s, 0.6, 5.75, 12.1, 1.0);
  badge(s, 0.85, 6.02, 2, C.text1, 0.45);
  text(s, "仮定2　局所的には、AIの出力が線形で近似できる（通常のLIMEと同じ仮定）", { x: 1.45, y: 5.75, w: 11, h: 1.0, fontSize: 19, bold: true, valign: "middle" });
}

startSection("実験");
agenda(3);

{
  const dev = DATA.deviation;
  const sh = dev.share;
  const s = newSlide("CONTENT", "仮定1の確かめ方（1）動くクラスの数", "仮定1は、手書き数字の実際の説明点1つを例に説明します。この点では、AIの確率は2が0.39、3が0.39、9が0.12でした。まず、この点のまわりに摂動データを600点撒き、それぞれの確率を出します。次に、摂動データの確率から元の点の確率を引いて「ずれ」を求めます。引き算をするのは、確率が高くても動かないクラスを数えないためです。最後に、クラスごとにずれの2乗を600点分足し合わせ、大きい順に並べます。この例では、2・3・9の3クラスでずれ全体の97%を占めるので、動くクラスは3個です。");
  const steps = [
    ["摂動データを撒く", "説明する点のまわりに、小さい半径で600点"],
    ["元の点の確率との差（ずれ）を取る", ""],
    ["クラスごとにずれを集計する", "ずれの2乗を600点分足し、大きい順に全体の95%に届くまでのクラス数を数える"],
  ];
  const ys = [1.4, 2.45, 5.05];
  steps.forEach(([t, d], i) => {
    badge(s, 0.6, ys[i], i + 1, i === 2 ? C.accent1 : C.text1, 0.45);
    text(s, t, { x: 1.2, y: ys[i] - 0.03, w: 4.8, h: 0.45, fontSize: 17, bold: true });
    if (d) text(s, d, { x: 1.2, y: ys[i] + 0.45, w: 4.8, h: 0.8, fontSize: 13, color: C.text2 });
  });
  const r2 = (x) => Math.round(x * 100) / 100;
  const pick = (v) => { const t = [v[2], v[3], v[9]].map(r2); return [...t, r2(1 - t[0] - t[1] - t[2])]; };
  const fmt = (x) => x.toFixed(2);
  const sig = (x) => (x > 0.004 ? "+" : x < -0.004 ? "−" : "") + Math.abs(x).toFixed(2);
  const p0 = pick(dev.probs), p1 = pick(dev.example_perturbed);
  const d1 = p1.map((v, i) => r2(v - p0[i]));
  const cellc = (t, o) => ({ text: t, options: { align: "center", fill: { color: C.background1 }, ...o } });
  s.addTable([
    [cellc("", { fill: { color: C.text1 } }), cellc("2", { bold: true, color: C.background1, fill: { color: C.text1 } }), cellc("3", { bold: true, color: C.background1, fill: { color: C.text1 } }), cellc("9", { bold: true, color: C.background1, fill: { color: C.text1 } }), cellc("その他", { bold: true, color: C.background1, fill: { color: C.text1 } })],
    [cellc("説明する点", { align: "left" }), ...p0.map((v) => cellc(fmt(v)))],
    [cellc("摂動データの1つ", { align: "left" }), ...p1.map((v) => cellc(fmt(v)))],
    [cellc("ずれ", { align: "left", bold: true, color: C.accent1, fill: { color: C.background2 } }), ...d1.map((v) => cellc(sig(v), { bold: true, color: C.accent1, fill: { color: C.background2 } }))],
  ], { x: 1.2, y: 3.0, w: 4.9, colW: [1.7, 0.8, 0.8, 0.8, 0.8], rowH: 0.42, fontSize: 13, color: C.text1, border: { type: "solid", pt: 1, color: "DDE2EA" }, valign: "middle", margin: [2, 4, 2, 4] });
  const pct = sh.map((v) => +(100 * v).toFixed(1));
  s.addChart(pres.charts.BAR, [{ name: "ずれの割合", labels: pct.map((_, k) => String(k)), values: pct }], {
    x: 6.4, y: 1.35, w: 6.3, h: 4.45, barDir: "col", barGapWidthPct: 40,
    chartColors: pct.map((_, k) => (k === 2 || k === 3 || k === 9 ? HEX.accent1 : HEX.accent5)),
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 0, valAxisMaxVal: 50, valAxisLabelFormatCode: "0", showLegend: false,
    showTitle: true, title: "クラスごとのずれの割合（%、600点分の合計）", catAxisTitle: "クラス", ...chartText, ...chartFrame(),
  });
  card(s, 6.4, 5.95, 6.3, 0.8, C.background2);
  s.addText([
    { text: "2・3・9で全体の97%　→　動くクラス ", options: { color: C.text1 } }, { text: "3個", options: { bold: true, color: C.accent1 } },
  ], { isTextBox: true, x: 6.65, y: 5.95, w: 5.9, h: 0.8, margin: 0, fontSize: 17, valign: "middle" });
}

{
  const dev = DATA.deviation;
  const [a1, a2] = dev.axis_share;
  const s = newSlide("CONTENT", "仮定1の確かめ方（2）必要な軸の数", `次は必要な軸の数です。600個のずれは、それぞれ10個の数の組、つまり10次元の点です。この点の集まりを、何本の向きの組み合わせで表せるかを数えます。向きはPCAで、ずれをよく表す順に求めます。この例では、1本目の向き（2が下がり、3と9が上がる）でずれ全体の${Math.round(a1 * 100)}%、2本目（9が下がり、3と2が上がる）まで使うと${Math.round((a1 + a2) * 100)}%を表せました。95%に届くのが2本目なので、必要な軸は2本です。右の図は、ずれを2本の軸の上に並べたもので、点が1本目の方向に長く伸び、2本目の方向にも少し広がっています。動くクラスが3個なので、軸は多くても2本です。`);
  const items = [
    ["600個のずれを、10次元の点として並べる", "ずれは10個の数の組。和はいつも0"],
    ["向き（軸）をPCAで、よく表す順に求める", "軸ごとに、ずれ全体の何%を表すかが分かる"],
    ["95%に届くまでの軸の本数を数える", ""],
  ];
  items.forEach(([t, d], i) => {
    const y = 1.4 + i * 1.15;
    badge(s, 0.6, y, i + 1, i === 2 ? C.accent1 : C.text1, 0.45);
    text(s, t, { x: 1.2, y: y - 0.03, w: 5.0, h: 0.45, fontSize: 16, bold: true });
    if (d) text(s, d, { x: 1.2, y: y + 0.45, w: 5.0, h: 0.5, fontSize: 13, color: C.text2 });
  });
  const axisCard = (y, name, share, desc, weights) => {
    card(s, 1.2, y, 5.0, 0.95, C.background2);
    s.addText([{ text: `${name}　`, options: { bold: true } }, { text: `${share}%`, options: { bold: true, color: C.accent1 } }, { text: `　${desc}`, options: {} }], { isTextBox: true, x: 1.4, y: y + 0.06, w: 4.7, h: 0.45, margin: 0, fontSize: 15, color: C.text1, valign: "middle" });
    text(s, weights, { x: 1.4, y: y + 0.5, w: 4.7, h: 0.35, fontSize: 12, color: C.text2 });
  };
  const L = dev.axis_loadings;
  const w = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
  axisCard(4.15, "軸1", Math.round(a1 * 100), "2が下がり、3と9が上がる", `重み　2: ${w(L[0][2])}　3: ${w(L[0][3])}　9: ${w(L[0][9])}`);
  axisCard(5.2, "軸2", Math.round(a2 * 100), "9が下がり、3と2が上がる", `重み　2: ${w(L[1][2])}　3: ${w(L[1][3])}　9: ${w(L[1][9])}`);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.2, y: 6.25, w: 5.0, h: 0.5, rectRadius: 0.08, fill: { color: C.text1 }, line: { color: C.text1 } });
  s.addText([{ text: `軸1で${Math.round(a1 * 100)}%、軸2まで${Math.round((a1 + a2) * 100)}% → 必要な軸 `, options: { color: C.background1 } }, { text: "2本", options: { bold: true, color: C.accent3 } }], { isTextBox: true, x: 1.35, y: 6.25, w: 4.8, h: 0.5, margin: 0, fontSize: 14, valign: "middle" });
  const sc = dev.scores;
  s.addChart(pres.charts.SCATTER, [{ name: "軸1", values: sc.map((v) => v[0]) }, { name: "軸2", values: sc.map((v) => v[1]) }], {
    x: 6.6, y: 1.35, w: 6.1, h: 5.4, lineSize: 0, lineDataSymbol: "circle", lineDataSymbolSize: 5, chartColors: [HEX.accent1],
    valAxisMinVal: -0.3, valAxisMaxVal: 0.3, catAxisMinVal: -0.3, catAxisMaxVal: 0.3, valAxisLabelFormatCode: "0.0", catAxisLabelFormatCode: "0.0",
    valAxisCrossesAt: -0.3, catAxisCrossesAt: -0.3,
    showLegend: false, showTitle: true, title: "ずれを軸1・軸2の上に並べた図（200点）",
    showCatAxisTitle: true, catAxisTitle: `軸1（${Math.round(a1 * 100)}%）`, showValAxisTitle: true, valAxisTitle: `軸2（${Math.round(a2 * 100)}%）`,
    catAxisTitleFontSize: 12, valAxisTitleFontSize: 12, catAxisTitleColor: HEX.dk2, valAxisTitleColor: HEX.dk2, ...chartText, ...chartFrame(),
  });
}

{
  const s = newSlide("CONTENT", "実験の設定", "実験では2つのことを確かめました。1つ目は仮定1、2つ目は軸を減らしたときに通常のLIMEと比べてどれだけ忠実さを失うかです。忠実さは、学習に使っていない摂動で、説明のモデルがAIの確率をどれだけ再現できるかで測ります。データは公開されている実データを4つ使いました。手書き数字は今日の例と同じデータです。なお実験では、スーパーピクセルを隠す代わりに、画素の値に小さな正規分布の摂動を加えています。");
  [["1", "近傍で、出力は少数の方向にしか動かないか（仮定1）"], ["2", "少数の軸にまとめても、通常のLIMEと同じくらい忠実か"]].forEach(([n, t], i) => {
    card(s, 0.6 + i * 6.2, 1.35, 5.9, 1.0, C.background2);
    badge(s, 0.85 + i * 6.2, 1.6, n, C.accent1, 0.5);
    text(s, t, { x: 1.55 + i * 6.2, y: 1.35, w: 4.8, h: 1.0, fontSize: 17, bold: true, valign: "middle" });
  });
  table(s, ["項目", "内容"], [
    ["データ", "手書き数字（10クラス、8×8画素）、文字認識（26クラス、16特徴）、yeast（9クラス、8特徴）、ワイン品質（6クラス、11特徴）"],
    ["分類器", "MLP、RBFカーネルのSVM（どちらも出力が滑らか）"],
    ["説明する点", "予測の迷い（1位と2位の確率差）が大・中・小から各10点、3回の学習で各データ90点"],
    ["近傍", "標準化した特徴に、データの共分散に沿った正規分布の摂動を加える。半径0.15と0.4"],
    ["忠実さの評価", "学習に使っていない摂動600点での決定係数 R²（全クラスの確率で計算）"],
  ], { x: 0.6, y: 2.65, w: 12.1, colW: [2.3, 9.8], rowH: 0.6, fontSize: 15 });
}

{
  const s = newSlide("CONTENT", "忠実さの測り方　決定係数 R²", "忠実さは決定係数 R² で測ります。学習に使っていない摂動データを600点用意し、各点でAIの確率と説明の予測を比べます。10クラス分の差の2乗を足したものが、その点での説明のずれです。比べる基準として、いつも平均の確率を答えた場合のずれも計算します。R² は1から「説明のずれの合計 ÷ 平均で答えたときのずれの合計」を引いた値です。1に近いほど、説明がAIの振る舞いを再現できています。手書き数字の例では、平均で答えたときのずれを1とすると、通常のLIMEのずれは0.126、この手法の3本では0.132で、R² はそれぞれ0.874と0.868でした。");
  const steps = [
    ["学習に使っていない摂動データを用意する", "説明を作るのに使った点とは別の600点"],
    ["各点で、AIの確率と説明の予測のずれを出す", "10クラス分の差の2乗を足す"],
    ["基準として「いつも平均の確率を答える」場合のずれも出す", ""],
    ["R² ＝ 1 − 説明のずれの合計 ÷ 平均で答えたときのずれの合計", "1に近いほど、AIの振る舞いを再現できている"],
  ];
  steps.forEach(([t, d], i) => {
    const y = 1.45 + i * 1.2;
    badge(s, 0.6, y, i + 1, i === 3 ? C.accent1 : C.text1, 0.45);
    text(s, t, { x: 1.2, y: y - 0.03, w: 5.4, h: 0.5, fontSize: 15, bold: true });
    if (d) text(s, d, { x: 1.2, y: y + (i === 3 ? 0.75 : 0.5), w: 5.4, h: 0.4, fontSize: 12, color: C.text2 });
  });
  card(s, 7.0, 1.4, 5.7, 5.35);
  text(s, "ずれの合計（平均で答えたときを1とする）", { x: 7.25, y: 1.55, w: 5.2, h: 0.4, fontSize: 13, color: C.text2 });
  const bars = [["平均で答える", 1.0, C.accent5, "1.000", ""], ["通常のLIME", 0.126, C.text1, "0.126", "R² = 0.874"], ["この手法（3本）", 0.132, C.accent1, "0.132", "R² = 0.868"]];
  const bx = 9.0, bw = 2.7;
  bars.forEach(([name, v, col, lab, r2v], i) => {
    const y = 2.15 + i * 1.0;
    text(s, name, { x: 7.25, y, w: 1.7, h: 0.5, fontSize: 13, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y + 0.05, w: Math.max(0.04, bw * v), h: 0.4, fill: { color: col }, line: { color: col } });
    text(s, lab, { x: bx + Math.max(0.04, bw * v) + 0.08, y, w: 0.9, h: 0.5, fontSize: 12, valign: "middle", color: C.text2 });
    if (r2v) text(s, r2v, { x: bx + 1.15, y, w: 2.0, h: 0.5, fontSize: 15, bold: true, valign: "middle", color: col });
  });
  text(s, "手書き数字、MLP、半径0.15の平均", { x: 7.25, y: 5.15, w: 5.2, h: 0.3, fontSize: 11, color: C.text2 });
  text(s, "R² ＝ 1 なら完全に再現、0 なら平均を答えるのと同じ", { x: 7.25, y: 5.6, w: 5.2, h: 0.9, fontSize: 15, bold: true });
}


{
  const s = newSlide("CONTENT", "結果1　局所的には少数の方向にしか動かない", "どのデータでも、近傍で確率が動くクラスは3個前後で、出力は2本前後の軸で表せました。26クラスの文字認識でも同じです。必要な軸の数は動くクラスの数より1つ前後少なく、2クラスが動くなら1本、3クラスなら2本という仮定1の説明とおおむね合います。つまり、低次元になる主な理由は、近傍で競合するクラスが少ないことです。ただ、手書き数字とyeastでは、軸の数がそれよりさらに少ない近傍が3〜4割ありました。これは、複数のクラスがまとまって動いていることを表しています。半径を0.4に広げると、文字認識では動くクラスが5個程度に増えます。SVMでも同じ傾向で、軸の数は平均1.4〜1.9本でした。");
  const hd = (t, al) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: al || "center" } });
  const cl = (t, i, o) => ({ text: t, options: { align: "center", fill: { color: i % 2 ? C.background1 : C.background2 }, ...o } });
  const rows = [["手書き数字", "10", "3.04", "1.71"], ["文字認識", "26", "2.85", "1.90"], ["yeast", "9", "3.88", "2.17"], ["ワイン品質", "6", "2.81", "1.68"]];
  s.addTable([
    [hd("データ", "left"), hd("クラス数"), hd("動くクラス（個）"), { text: "必要な軸（本）", options: { bold: true, color: C.background1, fill: { color: C.accent1 }, align: "center" } }],
    ...rows.map((r, i) => [cl(r[0], i, { align: "left" }), cl(r[1], i), cl(r[2], i), cl(r[3], i, { bold: true, color: C.accent1 })]),
  ], { x: 0.6, y: 1.5, w: 7.5, colW: [2.1, 1.5, 2.0, 1.9], rowH: 0.75, fontSize: 18, color: C.text1, border: { type: "solid", pt: 1, color: "FFFFFF" }, valign: "middle", margin: [4, 10, 4, 10] });
  text(s, "MLP、半径0.15。元の出力からのずれで測った平均（出力がほぼ一定の近傍は除く）", { x: 0.6, y: 5.45, w: 7.5, h: 0.6, fontSize: 12, color: C.text2 });
  card(s, 8.4, 1.5, 4.3, 5.2);
  bullets(s, [
    "26クラスの文字認識でも、動くクラスは3個前後、軸は2本前後",
    "必要な軸は、動くクラスより1つ前後少ない",
    { sub: true, text: "低次元の主な理由は、近傍で競合するクラスが少ないこと" },
    { sub: true, text: "手書き数字とyeastでは、複数のクラスがまとまって動く近傍もあった（32%、45%）" },
    "SVMでも同じ傾向（軸は平均1.4〜1.9本）",
  ], { x: 8.65, y: 1.75, w: 3.85, h: 4.8, fontSize: 15, subSize: 13 });
}

{
  const s = newSlide("CONTENT", "結果2　通常のLIMEに劣らないか", "4つのデータのどれでも、3本にまとめたときに失う R² は0.02未満でした。説明の量は大きく減りますが、忠実さはほとんど変わりません。一方で、軸を1本まで減らすと、yeastでは0.2ほど落ちます。結果1で見たとおり、必要な軸は2本前後なので、それより少なくすると表しきれません。SVMでも、3本で失う R² は0.001〜0.015でした。なお、文字認識では通常のLIME自体の R² が0.5程度と低く、これは次の課題につながります。");
  const hd = (t, fill) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: fill || C.text1 }, align: "center" } });
  const cl = (t, i, o) => ({ text: t, options: { align: "center", fill: { color: i % 2 ? C.background1 : C.background2 }, ...o } });
  const rows = [["手書き数字", "0.783", "0.855", "0.868", "0.874"], ["文字認識", "0.434", "0.483", "0.491", "0.494"], ["yeast", "0.765", "0.936", "0.967", "0.974"], ["ワイン品質", "0.863", "0.969", "0.976", "0.977"]];
  s.addTable([
    [{ text: "データ", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }, hd("軸1本"), hd("軸2本"), hd("軸3本", C.accent1), hd("通常のLIME")],
    ...rows.map((r, i) => [cl(r[0], i, { align: "left" }), cl(r[1], i, { color: C.text2 }), cl(r[2], i), cl(r[3], i, { bold: true, color: C.accent1 }), cl(r[4], i, { bold: true })]),
  ], { x: 0.6, y: 1.5, w: 7.6, colW: [2.0, 1.4, 1.4, 1.4, 1.4], rowH: 0.75, fontSize: 18, color: C.text1, border: { type: "solid", pt: 1, color: "FFFFFF" }, valign: "middle", margin: [4, 10, 4, 10] });
  text(s, "学習に使っていない摂動での R²（MLP、半径0.15）", { x: 0.6, y: 5.45, w: 7.6, h: 0.4, fontSize: 12, color: C.text2 });
  card(s, 8.5, 1.5, 4.2, 1.9, C.text1);
  text(s, "3本にまとめたときに失う R²", { x: 8.75, y: 1.65, w: 3.8, h: 0.35, fontSize: 13, color: C.accent5 });
  text(s, "0.017以下", { x: 8.75, y: 2.05, w: 3.8, h: 0.8, fontSize: 40, bold: true, color: C.background1 });
  text(s, "4データ・2半径、MLP", { x: 8.75, y: 2.9, w: 3.8, h: 0.35, fontSize: 12, color: C.accent5 });
  bullets(s, [
    "係数は640個（64画素×10クラス）から222個に減る（手書き数字）",
    "1本だと失う量は0.06〜0.22",
    { sub: true, text: "2本前後より減らすと落ちる" },
    "SVMでも3本の損失は0.015以下",
  ], { x: 8.5, y: 3.65, w: 4.2, h: 3.0, fontSize: 15, subSize: 13 });
}

startSection("課題と今後");
agenda(4);

{
  const s = newSlide("CONTENT", "課題", "課題は2つあります。1つ目は軸の読みやすさです。PCAの軸は、複数のクラスに重みが分かれることがあります。柴犬が上がって秋田犬と三毛猫が下がる形なら「柴犬の確率は主に秋田犬から来た」と読めますが、柴犬と三毛猫が同じ側に並ぶと意味が取れません。次元を減らすだけでは足りず、減らした後の軸が理解できることも条件になります。2つ目は、通常のLIME自体の忠実さが低い条件があることです。この手法は通常のLIMEを少数の軸で近似するものなので、その限界は引き継ぎます。");
  card(s, 0.6, 1.35, 7.6, 5.4);
  badge(s, 0.85, 1.55, 1, C.accent1, 0.45);
  text(s, "軸が読めるとは限らない", { x: 1.45, y: 1.52, w: 6.5, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
  const ok = (t) => ({ text: t, options: { color: C.accent4, bold: true, align: "center", fill: { color: C.background1 } } });
  const ng = (t) => ({ text: t, options: { color: C.accent1, bold: true, align: "center", fill: { color: C.background1 } } });
  const c = (t) => ({ text: t, options: { fill: { color: C.background1 } } });
  s.addTable([
    [{ text: "軸の形", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }, { text: "例", options: { bold: true, color: C.background1, fill: { color: C.text1 } } }, { text: "読めるか", options: { bold: true, color: C.background1, fill: { color: C.text1 }, align: "center" } }],
    [c("2クラスの対比"), c("柴犬 + ／ 秋田犬 −"), ok("読める")],
    [c("1対多"), c("柴犬 +0.8 ／ 秋田犬 −0.5、三毛猫 −0.3"), ok("読める")],
    [c("混在"), c("柴犬 +0.37、三毛猫 +0.21 ／ 秋田犬 −0.46"), ng("読めない")],
  ], { x: 0.9, y: 2.3, w: 7.0, colW: [1.55, 4.35, 1.1], rowH: 0.7, fontSize: 13, color: C.text1, border: { type: "solid", pt: 1, color: "DDE2EA" }, valign: "middle", margin: [3, 6, 3, 6] });
  text(s, "次元が減っても、軸の意味が分からなければ解釈しやすくならない", { x: 0.9, y: 5.4, w: 7.0, h: 0.9, fontSize: 16, bold: true, color: C.accent1 });

  card(s, 8.5, 1.35, 4.2, 5.4);
  badge(s, 8.75, 1.55, 2, C.text1, 0.45);
  text(s, "通常のLIME自体が忠実でない条件がある", { x: 9.35, y: 1.5, w: 3.2, h: 0.9, fontSize: 18, bold: true });
  bullets(s, [
    "文字認識（26クラス）や半径0.4の手書き数字では、通常のLIMEの R² が0.3〜0.6にとどまる",
    "この手法は通常のLIMEの近似なので、そこは改善できない",
  ], { x: 8.8, y: 2.65, w: 3.7, h: 3.8, fontSize: 15 });
}

{
  const s = newSlide("CONTENT", "今後の計画と最終ゴール", "次は、軸を読めるようにする方法を考えます。たとえば、軸の形を「2クラスの対比」や「1対多」に限った中で、確率の動きを最もよく表す軸を選ぶ方法です。データも増やして評価し、最後は人に説明を見てもらって、本当に分かりやすくなったかを確かめたいと考えています。");
  const steps = [
    ["読める軸を作る", "軸の形を対比や1対多に限る、回転、スパース化", "忠実さを保ったまま軸が読めるか"],
    ["データを広げる", "クラス数が多く、似たクラスが群になったデータなど", "仮定が成り立つ範囲"],
    ["説明の量をそろえる", "上位クラスの対比や特徴選択ありのLIMEと", "同じ量の説明で、より忠実か"],
    ["人による評価", "説明を読んで判断してもらう", "本当に分かりやすくなったか"],
  ];
  const sw = 2.85, gap = 0.233;
  s.addShape(pres.shapes.LINE, { x: 0.6 + 0.3, y: 1.85, w: 3 * (sw + gap), h: 0, line: { color: C.accent5, width: 2 } });
  steps.forEach(([t, d, q], i) => {
    const x = 0.6 + i * (sw + gap);
    badge(s, x + 0.05, 1.6, i + 1, i === 0 ? C.accent1 : C.text1, 0.5);
    card(s, x, 2.35, sw, 2.95);
    text(s, t, { x: x + 0.2, y: 2.5, w: sw - 0.4, h: 0.8, fontSize: 16, bold: true });
    text(s, d, { x: x + 0.2, y: 3.3, w: sw - 0.4, h: 0.95, fontSize: 13, color: C.text2 });
    text(s, "確かめること　" + q, { x: x + 0.2, y: 4.3, w: sw - 0.4, h: 0.9, fontSize: 13, color: C.accent1 });
  });
  card(s, 0.6, 5.6, 12.1, 1.15, C.text1);
  text(s, "最終ゴール", { x: 0.9, y: 5.68, w: 3, h: 0.35, fontSize: 13, bold: true, color: C.accent3 });
  text(s, "多クラス分類器の予測を「どのクラスからどのクラスへ、どの特徴によって確率が移ったか」という少数の軸で説明し、通常のLIMEより人が理解しやすいことを示す", { x: 0.9, y: 6.0, w: 11.6, h: 0.7, fontSize: 16, bold: true, color: C.background1 });
}

{
  const s = newSlide("CLOSING", "まとめ", "");
  const pts = [
    "多クラスLIMEは、比べるクラスを人が決めていて、クラス間の関係も見えない",
    "AIの出力を説明したい点のまわりで次元削減し、「0から6への移動」のような軸ごとに説明する",
    "実データ4種では、出力は局所的に2本前後の軸で表せ、3軸にまとめても通常のLIMEにほぼ劣らない",
    "次の課題は、減らした軸を人が読めるようにすること",
  ];
  pts.forEach((t, i) => {
    badge(s, 0.8, 1.6 + i * 1.25, i + 1, i === 3 ? C.accent1 : C.accent2, 0.55);
    text(s, t, { x: 1.6, y: 1.55 + i * 1.25, w: 10.9, h: 0.7, fontSize: 20, color: C.background1, valign: "middle" });
  });
}

startSection("付録");

// ---------- tabular-data version of the LIME explanation ----------
const TABCOL = { A: C.accent2, B: C.accent1, C: C.accent4 };
function coefBar(s, x, y, w, name, v, col, maxAbs) {
  const half = w / 2, len = half * Math.min(1, Math.abs(v) / maxAbs);
  s.addShape(pres.shapes.RECTANGLE, { x: v >= 0 ? x + half : x + half - len, y, w: Math.max(0.03, len), h: 0.34, fill: { color: col }, line: { color: col } });
  s.addText((v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2).replace(/^0/, "0"), { isTextBox: true, x: v >= 0 ? x + half + len + 0.05 : x + half - len - 0.75, y, w: 0.7, h: 0.34, margin: 0, fontSize: 12, valign: "middle", align: v >= 0 ? "left" : "right", color: C.text1 });
  text(s, name, { x: x - 1.75, y, w: 0.8, h: 0.34, fontSize: 13, valign: "middle", align: "left" });
}

{
  const s = newSlide("CONTENT", "付録E　テーブルデータ（1）局所的な直線を引く", "テーブルデータの場合は、データを特徴量空間の点として考えます。AIは、この空間に境界を引いてクラスを分けています。LIMEは、説明したい点（星）のまわりで、AIの確率を直線（特徴が多ければ平面）で近似します。その直線の傾き、つまり各特徴の係数が説明です。係数が正の特徴は、増やすとそのクラスの確率が上がります。");
  s.addImage({ path: tabImg("space"), x: 0.6, y: 1.35, w: 5.6, h: 5.11 });
  text(s, "（説明のためのイメージ）", { x: 0.6, y: 6.5, w: 5.6, h: 0.3, fontSize: 11, color: C.text2 });
  card(s, 6.6, 1.5, 6.1, 5.2);
  bullets(s, [
    "データは、特徴量空間の点",
    "AIは、空間に境界（灰色の線）を引いてクラスを分ける",
    "LIMEは、説明したい点（★）のまわりで、AIの確率を直線で近似する",
    "直線の傾き（各特徴の係数）が説明",
    { sub: true, text: "係数が正なら、増やすとそのクラスの確率が上がる" },
  ], { x: 6.9, y: 1.8, w: 5.6, h: 4.7, fontSize: 17, subSize: 15 });
}

{
  const s = newSlide("CONTENT", "付録F　テーブルデータ（2）クラスごとに直線を引く", "多クラスでは、「クラスAかそれ以外か」「クラスBかそれ以外か」のように、クラスごとに別々の直線を引きます。直線ごとに係数が出るので、説明もクラスの数だけできます。この例では、特徴1を増やすとAの確率が上がり、Cの確率が下がります。特徴2を増やすとBの確率が下がります。");
  const maxAbs = 0.6;
  ["A", "B", "C"].forEach((k, i) => {
    const y = 1.25 + i * 1.82;
    s.addImage({ path: tabImg(`ovr_${k}`), x: 0.9, y, w: 1.88, h: 1.7 });
    arrow(s, 3.25, y + 0.66, 0.9, 0.45, C.accent2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.5, y: y + 0.12, w: 8.2, h: 1.55, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: TABCOL[k], width: 2 } });
    text(s, `クラス${k}以外`, { x: 4.75, y: y + 0.12, w: 1.4, h: 1.55, fontSize: 14, valign: "middle", color: C.text2 });
    text(s, `クラス${k}`, { x: 11.25, y: y + 0.12, w: 1.3, h: 1.55, fontSize: 18, bold: true, valign: "middle", color: TABCOL[k] });
    s.addShape(pres.shapes.LINE, { x: 8.9, y: y + 0.3, w: 0, h: 1.2, line: { color: C.accent5, width: 1, dashType: "dash" } });
    const [c1, c2] = TAB.display[k];
    coefBar(s, 7.9, y + 0.4, 2.0, "特徴1", c1, TABCOL[k], maxAbs);
    coefBar(s, 7.9, y + 0.95, 2.0, "特徴2", c2, TABCOL[k], maxAbs);
  });
  text(s, "（説明のためのイメージ。係数は図の境界から計算した値）", { x: 0.6, y: 6.68, w: 8, h: 0.3, fontSize: 10, color: C.text2 });
}

{
  const s = newSlide("CONTENT", "付録G　テーブルデータ（3）クラスごとの説明を並べる", "クラスごとの説明を並べると、こうなります。読みやすくするために係数の大きい特徴を3つずつ選ぶと、クラスごとに選ばれる特徴が違います。また、特徴1はクラスAでは+0.4、クラスBでは−0.5です。つまり特徴1は、BからAへ確率を移す特徴ですが、それは2つの説明を見比べて初めて分かります。この研究は、この「BからAへ」を1本の軸として直接説明することを目指しています。");
  const cols = [
    ["A", [["特徴1", 0.4, true], ["特徴2", -0.6, false], ["特徴5", 0.3, false]]],
    ["B", [["特徴2", 0.3, false], ["特徴4", -0.5, false], ["特徴1", -0.5, true]]],
    ["C", [["特徴3", -0.6, false], ["特徴5", 0.7, false], ["特徴2", -0.6, false]]],
  ];
  cols.forEach(([k, rows], i) => {
    const x = 0.6 + i * 4.15;
    text(s, `クラス${k}`, { x, y: 1.35, w: 3.8, h: 0.5, fontSize: 22, bold: true, color: TABCOL[k] });
    s.addShape(pres.shapes.LINE, { x, y: 1.95, w: 3.8, h: 0, line: { color: C.accent5, width: 0.75 } });
    s.addShape(pres.shapes.LINE, { x: x + 2.35, y: 2.2, w: 0, h: 2.9, line: { color: C.accent5, width: 0.75, dashType: "dash" } });
    rows.forEach(([f, v, hl], j) => {
      const y = 2.3 + j * 0.95;
      if (hl) s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x - 0.1, y: y - 0.2, w: 4.0, h: 0.8, rectRadius: 0.1, fill: { color: "D6E6F2" }, line: { color: C.accent2, width: 1 } });
      text(s, f, { x, y, w: 1.0, h: 0.4, fontSize: 15, bold: true, valign: "middle" });
      const len = 1.25 * Math.abs(v) / 0.7;
      s.addShape(pres.shapes.RECTANGLE, { x: v >= 0 ? x + 2.35 : x + 2.35 - len, y: y + 0.03, w: len, h: 0.34, fill: { color: TABCOL[k] }, line: { color: TABCOL[k] } });
      s.addText((v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1), { isTextBox: true, x: v >= 0 ? x + 2.35 : x + 2.35 - len, y: y + 0.03, w: len, h: 0.34, margin: 0, align: "center", valign: "middle", fontSize: 12, bold: true, color: C.background1 });
    });
  });
  card(s, 0.6, 5.35, 12.1, 1.4, C.background2);
  bullets(s, [
    "クラスごとに選ばれる特徴が違い、そのままでは比べにくい",
    "特徴1はAを上げ（+0.4）Bを下げる（−0.5）。BからAへ確率を移す特徴だが、見比べないと分からない",
  ], { x: 0.85, y: 5.5, w: 11.6, h: 1.2, fontSize: 15 });
  text(s, "（数値は説明のための例）", { x: 0.6, y: 6.85, w: 6, h: 0.25, fontSize: 10, color: C.text2 });
}

{
  const s = newSlide("CONTENT", "付録A　記号と用語", "");
  table(s, ["記号・用語", "意味"], [
    ["K", "クラス数"], ["d", "特徴（スーパーピクセル）の数"], ["p(z)", "摂動 z に対するAIの確率（K個の数）"],
    [{ text: [{ text: "V" }, { text: "q", options: { subscript: true } }], options: { fill: { color: C.background1 } } }, { text: "PCAで求めた q 本の軸（各列がクラスの重み）", options: { fill: { color: C.background1 } } }], ["B", "通常のLIMEの係数行列（d × K）"],
    ["必要な軸の数", "元の出力からのずれの95%を表すのに必要な最小の向きの数"],
    ["動くクラスの数", "元の出力からのずれの95%を担う最小のクラス数。0と6だけが動くなら2"],
  ], { x: 0.6, y: 1.4, w: 12.1, colW: [2.8, 9.3], rowH: 0.62, fontSize: 15 });
}

{
  const s = newSlide("CONTENT", "付録B　通常のLIMEとの関係", "");
  text(s, "全特徴を使う線形回帰（Ridge）は目的変数について線形なので、次が厳密に成り立つ", { x: 0.6, y: 1.4, w: 12, h: 0.5, fontSize: 17 });
  card(s, 0.6, 2.1, 12.1, 1.4, C.background2);
  s.addText([
    { text: "p̂（この手法） ＝ μ ＋ （p̂（通常のLIME） − μ） V" }, { text: "q", options: { subscript: true } },
    { text: " V" }, { text: "q", options: { subscript: true } }, { text: "ᵀ" },
  ], { isTextBox: true, x: 0.9, y: 2.1, w: 11.6, h: 1.4, margin: 0, fontSize: 26, bold: true, valign: "middle", align: "center", color: C.text1 });
  bullets(s, [
    "この手法の予測は、通常のLIMEの予測を q 本の軸に射影したものと一致する",
    "そのため、忠実さで通常のLIMEを上回ることはない",
    "この研究では、通常のLIMEに劣らない忠実さを保ったまま、説明の量を減らし、クラス間の関係を読めるようにすることを目指す",
  ], { x: 0.6, y: 3.9, w: 12.1, h: 2.8 });
}

{
  const s = newSlide("CONTENT", "付録C　想定質問", "");
  const qa = [
    ["上位2クラスの差を説明すれば十分では？", "2クラスだけが競合する近傍では同じ結果になる。実データ4種で軸1本同士を比べた予備実験では、動くクラスが2つ以下の近傍で差はほぼなく、3つ以上の近傍でこの手法の R² が高かった"],
    ["入力側（特徴）をまとめればよいのでは？", "入力側をまとめる方法は既にあり、この研究と組み合わせられる。この研究は、まだ扱いの少ないクラス側をまとめる"],
    ["なぜPCAなのか？", "出力の動きを最もよく再現する軸が得られるため、まず基準としてPCAを使った。読みやすさのために、軸の形を制限する方法を今後試す"],
    ["ランダムフォレストでは使えないのか？", "合成データの実験（付録D）では、近傍で多くのクラスの票が少しずつ揺れるため軸を減らせなかった。通常のLIMEの R² も0.6〜0.7台にとどまる"],
  ];
  qa.forEach(([q, a], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.35 + Math.floor(i / 2) * 2.75;
    card(s, x, y, 5.9, 2.5);
    text(s, "Q　" + q, { x: x + 0.25, y: y + 0.18, w: 5.4, h: 0.5, fontSize: 16, bold: true, color: C.accent1 });
    text(s, a, { x: x + 0.25, y: y + 0.75, w: 5.4, h: 1.65, fontSize: 14 });
  });
}

{
  const s = newSlide("CONTENT", "付録D　合成データでの結果", "");
  text(s, "合成データ（10・20クラス、特徴40個）で、MLPとランダムフォレストを比べた（半径0.15）", { x: 0.6, y: 1.35, w: 12.1, h: 0.6, fontSize: 15, color: C.text2 });
  table(s, ["条件", "動くクラスの数", "必要な軸の数", "3軸の R²（通常のLIME → この手法）"], [
    ["MLP 10クラス", "2.54", "1.49", "0.828 → 0.825"], ["MLP 20クラス", "3.15", "2.09", "0.793 → 0.785"],
    ["ランダムフォレスト 10クラス", "6.94", "5.13", "0.735 → 0.678"], ["ランダムフォレスト 20クラス", "14.31", "11.06", "0.638 → 0.519"],
  ], { x: 0.6, y: 2.1, w: 12.1, colW: [3.6, 2.2, 2.2, 4.1], rowH: 0.6, fontSize: 15 });
  bullets(s, [
    "MLPは実データと同じく、少数の軸で表せ、3軸でほぼ劣らない",
    "ランダムフォレストは票が多くのクラスで少しずつ揺れるため、動くクラスも軸も多く、3軸では足りない",
  ], { x: 0.6, y: 5.3, w: 12.1, h: 1.4 });
}

(async () => {
  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  // pptxgenjs leaves the theme's East Asian font empty; set it so Japanese text uses the theme font.
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(outPath));
  const themePath = "ppt/theme/theme1.xml";
  let xml = await zip.file(themePath).async("string");
  xml = xml.replace(/<a:ea typeface=""\s*\/>/g, `<a:ea typeface="${THEME.bodyFontFace}"/>`);
  zip.file(themePath, xml);
  fs.writeFileSync(outPath, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log("wrote", outPath);
})();
