const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "T. Vansnick et al.";
pres.title = "Classification multimodale Alzheimer";

// Colors
const NAVY = "1E2761";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const LIGHT_BG = "F4F6FA";
const MRI_BLUE = "4285F4";
const TAB_GREEN = "34A853";
const FUSION_GOLD = "FBBC05";
const OUTPUT_RED = "EA4335";
const DARK_TEXT = "1E293B";
const MUTED = "64748B";

// Paths
const BASE = path.resolve(__dirname, "..");
const PREPROCESS_IMG = path.join(BASE, "paper/figures/preprocessing_pipeline.png");
const GRADCAM_IMG = path.join(BASE, "experiments/report_multi_seed/interpretability/mlp_early_fusion/AD_01.png");

// ============================================================
// SLIDE 1: Contexte & Architecture
// ============================================================
const slide1 = pres.addSlide();
slide1.background = { color: WHITE };

// Title bar
slide1.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.7, fill: { color: NAVY }
});
slide1.addText("Contexte & Architecture", {
  x: 0.4, y: 0.05, w: 9, h: 0.6,
  fontSize: 24, fontFace: "Georgia", color: WHITE, bold: true, margin: 0
});

// --- LEFT COLUMN ---
const leftX = 0.4;

// Problem section
slide1.addText("Problème", {
  x: leftX, y: 0.9, w: 4.5, h: 0.35,
  fontSize: 16, fontFace: "Georgia", color: OUTPUT_RED, bold: true, margin: 0
});
slide1.addText([
  { text: "Alzheimer = 60-70% des démences (57M cas)", options: { bullet: true, breakLine: true } },
  { text: "Diagnostic invasif (PET, ponction lombaire)", options: { bullet: true, breakLine: true } },
  { text: "IRM + données cliniques = alternative non-invasive", options: { bullet: true } }
], {
  x: leftX, y: 1.25, w: 4.5, h: 1.1,
  fontSize: 11, fontFace: "Calibri", color: DARK_TEXT, paraSpaceAfter: 4
});

// Bias section
slide1.addText("Biais dans la littérature", {
  x: leftX, y: 2.4, w: 4.5, h: 0.35,
  fontSize: 16, fontFace: "Georgia", color: OUTPUT_RED, bold: true, margin: 0
});
slide1.addText([
  { text: "<20% des études valident sur un dataset externe", options: { bullet: true, breakLine: true } },
  { text: "Même sujet dans train & test (data leakage) → les modèles reconnaissent le ", options: { bullet: true } },
  { text: "patient", options: { italic: true } },
  { text: ", pas la maladie", options: { breakLine: true } },
], {
  x: leftX, y: 2.75, w: 4.5, h: 0.85,
  fontSize: 11, fontFace: "Calibri", color: DARK_TEXT, paraSpaceAfter: 4
});

// Data section
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: leftX, y: 3.75, w: 4.5, h: 0.75,
  fill: { color: ICE }, rectRadius: 0.08
});
slide1.addText([
  { text: "Données : ", options: { bold: true } },
  { text: "6 065 sujets, 3 cohortes (ADNI, OASIS, NACC)", options: { breakLine: true } },
  { text: "5-fold CV × 3 seeds, split strict par sujet" }
], {
  x: leftX + 0.15, y: 3.8, w: 4.2, h: 0.65,
  fontSize: 11, fontFace: "Calibri", color: NAVY
});

// --- RIGHT COLUMN ---
const rightX = 5.2;

// Architecture diagram as cards
// MRI Branch title
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: rightX, y: 0.85, w: 2.0, h: 0.3,
  fill: { color: MRI_BLUE }
});
slide1.addText("Branche IRM", {
  x: rightX, y: 0.85, w: 2.0, h: 0.3,
  fontSize: 10, fontFace: "Calibri", color: WHITE, bold: true, align: "center", valign: "middle", margin: 0
});

// MRI steps
const mriSteps = ["3D IRM 128³", "Patch 16³", "ViT 12 blocs (MAE)", "Features 768-d"];
mriSteps.forEach((txt, i) => {
  slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rightX, y: 1.2 + i * 0.35, w: 2.0, h: 0.3,
    fill: { color: "E8F0FE" }, rectRadius: 0.05
  });
  slide1.addText(txt, {
    x: rightX, y: 1.2 + i * 0.35, w: 2.0, h: 0.3,
    fontSize: 9, fontFace: "Calibri", color: NAVY, align: "center", valign: "middle", margin: 0
  });
});

// Tab Branch title
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: rightX + 2.4, y: 0.85, w: 2.0, h: 0.3,
  fill: { color: TAB_GREEN }
});
slide1.addText("Branche Tabulaire", {
  x: rightX + 2.4, y: 0.85, w: 2.0, h: 0.3,
  fontSize: 10, fontFace: "Calibri", color: WHITE, bold: true, align: "center", valign: "middle", margin: 0
});

// Tab steps
const tabSteps = ["16 features cliniques", "Feature Embed 64-d", "FT-Transformer 3 blocs", "Features 64-d"];
tabSteps.forEach((txt, i) => {
  slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: rightX + 2.4, y: 1.2 + i * 0.35, w: 2.0, h: 0.3,
    fill: { color: "E6F4EA" }, rectRadius: 0.05
  });
  slide1.addText(txt, {
    x: rightX + 2.4, y: 1.2 + i * 0.35, w: 2.0, h: 0.3,
    fontSize: 9, fontFace: "Calibri", color: "1B5E20", align: "center", valign: "middle", margin: 0
  });
});

// Fusion block
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: rightX + 0.7, y: 2.7, w: 3.0, h: 0.35,
  fill: { color: FUSION_GOLD }, rectRadius: 0.05
});
slide1.addText("Cross-Attention bidirectionnelle (8 heads)", {
  x: rightX + 0.7, y: 2.7, w: 3.0, h: 0.35,
  fontSize: 10, fontFace: "Calibri", color: DARK_TEXT, bold: true, align: "center", valign: "middle", margin: 0
});

// Output block
slide1.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x: rightX + 1.2, y: 3.15, w: 2.0, h: 0.3,
  fill: { color: OUTPUT_RED }, rectRadius: 0.05
});
slide1.addText("CN / AD-trajectoire", {
  x: rightX + 1.2, y: 3.15, w: 2.0, h: 0.3,
  fontSize: 10, fontFace: "Calibri", color: WHITE, bold: true, align: "center", valign: "middle", margin: 0
});

// Arrows (vertical lines for MRI branch)
for (let i = 0; i < 3; i++) {
  slide1.addShape(pres.shapes.LINE, {
    x: rightX + 1.0, y: 1.5 + i * 0.35, w: 0, h: 0.05,
    line: { color: MRI_BLUE, width: 1.5 }
  });
}
// Arrows for tab branch
for (let i = 0; i < 3; i++) {
  slide1.addShape(pres.shapes.LINE, {
    x: rightX + 3.4, y: 1.5 + i * 0.35, w: 0, h: 0.05,
    line: { color: TAB_GREEN, width: 1.5 }
  });
}

// Preprocessing pipeline image
slide1.addText("Pipeline de prétraitement IRM", {
  x: rightX, y: 3.65, w: 4.4, h: 0.3,
  fontSize: 11, fontFace: "Calibri", color: MUTED, italic: true, align: "center", margin: 0
});
slide1.addImage({
  path: PREPROCESS_IMG,
  x: rightX + 0.1, y: 3.95, w: 4.3, h: 1.1,
  sizing: { type: "contain", w: 4.3, h: 1.1 }
});

// Footer
slide1.addText("Vansnick et al. — ILIA, Université de Mons", {
  x: 0.4, y: 5.2, w: 9, h: 0.3,
  fontSize: 8, fontFace: "Calibri", color: MUTED, italic: true, margin: 0
});


// ============================================================
// SLIDE 2: Résultats & Explicabilité
// ============================================================
const slide2 = pres.addSlide();
slide2.background = { color: WHITE };

// Title bar
slide2.addShape(pres.shapes.RECTANGLE, {
  x: 0, y: 0, w: 10, h: 0.7, fill: { color: NAVY }
});
slide2.addText("Résultats & Explicabilité", {
  x: 0.4, y: 0.05, w: 9, h: 0.6,
  fontSize: 24, fontFace: "Georgia", color: WHITE, bold: true, margin: 0
});

// --- LEFT COLUMN: Tables ---
const tLeftX = 0.3;

// Table 1: Transformers
slide2.addText("Transformers (ViT + FT-Transformer)", {
  x: tLeftX, y: 0.85, w: 4.8, h: 0.3,
  fontSize: 13, fontFace: "Georgia", color: MRI_BLUE, bold: true, margin: 0
});

const tHeader = [
  { text: "Modèle", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
  { text: "Acc", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
  { text: "Sens", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
  { text: "Spéc", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
  { text: "AUC", options: { bold: true, color: WHITE, fill: { color: NAVY } } }
];

const tRows1 = [
  tHeader,
  ["IRM seule (ViT)", "83.3", "56.2", "90.8", ".826"],
  ["Tabulaire (FT-T)", "84.9", "83.5", "85.2", ".925"],
  [
    { text: "Fusion ViT+FT-T", options: { bold: true } },
    { text: "92.4", options: { bold: true } },
    { text: "78.8", options: { bold: true } },
    { text: "96.2", options: { bold: true } },
    { text: ".959", options: { bold: true } }
  ]
];

slide2.addTable(tRows1, {
  x: tLeftX, y: 1.15, w: 4.8,
  fontSize: 9, fontFace: "Calibri", color: DARK_TEXT,
  colW: [1.7, 0.7, 0.7, 0.7, 0.7],
  rowH: [0.28, 0.25, 0.25, 0.28],
  border: { pt: 0.5, color: "DEE2E6" },
  autoPage: false
});

// Highlight fusion row
slide2.addShape(pres.shapes.RECTANGLE, {
  x: tLeftX, y: 1.15 + 0.28 + 0.25 + 0.25, w: 4.8, h: 0.28,
  fill: { color: FUSION_GOLD, transparency: 75 }
});

// Table 2: CNN 3D
slide2.addText("CNN 3D (ResNet3D)", {
  x: tLeftX, y: 2.35, w: 4.8, h: 0.3,
  fontSize: 13, fontFace: "Georgia", color: OUTPUT_RED, bold: true, margin: 0
});

const tRows2 = [
  tHeader.map(h => ({ ...h, options: { ...h.options } })),
  ["ResNet3D seul", "87.3", "76.3", "90.3", ".914"],
  [{ text: "Early fusion :", options: { italic: true, color: MUTED, colspan: 5 } }],
  ["  R3D + MLP", { text: "93.5", options: { bold: true } }, { text: "85.9", options: { bold: true } }, "95.6", ".934"],
  ["  R3D + XGBoost", "90.7", "79.3", "93.8", ".944"],
  [{ text: "Late fusion :", options: { italic: true, color: MUTED, colspan: 5 } }],
  ["  R3D + MLP (wt)", "91.4", "86.4", "92.8", { text: ".955", options: { bold: true } }],
  ["  R3D + XGB (wt)", "92.5", "82.8", "95.2", ".952"],
];

slide2.addTable(tRows2, {
  x: tLeftX, y: 2.65, w: 4.8,
  fontSize: 9, fontFace: "Calibri", color: DARK_TEXT,
  colW: [1.7, 0.7, 0.7, 0.7, 0.7],
  rowH: [0.28, 0.23, 0.2, 0.23, 0.23, 0.2, 0.23, 0.23],
  border: { pt: 0.5, color: "DEE2E6" },
  autoPage: false
});

// --- RIGHT COLUMN: GradCAM ---
const rX = 5.4;

slide2.addText("Explicabilité — Integrated Gradients", {
  x: rX, y: 0.85, w: 4.3, h: 0.3,
  fontSize: 13, fontFace: "Georgia", color: OUTPUT_RED, bold: true, margin: 0
});

slide2.addText("ResNet3D + MLP (early fusion)", {
  x: rX, y: 1.15, w: 4.3, h: 0.25,
  fontSize: 10, fontFace: "Calibri", color: MUTED, italic: true, margin: 0
});

// GradCAM image
slide2.addImage({
  path: GRADCAM_IMG,
  x: rX, y: 1.5, w: 4.3, h: 2.0,
  sizing: { type: "contain", w: 4.3, h: 2.0 }
});

slide2.addText("Patient AD — p(AD) = 0.997", {
  x: rX, y: 3.55, w: 4.3, h: 0.25,
  fontSize: 10, fontFace: "Calibri", color: DARK_TEXT, align: "center", margin: 0
});

// Interpretation bullets
slide2.addText([
  { text: "Zones activées : ", options: { bullet: true, breakLine: true } },
  { text: "hippocampe", options: { bold: true } },
  { text: " et ", options: {} },
  { text: "lobe temporal médial", options: { bold: true, breakLine: true } },
  { text: "Régions connues pour l'atrophie dans Alzheimer", options: { bullet: true, breakLine: true } },
  { text: "Patterns cohérents avec la neuropathologie", options: { bullet: true } }
], {
  x: rX, y: 3.9, w: 4.3, h: 1.0,
  fontSize: 11, fontFace: "Calibri", color: DARK_TEXT, paraSpaceAfter: 4
});

// Footer
slide2.addText("Vansnick et al. — ILIA, Université de Mons", {
  x: 0.4, y: 5.2, w: 9, h: 0.3,
  fontSize: 8, fontFace: "Calibri", color: MUTED, italic: true, margin: 0
});


// ============================================================
// SAVE
// ============================================================
const outPath = path.join(__dirname, "slides.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Created: " + outPath);
});
