import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error(error?.message ?? String(error));
  process.exit(1);
});
process.on("unhandledRejection", (error) => {
  console.error(error?.message ?? String(error));
  process.exit(1);
});

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(scriptDir, "..");
const metadataDir = path.join(root, "metadata");
const previewDir = "/private/tmp/air-quality-artifact-workbook/previews";
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");

const csvSheets = [
  ["Stations", "stations.csv"],
  ["Sources", "source_references.csv"],
  ["Dataset_Manifest", "qld_dataset_manifest.csv"],
  ["Resource_Manifest", "qld_resource_manifest.csv"],
  ["Data_Dictionary", "data_dictionary.csv"],
  ["Quality", "quality_by_resource.csv"],
];

const numericColumns = {
  Stations: [4, 5, 11, 13, 14, 15, 16, 17, 18],
  Dataset_Manifest: [0],
  Resource_Manifest: [0, 7, 8, 16],
  Quality: [6, 7, 8, 9, 10, 11, 12],
};

for (const [sheetName, fileName] of csvSheets) {
  const csvText = await fs.readFile(path.join(metadataDir, fileName), "utf8");
  const imported = await Workbook.fromCSV(csvText, { sheetName });
  const sourceSheet = imported.worksheets.getItem(sheetName);
  const values = sourceSheet.getUsedRange().values;
  for (let row = 1; row < values.length; row += 1) {
    for (const col of numericColumns[sheetName] ?? []) {
      const current = values[row]?.[col];
      if (current !== null && current !== undefined && current !== "") {
        const parsed = Number(current);
        if (Number.isFinite(parsed)) values[row][col] = parsed;
      }
    }
  }
  const targetSheet = workbook.worksheets.add(sheetName);
  if (values.length > 0 && values[0]?.length > 0) {
    targetSheet.getRangeByIndexes(0, 0, values.length, values[0].length).values = values;
  }
}

const paper = workbook.worksheets.add("Paper_Guide");

const navy = "#17324D";
const teal = "#0F766E";
const paleTeal = "#DFF3EF";
const paleBlue = "#EAF2F8";
const paleGold = "#FFF4CC";
const paleRed = "#FCE8E6";
const textDark = "#17202A";
const white = "#FFFFFF";
const muted = "#52606D";

overview.showGridLines = false;
overview.mergeCells("A1:H1");
overview.getRange("A1").values = [["New Dataset — Australian Air Quality Research Catalog"]];
overview.getRange("A1:H1").format = {
  fill: navy,
  font: { bold: true, color: white, size: 18 },
  verticalAlignment: "center",
};
overview.getRange("A1:H1").format.rowHeight = 34;
overview.mergeCells("A2:H2");
overview.getRange("A2").values = [["Victoria existing daily data + Queensland official hourly data | Version 2026-08-20"]];
overview.getRange("A2:H2").format = {
  fill: paleBlue,
  font: { color: muted, italic: true },
  verticalAlignment: "center",
};

overview.getRange("A4:H4").values = [[
  "Total stations", "", "Queensland stations", "", "Daily records", "", "Queensland hourly records", "",
]];
overview.getRange("A5:H5").formulas = [[
  "=COUNTA('Stations'!$A$2:$A$70)", "", "=COUNTIF('Stations'!$C$2:$C$70,\"Queensland\")", "",
  "=SUM('Stations'!$L$2:$L$70)", "", "=SUMIFS('Quality'!$G$2:$G$263,'Quality'!$C$2:$C$263,\"Queensland\",'Quality'!$D$2:$D$263,\"hourly\")", "",
]];
for (const range of ["A4:B5", "C4:D5", "E4:F5", "G4:H5"]) {
  overview.getRange(range).format = {
    fill: paleTeal,
    borders: { preset: "outside", style: "thin", color: "#9CCFC5" },
  };
}
overview.getRange("A4:H4").format.font = { bold: true, color: teal };
overview.getRange("A5:H5").format.font = { bold: true, color: textDark, size: 15 };
overview.getRange("A5:H5").format.numberFormat = "#,##0";

overview.getRange("A7:H7").values = [[
  "Raw Queensland resources", "", "Duplicates removed", "", "Queensland PM2.5 valid hours", "", "Negative PM2.5 hours retained", "",
]];
overview.getRange("A8:H8").formulas = [[
  "=COUNTA('Resource_Manifest'!$B$2:$B$252)", "", "=SUM('Quality'!$J$2:$J$263)", "",
  "=SUMIFS('Quality'!$K$2:$K$263,'Quality'!$C$2:$C$263,\"Queensland\")", "", "=SUMIFS('Quality'!$M$2:$M$263,'Quality'!$C$2:$C$263,\"Queensland\")", "",
]];
for (const range of ["A7:B8", "C7:D8", "E7:F8", "G7:H8"]) {
  overview.getRange(range).format = {
    fill: paleGold,
    borders: { preset: "outside", style: "thin", color: "#E4C965" },
  };
}
overview.getRange("A7:H7").format.font = { bold: true, color: "#735C0F" };
overview.getRange("A8:H8").format.font = { bold: true, color: textDark, size: 14 };
overview.getRange("A8:H8").format.numberFormat = "#,##0";

overview.mergeCells("A10:H10");
overview.getRange("A10").values = [["Scope and provenance"]];
overview.getRange("A10:H10").format = { fill: teal, font: { bold: true, color: white } };
overview.getRange("A11:B16").values = [
  ["Coverage", "2021-01-01 to 2025-12-31"],
  ["Daily scope", "69 stations across Victoria and Queensland"],
  ["Hourly scope", "58 Queensland stations; 2,157,414 de-duplicated rows"],
  ["Data license", "Queensland CC BY 4.0; Victoria AirWatch portal CC BY 4.0 International"],
  ["BOM notice", "Check product-specific Bureau terms before redistribution or commercial use"],
  ["Audit trail", "251 raw CSVs retained with exact source URL, file size and SHA-256"],
];
overview.mergeCells("B11:H11");
overview.mergeCells("B12:H12");
overview.mergeCells("B13:H13");
overview.mergeCells("B14:H14");
overview.mergeCells("B15:H15");
overview.mergeCells("B16:H16");
overview.getRange("A11:A16").format = { fill: paleBlue, font: { bold: true, color: navy } };
overview.getRange("B11:H16").format.wrapText = true;

overview.mergeCells("A18:H18");
overview.getRange("A18").values = [["Critical interpretation notes"]];
overview.getRange("A18:H18").format = { fill: teal, font: { bold: true, color: white } };
const notes = [
  ["Units", "Queensland NO2/O3/SO2/NO/NOx were converted from ppm to ppb. CO remains ppm."],
  ["BSP", "bsp is particle light scattering (1/Mm), not barometric pressure; pressure is barometric_pressure_hpa."],
  ["Missingness", "Blank means missing or not monitored. It is never converted to zero."],
  ["Negative PM", "Published negative PM values are retained and flagged; choose and report a sensitivity rule."],
  ["Daily validity", "Use pm25_valid_hours and pm25_completeness. A 75% rule is a research filter, not legal compliance."],
  ["NEPM", "Use the required averaging period. Daily means cannot assess 8-hour CO/O3 or hourly NO2/SO2 standards."],
  ["Scope", "This release covers two states, not every Australian jurisdiction."],
];
overview.getRange("A19:B25").values = notes;
for (let r = 19; r <= 25; r += 1) overview.mergeCells(`B${r}:H${r}`);
overview.getRange("A19:A25").format = { fill: paleGold, font: { bold: true, color: "#735C0F" } };
overview.getRange("B19:H25").format.wrapText = true;

overview.mergeCells("A27:H27");
overview.getRange("A27").values = [["Recommended starting files"]];
overview.getRange("A27:H27").format = { fill: teal, font: { bold: true, color: white } };
overview.getRange("A28:B32").values = [
  ["Daily modeling", "daily/all_stations_daily.csv"],
  ["Hourly modeling", "hourly/by_station/qld_<station>_hourly.csv"],
  ["Station selection", "metadata/stations.csv"],
  ["Column definitions", "metadata/data_dictionary.csv"],
  ["Paper methods", "PAPER_GUIDE.md"],
];
for (let r = 28; r <= 32; r += 1) overview.mergeCells(`B${r}:H${r}`);
overview.getRange("A28:A32").format = { fill: paleBlue, font: { bold: true, color: navy } };
overview.getRange("A1:H32").format.font.name = "Aptos";
overview.getRange("A1:H32").format.verticalAlignment = "center";
overview.getRange("A1:H32").format.rowHeight = 23;
overview.getRange("A1:H1").format.rowHeight = 36;
overview.getRange("A2:H2").format.rowHeight = 28;
overview.getRange("B11:H16").format.rowHeight = 30;
overview.getRange("B19:H25").format.rowHeight = 34;
overview.getRange("A1:H32").format.columnWidth = 16;
overview.getRange("B1:H32").format.columnWidth = 18;
overview.freezePanes.freezeRows(2);

paper.showGridLines = false;
paper.mergeCells("A1:H1");
paper.getRange("A1").values = [["Paper Guide — Methods, Standards and Reporting Checklist"]];
paper.getRange("A1:H1").format = { fill: navy, font: { bold: true, color: white, size: 17 } };
paper.getRange("A1:H1").format.rowHeight = 34;
paper.mergeCells("A3:H3");
paper.getRange("A3").values = [["AAQ NEPM concentration standards — always use the specified averaging period"]];
paper.getRange("A3:H3").format = { fill: teal, font: { bold: true, color: white } };
paper.getRange("A4:C14").values = [
  ["Pollutant", "Averaging period", "Standard"],
  ["CO", "8 hours", "9.0 ppm"],
  ["NO2", "1 hour", "80 ppb"],
  ["NO2", "1 year", "15 ppb"],
  ["O3", "8 hours", "65 ppb"],
  ["SO2", "1 hour", "100 ppb; 75 ppb from 2025-01-01"],
  ["SO2", "1 day", "20 ppb"],
  ["PM10", "1 day", "50 ug/m3"],
  ["PM10", "1 year", "25 ug/m3"],
  ["PM2.5", "1 day", "25 ug/m3"],
  ["PM2.5", "1 year", "8 ug/m3"],
];
paper.getRange("A4:C4").format = { fill: paleBlue, font: { bold: true, color: navy } };
paper.getRange("A4:C14").format.borders = { preset: "inside", style: "thin", color: "#D9E2EA" };
paper.mergeCells("E4:H4");
paper.getRange("E4").values = [["Do not over-claim compliance"]];
paper.getRange("E4:H4").format = { fill: paleRed, font: { bold: true, color: "#8A1C1C" } };
paper.mergeCells("E5:H9");
paper.getRange("E5").values = [["Formal NEPM evaluation also depends on monitoring-station eligibility, data-capture rules, exceptional-event treatment and jurisdictional reporting procedures. Describe threshold exceedances as concentration comparisons unless all compliance requirements have been applied."]];
paper.getRange("E5:H9").format = { fill: "#FFF7F6", wrapText: true, verticalAlignment: "top" };

paper.mergeCells("A16:H16");
paper.getRange("A16").values = [["Minimum model-reporting checklist"]];
paper.getRange("A16:H16").format = { fill: teal, font: { bold: true, color: white } };
const checklist = [
  ["1", "Exact stations, years, target variable and units"],
  ["2", "Chronological train/validation/test boundaries and sample counts"],
  ["3", "Missingness threshold, negative-value rule, duplicate rule and any interpolation"],
  ["4", "Persistence and seasonal-naive baselines before deep models"],
  ["5", "MAE and RMSE, plus per-station and macro/weighted summaries"],
  ["6", "Random seeds, software versions, hyperparameter search and stopping rule"],
  ["7", "Station class, operator, instrument/method differences and sensitivity tests"],
  ["8", "Source URLs, access date, license and SHA-256 for raw files"],
  ["9", "Exceptional-event handling for fire, planned burns and dust"],
];
paper.getRange("A17:B25").values = checklist;
for (let r = 17; r <= 25; r += 1) paper.mergeCells(`B${r}:H${r}`);
paper.getRange("A17:A25").format = { fill: paleGold, font: { bold: true, color: "#735C0F" }, horizontalAlignment: "center" };
paper.getRange("B17:H25").format.wrapText = true;

paper.mergeCells("A27:H27");
paper.getRange("A27").values = [["Core references"]];
paper.getRange("A27:H27").format = { fill: teal, font: { bold: true, color: white } };
paper.getRange("A28:B33").values = [
  ["Data", "Queensland Government Air Quality Monitoring 2021–2025"],
  ["Victoria", "EPA Victoria AirWatch yearly hourly averages"],
  ["Meteorology", "Bureau of Meteorology Climate Data Online"],
  ["Standards", "Federal Register: National Environment Protection (Ambient Air Quality) Measure"],
  ["Policy", "NEPC 2021 variation for O3, NO2 and SO2"],
  ["Health", "Australian Government enHealth PM2.5 public-health advice"],
];
for (let r = 28; r <= 33; r += 1) paper.mergeCells(`B${r}:H${r}`);
paper.getRange("A28:A33").format = { fill: paleBlue, font: { bold: true, color: navy } };
paper.getRange("B28:H33").format.wrapText = true;
paper.getRange("A1:H33").format.font.name = "Aptos";
paper.getRange("A1:H33").format.rowHeight = 24;
paper.getRange("B17:H25").format.rowHeight = 32;
paper.getRange("A1:H33").format.columnWidth = 16;
paper.getRange("B1:H33").format.columnWidth = 18;
paper.freezePanes.freezeRows(1);

function styleDataSheet(sheetName, urlColumns = [], wideTextColumns = []) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const header = used.getRow(0);
  header.format = {
    fill: navy,
    font: { bold: true, color: white },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: "#8AA4B8" } },
  };
  header.format.rowHeight = 34;
  used.format.font.name = "Aptos";
  used.format.font.size = 10;
  used.format.autofitColumns();
  used.format.autofitRows();
  for (const col of urlColumns) {
    sheet.getRange(`${col}:${col}`).format.columnWidth = 42;
    sheet.getRange(`${col}:${col}`).format.wrapText = true;
  }
  for (const col of wideTextColumns) {
    sheet.getRange(`${col}:${col}`).format.columnWidth = 34;
    sheet.getRange(`${col}:${col}`).format.wrapText = true;
  }
}

styleDataSheet("Stations", [], ["B", "G", "H", "M"]);
styleDataSheet("Sources", ["E"], ["C", "D", "F", "G"]);
styleDataSheet("Dataset_Manifest", ["G", "J", "K"], ["D", "E", "L", "M", "N"]);
styleDataSheet("Resource_Manifest", ["L", "M"], ["C", "D", "J", "O", "P", "R"]);
styleDataSheet("Data_Dictionary", [], ["A", "B", "D", "E", "F"]);
styleDataSheet("Quality", ["N"], ["B", "N"]);

workbook.worksheets.getItem("Stations").getRange("A2:S70").format.rowHeight = 24;
workbook.worksheets.getItem("Sources").getRange("A2:H13").format.rowHeight = 42;
workbook.worksheets.getItem("Dataset_Manifest").getRange("A2:N6").format.rowHeight = 42;
workbook.worksheets.getItem("Resource_Manifest").getRange("A2:T252").format.rowHeight = 48;
workbook.worksheets.getItem("Resource_Manifest").getRange("O:O").format.columnWidth = 48;
workbook.worksheets.getItem("Resource_Manifest").getRange("R:R").format.columnWidth = 28;
workbook.worksheets.getItem("Data_Dictionary").getRange("A2:F68").format.rowHeight = 44;
workbook.worksheets.getItem("Data_Dictionary").getRange("E:F").format.columnWidth = 42;
workbook.worksheets.getItem("Quality").getRange("A2:N263").format.rowHeight = 24;

workbook.worksheets.getItem("Stations").getRange("E2:F70").format.numberFormat = "0.0000";
workbook.worksheets.getItem("Stations").getRange("L2:L70").format.numberFormat = "#,##0";
workbook.worksheets.getItem("Quality").getRange("G2:K263").format.numberFormat = "#,##0";
workbook.worksheets.getItem("Quality").getRange("L2:L263").format.numberFormat = "0.0%";
workbook.worksheets.getItem("Quality").getRange("M2:M263").format.numberFormat = "#,##0";
workbook.worksheets.getItem("Quality").getRange("L2:L263").conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0.25,
  format: { fill: "#FCE8E6", font: { color: "#8A1C1C" } },
});
workbook.worksheets.getItem("Quality").getRange("L2:L263").conditionalFormats.add("cellIs", {
  operator: "lessThanOrEqual",
  formula: 0.1,
  format: { fill: "#E6F4EA", font: { color: "#176B32" } },
});
workbook.worksheets.getItem("Resource_Manifest").getRange("Q2:Q252").format.numberFormat = "#,##0";

const overviewCheck = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:H32",
  include: "values,formulas",
  tableMaxRows: 32,
  tableMaxCols: 8,
});
console.log(overviewCheck.ndjson);

const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(formulaErrors.ndjson);

const renderRanges = {
  Overview: "A1:H32",
  Stations: "A1:S16",
  Sources: "A1:H13",
  Dataset_Manifest: "A1:N6",
  Resource_Manifest: "A1:T12",
  Data_Dictionary: "A1:F24",
  Quality: "A1:N20",
  Paper_Guide: "A1:H33",
};
for (const [sheetName, range] of Object.entries(renderRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(root, "New_Dataset_catalog.xlsx"));
console.log(JSON.stringify({ output: path.join(root, "New_Dataset_catalog.xlsx"), previewDir }));
