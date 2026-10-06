# Tableau Public dashboard: build guide

Three dashboards in one Tableau Public workbook, built from `m5_tableau_data.xlsx`:

1. **Store Outcomes**: fill rate, waste and stockouts by store and state, for a chosen model and ordering policy
2. **Model Comparison**: stockout-vs-waste frontier, cost vs baseline, accuracy by horizon, calibration
3. **Data Quality**: missing dates and cells, store closures, unlisted items, zero-sales rates, demand classes

Expect about 3–4 hours the first time. The **Check values** section at the end lists numbers your
Tableau views should show, so you can confirm each calculated field before moving on.

---

## 0. Setup (10 min)

1. Install **Tableau Public** (free desktop app) from public.tableau.com and sign in with a free account.
   The desktop app is easier than web authoring for multi-sheet dashboards.
2. Open Tableau Public → **Connect → Microsoft Excel** → choose `m5_tableau_data.xlsx`.
3. Drag the sheet **newsvendor_store_day** onto the canvas. Rename the data source (top-left)
   to **Store Day**.
4. Add the other sheets as **separate data sources**: **Data → New Data Source → Microsoft Excel**
   → same file → drag one sheet each. Name them:

| Sheet | Data source name | Rows | Grain |
|---|---|---|---|
| newsvendor_store_day | Store Day | 5,040 | store × day × model × critical ratio |
| newsvendor_item_store | Item Store | 148,140 | item × store × model × critical ratio (28 days summed) |
| accuracy_store_week | Accuracy | 120 | model × store × horizon week |
| calibration | Calibration | 21 | model × quantile level |
| dq_store_day | DQ Store Day | 19,690 | store × day, full history 2011–2016 |
| dq_series | DQ Series | 8,230 | item × store profile |

Tableau cleans field names on import (`units_demanded` becomes **Units Demanded**); this guide uses
the cleaned names.

5. In **Store Day**, **Item Store** and **Accuracy**, right-click **Critical Ratio** (where present)
   → **Convert to Dimension**, then **Convert to Discrete**. Do the same for **Quantile Level** in
   **Calibration**.
6. In **DQ Store Day** and **DQ Series**, right-click **State Name** → **Geographic Role → State/Province**.
   Do the same for **State Name** in **Store Day**.

**Colors used throughout** (Edit Colors → click a value → enter hex):

| Value | Hex |
|---|---|
| Baseline | `#9CA3AF` (grey) |
| LightGBM | `#2563EB` (blue) |
| Hierarchical Bayes | `#EA580C` (orange) |
| CA / TX / WI | `#0F766E` / `#B45309` / `#7C3AED` |

Font: Tableau Book, titles bold 12–14 pt. Dashboard size: **Fixed, 1200 × 850**.

---

## 1. Calculated fields

Create each with **Analysis → Create Calculated Field** while the named data source is selected.
Ratios are always *sum over sum*, so they stay correct at store, state and total level.

### Store Day (also create the first five in Item Store)

| Name | Formula | Format |
|---|---|---|
| Fill Rate | `SUM([Units Sold]) / SUM([Units Demanded])` | Percentage, 1 dp |
| Waste % | `SUM([Units Wasted]) / SUM([Units Ordered])` | Percentage, 1 dp |
| Stockout Rate | `SUM([Stockout Item Days]) / SUM([Item Days])` | Percentage, 1 dp |
| Fresh Cost | `[Lost Margin] + [Waste Cost]` | Currency, 0 dp |
| Baseline Cost (CR) | `{FIXED [Critical Ratio] : SUM(IIF([Model]="Baseline", [Newsvendor Cost Implied], 0))}` | Currency |
| Cost vs Baseline | `SUM([Newsvendor Cost Implied]) / MIN([Baseline Cost (CR)]) - 1` | Percentage, 1 dp |
| Store Days | `COUNTD(STR([Store Id]) + "-" + STR([D]))` | Number |
| Lost Revenue per Store-Day | `SUM([Lost Revenue]) / [Store Days]` | Currency, 0 dp |
| Waste Cost per Store-Day | `SUM([Waste Cost]) / [Store Days]` | Currency, 0 dp |
| Stockouts per Store-Day | `SUM([Stockout Item Days]) / [Store Days]` | Number, 0 dp |

### Item Store (velocity view)

| Name | Formula |
|---|---|
| Fresh Cost | `[Lost Margin] + [Waste Cost]` |
| Baseline Fresh Cost (CR, Velocity) | `{FIXED [Critical Ratio], [Velocity] : SUM(IIF([Model]="Baseline", [Lost Margin] + [Waste Cost], 0))}` |
| Fresh Cost vs Baseline | `SUM([Fresh Cost]) / MIN([Baseline Fresh Cost (CR, Velocity)]) - 1` |
| Velocity Order | `CASE [Velocity] WHEN "Slow (<0.5/day)" THEN 1 WHEN "Medium (0.5-2/day)" THEN 2 ELSE 3 END` |

### Accuracy

| Name | Formula |
|---|---|
| Pinball Loss | `SUM([Pinball Sum]) / SUM([Item Days])` |
| MAE | `SUM([Abs Err Median Sum]) / SUM([Item Days])` |
| In-Stock at q90 | `SUM([In Stock Q90]) / SUM([Item Days])` |

### Calibration

| Name | Formula |
|---|---|
| Perfect Calibration | `AVG(FLOAT([Quantile Level]))` |

### DQ Store Day

| Name | Formula |
|---|---|
| Zero-Sales Rate | `SUM([Listed Items Zero Sales]) / SUM([Items Listed])` |
| Not-Listed Share | `SUM([Items Not Listed]) / SUM([Items In Catalog])` |
| Calendar Days | `COUNTD([Date])` |
| Missing Dates | `DATEDIFF('day', {MIN([Date])}, {MAX([Date])}) + 1 - COUNTD([Date])` |
| Missing Cells | `SUM([Missing Cells])` |
| Store Closure Days | `SUM([Store Closed])` |
| Low-Sales Anomaly Days | `SUM([Low Sales Anomaly])` |
| Flagged Day Units | `IF [Store Closed] = 1 OR [Low Sales Anomaly] = 1 THEN [Units Sold] END` |
| Day Flag | `IF [Store Closed] = 1 THEN "Store closed" ELSEIF [Low Sales Anomaly] = 1 THEN "Low-sales anomaly" ELSE "Normal" END` |

### DQ Series

| Name | Formula |
|---|---|
| Series Count | `COUNTD([Series Id])` |
| Possible Stockout Share | `SUM([Suspected Stockout 28D]) / COUNT([Series Id])` |

---

## 2. Dashboard 1: Store Outcomes

Global filters: **Model** (single value, dropdown, default **LightGBM**) and **Critical Ratio**
(single value list, default **0.9**). Right-click each filter → **Apply to Worksheets → All Using This
Data Source**. Show both on the dashboard.

| Sheet | Build |
|---|---|
| **KPI – Fill Rate** (+ 4 more) | Text mark. Drag **Fill Rate** to Text. Format text: big number (24 pt bold) + small label. Duplicate for **Waste %**, **Stockout Rate**, **Lost Revenue per Store-Day**, **Waste Cost per Store-Day**. |
| **State Map** | Double-click **State Name** (map appears). Color: **Fill Rate** (sequential blue). Label: **State Name**, **Fill Rate**, **Waste %**. Turn off unknown-location indicator; zoom to CA/TX/WI or keep the full US. |
| **Store Fill Rate vs Waste** | Rows: **State Id**, **Store Id**. Columns: **Fill Rate**, **Waste %** (two side-by-side bar panes). Color: **State Id**. Labels on. Add a reference line (Analytics → Average Line, per pane). |
| **Store Scatter** | Columns: **Waste %**. Rows: **Fill Rate**. Detail: **Store Id**. Color: **State Id**. Label: **Store Id**. Circle size ~large. Title: "Up and left is better". |
| **Daily Fill Rate** | Columns: **Date** (exact date, continuous). Rows: **Fill Rate**. Color: **State Id**. Line marks. |

Layout: filters + 5 KPIs across the top; map and scatter in the middle row; store bars and daily
line at the bottom. Add **Dashboard → Actions → Filter**: source **State Map**, on **Select**,
target all sheets, "Clearing the selection will: Show all values".

Subtitle text box: *"Each store orders every FOODS_3 item daily at the chosen quantile of the
forecast (critical ratio). Fill rate = units sold ÷ units demanded; waste = unsold units ÷ units
ordered. Test period 23 May – 19 June 2016."*

---

## 3. Dashboard 2: Model Comparison

These sheets compare all models and all critical ratios, so **don't** apply the global Model/CR filters
to them (when setting the filter scope on Dashboard 1, use **Selected Worksheets** if needed).

| Sheet | Build |
|---|---|
| **Trade-off Frontier** (Store Day) | Columns: **Waste %**. Rows: **Fill Rate**. Marks: **Line**. Color: **Model**. Path: **Critical Ratio**. Duplicate **Fill Rate** on Rows → dual axis → circle marks → synchronize axis. Label circles with **Critical Ratio**. Title: "Each point is an ordering policy; up and left is better". |
| **Cost vs Baseline** (Store Day) | Columns: **Critical Ratio**, **Model**. Rows: **Cost vs Baseline**. Filter **Model** to exclude Baseline. Color: **Model**. Labels on. (The FIXED calculation still sees the baseline rows because FIXED is computed before dimension filters.) |
| **Accuracy by Week** (Accuracy) | Columns: **Horizon Week**. Rows: **Pinball Loss**. Color: **Model**. Line + circle marks. |
| **Cost by Velocity** (Item Store) | Filter **Critical Ratio** = 0.3 and **Model** excludes Baseline. Columns: **Velocity** (sort by **Velocity Order**), **Model**. Rows: **Fresh Cost vs Baseline**. Color: **Model**. |
| **Calibration** (Calibration) | Columns: **Quantile Level**. Rows: **Coverage Mid**, then **Perfect Calibration** → dual axis → synchronize. Coverage Mid: circles colored by **Model**. Perfect Calibration: grey dashed line. |
| **Model Scorecard** (Accuracy) | Rows: **Model**. Columns: **Measure Names**. Text: **Measure Values** with **Pinball Loss**, **MAE**, **In-Stock at q90** only. |

Layout: frontier large on the left; scorecard, cost vs baseline and accuracy-by-week stacked on
the right; velocity and calibration along the bottom.

---

## 4. Dashboard 3: Data Quality

| Sheet | Build |
|---|---|
| **DQ KPIs** (DQ Store Day) | One text sheet per KPI: **Calendar Days**, **Missing Dates**, **Missing Cells**, **Store Closure Days**, **Low-Sales Anomaly Days**, **Zero-Sales Rate**, **Not-Listed Share**. |
| **Zero-Sales Heatmap** (DQ Store Day) | Columns: **MONTH(Date)** (continuous month, green pill). Rows: **State Id**, **Store Id**. Marks: **Square**. Color: **Zero-Sales Rate** (sequential orange). |
| **Daily Units + Flags** (DQ Store Day) | Columns: **Date** (exact, continuous). Rows: **SUM(Units Sold)**, then **SUM(Flagged Day Units)** → dual axis → synchronize. First axis: thin grey line. Second: circles colored by **Day Flag** (red = closed, amber = anomaly). Filter: **Store Id** (single value dropdown). |
| **Catalog Coverage** (DQ Store Day) | Columns: **MONTH(Date)** continuous. Rows: **Not-Listed Share**. Color: **Store Id**. Shows items launching over time. |
| **Demand Classes** (DQ Series) | Rows: **Demand Class**. Columns: **Series Count**. Color: **Demand Class**. Sort descending. Tooltip/caption: "Syntetos–Boylan classes: ADI = average days between sales, CV² = variability of non-zero sales." |
| **Longest Zero Run** (DQ Series) | Right-click **Longest Zero Run Days → Create → Bins** (size 14). Columns: the bin field. Rows: **CNT(Series Id)**. Bars. |

Text box for this dashboard:

> *The M5 calendar has no missing dates and no empty cells. Gaps show up in other ways: stores close
> on Christmas (24 store-days with zero sales), 21% of item-days are before an item was listed
> (no shelf price), and 47% of listed item-days have zero sales. 78% of series have at least one
> run of 28+ zero-sales days while listed: slow demand, a stockout or a temporary delisting. M5 has
> no inventory data to tell these apart, which is why sales understate true demand.*

---

## 5. Publish (15 min)

1. Optional: **Dashboard → New Story**, or add navigation buttons (**Objects → Navigation**) linking
   the three dashboards.
2. Hide the individual worksheets: right-click each dashboard tab → **Hide All Sheets**.
3. **File → Save to Tableau Public As…** → title: **M5 Grocery Forecasting: Fill Rate vs Waste**.
4. On your Tableau Public profile, open the viz → **Edit Details**: add the description and the
   GitHub link (`https://github.com/josethomas568/m5-grocery-newsvendor-forecasting`), set the
   first dashboard as the thumbnail, and copy the share link.
5. Add the link to the repo README (replace the placeholder in the Tableau section).

The data is public M5 data, so publishing it to Tableau Public is fine.

---

## 6. Check values

Tableau should reproduce these exactly (computed from the same extracts in Python).

**Dashboard 1, Model = LightGBM, Critical Ratio = 0.9**

| | Value |
|---|---|
| Fill rate (all stores) | 90.5% |
| Waste % | 55.2% |
| Stockout rate | 7.3% |
| Stockouts per store-day | 60.5 |
| Lost revenue per store-day | $429 |
| Waste cost per store-day | $4,085 |
| Fill rate by state: CA / TX / WI | 90.2% / 90.4% / 91.1% |
| Highest fill rate store | WI_1, 91.6% |
| Highest waste store | CA_4, 61.4% |

**Dashboard 2**

| | Baseline | LightGBM | Hierarchical Bayes |
|---|---|---|---|
| Pinball loss | 0.610 | 0.572 | 0.583 |
| MAE | 1.680 | 1.632 | 1.635 |
| In-stock at q90 | 91.4% | 92.7% | 91.8% |
| Cost vs baseline at CR 0.3 | – | −3.0% | −3.7% |
| Cost vs baseline at CR 0.95 | – | −10.0% | −6.0% |
| Fresh cost vs baseline, CR 0.3: slow / medium / fast | – | +0.8% / −3.7% / −3.1% | 0.0% / −2.4% / −4.6% |

**Dashboard 3**

| | Value |
|---|---|
| Calendar days | 1,969 (29 Jan 2011 – 19 Jun 2016) |
| Missing dates | 0 |
| Missing cells | 0 |
| Store closure days | 24 (all on 25 Dec) |
| Low-sales anomaly days | 33 |
| Zero-sales rate (listed item-days) | 47.3% (CA_3 lowest at 36.0%, TX_1 highest at 53.1%) |
| Not-listed share | 21.0% |
| Demand classes | Intermittent 56.2%, Lumpy 27.7%, Smooth 10.6%, Erratic 5.5% |

**Definitions**: *store closed* = the store sold zero FOODS_3 units that day; *low-sales anomaly* =
an open day with sales below half the store's centred 28-day median; *listed* = the item had a
shelf price that week in `sell_prices.csv`.
