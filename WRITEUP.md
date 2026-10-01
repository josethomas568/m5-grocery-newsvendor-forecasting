# How forecast errors affect stores

*Write-up for the M5 FOODS_3 probabilistic forecasting and newsvendor project.*

## The setup

A store orders every item every day. If it orders too little, the shelf goes empty and the sale
is lost. If it orders too much, the leftover is thrown away. I forecast demand for 823 FOODS_3
items in 10 Walmart stores over the official M5 evaluation month (23 May to 19 June 2016) with
three models: a simple baseline, LightGBM quantile regression, and a hierarchical Bayesian model.
I then turned each forecast into daily orders and counted what happened on the shelf.

## 1. The two kinds of error do not cost the same

An under-forecast costs the margin on the sale you missed. An over-forecast of a perishable costs
the full price you paid for the unit you threw away. At a 30% grocery margin, one unit of waste
costs 70 cents per dollar of price; one lost sale costs 30 cents. Waste is 2.3 times as expensive.

That asymmetry decides the order. The newsvendor answer is to stock up to the demand quantile at
the critical ratio, margin / (margin + cost) = 30%. Ordering the "best guess" (the median) instead
raised total cost by 14% for LightGBM and 20% for the baseline. **A point forecast, however
accurate, gives the wrong order. The store needs the forecast's distribution.**

## 2. What the errors look like in one store, on one day

FOODS_3 demand in one store averages about 2,040 units a day. Under a high-availability policy
(order the 90th percentile):

| Per store, per day | Baseline | LightGBM | Hierarchical |
|---|---|---|---|
| Items that run out | 70 | 61 | 68 |
| Units of demand lost | 258 | 193 | 219 |
| Lost sales ($ revenue) | $511 | $429 | $479 |

Waste stays about the same across models at this setting, so the better forecast buys roughly
a quarter fewer lost units for free. Read the other way, along the trade-off curve: to serve 90%
of demand, LightGBM needs **19% less waste** than the baseline, and the hierarchical model 13% less.

Under the cost-optimal fresh policy (30th percentile) the trade runs the other direction:
LightGBM cut waste from $442 to $265 per store-day (−40%) but lost more sales, for a net 3% lower
total cost. Better forecasts don't remove the trade-off; they move the whole curve.

## 3. Where errors hurt most

- **Slow movers can't be forecast into profitability.** A quarter of item-days come from items
  selling under one unit every two days. At fresh-grocery economics every model orders zero for
  them, and no model beats the baseline. Whether to carry them is an assortment decision, not a
  forecasting one.
- **Fast movers carry the money.** Items selling over two a day account for about 65% of total
  newsvendor cost, and that's where both models save the most (3–5% vs baseline).
- **Errors grow with the horizon.** Pinball loss rises 18–33% from week 1 to week 4, depending
  on the model. The hierarchical model is slightly better in week 1 and level in week 2;
  LightGBM pulls ahead in weeks 3–4 because its price and calendar features keep working when
  recent sales history goes stale. For a store, ordering more often (shorter lead time) is worth
  as much as a better model.
- **Calibration sets the stockout rate; information sets the waste.** Ordering each model's 90th
  percentile kept items in stock on 91–93% of item-days, so all three keep the promise the
  policy makes. The difference is where each puts the stock: LightGBM uses prices, weekday and
  recent trend to shift quantity toward the days and items where demand actually shows up, so it
  reaches the same service with less excess on the shelf.

## 4. Caveats that matter in a real store

- **Sales are not demand.** When an item sold out, M5 records the sales, not the demand that walked
  away. Models trained on censored sales learn to under-forecast popular items, which causes more
  stockouts, which are recorded as lower sales. Breaking that loop needs stockout flags or
  on-hand inventory data.
- **No carry-over.** I treated every unit as perishable within the day. Many FOODS_3 items are
  shelf-stable, so absolute waste here is far too high (at the 90% policy it exceeds demand).
  The comparisons between models still hold; the levels don't.
- **Real ordering has friction:** lead times, case packs, minimum shelf presentation, markdowns
  before items expire, and the goodwill cost of an empty shelf, which pushes the target service
  level above the pure-margin 30%.
- **One evaluation month.** Results are for a single 28-day window and should be confirmed on
  rolling origins.

## 5. What I'd tell a store operations team

1. Set service targets by item economics, not one number for the whole store. Perishables
   with thin margins should target lower in-stock levels than shelf-stable staples.
2. Ask for forecast quantiles, not just a number; the order is a quantile.
3. Judge forecasts by waste at a fixed service level, or stockouts at a fixed waste budget.
   A 6% improvement in pinball loss here became 13–19% less waste at the same fill rate.
4. Start capturing out-of-stock events. Without them, every model learns to repeat yesterday's
   empty shelf.
