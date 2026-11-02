# Forecasting literature for intermittent pharmacy demand

Part of our follow-up after the ML results: what the literature says about
forecasting demand like ours. A review, not experiments. Every claim about a paper is
linked to a source I opened or found in search results. Where I only saw an
abstract or a secondary summary, I say so. Where something comes from my own
memory and I could not confirm it, it is marked "(unverified)".

## The setting, briefly

About 3,000 drugs, weekly units, 1-8 week horizons, roughly 2.5 years of
history (stock data only from Oct 2023). Many series are low-count or
intermittent. Weeks with a total stock-out are dropped from fitting and
scoring; partial shelf-outs are not. Eid al-Fitr is the one strong calendar
shock, with only two or three past Eids to learn from. The forecasts feed an
(s,S) policy through a negative binomial (NB) lead-time quantile at 95%
service, so the upper tail of 1-2 week lead-time demand matters more than the
point forecast.

Two forecasters exist. The current one is per-drug SES with a store-wide,
shrunk Eid uplift (seasonal naive for rare drugs), NB spread with per-drug
dispersion. The ML one is a global LightGBM, Tweedie 1.3, raw units, direct
multi-horizon (horizon as a feature), masked censored lags, 3 seeds. ML wins
selection WAPE 0.451 vs 0.468, wins slightly in every tier on 2025, but the
decision replay difference (+0.4% lost revenue, CI spans zero) is not
significant.

That last point frames everything below. The point forecast is already about
as good as the two very different families can make it. The likeliest gains
are in the distribution (the tail the policy uses), in censoring, and in
combining what already exists, not in a new point model.

## Ranked shortlist

Ranked by expected decision value for this dataset per unit of effort.

### 1. Model lead-time demand uncertainty properly, not as a sum of independent weeks

**What.** The policy sums per-week NB means and variances over the lead time.
That treats each week's forecast error as independent. But the error in the
estimated level is shared across all weeks of the lead time, so errors are
positively correlated and the summed variance is too small. Fixes, from
simplest: (a) calibrate the lead-time variance empirically from backtest
errors of the 1-2 week sum, per tier or per dispersion bucket; (b) add the
level-estimation variance explicitly (for SES this has a closed form); (c)
forecast the lead-time sum directly as its own target (see item 4).

**Why here.** The decision uses the 95% quantile of lead-time demand. Under-
stated lead-time variance means reorder points that are too low, which shows
up as stock-outs, which then censor the data the next forecast learns from.
The existing p90 coverage check is per week; it does not show whether the
2-week sum is covered at 95%.

**Evidence.** [Prak, Teunter & Syntetos (2017), EJOR 256(2)](https://ideas.repec.org/a/eee/ejores/v256y2017i2p454-461.html)
show that forecast errors over the lead time are positively correlated even
when demand itself is not autocorrelated, so the standard per-period-variance
x lead-time approach underestimates variance and sets safety stocks too low
(abstract; the "up to 30% too low" figure comes from a secondary summary, not
checked in the paper). [Snyder, Ord & Beaumont (2012), IJF](https://ideas.repec.org/a/eee/intfor/v28y2012i2p485-496.html)
argue inventory planning for slow movers should use prediction distributions
from dynamic models with distributions more flexible than Poisson; later work
reports the NB did well in their study (abstract and secondary summary only).

**Limitations.** Needs a backtest of lead-time sums at the service quantile,
which exists only partly. Calibrating per segment needs enough stock-out-free
lead-time windows.

**Effort.** Low for (a): a few lines on existing backtest predictions.
Medium for (b).

**Segments.** All, most for fast and regular drugs where the policy holds
real stock.

### 2. Equal-weight combination of the ML and statistical forecasts

**What.** Average the ML mean with SES+Eid (and possibly TSB or SBA) per drug-
week, with fixed equal weights. No learned weights.

**Why here.** The two forecasters are built very differently (global trees
on features vs. local smoothing), so their errors are probably not strongly
correlated. ML wins by a small margin everywhere, with no tier where it is
clearly better. That is the textbook case where a simple average beats
picking one. It also hedges the "flat" ML forecasts with a method that reacts
to the drug's own recent level.

**Evidence.** [Petropoulos & Kourentzes (2015), JORS](https://eprints.lancs.ac.uk/id/eprint/69427)
found that combining intermittent-demand methods, and combining forecasts
from different temporal aggregation levels, beats single methods and removes
the need for per-series selection (abstract). [Wang, Kang & Petropoulos
(2023), arXiv](https://arxiv.org/abs/2304.03092) found probabilistic
combinations beat individual methods for intermittent demand, with a trade-
off between forecast accuracy and inventory performance (abstract only).
[Li, Kang, Petropoulos & Li (2022), arXiv](https://arxiv.org/abs/2204.08283)
report feature-based combination weights improving point and quantile
forecasts on two intermittent datasets (abstract). The general finding that
combinations are hard to beat runs through the M competitions; the M4 and M5
result papers both stress it (I did not re-read the full papers for this
review). [Kourentzes (2014), IJPE](https://ideas.repec.org/a/eee/proeco/v156y2014icp180-190.html)
found automatic per-series model selection did no better than using one model
(abstract), another argument against fine-grained selection.

**Limitations.** If one forecaster is much better in a segment, averaging
dilutes it. The ML NB spread is borrowed from the baseline recipe, so the
combination's spread needs the same calibration as item 1.

**Effort.** Very low. Both forecasts already exist on the 2025 eval origins
(`data/ml/eval/predictions/*.parquet`); the combination can be scored there
as a diagnostic and confirmed on the selection origins.

**Segments.** Fast, regular, slow. Probably not rare (see item 5).

### 3. Treat stock-out censoring as a bias problem, not just a masking problem

**What.** Dropping weeks with a stock-out removes exactly the weeks where
demand was high relative to stock. The surviving weeks are a biased sample:
the model learns a level that is too low, and scoring on the same filtered
weeks hides that. Options, from simplest: (a) measure how much demand is
lost by comparing sales rates on days before the stock-out within the same
week (partial-week rate scaling, a common practical fix); (b) keep censored
weeks as lower bounds and fit with a censored likelihood (Tobit-style), which
SES-style models support; (c) two-stage: recover latent demand first, then
train the forecaster on it.

**Why here.** The stock data begins Oct 2023, so the censoring flag exists for
most of the training span. The ML plan already planned an "impute" version
but did not run it. Shelf-outs (floor empty, store stocked) are not treated
as censoring at all. Any downward bias feeds straight into the reorder point.

**Evidence.** Censored sales understate demand; ignoring it leads to
"systematic degradation" in airline revenue management, where the fix is
called unconstraining ([review, 2012](https://www.doi.org/10.1155/2012/270910),
abstract only). [Pedregal, Trapero & Holgado, Tobit Exponential Smoothing
(arXiv 2024)](https://arxiv.org/abs/2407.17920) extend ETS to censored data
and report a large reduction in forecast bias; a [follow-up with temporal
aggregation](https://arxiv.org/abs/2409.05412) handles mixed frequencies.
[FreshRetailNet-50K (arXiv 2025)](https://arxiv.org/abs/2505.16319) reports
that a two-stage recover-then-forecast approach cut systematic underestimation
from 7.37% to near zero with a 2.73% accuracy gain (preprint, self-reported).

**Limitations.** Latent demand is unobserved, so you cannot score the
correction directly; the decision replay is the only honest test. Substitution
between drugs during stock-outs (demand moves to a sibling, not lost) makes
the correction overstate total demand for the group. Tobit ETS is new and only
in preprints.

**Effort.** (a) Low-medium. (b) Medium. (c) Medium-high.

**Segments.** Fast and regular drugs that stock out. Little effect on rare.

### 4. Temporal aggregation (ADIDA / iMAPA / forecasting the lead-time sum)

**What.** Sum weekly demand into larger buckets (2, 4 weeks, or the lead time
plus review period), forecast there, then split back down if needed. iMAPA
does this at several levels and averages them. Temporal hierarchies (THieF)
reconcile forecasts across levels.

**Why here.** Aggregation reduces intermittence: a drug that sells in 30% of
weeks may sell in most 4-week blocks. The policy needs lead-time totals, not
weekly values, so forecasting the 2-week or 4-week sum directly targets what
the decision uses and sidesteps the correlated-error problem in item 1.

**Evidence.** [Nikolopoulos et al. (2011), JORS](https://eprints.lancs.ac.uk/id/eprint/58330)
proposed ADIDA on 5,000 RAF SKUs and studied aggregating into buckets equal
to lead time plus review period (abstract). [Petropoulos & Kourentzes (2015)](https://eprints.lancs.ac.uk/id/eprint/69427)
combine across aggregation levels; their iMAPA is in the R package
[tsintermittent](https://rdrr.io/cran/tsintermittent/man/imapa.html).
[Athanasopoulos, Hyndman, Kourentzes & Petropoulos (2017), EJOR](https://www.research.lancs.ac.uk/portal/en/publications/forecasting-with-temporal-hierarchies(1ae6e252-ec65-4fe0-aceb-d6b8af52cb05).html)
report temporal hierarchies improve accuracy especially when model
uncertainty is high (working paper abstract).

**Limitations.** Aggregation blurs the Eid weeks, so the Eid uplift must be
applied at the bucket level. Disaggregating back to weekly is mostly flat
(that is fine; see the flat-forecast section). The literature is mostly on
spare parts, not pharmacy retail. iMAPA lives in R.

**Effort.** Low for ADIDA with SES (a Python reimplementation is short).
Medium for full iMAPA or THieF.

**Segments.** Slow and rare drugs most. Also the lead-time target for all
tiers.

### 5. Pooled (shrinkage) level estimates for rare and slow drugs

**What.** For a drug with few sales, estimate its rate as a weighted mix of
its own history and a group prior (tier, form, ingredient group, or all
drugs), with weights set by how much data it has. The simplest form is a
Gamma-Poisson (empirical Bayes) rate. The full form is a Bayesian
hierarchical NB state-space model.

**Why here.** Seasonal naive wins on rare drugs today. That says the local
smoothers are too noisy on very sparse series, not that last year's same
week is a good predictor. Shrinkage is the standard answer to "too little
data per series," and it gives a proper predictive distribution, which the
policy needs.

**Evidence.** [Chapados (2014), ICML](https://proceedings.mlr.press/v32/chapados14.html)
builds a hierarchical NB state-space model for groups of short count series
from supply chain planning, sharing hyperparameters across series (abstract).
[Montero-Manso & Hyndman (2021)](https://arxiv.org/abs/2008.00444) show
global (pooled) models are not more restrictive than local ones and have
complexity that does not grow with the number of series (abstract), which
supports pooling in general. The [Bayesian hierarchical sparse count
retail paper](https://arxiv.org/abs/1805.05657) is another example (I saw
only the search listing).

**Limitations.** Ingredient-group features made the ML model slightly worse,
so the right grouping is unclear; tier-level priors may be all that helps.
Full Bayesian fits on 3,000 series are slow without care.

**Effort.** Low for Gamma-Poisson empirical Bayes. High for the full
hierarchical model.

**Segments.** Rare and slow.

### 6. TSB (occurrence-probability smoothing) for slow and rare drugs

**What.** Teunter-Syntetos-Babai: smooth the probability that a sale occurs
in a week (updated every week, including zero weeks) and the size when it
does; forecast = probability x size.

**Why here.** Croston and SBA only update after a sale, so a drug that stops
selling keeps its old forecast. In a pharmacy with delistings, generics
switching and a store change, obsolescence is real. TSB is already
implemented in `forecast.py` and ran in the baselines, so the cost is only a
fair comparison on rare drugs and inside a combination.

**Evidence.** [Teunter, Syntetos & Babai (2011), EJOR](https://orca.cardiff.ac.uk/id/eprint/42067/)
introduce the method and argue it suits obsolescence because the forecast
falls after zeros (abstract; tested on simulated data).
[Babai, Syntetos & Teunter (2014), IJPE](https://bnu.repository.guildhe.ac.uk/id/eprint/17358/)
is an empirical follow-up on accuracy and obsolescence risk (I saw only the
record). On SBA: [Syntetos & Boylan (2005), IJF](https://salford-repository.worktribe.com/output/1474808/the-accuracy-of-intermittent-demand-estimates)
found their Croston correction beat SES, Croston and a moving average on
3,000 automotive series; but an independent 16,000-item warehouse study cited
in the same search found Croston and SBA worse than SES (secondary source, not
verified). The project's own finding that SBA runs 4-6% low on fast drugs
fits the known downward bias of SBA on non-intermittent series.

**Limitations.** Gains over SES are usually modest. The
[Syntetos-Boylan-Croston (2005) ADI/CV² cut-offs (1.32, 0.49)](https://orca.cardiff.ac.uk/45026)
were derived for choosing between Croston and SBA, not as a general router;
using them to route methods is common practice but not what the paper
showed.

**Effort.** Very low.

**Segments.** Slow, rare, and drugs in decline.

### 7. Distribution-aware global model (keep LightGBM, change what it outputs)

**What.** Keep the global model for the mean but model the spread too: either
a second model for the NB dispersion (GAMLSS style, location and scale as
functions of features), quantile LightGBM at the service quantile, or a
hurdle split (occurrence classifier x size regressor).

**Why here.** The ML model borrows the baseline's per-drug dispersion. A
global spread model could use the same features (censoring history, Eid,
tier) to widen or narrow the tail where it matters.

**Evidence.** [Ziel (2021), M5 uncertainty](https://arxiv.org/abs/2107.06675)
argues M5 data is strongly overdispersed with many zeros, that LightGBM and
XGBoost with their usual objectives do not capture this well, and that
distributional (GAMLSS, NB location-scale) models fit better (abstract). The
[M5 uncertainty results paper (IJF 2022)](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1365-1385.html)
asked for nine quantiles on 42,840 series (I did not read the full findings).
[Damato et al. (2026), arXiv](https://arxiv.org/abs/2601.14031), on 40,000+
intermittent series, found a simple global network (TiDE) beat local models
and that a Tweedie head gave the best high quantiles among NB, hurdle-NB and
Tweedie heads (abstract; preprint).

**Limitations.** Quantile models per quantile can cross and are hard to sum
over a lead time; a parametric (NB) head is easier to aggregate. Extra
moving part to maintain.

**Effort.** Medium.

**Segments.** Fast and regular, where there is enough data to learn a spread.

### 8. Small refinements to the existing global model

**What.** Per-series scaling of the target (planned, not adopted), a Tweedie
power sweep (only 1.3 used), direct models per horizon group instead of one
model with h as a feature, and dropping features that only add variance.

**Why here.** These are the standard M5 levers. The plan already scoped them.

**Evidence.** Tree models dominated M5, and the availability of Tweedie loss
is cited as one reason ([Januschowski et al. (2022), "Forecasting with
trees", IJF](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1473-1481.html),
abstract and secondary summaries; [M5 accuracy results, IJF 2022](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1346-1364.html)).
My recollection (unverified) is that the M5 winner averaged LightGBM Tweedie
models trained per store and per store-category/department, mixing recursive
and non-recursive versions. [Spiliotis et al. (2022)](https://ideas.repec.org/a/spr/operea/v22y2022i3d10.1007_s12351-020-00605-2.html)
found some cross-learned ML methods beat Croston-type methods on daily SKU
data in accuracy and bias (abstract). [Petropoulos et al., "Wielding Occam's
razor"](https://arxiv.org/abs/2102.13209) argue a small, well-chosen model set
loses little accuracy at far lower cost (abstract).

**Limitations.** M5 was daily, Walmart-scale (30,490 bottom series, about
five years), with prices and promotions. This set is weekly, one store, 2.5
years, with almost no promotion signal. Expect much smaller gains. The
current ML-vs-baseline gap is already within noise at the decision level.

**Effort.** Low to medium each; the risk is spending tuning budget for
WAPE points that do not move the replay.

**Segments.** Fast and regular.

### 9. Lower priority: iETS, deep global models, Prophet

[iETS (Svetunkov & Boylan, IJPE 2023)](https://www.r-bloggers.com/2023/09/iets-state-space-model-for-intermittent-demand-forecasting/)
is a state-space model that handles occurrence and size together, covers
Croston and TSB as special cases, and removes the need to classify series
first. It is attractive in principle (a proper predictive distribution, one
model for all tiers) but is in the R package `smooth` only. Worth it only if
items 1, 5 and 6 leave a gap.

Deep global models (DeepAR, TiDE, [deep renewal processes, Türkmen et al.
2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8629246/)) have some
evidence on intermittent data, but with 2.5 years of weekly data from one
store the sample is small, and the plan excluded them on scope. I would not
start here.

Prophet with one or two past Eids has very little to learn the holiday
effect from; the store-wide shrunk uplift already pools that information more
efficiently.

## Are flat forecasts a bug or the right answer?

Mostly the right answer, with three places where flatness would hide real
signal.

**Why flat is often optimal.** If a series has no trend and no seasonality,
the best estimate of its mean is the same for every future week, and only the
uncertainty grows with the horizon. SES makes this explicit: in Hyndman &
Athanasopoulos's textbook, SES "has a 'flat' forecast function", all
forecasts equal the last level, and it suits data with no clear trend or
seasonal pattern ([FPP3, 8.1](https://otexts.com/fpp3/ses.html)). Croston,
SBA and TSB are flat by construction too: they forecast a rate, not a path.
[Shenstone & Hyndman (2005)](https://ideas.repec.org/p/msh/ebswps/2003-1.html)
found no consistent stochastic model underlies Croston's method but that its
point forecasts can still be useful (abstract). For a low-count drug selling
two or three units a week, week-to-week variation is mostly Poisson-type
noise. Any horizon-specific pattern estimated from 2.5 years would mostly fit
noise. A flat forecast is the honest output.

**Why the ML model in particular comes out flat.** It is direct multi-horizon
with h as a feature. Every feature except h and the target-week calendar
features (week of year, Eid distance, Ramadan flag, Eid uplift) is the same
across the eight rows of a drug-origin. Outside the Eid window those calendar
features carry little signal at weekly resolution, so trees split mostly on
own-demand features and the eight predictions land in nearly the same leaves.
h1 = 2.4536 vs h2 = 2.4584 is what this design should produce for an
ordinary drug in an ordinary month. It is not by itself a bug.

**Where flatness would mean missed signal.** Check these, in order:

1. **Eid and Ramadan weeks.** Forecasts made before Eid should rise in the
   Eid window and fall after. If ML forecasts stay flat through an Eid that
   falls inside the 8-week horizon, the calendar features are not being used.
   This is the single most important test.
2. **Mean reversion after an unusual week.** If the last week was a spike,
   the h1 forecast should sit a bit closer to it than h8, which should sit
   closer to the long-run mean. A global model can learn this; SES cannot.
   Check by regressing realised demand at each h on recent lags vs. long
   means: if the weight on recent lags falls with h in the actuals but not in
   the forecasts, there is horizon signal left on the table.
3. **Lifecycle drift.** New drugs ramping up and dying drugs fading have a
   real slope. A flat forecast will be biased on them in a predictable
   direction. Check bias by horizon on drugs younger than ~26 weeks and on
   drugs whose 13-week mean is far below their 52-week mean.

A simple diagnostic covers all three: compute bias (sum of forecast / sum of
actual) by horizon, separately for Eid-window and non-Eid targets and by
lifecycle stage. If bias is flat across h, flat forecasts are fine. If it
drifts with h, there is signal to recover. Note too that uncertainty must
grow with h even when the mean is flat; for the policy, that matters more
than the mean path (item 1).

## Evaluation lessons from the literature

**WAPE and MAE reward the median, which can be zero.** A forecast that
minimises MAE targets the median, and one that minimises squared error
targets the mean ([FPP3, 5.8](https://otexts.com/fpp3/accuracy.html)). WAPE is
total absolute error over total actuals, so it is MAE rescaled and inherits
this. On a series that sells in fewer than half its weeks, the median is
zero, and a forecast of zero often scores well on MAE/WAPE
([Li et al. 2022](https://arxiv.org/abs/2204.08283), citing Kolassa 2020).
Kolassa's 2016 slides on count forecasts make the same point and recommend
always reporting bias next to MAE, wMAPE or MASE, since those are all
multiples of MAD (from a search summary of the slides; I could not open the
PDF). For a reorder policy, a zero-ish forecast is the worst kind of good
score. The project's bias band (0.97-1.03 on fast and regular) is the right
guard. Rare and slow drugs need the same guard, and ideally a metric that is
minimised by the mean, such as RMSE scaled per series.

**Use scaled errors to compare across drugs.** MASE divides by the in-sample
error of a naive method, so it is scale-free and stays defined when a series
has zeros ([Hyndman & Koehler 2006](https://ideas.repec.org/a/eee/intfor/v22y2006i4p679-688.html);
[Hyndman 2006, Foresight](https://ideas.repec.org/a/for/ijafaa/y2006i4p43-46.html)).
But MASE is still MAE-based, so it also leans toward the median and toward
under-forecasting ([Doszyń 2020](https://ideas.repec.org/a/ers/journl/vxxiiiy2020ispecial1p1113-1127.html),
from search summary). A scaled squared error (RMSSE, the M5 accuracy metric)
targets the mean, which is what a reorder policy builds on.
[Kourentzes (2014)](https://ideas.repec.org/a/eee/proeco/v156y2014icp180-190.html)
proposes rate-based metrics (mean absolute rate, mean squared rate) for
tuning intermittent methods, which compare cumulative forecast against
cumulative demand.

**Score the distribution, at the quantile the policy uses.** The policy uses
a 95% lead-time quantile, so the relevant scores are pinball loss at that
quantile on the 1-2 week lead-time sum, and coverage of that sum. CRPS
scores the whole distribution and is strictly proper ([Gneiting & Raftery
2007, JASA](https://doi.org/10.1198/016214506000001437)). For count
forecasts, PIT histograms need randomising because the outcomes are
discrete ([Kolassa 2016, IJF](https://ideas.repec.org/a/eee/intfor/v32y2016i3p788-803.html),
which argues discrete predictive distributions are needed for inventory and
that the right tools are rarely used in sales forecasting). The M5
uncertainty competition used a scaled pinball loss over nine quantiles
([results paper](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1365-1385.html)).

**Accuracy and inventory performance can disagree.** [Wang, Kang &
Petropoulos (2023)](https://arxiv.org/abs/2304.03092) report a trade-off
between forecast accuracy and inventory performance for intermittent demand.
The project already sees this: ML wins WAPE everywhere but the replay
difference is within noise. The decision replay (lost revenue, stock-out
days, stock held) should stay the final judge, with forecast metrics as
diagnostics.

**Censoring contaminates the scorecard too.** Scoring only stock-out-free
weeks rewards forecasts that are low exactly where demand was high. Any
method that corrects censoring (item 3) will look worse on that scorecard
even if it is better for the policy. This is another reason to judge on the
replay.

**Small samples need intervals.** With one Eid in the selection window and
one in 2025, any Eid-specific conclusion rests on one event. Report
per-origin spread or block-bootstrap intervals, as the project already does
for the replay.

## Sources not verified in full

Most entries above rest on abstracts, repository records or secondary
summaries, not full texts. The specific claims I relied on most and did not
confirm in the original: the "up to 30%" safety-stock shortfall attributed to
Prak et al.; the M5 winner's exact model design; the 16,000-item study
showing Croston/SBA worse than SES; and the Kolassa slide wording.
