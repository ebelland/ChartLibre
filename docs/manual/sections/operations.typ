#import "helpers.typ": *

= Series operations <series-operations>

A *series operation* takes the data of one or more series already on a chart and computes something from it: a smoothed curve, a fitted model, a spectrum, the position of the peaks, a statistical test. The result is added to the chart as new series (and, where it is data, as a new table), so it can be styled, exported and operated on again like anything else.

== The Series Operations page

The *Series Operations* page of the navigation rail is a list, organized in sections like a system sidebar. Each row has an icon and a name; resting the pointer on a row shows a one-line description in the bar at the bottom of the page. A search box at the top filters the list by name or description as you type, and *Return* opens the first match.

#defs(
  [*Plot*], [*Plot* — creates a new chart (see @creating-a-chart).],
  [*Data*], [*Query Builder* — writes and saves SQL queries (see @query-builder).],
  [*Analysis*], [Peaks, Roots, Calculus.],
  [*Statistics*], [Statistics, Outliers, Clustering, Control Chart, Transform, Decomposition.],
  [*Geometry*], [Geometry.],
  [*Signal Processing*], [Smoothing, Spectral Analysis, Filtering, Baseline Correction.],
  [*Modeling*], [Fit, Interpolation, Function, Regression, GP Regression.],
)

An operation added by you or a plugin (see @advanced-operation) appears under *Other* unless it is given a section.

#figure(
  image("../screenshot_series_operations.png", width: 100%),
  caption: [The Series Operations page.],
)

== How every operation window works

Every operation opens the same kind of window, operating on the figure currently shown:

- *On the left*, in collapsible sections: the *Axis / Series* selector (which axis of the figure, and which of its series to read), the *Model* (the method, with a link to its documentation), and the *Parameters* of that model. Each parameter explains itself in its tooltip.
- *On the right*, the *Results* pane: a formatted report of what was computed — tables of numbers, fit statistics, warnings — and below it the log of the computation.
- *At the bottom*, the buttons: *Preview*, *Copy* (copies the report), *OK* and *Close*.

The workflow is always the same:

+ Choose the axis and tick the series to operate on.
+ Choose a model and set its parameters.
+ Click *Preview*. The calculation runs, the report fills the Results pane and the result is drawn on the chart, so you can judge it in place.
+ Adjust and *Preview* again as often as needed. Each Preview replaces the previous one.
+ Click *OK* to keep the result and close the window — or *Close* to discard the preview and leave the chart exactly as it was.

#note[
  Nothing is computed while you are editing. Changing a model, a parameter or the selected series only marks the current result as out of date; *Preview* and *OK* are what run the calculation. A slow operation can be set up calmly, and a half-typed number is never computed.
]

#note[
  A calculation that takes a while — a *Fit* above all — runs in the background. The window stays usable, a *Stop* button and a progress bar appear beside *Preview*, and *Stop* ends the calculation at its next step without changing anything. Closing the window stops it too.
]

=== What an operation writes

When *OK* is clicked:

- Results that are data (a smoothed curve, a spectrum, fitted values) are saved as new *tables* in the project. Their names start with an underscore (`_Smoothing_temperature`, ...) so they are easy to tell apart from imported data in the table list.
- New *series* reading those tables are added to the chart: on the same axis when the result is in the same units as the source (a smoothed or fitted curve), on a new axis of the same figure when it is not (a derivative, a spectrum, a control chart).
- The report is added to the figure's *notes*: a pane below the chart that collects the reports of every operation applied to it, and that goes into exported reports (see @output).
- The operation, its parameters and its report are recorded in the project's *history* (see @history), so it can be reviewed later.

*Statistics* and *Geometry* are the two exceptions: Statistics writes no tables (it reports, and draws a chart only on request), and Geometry writes a query instead of a table.

=== One series or several

Most operations work on *each ticked series separately*, producing one result per series — smooth three series at once, get three smoothed curves. *Decomposition* needs several series and combines them; *Statistics* reports on each series and also compares them in pairs and as groups; *Spectral Analysis* pairs the first ticked series with each of the others for its two-signal estimators.

=== Curves, surfaces and fields

The *Axis / Series* selector labels each series with the shape it found:

#defs(
  [`2D (x, y)`], [An ordinary curve — every operation's usual case.],
  [`3D on a grid (x, y, z)`], [The rows form a complete x/y grid, so the surface is read exactly as measured.],
  [`3D scattered (x, y, z)`], [Points with no grid. Operations that need a grid interpolate onto one and leave everything outside the samples' convex hull blank, rather than inventing values where nothing was measured.],
  [`vector field (x, y, u, v)`], [Shown for information; no operation computes on vector fields yet.],
)

Five operations also work on surfaces: *Fit*, *Function*, *Peaks*, *Roots* and *Calculus*. *Smoothing* has its own 2D and 3D models, and *Geometry* moves 3D series in space. Each is described below.

// ======================================================================
== Analysis

=== Peaks — find and measure peaks
Finds the local maxima (and/or minima) of a series and measures each one: position, height, *prominence*, width at half prominence and the x bounds of that width. The peaks are marked on the chart and listed in the report.

A raw search for local maxima on real data returns hundreds of hits, almost all noise, so Peaks is built around the filters that decide what counts as a peak:

#defs(
  [*Model*], [*Maxima*, *Minima*, or *Maxima and minima*.],
  [*Filter by*], [*Prominence* (default): how far a peak stands above the higher of the two valleys either side of it. It works on a sloping or raised baseline. *Height*: the value compared with zero, meaningful only when the baseline is at zero.],
  [*Threshold*], [The minimum prominence or height, as a fraction of the signal's full range (0.1 keeps peaks standing at least a tenth of the range above their surroundings).],
  [*Distance*], [Minimum separation in points; of two peaks closer than this, the more prominent is kept. This is what stops one noisy peak being reported as several.],
  [*Min width*], [Minimum width at half prominence, in points. Raise it to reject single-sample spikes.],
  [*Limit*], [Keep at most this many peaks, the most prominent first.],
)

#use[Spectroscopy and chromatography (peak positions and widths), finding cycles and extremes in any signal. The half-prominence bounds are exactly what *Calculus* needs to integrate a peak's area.]

On a *surface*, Peaks finds local maxima and minima in two dimensions with the same parameters.

=== Roots — find where a series crosses a level
Finds every x at which the series crosses a chosen *level* — the zeros of y(x) − level. It first scans consecutive samples for a change of sign (each such pair *brackets* a crossing), then refines each bracket with a numerical solver on an interpolating curve. Because the solvers cannot leave their bracket, refining can only improve the answer. Each root is reported with its direction (rising or falling), the solver's iterations and its residual.

#defs(
  [*Model (solver)*], [*Brent* (default, fast and safe), *Bisection* (slowest, most robust), *TOMS 748* (often fewest iterations), *Newton (secant)*.],
  [*Level*], [The y value to solve for; 0 finds the zeros.],
  [*Interpolation*], [What the series does between samples: *Straight line* (claims nothing the samples do not say), *Cubic spline* or *Monotone cubic (PCHIP)* (better for smooth signals; a spline can overshoot noisy data into crossings that are not there).],
  [*Tolerance digits*], [How many decimal places of x the solver must settle.],
  [*Max iterations*], [Per crossing.],
  [*Limit*], [Keep the first N crossings in x order.],
)

#use[Titration endpoints, the time a response falls to half its value, threshold crossings, the temperature at which a difference changes sign, the zero crossings of a derivative (maxima and minima).]

On a *surface*, Roots traces the *level curve* z = level — a contour line — instead of a list of points.

=== Calculus — differentiate or integrate
Computes the numerical derivative or integral of a series. Two things that are easy to get wrong are built in:

- *Smoothing inside the derivative.* Differentiation amplifies noise: a finite difference divides by a small step, so small noise becomes large noise. The default derivative fits a low-order polynomial over a moving window and differentiates that (Savitzky–Golay), smoothing and differentiating in one step.
- *Baseline removal inside the integral.* The area under a peak is meaningless if the signal does not return to zero, because a constant offset adds offset × width to the area.

#defs(
  [*Derivative (Savitzky-Golay)*], [Default. *Window* (points per local fit; larger smooths more), *polynomial order*, and *derivative order* (1 for the slope, 2 for the curvature).],
  [*Derivative (finite difference)*], [Plain central differences: exact on clean data, noisy on measured data.],
  [*Derivative (spline)*], [Differentiates a smoothing spline; *smoothing* 0 passes through every point, larger values allow the spline to depart from noisy data.],
  [*Integral (cumulative)*], [The running integral (trapezoidal rule), as a new curve.],
  [*Integral (total area)*], [One number: the area under the curve. *Simpson's rule* is more accurate on smooth data. *Baseline*: none, subtract the minimum, or subtract the straight line between the endpoints.],
  [*Derivative (surface gradient)*], [For a surface: the gradient magnitude at each point, with ∂z/∂x and ∂z/∂y alongside.],
  [*Integral (surface volume)*], [For a surface: the volume under it.],
)

Derivatives are drawn on a new axis by default (a slope is in different units from the data). On a date axis, the derivative is expressed per natural unit of the sampling interval — per day for daily data — rather than per second.

#use[Rates of change (velocity from position, growth rate), finding inflection points, peak areas (concentration from a chromatogram), totals from rates (energy from power).]

// ======================================================================
== Statistics

=== Statistics — tests and summaries
Reports descriptive statistics and hypothesis tests on the ticked series, and can add a chart of what was tested. It is described fully in @statistics.

=== Outliers — detect anomalies
Flags the points of a series that do not belong with the rest, and either *hides* them from the chart or *colours* them.

#defs(
  [*Z-score threshold*], [Points more than *threshold* standard deviations from the mean. Simple, but the outliers themselves inflate the standard deviation.],
  [*Interquartile range*], [Points beyond Q1 − k·IQR or Q3 + k·IQR (*IQR factor* k, 1.5 by default — the box-plot rule). Robust to the outliers it is looking for.],
  [*Median absolute deviation*], [Like the z-score but built on the median and the MAD, so a few extreme values cannot hide each other.],
  [*Rolling median residual*], [Compares each point with the median of a moving *window* around it — finds spikes in a signal that trends or oscillates, where a global rule would flag the peaks of the signal itself.],
  [*Isolation Forest*], [Machine learning: points that are easy to isolate by random splits are anomalous. *Contamination* is the expected fraction of outliers.],
  [*Local Outlier Factor*], [Points in regions much sparser than their *neighbours'* regions.],
  [*One-Class SVM*], [Learns the boundary of the normal points; *nu* bounds the fraction treated as outliers.],
  [*Elliptic Envelope*], [Fits a robust ellipse (Minimum Covariance Determinant) to the cloud; points far outside it are outliers.],
)

The last four judge a point by its position relative to the others in (x, y), not by its value alone, so they can find points that are unusual for their x even when their y is ordinary. Both coordinates are rescaled robustly first, so x and y in different units weigh equally.

*Action*: *Hide them* marks the rows in the table's `Hide` column, which every chart skips — the data stays in the table, and the rows can be un-hidden from the table editor (see @table-editor). *Colour them* keeps every point on the chart and draws the outliers in the chosen *colour*.

#use[Cleaning measurement glitches before a fit, flagging anomalous batches, checking how sensitive a result is to a few points.]

=== Clustering — group similar data
Groups the points of a series into clusters and writes each row's cluster number into a `ClusterId` column of the source table. The result is drawn either as *one series coloured by ClusterId* or as *separate series per cluster* (each with its own legend entry and style).

#defs(
  [*K-means / vector quantization*], [SciPy's k-means: a fixed *number of clusters*, each the set of points nearest one centre. *Whiten* rescales features first; *iterations* and *threshold* control convergence.],
  [*Hierarchical / agglomerative*], [Merges the closest points and groups step by step. *Linkage*: single, complete, average, weighted, centroid, median, Ward. Cut either at a number of clusters or at a *distance threshold*.],
  [*scikit-learn*], [K-Means, Mini-Batch K-Means, Bisecting K-Means, Agglomerative, *DBSCAN* and *OPTICS* (density-based: find clusters of any shape and leave noise points unassigned; set the neighbourhood radius *eps* and *min samples*), Birch, Mean Shift, Spectral Clustering, Gaussian Mixture.],
)

*Features* chooses which numeric columns of the series are used. A *max runtime* stops algorithms that scale badly on large tables.

#use[Finding natural groups — material phases, customer segments, operating regimes — and then analysing each group separately.]

=== Control Chart — monitor process stability
Builds a Shewhart control chart and flags the points that signal a change in the process. The chart is drawn on a new axis with its centre line, control limits and, optionally, the A/B/C zones.

The key point of a control chart is where its limits come from: never from the standard deviation of all the data (a process that drifted has a large overall spread *because* it drifted), but from the variation *within* short stretches of it — the moving range for individual values, the range or standard deviation within subgroups otherwise.

#defs(
  [*Individuals (I-MR)*], [One measurement per point; sigma from the average moving range.],
  [*Moving range (MR)*], [The companion chart of point-to-point ranges.],
  [*X-bar and R*], [Means of consecutive *subgroups* of size n (2–25), sigma from the average range. For n up to about 8.],
  [*X-bar and S*], [The same, sigma from the subgroup standard deviations. Better for larger subgroups.],
  [*p*], [Fraction defective per sample. Needs a *sample size column*; limits widen and narrow with the sample size.],
  [*np*], [Count defective per sample, constant sample size.],
  [*c*], [Defects per unit, constant area of opportunity.],
  [*u*], [Defects per unit with a varying sample size (needs the sample size column).],
)

#defs(
  [*Sigma limit*], [Distance of the limits from the centre, in sigmas; 3 is the Shewhart convention (about one false alarm per 370 points on a stable process).],
  [*Nelson rules*], [Besides points beyond the limits (rule 1), flag patterns inside them: 9 points in a row on one side of the centre (2), 6 steadily rising or falling (3), 14 alternating up and down (4), 2 of 3 beyond 2σ (5), 4 of 5 beyond 1σ (6), 15 in a row within 1σ (7), 8 in a row outside 1σ on both sides (8). The zone rules (5–8) are switched off on charts whose limits move with the sample size.],
  [*Exclude violations*], [Recompute the limits without the flagged points — the "trial limits, then revised limits" step. Use it only when the flagged points have a known cause that has been removed.],
  [*Draw*], [Centre line, limits, zones, and violations as separate markers.],
)

The report lists each signalling point and the rule it broke. A chart with no signals says the process is *stable* — which is not the same as meeting a specification.

#use[Manufacturing and laboratory quality control, monitoring any process over time.]

=== Transform — rescale or reshape a distribution
Replaces y with a transformed version of itself, point for point (same x, same number of points), on a new axis.

#defs(
  [*Power transform*], [Makes a skewed distribution closer to normal. *Yeo-Johnson* accepts any values; *Box-Cox* often fits better but needs strictly positive values. The fitted λ is reported.],
  [*Quantile transform*], [Maps values through their ranks onto a *uniform* or *normal* distribution. Very strong: any shape becomes the target shape.],
  [*Standard scaler*], [z-scores: subtract the mean, divide by the standard deviation.],
  [*Robust scaler*], [Subtract the median, divide by the IQR — unaffected by outliers. *With centering* and *with scaling* switch the two steps.],
)

#use[Preparing data for a method that assumes normality or comparable scales — a test, a control chart, a clustering on several variables.]

=== Decomposition — several series together
The one operation that combines *several* series into one calculation. Every ticked series is resampled onto a common grid of *n grid* points over their shared x range, giving a matrix with one column per series.

#defs(
  [*PCA*], [Principal components: the directions of greatest shared variation, ranked by the share of variance each explains (reported).],
  [*Fast ICA*], [Independent components: separates signals that were mixed together (the classic "cocktail party" problem).],
  [*NMF*], [Non-negative factors: parts that add up to the series, for data that cannot be negative (spectra, concentrations).],
  [*t-SNE*, *Isomap*, *Locally Linear Embedding*], [Manifold learning: places each grid sample as a point in 2D so that similar samples are close, drawn as a scatter on a new axis. Exploratory — the axes of the map have no units. *Perplexity* (t-SNE) and *neighbours* (Isomap, LLE) set how local the structure is.],
)

*Components* sets how many to extract. PCA, ICA and NMF draw each component as a curve over the shared x.

#use[Finding the few patterns behind many correlated signals (several sensors, many spectra), separating mixed sources, exploring similarity.]

// ======================================================================
== Geometry <geometry>

=== Geometry — move a series' coordinates
Rotates, translates, scales, mirrors or shears a series. It answers "put this where that is": rotate a scan onto a reference frame, shift a profile, flip it, scale a model onto measured units.

#defs(
  [*Rotate*], [Turns the series counter-clockwise about a centre, by an angle in degrees.],
  [*Translate*], [Adds a fixed amount to every x and y (and z).],
  [*Roto-translation*], [A rotation about the centre followed by a translation: a rigid motion; shape and size never change.],
  [*Scale*], [Multiplies x and y (and z) by their own factors about the centre. Unequal factors change the shape.],
  [*Mirror*], [Reflects across a horizontal, vertical or diagonal line through the centre.],
  [*Shear*], [Leans the series over: so much x added per unit of y, or y per unit of x.],
)

*Centre on the data* (on by default) uses the middle of the series' own extent as the fixed point — "turn this shape" rather than "turn it about the origin". Turn it off to type the centre.

*In 3D.* A series with a z column is moved in space: three rotation angles (about x, then y, then z, each by the right-hand rule), and translation, scale and centre gain a z value. Mirror and Shear act in the x–y plane. The preview becomes a 3D view.

The Results pane shows a picture: the source in blue, the result in red and, with *Show the deformation grid*, a square mesh before and after, which makes a shear or an uneven scale easy to read.

#note[
  Geometry computes nothing and writes no table. The new series carries a SQL query — the source series' own query wrapped in the transform's arithmetic, with a 2D rotation written as `cos(radians(30))` so it stays readable. The moved series therefore *follows its source* (change the source and the transform applies to the new data), costs no extra space, and can be adjusted afterwards in the Query Builder.
]

// ======================================================================
== Signal processing

=== Smoothing — reduce noise
Produces a smoothed copy of a series, drawn beside the original. The *data type* selector chooses between models for curves (1D), surfaces (2D) and volumes (3D).

*Curves (1D):*
#defs(
  [*Moving Average*], [Mean over a sliding *window*, optionally centred. Simple; flattens peaks.],
  [*Savitzky-Golay*], [Local polynomial fit over a window. Smooths while preserving peak heights and widths better than a moving average. Can also return a derivative.],
  [*Gaussian Filter*], [Weighted average with a Gaussian kernel of width *sigma*.],
  [*Median Filter*], [Median over a window. Removes isolated spikes and keeps sharp steps.],
  [*Wiener Filter*], [Adaptive: smooths more where the local variance is close to the noise level.],
  [*Smoothing Spline*], [A spline of degree *k* with smoothing factor *s* (0 passes through every point).],
  [*LOWESS / LOESS*], [Locally weighted regression on a *fraction* of the data around each point, with robustness *iterations* that down-weight outliers.],
  [*Kalman Smoother*], [A state-space smoother; set the *process* and *measurement variances*. Good for tracking a slowly changing level in noisy readings.],
  [*FFT Low-Pass*], [Removes frequencies above a *cutoff ratio* of the spectrum.],
  [*Butterworth Filter*], [A classic low-, high- or band-pass filter of a given *order* and *cutoff*, applied forward and backward (no phase shift).],
  [*Wavelet Denoising*], [Thresholds the wavelet coefficients (*wavelet*, *level*, *threshold mode* and *factor*). Keeps sharp features that linear filters blur.],
  [*Whittaker-Eilers*], [Penalized least squares with smoothness *lambda* and difference *order*. Fast, handles any length.],
  [*Hodrick-Prescott*], [Separates a smooth trend from cycles (*lambda*: 1600 is the convention for quarterly data).],
  [*Total Variation Denoising*], [Removes noise while keeping steps sharp (*weight*). Ideal for piecewise-constant signals.],
)

*Surfaces (2D) and volumes (3D):* Gaussian, Median, smoothing and rectangular bivariate splines, RBF (radial basis function) smoothing, FFT low-pass and Total Variation, each with per-axis sizes where it applies.

#use[Making a trend visible in noisy data, preparing a signal for peak finding or differentiation. When in doubt, start with Savitzky-Golay or LOWESS.]

=== Spectral Analysis — analyse frequencies
Computes how a signal's energy is distributed over frequency, or how it correlates with itself or another signal over lag. The results are drawn on a new axis.

#defs(
  [*Power spectral density (Welch)*], [The standard estimate of power per frequency for a noisy signal: averages the spectra of overlapping segments.],
  [*Cross spectral density (Welch)*], [For a pair of signals: where they share power, and with what phase.],
  [*Coherence*], [The normalized cross spectrum: how linearly related two signals are at each frequency, from 0 to 1.],
  [*Magnitude / Phase (unwrapped) / Angle (wrapped) spectrum*], [The raw one-sided FFT, for deterministic signals where Welch's averaging would smear sharp peaks.],
  [*Laplace transform (damped FFT)*], [The FFT of the signal weighted by e#super[−σt], for decaying or growing signals. *Damping σ* = 0 reduces it to the Fourier transform.],
  [*Wavelet power spectrum (Morlet)*], [Frequency content *over time* — how the spectrum changes along the signal. *Morlet w0* trades frequency resolution for time resolution; *scales* sets how many frequencies are evaluated.],
  [*Autocorrelation*], [How similar the signal is to itself shifted by each lag — reveals periodicity.],
  [*Cross-correlation*], [The same between two signals — reveals which one leads and by how much.],
)

The *sampling frequency* is derived from the spacing of x (with a warning if the spacing is not uniform) or typed in. Welch estimators take a *segment length*, *overlap*, *window* and *detrend*; spectra can be *one-sided* and in *decibels*; correlations take *max lags* and a *normalisation* (unbiased, biased, none). Two-signal estimators pair the *first* ticked series with each of the others.

#use[Vibration analysis, finding periodicities (seasonal cycles, mains hum), comparing two sensors, checking what a filter should remove.]

=== Filtering — filter, detrend or demodulate
Changes the signal itself in the frequency domain — the complement of Spectral Analysis, which only measures it.

#defs(
  [*IIR filter*], [Family *Butterworth* (maximally flat), *Chebyshev I/II* or *Elliptic* (sharper roll-off with ripple: set *ripple* and *attenuation* in dB), or *Bessel* (best preserves wave shape). Response *lowpass*, *highpass*, *bandpass* or *bandstop*, with an *order* and one or two *cutoffs*. Applied forward and backward, so there is no phase shift.],
  [*FIR filter*], [Linear-phase filter with a number of *taps* and a *window*. No ringing on steps.],
  [*Analytic signal*], [Via the Hilbert transform: the *amplitude envelope*, *instantaneous phase* or *instantaneous frequency*. Used for AM demodulation and vibration envelopes.],
  [*Detrend*], [Removes a *linear* trend or a *constant* (the mean), on its own or before filtering.],
)

The cutoffs are in the same units as the sampling frequency, which is taken from x as in Spectral Analysis.

#use[Removing mains hum (bandstop at 50/60 Hz), separating slow drift from fast signal, extracting an envelope.]

=== Baseline Correction — subtract a background
Estimates the slowly varying background under a spectrum or chromatogram and subtracts it. Peak heights and especially peak areas are wrong until this is done.

#defs(
  [*Asymmetric Least Squares (AsLS)*], [A smooth curve fitted to stay below the data: points above it cost little, points below cost a lot. *Lambda* sets its stiffness (larger bends less into narrow peaks), *p* its asymmetry (smaller pushes it further below; keep it well under 0.5), *iterations* the re-weighting passes. Copes with curved and overlapping baselines.],
  [*Rubber band*], [The lower convex hull of the data — a band stretched under it. No parameters, but it cannot follow a baseline that dips inward.],
)

*Draw the baseline* adds the estimated background as its own series, so the correction can be checked by eye.

#use[Raman, IR, XRD and fluorescence spectra; chromatograms with drift — always before measuring peak areas.]

// ======================================================================
== Modeling

=== Fit — fit a model to data
Fits a mathematical model y = f(x; parameters) to a series by least squares and adds the fitted curve beside the data. It is the most complete operation in ChartLibre.

*The model.* The model tree lists the function library by family, with a search box:

#defs(
  [*Basic functions*], [Constant, linear, polynomials up to degree 10, reciprocal, logarithmic, power law, rational.],
  [*Growth and saturation*], [Exponential growth and decay, double exponential, saturation, Michaelis–Menten, Hill.],
  [*Sigmoidal curves*], [4- and 5-parameter logistic, Richards, Gompertz, error-function, tanh and arctan sigmoids, double-logistic pulse.],
  [*Peak functions*], [Gaussian, Lorentzian, pseudo-Voigt, Voigt, asymmetric Gaussian, Pearson VII.],
  [*Periodic functions*], [Sine, cosine, damped sine, linear chirp.],
  [*Statistical distributions*], [Normal, lognormal, Weibull, gamma, beta and Cauchy densities and distribution functions.],
  [*Reliability models*], [Exponential and Weibull survival, Weibull hazard.],
  [*Semiconductor and process*], [Arrhenius, inverse-temperature linear, dose-response power, Poisson yield, process window.],
  [*Adsorption and surface*], [Langmuir, Freundlich, Temkin isotherms.],
  [*NIST reference models*], [The models of the NIST StRD nonlinear regression benchmarks.],
  [*Surfaces*], [For z = f(x, y): plane, paraboloid, 2D Gaussian, saddle, ripple.],
  [*User functions / User surfaces*], [Your own, written with the Function Creator (see @advanced).],
)

The *Multi-peak builder* creates a sum of *N* Gaussian, Lorentzian or pseudo-Voigt peaks plus an offset, optionally with one *shared width* (and shared η) for all peaks — for overlapping peaks in a spectrum.

*The parameters.* The table lists each parameter with its *initial value*, *lower and upper bounds* and a *fixed* flag. *Estimate* fills the initial values from the data — with the function's own estimator when it has one, or a random search of the parameter space when it does not. A good starting point matters: a least-squares fit started in the wrong place converges confidently to the wrong answer.

*The algorithm.*
#defs(
  [*Trust region* (default)], [Bounded least squares. Right for most fits.],
  [*Dogleg box*], [Bounded least squares, often better when a parameter ends up on a bound.],
  [*Levenberg-Marquardt*], [Classic unbounded least squares; bounds are ignored.],
  [*Nelder-Mead*, *Powell*], [Derivative-free local methods, for rough or noisy objective functions.],
  [*Differential evolution*, *Dual annealing*, *Basin hopping*, *SHGO*, *Monte Carlo*], [Global methods: search the whole box defined by the bounds, then finish with a local fit. Use them when the result depends on the starting point.],
)
*Loss* sets how large residuals are weighted: plain least squares trusts every point; the robust losses (soft L1, Huber, Cauchy, arctan) limit how much one bad point can pull the fit. *Max evaluations* caps the work.

*The report* gives the fitted expression, each parameter with its standard error, t value, p value and 95% confidence interval, the parameter correlation matrix, and goodness of fit (R², RMSE, residual sum of squares, reduced chi-squared, AIC and BIC). Parameters can be copied, saved as CSV, or reused as the starting point of the next fit.

*Extra output:* *95% confidence band* (where the true curve lies, given the parameter uncertainty), *95% prediction band* (where a new measurement would fall), a *residuals vs x* chart (structure in the residuals means the model is missing something) and a *measured vs fit* chart (a perfect fit lies on the diagonal). Fitted values and residuals are saved to the *output table*.

#use[Extracting physical parameters (a rate constant, a peak position, an EC50), calibration curves, comparing candidate models. ChartLibre's fit has been checked against the certified NIST StRD reference results.]

=== Interpolation — fill gaps and resample
Evaluates a smooth or fitted curve through a series at new x values: to fill gaps, to resample onto a regular grid, or to extend a little beyond the data.

#defs(
  [*NumPy interp*, *Linear*], [Straight lines between points.],
  [*SciPy CubicSpline*], [A smooth cubic through every point, with a choice of *boundary condition*.],
  [*SciPy PCHIP*], [Monotone cubic: never overshoots between points — the safe choice for data that must not wiggle.],
  [*SciPy Akima*], [A cubic that is less affected by outliers than a natural spline.],
  [*SciPy spline family*], [UnivariateSpline / B-spline of degree *k* with smoothing *s*.],
  [*Polynomial*], [A least-squares polynomial of a given *degree*.],
  [*Exponential*, *Logarithmic*, *Power*, *Gaussian*, *Sigmoid*], [Simple fitted models, with optional starting parameters.],
)

*X spacing* chooses where the new values are computed: a number of *points* spaced linearly, logarithmically, geometrically or at Chebyshev nodes over the data range (optionally *extended* by a percentage on both sides, or over an explicit *X range*); a fixed integer *step*; the *original data x* values; or a custom list of x values. *Allow extrapolation* permits values outside the data's x range.

#use[Putting two series on the same x grid before comparing or subtracting them; filling missing readings; producing a smooth curve for display.]

=== Function — plot a function
Evaluates a function from the same library as Fit over a range and draws it, with no data and nothing fitted. Choose the function, type its parameter values, the *start* and *stop* of the range, the number of *points* and the *spacing* (linear, or logarithmic for log axes). For a surface function, the range is a grid in x and y.

#use[Seeing the shape of a model before choosing it, drawing a theoretical or reference curve beside measurements, checking starting values before a fit, generating synthetic test data.]

=== Regression — robust and machine-learning regression
Fits a curve through a series with models that do not assume an algebraic formula, and evaluates it on a dense grid over the data's x range.

#defs(
  [*RANSAC*], [Robust straight line: fits random subsets (*max trials*) and keeps the one most points agree with. Ignores gross outliers completely.],
  [*Huber*], [Robust straight line that down-weights large residuals; *epsilon* near 1 is more robust, larger behaves like ordinary least squares; *alpha* regularizes.],
  [*Isotonic*], [The best-fitting curve that only rises (or only falls, or *auto*). No shape assumed beyond monotonicity.],
  [*Random Forest*], [Average of many regression trees (*estimators*, *max depth*). Flexible, step-like.],
  [*Gradient Boosting*], [Trees added one after another, each correcting the last (*learning rate*). Flexible and often accurate.],
)

#use[A trend through outlier-heavy data (RANSAC, Huber), a calibration that must be monotone (Isotonic), a flexible non-parametric trend when no formula is known.]

=== GP Regression — a fit with its own uncertainty
Gaussian process regression: a smooth curve through the data *and* a ±2σ band saying how certain the curve is at each x — narrow where the data are dense, wide in gaps and beyond the ends. The result is three series: the mean and the two band edges.

#defs(
  [*RBF*], [Very smooth curves.],
  [*Matern (ν = 1.5)* / *(ν = 2.5)*], [Rougher curves (once / twice differentiable) — often more realistic for physical data.],
  [*Rational Quadratic*], [A mix of length scales — for data with both slow and fast variation.],
)

*Length scale* (how quickly the curve may vary with x) and *noise level* are starting guesses; both are optimized during the fit, and the fitted kernel and log marginal likelihood are reported.

#use[Interpolating sparse or expensive measurements with an honest uncertainty; deciding where the next measurement would be most informative.]
