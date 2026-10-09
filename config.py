"""Central settings: allometry exponents, body-weight references, traffic-light thresholds."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MODELS_DIR = DATA_DIR / "src" / "models"

# --- V definition ---------------------------------------------------
# Reported volumes (Vz, Vss, Vc, ...) are all converted to L and geometric-averaged, same rule for human and monkey.
# Apparent volumes (/F) are excluded.
V_MIX = True

# --- Default SC absorption rate ---------------------------------------------
# Many SC drugs have no ka; without one nothing is absorbed and the whole profile stays 0.
# The default corresponds to an antibody SC absorption half-life of 2-4 days.
DEFAULT_KA_SC = 0.25    # 1/day

# --- New Drug tab simulation --------------------------------------------------
# Medians of published popPK models, validated against Top-5 antibody observed profiles.
NEWDRUG_F_SC = 0.71          # median SC antibody bioavailability (IQR 0.64–0.76)
NEWDRUG_VC_FRACTION = 0.58   # median V1 / (V1 + V2) of antibody 2-compartment models (unused: 1-compartment)
NEWDRUG_Q_LDAY = 0.50        # median inter-compartment clearance Q [L/day] of the same models (unused: 1-compartment)

# --- Body weight / BSA references ---
HUMAN_BW = 70.0     # kg
# Standard adult body surface area; converts mg/m^2 dosing to mg
HUMAN_BSA = 1.73    # m^2

# Hybrid allometry applies only when these columns are all measured.
# The base value is all the hybrid truly needs; other features are filled with training-fold medians.
HYBRID_CORE = ("log_allo",)
HUMAN_BSA = 1.73    # m^2
DEFAULT_MONKEY_BW = 3.0  # kg (when the file has no value)

# --- Physiologically plausible range (per kg); CL/V outside it are treated as unit errors and dropped ---
# Antibody V: plasma (0.04 L/kg) to interstitial (~0.2 L/kg), up to ~1 L/kg with target binding; 95% of normal values are below 0.88 L/kg.
# The 5 L/kg cap catches only clear errors (e.g. 16.9 L/kg, an L/kg vs mL/kg typo) and leaves normal values alone.
# The CL cap is generous enough to keep fast-clearing fragments (~8 L/day/kg).
# Out-of-range values are filtered per measurement unit to avoid contaminating the geometric mean.
V_MAX_L_PER_KG = 5.0
V_MIN_L_PER_KG = 0.005
CL_MAX_LDAY_PER_KG = 15.0
CL_MIN_LDAY_PER_KG = 1e-5

# --- Fixed allometry exponents (mAb, single species) ---
ALLO_EXP_CL = 0.85
# Slope of the log(body weight) ~ log(V) regression with human and monkey pooled.
# Must match the exponents the bundled models were trained with, otherwise hybrid base values drift.
ALLO_EXP_V = 1.092

# --- Traffic-light thresholds (fold-error) ---
# Cutoffs differ per target and are fixed, not training-set quantiles:
# quantiles would always mark the worst 20% red even as the model improves.
# 2-fold is a common PK tolerance; 3-fold is failure.
FLAG_CUT = {"CL": {"green": 2.0, "red": 3.0},
            "V":  {"green": 1.5, "red": 2.0}}
FLAG_GREEN = 1.5   # legacy name; still used by observed-vs-predicted badges
FLAG_RED = 3.0

# --- Model selection weights (CL and V selected independently) ---
# Score = W_CLOSENESS*GMFE + W_RMSE*RMSE  (lower is better).
#   GMFE = exp(mean|log(Pred/Obs)|) = geometric-mean fold-error vs obs (closeness)
#   RMSE = sqrt(mean(log-fold^2))            -> penalizes large outliers more
# W_CLOSENESS > W_RMSE so that Predicted:Observed closeness (GMFE) weighs more.
SEL_W_CLOSENESS = 0.85
SEL_W_RMSE = 0.15

# --- OOD (out-of-distribution) fallback ---
# ML is reliable only where training data is dense. If the target drug's molecular features are far from the training set
# (mean distance to its 3 nearest neighbors above the threshold), fall back to allometry instead of the selected ML.
# Settings were fixed by nested validation (feature set and threshold quantile chosen inside each fold).
# Re-validation after adding SCIV and monkey t1/2 features:
#   V  GMFE 1.553x -> 1.459x (Wilcoxon p=0.041), 12 of 35 switched
#   CL already uses allometry, so the fallback is a no-op
# Distance uses numeric features (SIM_NUM_FEATS) only: categorical features make the density metric
# a proxy for class labels and performed worse.
OOD_FALLBACK = True
OOD_K = 3               # number of nearest neighbors
# Quantile of the training distance distribution used as the threshold.
# 65 flags 35% of training drugs as OOD by definition, which made the warning noise;
# 95 flags only the most unusual 5% of training drugs.
OOD_PCT = 95
OOD_USE_CAT = False     # include categorical features in the distance

# --- Baseline shrinkage: pull the ML prediction slightly toward allometry ---
# Experimental. Set SHRINK_TO_ALLO=False to turn it off (no code change needed).
# Final = exp( w*log(ML prediction) + (1-w)*log(allometry) ), w = SHRINK_W.
# Separate from residual correction and OOD fallback: pulls the ML output toward a stable baseline to reduce large errors.
# Automatically a no-op for targets where allometry is selected (currently CL).
# Nested validation (V): pure ML 1.568x -> 70/30 shrinkage 1.502x; on top of OOD fallback 1.477x -> 1.454x
# (the extra gain is not significant, p=0.38: a reasonable hedge, not a sure win).
SHRINK_TO_ALLO = False  # tried and reverted (extra gain over OOD fallback not significant, p=0.38)
SHRINK_W = 0.7          # ML weight (0.7 = 70% ML + 30% allometry); 1.0 means no shrinkage

# --- Inter-individual variability (IIV) for uncertainty bands ---
# The band shows how widely CL/V scatter between individuals given the same drug (inter-individual variability),
# not how accurate the prediction is.
# Aggregated over 43 drugs' virtual populations: median CL CV 32% (log-SD 0.31), V CV 25% (log-SD 0.25).
# Individual CL/V are sampled log-normally to draw the band.
IIV_CL_SD = 0.31        # log-normal SD (≈ CV 32%)
IIV_V_SD = 0.25         # log-normal SD (≈ CV 25%)
IIV_BAND_LO = 5         # band lower percentile
IIV_BAND_HI = 95        # band upper percentile
IIV_N = 120             # Monte Carlo virtual subjects

# Adult body-weight distribution for the band around the literature model (Model parameters, orange dashed line).
# Drugs with a body-weight covariate draw virtual weights from it and scale CL/V/Q
# (pksim.covariate_scaled_params); otherwise IIV_CL_SD/IIV_V_SD above are used.
POP_WT_REF = 70.0       # reference weight (kg), denominator of the covariate exponent
POP_WT_CV = 0.18        # adult weight variability (log-normal CV, rough approximation)

# --- Simulation ---
SIM_DAYS_DEFAULT = 84   # 12 weeks
