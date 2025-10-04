import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance
import os

xgb = joblib.load("../models/xgb_koi.joblib")
print("Loaded model from ../models/xgb_koi.joblib")

plt.figure(figsize=(10, 6))
plot_importance(xgb, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances (by Gain)")
plt.tight_layout()

os.makedirs("../plots", exist_ok=True)
plt.savefig("../plots/feature_importance.png")
print("Saved plot to ../plots/feature_importance.png")

# === Feature Meaning Reference ===
# koi_fpflag_nt:  "Not Transit-Like" False Positive Flag  
#   → 1 if the detected signal does not resemble a planet transit (e.g., caused by noise or instrumental artifact).
#
# koi_fpflag_co:  "Centroid Offset" False Positive Flag  
#   → 1 if the source of the dimming is offset from the target star (indicating contamination by a nearby star).
#
# koi_score:  Disposition Score  
#   → Confidence score (0–1) reflecting the likelihood that the object is a true planet candidate based on model fitting.
#
# koi_fpflag_ss:  "Stellar Eclipse" False Positive Flag  
#   → 1 if the signal is likely caused by a binary star (an eclipsing binary) instead of a planet transit.
#
# koi_fpflag_ec:  "Ephemeris Match Indicates Contamination" False Positive Flag  
#   → 1 if the periodicity matches another known variable star or eclipsing binary (possible contamination).
#
# koi_model_snr:  Transit Signal-to-Noise Ratio  
#   → The ratio of the depth of the transit to the background noise — higher values mean a stronger, more reliable detection.
#
# dur_over_per:  Derived feature: Transit Duration / Orbital Period  
#   → Indicates how long the planet is in front of the star relative to its orbit — helps differentiate planet vs. false signals.
#
# koi_depth:  Transit Depth (ppm)  
#   → How much the star's brightness drops during the transit; deeper dips mean larger planets or stellar companions.
#
# depth_sqrt_dur:  Derived feature: sqrt(Transit Depth) * Transit Duration  
#   → Combines signal depth and duration; helps capture the geometric characteristics of a transit.
#
# koi_impact:  Impact Parameter  
#   → How centrally the planet crosses the star’s disk (0 = center, 1 = grazing). Extreme values can indicate partial or false transits.
