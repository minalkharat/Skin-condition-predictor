"""
Skin Disease Diagnosis Support (educational prototype)
------------------------------------------------------
Takes skin manifestations + clinical history, then produces a diagnosis-
support report: ranked likely conditions, supporting findings, red flags,
and the confirmatory tests a doctor would usually consider.

No download needed: it trains on a simulated dataset built from general
textbook descriptions. It is NOT a medical device and NOT a real diagnosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. FEATURES
# ---------------------------------------------------------------
MANIFESTATIONS = [  # rated 0 (absent) to 3 (severe)
    "redness", "scaling", "itching", "sharp_borders", "thick_plaques",
    "oozing_crusting", "ring_shaped", "scalp_involvement",
    "knee_elbow_involvement", "flat_purple_bumps", "oily_areas",
]
CLINICAL = [  # patient history
    "age", "duration_weeks", "family_history", "allergy_history",
    "recent_viral_illness", "animal_or_soil_contact",
]
FEATURES = MANIFESTATIONS + CLINICAL

# Typical manifestation severity per condition (general, simplified)
PROFILES = {
    "Psoriasis": dict(redness=2.2, scaling=2.7, itching=1.2, sharp_borders=2.6,
                      thick_plaques=2.5, scalp_involvement=1.8, knee_elbow_involvement=2.3),
    "Eczema (atopic dermatitis)": dict(redness=2.0, scaling=1.5, itching=2.8,
                                       oozing_crusting=1.8, sharp_borders=0.6),
    "Seborrheic dermatitis": dict(redness=1.6, scaling=2.0, itching=1.3,
                                  scalp_involvement=2.3, oily_areas=2.5),
    "Lichen planus": dict(redness=1.4, itching=2.5, sharp_borders=1.8,
                          flat_purple_bumps=2.8),
    "Pityriasis rosea": dict(redness=1.8, scaling=1.2, itching=1.5,
                             ring_shaped=1.8, sharp_borders=1.3),
    "Ringworm (tinea)": dict(redness=1.8, scaling=1.8, itching=2.0,
                             ring_shaped=2.7, sharp_borders=2.2),
}

# age mean, age sd, duration mean (weeks), P(family hx), P(allergy hx),
# P(recent viral illness), P(animal/soil contact)
CLINICAL_PROFILES = {
    "Psoriasis": (40, 15, 30, 0.50, 0.15, 0.10, 0.05),
    "Eczema (atopic dermatitis)": (25, 20, 40, 0.30, 0.70, 0.10, 0.10),
    "Seborrheic dermatitis": (45, 20, 20, 0.15, 0.10, 0.05, 0.05),
    "Lichen planus": (50, 12, 20, 0.10, 0.10, 0.10, 0.05),
    "Pityriasis rosea": (25, 10, 6, 0.05, 0.10, 0.60, 0.05),
    "Ringworm (tinea)": (30, 18, 8, 0.05, 0.10, 0.05, 0.60),
}

# What a doctor typically considers next (general information only)
NEXT_STEPS = {
    "Psoriasis": "Exam of scalp, nails and joints; skin biopsy if the picture is unclear.",
    "Eczema (atopic dermatitis)": "History and exam; patch testing if a contact allergen is suspected.",
    "Seborrheic dermatitis": "Clinical exam; rule out psoriasis and fungal infection.",
    "Lichen planus": "Skin biopsy; check mouth and nails; further blood tests may be considered.",
    "Pityriasis rosea": "Look for an initial 'herald patch'; tests to rule out lookalikes (e.g. KOH, syphilis blood test) when appropriate.",
    "Ringworm (tinea)": "KOH microscopy or fungal culture of skin scrapings.",
}

# ---------------------------------------------------------------
# 2. SIMULATED DATASET
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
rows = []
for disease, prof in PROFILES.items():
    age_m, age_sd, dur_m, p_fam, p_all, p_vir, p_con = CLINICAL_PROFILES[disease]
    for _ in range(300):
        row = {f: int(np.clip(round(rng.normal(prof.get(f, 0.2), 0.8)), 0, 3))
               for f in MANIFESTATIONS}
        row["age"] = int(np.clip(rng.normal(age_m, age_sd), 1, 90))
        row["duration_weeks"] = int(max(1, rng.normal(dur_m, dur_m * 0.4)))
        row["family_history"] = int(rng.random() < p_fam)
        row["allergy_history"] = int(rng.random() < p_all)
        row["recent_viral_illness"] = int(rng.random() < p_vir)
        row["animal_or_soil_contact"] = int(rng.random() < p_con)
        row["diagnosis"] = disease
        rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("skin_clinical_sample.csv", index=False)
print(f"Created sample dataset: {len(df)} simulated patients, {len(PROFILES)} conditions\n")

# ---------------------------------------------------------------
# 3. TRAIN + EVALUATE
# ---------------------------------------------------------------
X, y = df[FEATURES], df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy on unseen simulated patients: {accuracy_score(y_test, pred)*100:.1f}%\n")
print(classification_report(y_test, pred))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[0], cmap="Blues",
                                        xticks_rotation=40, colorbar=False)
axes[0].set_title("Predicted vs actual diagnosis")
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", ax=axes[1], color="seagreen")
axes[1].set_title("What drives the diagnosis?")
plt.tight_layout()
plt.savefig("skin_diagnosis_results.png", dpi=130)
print("Saved plots to skin_diagnosis_results.png\n")


# ---------------------------------------------------------------
# 4. DIAGNOSIS-SUPPORT REPORT
# ---------------------------------------------------------------
def check_red_flags(patient, flags):
    out = []
    if flags.get("fever"):
        out.append("Fever together with a rash: see a doctor promptly.")
    if flags.get("rapid_spread"):
        out.append("Rapidly spreading rash: get medical review soon.")
    if flags.get("changing_dark_spot"):
        out.append("New or changing dark spot: needs an in-person skin cancer check. "
                   "This tool does NOT assess skin cancer.")
    if patient.get("oozing_crusting", 0) >= 3:
        out.append("Heavy oozing or crusting can mean infection: get it checked.")
    if patient.get("redness", 0) >= 3 and patient.get("scaling", 0) >= 3:
        out.append("Intense redness with heavy scaling can need urgent care.")
    return out


def diagnose(patient, flags=None, name="Patient"):
    flags = flags or {}
    case = pd.DataFrame([{f: patient.get(f, 0) for f in FEATURES}])
    probs = model.predict_proba(case)[0]
    top = np.argsort(probs)[::-1][:3]

    L = ["=" * 56, f"DIAGNOSIS-SUPPORT REPORT: {name}", "=" * 56,
         f"Age {patient.get('age', '?')}, rash present for {patient.get('duration_weeks', '?')} week(s)", ""]

    L.append("Differential diagnosis (most likely first):")
    for rank, i in enumerate(top, 1):
        L.append(f"  {rank}. {model.classes_[i]:28s} {probs[i]*100:5.1f}%")

    best = model.classes_[top[0]]
    prof = PROFILES[best]
    supports = [f.replace("_", " ") for f in MANIFESTATIONS
                if patient.get(f, 0) >= 2 and prof.get(f, 0) >= 1.5]
    L += ["", f"Findings supporting {best}:",
          "  " + (", ".join(supports) if supports else "no strongly typical findings")]

    if probs[top[0]] < 0.60:
        L += ["", "Confidence is LOW: several conditions look alike, so an",
              "in-person exam is needed to tell them apart."]

    L += ["", "Usual next step for the leading diagnosis:", f"  {NEXT_STEPS[best]}"]

    flags_found = check_red_flags(patient, flags)
    L += ["", "RED FLAGS:"] + (["  - " + f for f in flags_found] if flags_found else ["  none detected"])
    L += ["", "NOTE: Educational prototype trained on simulated data.",
          "Not a medical device. It cannot replace a dermatologist."]

    report = "\n".join(L)
    print(report, "\n")
    return report


# ---------------------------------------------------------------
# 5. DEMO CASES
# ---------------------------------------------------------------
reports = []
reports.append(diagnose(
    dict(age=46, duration_weeks=30, redness=2, scaling=3, itching=1, sharp_borders=3,
         thick_plaques=2, scalp_involvement=2, knee_elbow_involvement=3, family_history=1),
    name="Case A"))
reports.append(diagnose(
    dict(age=28, duration_weeks=3, redness=2, scaling=2, itching=2, sharp_borders=2,
         ring_shaped=3, oozing_crusting=3, animal_or_soil_contact=1),
    flags=dict(fever=False), name="Case B"))
open("diagnosis_report.txt", "w").write("\n\n".join(reports))
print("Saved example reports to diagnosis_report.txt\n")


# ---------------------------------------------------------------
# 6. TRY YOUR OWN CASE (press Enter to skip)
# ---------------------------------------------------------------
def ask(label, lo, hi, default=0):
    val = input(f"  {label} ({lo}-{hi}): ").strip()
    return int(val) if val.isdigit() and lo <= int(val) <= hi else default


if input("Enter your own case? (y/n): ").strip().lower() == "y":
    own = {"age": ask("age", 1, 90, 30), "duration_weeks": ask("weeks present", 1, 520, 4)}
    for f in MANIFESTATIONS:
        own[f] = ask(f.replace("_", " ") + " severity", 0, 3)
    for f in CLINICAL[2:]:
        own[f] = ask(f.replace("_", " ") + " (0=no, 1=yes)", 0, 1)
    own_flags = {
        "fever": ask("fever (0=no, 1=yes)", 0, 1) == 1,
        "rapid_spread": ask("rapidly spreading (0=no, 1=yes)", 0, 1) == 1,
        "changing_dark_spot": ask("new/changing dark spot (0=no, 1=yes)", 0, 1) == 1,
    }
    diagnose(own, own_flags, name="Your case")
