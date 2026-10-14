# # 🩺 AI Skin Disease Diagnosis Support (Educational Prototype)

A machine learning project in Python that takes visible skin manifestations and clinical history, then produces a diagnosis-support report with ranked likely conditions, supporting findings, red flags and the usual next clinical step.

> ⚠️ **Not for medical or clinical use.** This is a learning project trained on simulated data. It cannot diagnose anything and cannot replace a dermatologist.

![Results](skin_diagnosis_linkedin.png)

## What it does
- Creates a simulated dataset of 1,800 patients across 6 conditions: psoriasis, eczema, seborrheic dermatitis, lichen planus, pityriasis rosea and ringworm
- Uses **11 manifestations** (redness, scaling, itching, ring-shaped patches, plaques and more, rated 0 to 3) plus **6 clinical history features** (age, duration, family history, allergy history, recent viral illness, animal or soil contact)
- Trains a **Random Forest** classifier (94.7% accuracy on unseen *simulated* patients)
- Generates a **diagnosis-support report** for each case:
  - Differential diagnosis with confidence
  - Findings that support the leading diagnosis
  - Low-confidence warning when conditions look alike
  - Usual confirmatory step (for example, KOH test for ringworm)
  - **Red flags** (fever with rash, rapid spread, heavy oozing or crusting, new or changing dark spot)
- Lets you type in your own case

## How to run
```bash
pip3 install -r requirements.txt
python3 skin_disease_diagnosis.py
```

Outputs: `diagnosis_report.txt`, `skin_diagnosis_results.png`, `skin_clinical_sample.csv`

## Limitations
- The data is **simulated** from general textbook descriptions, not real patients, so the accuracy does not reflect real-world performance
- Does not assess skin cancer or other conditions outside the 6 listed
- Real diagnosis needs an in-person exam and sometimes tests such as biopsy
- Future work: train on real clinical data and images (for example the HAM10000 dataset) with a CNN

**Tech:** Python · pandas · NumPy · scikit-learn · matplotlib
