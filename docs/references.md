# Supporting References

This document lists references used to justify the synthetic-data generation,
augmentation, and image-degradation/noise assumptions used in the Drift-Sense
project.

## Reference 1 — Synthetic SEM Data for Semiconductor Inspection

**Title:** Defect detection in photolithographic patterns using deep learning models trained on synthetic data

**Authors:** [See the journal article for the complete author list]

**Year:** 2025

**Journal:** Heliyon, Volume 11, Issue 10

**DOI:** https://doi.org/10.1016/j.heliyon.2025.e43377

**URL:** https://www.sciencedirect.com/science/article/pii/S240584402501761X

**How this reference informed the project:**

The work demonstrates the use of artificially generated SEM images of
semiconductor line patterns with known defect distributions and automatic
annotations. This supports the use of synthetic semiconductor imagery when
large quantities of annotated experimental data are unavailable.

**Augmentation / synthetic data:**
- Synthetic semiconductor/SEM images can be used to construct annotated
  training datasets.
- Controlled variation of pattern characteristics and defects can increase
  dataset diversity.

**Noise / image degradation:**
- The reference motivates the use of realistic synthetic variations rather
  than relying exclusively on a small set of experimental images.

**Relevance to Drift-Sense:**
The Drift-Sense dataset generator follows the same general principle:
generate semiconductor-pattern image pairs synthetically while retaining
exact ground-truth coordinates.

---

## Reference 2 — Geometric Augmentation for Semiconductor Inspection

**Title:** Geometric transformation-based data augmentation on defect classification of segmented images of semiconductor materials using a ResNet50 convolutional neural network

**Year:** 2022

**Journal:** Applied Soft Computing

**DOI:** https://doi.org/10.1016/j.asoc.2022.108473

**URL:** https://www.sciencedirect.com/science/article/abs/pii/S0957417422010120

**How this reference informed the project:**

This study evaluates geometric data augmentation for semiconductor-defect
classification and reports that appropriate augmentation improves CNN
performance, particularly when the original dataset is imbalanced.

**Augmentation:**
- Geometric transformations are useful for increasing the diversity of
  semiconductor inspection images.
- Augmentation can help CNN models generalize across variations in the
  appearance and orientation of patterns.

**Noise / image degradation:**
- Although the main focus is geometric augmentation rather than SEM noise,
  the work supports introducing controlled visual variation during training
  instead of relying exclusively on a fixed image set.

**Relevance to Drift-Sense:**
The Drift-Sense synthetic dataset uses controlled transformations and
appearance variation to expose the CNN verification stage to different
instances of semiconductor patterns.

---

## Reference 3 — SEM Noise and Charging-Aware Imaging

**Title:** Deep learning denoising enables rapid SEM imaging under charging conditions for FE SEM, CD SEM, and review SEM

**Authors:** Hyungjoo Park, Beom-Seok Oh, Kuk Jin Jang

**Year:** 2025/2026

**Journal:** Scientific Reports, Volume 16, Article 3342

**DOI:** https://doi.org/10.1038/s41598-025-33273-3

**URL:** https://www.nature.com/articles/s41598-025-33273-3

**How this reference informed the project:**

This work studies SEM imaging under rapid-acquisition and charging-sensitive
conditions. It discusses the relationship between reduced acquisition time,
lower signal-to-noise ratio, and SEM image degradation.

**Augmentation:**
- SEM training and evaluation data should account for acquisition-condition
  variability when robustness to real imaging conditions is important.

**Noise modeling:**
- Rapid SEM acquisition can produce noisier images because of reduced
  electron signal.
- Charging can produce image distortion, local wash-out/saturation, and
  edge-related artifacts.
- Image quality can vary with acquisition conditions such as beam alignment,
  focus, accelerating voltage, and other instrument settings.

**Image degradation assumptions:**
- The reference supports treating SEM image quality as dependent on realistic
  acquisition conditions rather than assuming an ideal noise-free image.
- It also emphasizes preserving geometric/structural fidelity when evaluating
  SEM image processing systems.

**Relevance to Drift-Sense:**
These observations motivate controlled image-quality and degradation
variation in the synthetic semiconductor image-generation pipeline and
support evaluating localization using images that are not assumed to be
perfectly noise-free.

---

## Summary of How the References Were Used

| Project Component | Supporting Reference |
|---|---|
| Synthetic semiconductor image generation | Reference 1 |
| Semiconductor image augmentation | Reference 2 |
| Geometric variation / augmentation | Reference 2 |
| SEM noise considerations | Reference 3 |
| Charging / acquisition-related degradation | Reference 3 |
| Need for realistic synthetic training data | References 1 and 3 |

## Citation Note

These references are provided as supporting technical literature for the
synthetic-data, augmentation, and image-degradation assumptions used in
Drift-Sense.

The references should also be cited consistently in the project
presentation/PPT where these design choices are discussed.
