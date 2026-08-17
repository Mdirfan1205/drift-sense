# Supporting References

These references support the synthetic-data generation, image augmentation, and SEM image-quality/noise considerations used in the Drift-Sense project.

## 1. Synthetic SEM Data for Semiconductor Inspection

**Title:** Defect detection in photolithographic patterns using deep learning models trained on synthetic data

**Authors:** See the complete author list in the published article.

**Year:** 2025

**Journal:** Heliyon, Volume 11, Issue 10, Article e43377

**DOI:** https://doi.org/10.1016/j.heliyon.2025.e43377

**URL:** https://www.sciencedirect.com/science/article/pii/S240584402501761X

### Relevance to Drift-Sense

This work demonstrates the generation of synthetic scanning electron microscopy (SEM) images of semiconductor line patterns with known defect distributions and automatic annotations. It supports the use of synthetic semiconductor images when obtaining large quantities of manually annotated experimental data is difficult.

### Design choice supported

* Synthetic semiconductor/SEM image generation
* Automatic ground-truth annotation
* Controlled variation of semiconductor pattern appearance
* Development of deep-learning inspection systems using synthetic training data

The Drift-Sense dataset generator follows the same general principle by creating synthetic semiconductor image pairs while recording exact ground-truth coordinates.

---

## 2. Geometric Data Augmentation for Semiconductor Inspection

**Title:** Geometric transformation-based data augmentation on defect classification of segmented images of semiconductor materials using a ResNet50 convolutional neural network

**Authors:** Francisco López de la Rosa, José L. Gómez-Sirvent, Roberto Sánchez-Reolid, Rafael Morales, Antonio Fernández-Caballero

**Year:** 2022

**Journal:** Expert Systems with Applications, Volume 206, Article 117731

**DOI:** https://doi.org/10.1016/j.eswa.2022.117731

**URL:** https://www.sciencedirect.com/science/article/pii/S0957417422010120

### Relevance to Drift-Sense

This study evaluates geometric data-augmentation techniques for semiconductor-defect images and examines their effect on CNN performance. The authors report that appropriate synthetic augmentation can improve performance, particularly when datasets are limited or imbalanced.

### Design choice supported

* Geometric image augmentation
* Increasing training-data diversity
* Controlled transformations of semiconductor inspection images
* Improving CNN robustness through additional synthetic examples

This supports the use of controlled image variation in the Drift-Sense synthetic-data and CNN-verification pipeline.

---

## 3. SEM Noise, Signal-to-Noise Ratio, and Charging

**Title:** Deep learning denoising enables rapid SEM imaging under charging conditions for FE SEM, CD SEM, and review SEM

**Authors:** Hyungjoo Park, Beom-Seok Oh, Kuk Jin Jang

**Year:** 2026

**Journal:** Scientific Reports, Volume 16, Article 3342

**DOI:** https://doi.org/10.1038/s41598-025-33273-3

**URL:** https://www.nature.com/articles/s41598-025-33273-3

### Relevance to Drift-Sense

This study investigates SEM imaging under rapid acquisition and charging-sensitive conditions. It describes how shorter acquisition can reduce electron signal and therefore lower signal-to-noise ratio, while longer exposure can increase charging-related image distortion.

### Design choice supported

* SEM image-quality variability
* Noise and signal-to-noise considerations
* Acquisition-related image degradation
* Charging-related distortion
* Importance of robustness to SEM imaging conditions

These observations support considering realistic image-quality variation when developing synthetic semiconductor inspection data and evaluating image-localization systems.

---

## Reference-to-Project Mapping

| Drift-Sense component                       | Supporting reference |
| ------------------------------------------- | -------------------- |
| Synthetic semiconductor image generation    | Reference 1          |
| Ground-truth annotation of synthetic images | Reference 1          |
| Image augmentation / controlled variation   | Reference 2          |
| CNN training-data diversity                 | Reference 2          |
| SEM noise and SNR considerations            | Reference 3          |
| SEM acquisition-related degradation         | Reference 3          |
| Charging-related image distortion           | Reference 3          |

## Citation Consistency

These references are provided as technical support for the synthetic-data, augmentation, and SEM image-quality considerations used in Drift-Sense.

The presentation should cite the same references wherever these design choices are discussed.
