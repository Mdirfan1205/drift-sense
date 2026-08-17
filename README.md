\# Drift-Sense



AI-Assisted Multi-Scale Localization for Synthetic Semiconductor Pattern Images



\## 1. Overview



Drift-Sense localizes a reference semiconductor-pattern image inside a larger search image.



The system combines:



\- Multi-scale OpenCV template matching

\- Candidate generation and non-maximum suppression (NMS)

\- CNN-based candidate verification

\- Hybrid OpenCV + AI candidate selection

\- Targeted candidate-selection logic for difficult localization cases



The final localization output is the predicted center coordinate:



```text

(x, y)

