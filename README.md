# UniCast: A Universal Model for Multi-Lead-Time and Modality-Specific Predictions
## 📌 Overview

This is the official PyTorch implementation of **UniCast: A Universal Model for Multi-Lead-Time and Modality-Specific Predictions**.

**Key highlights of UniCast:**

- 🌦️ **Semi-autoregressive Strategy**: UniCast employs a semi-autoregressive forecasting strategy that effectively reconciles the conflicting temporal demands between precipitation and upper-air variables.
- 🎯 **Specialized Training Paradigm**: UniCast tailors training tasks to specific challenges—training a **[generalist](https://github.com/Yumenomae/UniCast-A-Universal-Model-for-Multi-Lead-Time-and-Modality-Specific-Predictions/tree/main/Generalist)** for upper-air variables alongside a **[specialist](https://github.com/Yumenomae/UniCast-A-Universal-Model-for-Multi-Lead-Time-and-Modality-Specific-Predictions/tree/main/Specialist)** for precipitation-specific predictions.

---

## 🙏 Acknowledgements & Code Base

This repository is built upon the following open-source projects:

- **[Stormer](https://github.com/tung-nd/Stormer)** by [Tung Nguyen](https://github.com/tung-nd)
- **[Earth2Studio](https://github.com/NVIDIA/earth2studio)** by NVIDIA
- **[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)** by [ashleve](https://github.com/ashleve)

We sincerely thank the authors for making their code publicly available.

---

## ⚠️ Data Availability & Code Restrictions

Due to strict data privacy policies and licensing agreements with the **China Meteorological Administration (CMA)**, the original datasets used in this study **cannot** be publicly released.

Specifically, the following proprietary data are **NOT** included in this repository:

- **High-resolution atmospheric reanalysis**: CMA-3DVar and CMA-GD regional reanalysis.
- **Precipitation observations**: China Meteorological Administration Land Data Assimilation System (CMPA).

To protect the data privacy, we only provide the core model architecture, training logic, and inference pipelines as a reference implementation. 

> 💡 **For researchers interested in reproducing our work:** You will need to obtain the equivalent regional reanalysis and gauge-based precipitation datasets from your local meteorological agencies or publicly available alternatives (e.g., ERA5, IMERG) and adapt the data loader accordingly.

---

## 📝 Citation

**Please Note:** 
This repository is currently under active development (🚧 Work in Progress), and the accompanying paper is under review (Submitted). 
An official BibTeX citation will be posted here upon publication.

In the meantime, if you find our code useful for your research, we recommend citing this repository directly as:

> Zheng et al. (2026). UniCast: A Universal Model for Multi-Lead-Time and Modality-Specific Predictions (Version v1.0) [Source code]. GitHub. https://github.com/Yumenomae/UniCast

We will update this section with the official journal citation once available. Stay tuned!
