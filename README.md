# iSyncTab (CVPR 2026 Submission)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" />
  <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C.svg" />
  <img src="https://img.shields.io/badge/modality-tabular+image-purple.svg" />
  <img src="https://img.shields.io/badge/status-double--blind%20review-orange.svg" />
  <img src="https://img.shields.io/badge/venue-CVPR%202026-black.svg" />
  <img src="https://img.shields.io/badge/Anonymous_Submission-green.svg" />
</p>

> **Note (anonymized partial release).**  
> This repository contains an anonymized, **partial artifact** for the CVPR 2026 submission.  
> It includes the **NS-PFS-based iSyncTab variants only**.  
> The **MBFS iSyncTab variant described in the paper is *not* included** in this code release.

---

## 1. Overview

**iSyncTab** is a multimodal architecture for problems where each example has:

- **Tabular metadata** (numeric + categorical + optional text-like fields), and  
- **Image data** (e.g., medical images, natural images).

The key idea is to treat **both tabular features and image features as tokens**, then use **Neural Synchrony-guided Paired Feature Sequencing (NS-PFS)** to learn a global permutation over all tokens before feeding them to a transformer-like encoder (linformer).

This repository provides:

- `iSyncTab` - a **simple NS-PFS variant** (default).
- `iSyncTabRefined` - a **refined NS-PFS variant** with more structured synchrony and sequencing.
- A **demo training script** for one concrete dataset (HAM10000) using Optuna, mainly to:
  - Demonstrate how to plug iSyncTab into a generic PyTorch pipeline.

iSyncTab itself is **not specific** to HAM10000: any dataset with tabular + image inputs can be used by providing a matching PyTorch `Dataset` / `DataLoader`.

---

## 2. Repository Structure

A typical layout is:

```text
.
├── iSyncTab/
│   ├── __init__.py
│   ├── iSyncTab.py              # simple NS-PFS variant (default)
│   └── iSyncTab_refined.py      # refined NS-PFS variant
├── iSyncTab Demo.ipynb  # example demo on HAM10000
├── requirements.txt
└── README.md

