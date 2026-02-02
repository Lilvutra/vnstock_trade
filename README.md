# vnstock_trade

A modular Python package for Vietnamese stock market analysis and ML-driven trading strategies.

---

## Overview

`vnstock_trade` implements a **sequential machine learning pipeline** for stock market analysis and algorithmic trading in Vietnam.

The pipeline is designed with data flowing **unidirectionally** through the following stages:

**Fetch → Features → Labels → Training → Strategy**


---

##  Key Features




---

##  Quick Start

### Prerequisites 

- macOS(Apple Silicon or Intel) or Linux
- Python 3.10+

### Installation




---

##  Package Structure 

```
vnstock_trade/
├── src/vnstock/
│   ├── fetch/          # Data acquisition from VCI API
│   ├── features/       # Technical indicator generation
│   ├── labels/         # Target variable creation
│   ├── training/       # Model training (RandomForest)
│   ├── strategy/       # Trading strategy logic
│   └── utils/          # Shared helper functions
│
├── scripts/
│   └── run_pipeline.py # Pipeline orchestration
│
└── data/               # Generated datasets (CSV outputs)

```
