# Bayesian Marketing Mix Modeling Project

This repository contains a **Bayesian marketing mix model (MMM)** built for
two e‑commerce channels: **NAVER Brand Store** and **Olive Young (OY)**.
The goal is to quantify how advertising spend and promotions drive
sales, estimate each channel’s baseline (기초 체력), and provide a
framework for optimising future advertising budgets.

> **Background:** 2025년 1월부터 12월까지의 광고비, 매출, 프로모션 데이터를
> 수집하여 MMM을 구축했습니다. 1–3월은 광고 데이터가 유실되어 모델
> 학습에는 4월 1일 이후의 데이터만 사용했습니다. 따라서 베이스라인
> 매출 추정치는 절대값보다 상대적인 비교 지표로 활용해야 합니다.

## Components

### 1. Data Loading

The consolidated dataset (e.g. `final_dataset.csv`) must contain the
following columns:

- `date` – 날짜 (`YYYY-MM-DD`)
- `naver_spend`, `oy_spend` – 각 채널의 광고비 (수치형)
- `naver_sales`, `oy_sales` – 각 채널의 매출 (수치형)
- `log_naver_sales`, `log_oy_sales` – 로그 변환된 매출 (`log(sales+1)`)  
- `naver_promo_any`, `promo_<category>` – 프로모션 더미 변수 (0/1)

Use the provided `load_dataset` function in `bayesian_mmm/model_hierarchical.py`
to import the dataset and produce month/weekday indices.

### 2. Hierarchical Model

The module `bayesian_mmm/model_hierarchical.py` defines a **hierarchical
Bayesian MMM** with monthly intercepts. Each channel’s baseline is
allowed to vary by month via hyper priors. The model includes:

- Geometric adstock and Hill saturation transforms (`bayesian_mmm/transforms.py`)
- Elasticities for NAVER and OY advertising
- Cross‑channel spillover effects
- Promotion effects (optionally with spillover)
- Weekday seasonality
- Monthly intercepts drawn from shared hyper priors

To run inference you need **PyMC v4** and **ArviZ**. Example usage:

```python
from bayesian_mmm.model_hierarchical import load_dataset, build_model
import pymc as pm

# Load data
data = load_dataset('final_dataset.csv')

# Build model
model = build_model(data)

# Sample (requires PyMC)
with model:
    idata = pm.sample(1000, tune=1000, target_accept=0.9)

# Analyse posterior using ArviZ
import arviz as az
az.summary(idata)
```

Due to environment restrictions, sampling cannot be executed here.  The
above code should be run in a local Python environment with the
required libraries installed.

### 3. Budget Optimization

The script `optimize_budget.py` provides a simple method to allocate a
future advertising budget across channels and months.  It requires:

1. A dataset with spend and sales (`final_dataset.csv`).  
2. A CSV with **monthly revenue targets** for each channel (columns
   `month`, `oy_target`, `naver_target`).  
3. A total annual budget to distribute.

The algorithm computes channel ROI estimates based on the model’s
elasticities and historical spend/sales ratio, splits the budget
between channels accordingly, and then distributes each channel’s
budget across months as a weighted average of equal distribution and
target share.  Adjust the `--mix` parameter (0–1) to control how
strongly the allocation follows targets vs. equal distribution.

Example usage:

```bash
python optimize_budget.py \
    --dataset final_dataset.csv \
    --targets targets_2026.csv \
    --budget 2400000000 \
    --mix 0.5 \
    --output allocation_2026.csv
```

This will output a CSV `allocation_2026.csv` with the columns
`month`, `naver_budget`, `oy_budget` showing the recommended spend.

## Additional Notes

* **Promotion Calendar:** When planning budgets, consider the monthly
  promotion calendar (e.g. 올영세일, 넾다세일, 블랙프라이데이) and
  align higher budgets with months containing high‑impact events.
* **Model Uncertainty:** A full Bayesian treatment would provide
  posterior distributions for every coefficient, allowing credible
  intervals for ROI estimates. To achieve this, run the hierarchical
  model with PyMC in a local environment.
* **Extensibility:** You can extend the model to include more
  channels, additional covariates (price, holidays), or custom
  saturation/adstock functions by editing the corresponding modules.

## Licence

This project is provided as an educational example for marketing
analytics.  Feel free to use and adapt the code for non‑commercial
purposes.  No warranties are provided.