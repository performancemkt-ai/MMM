"""Hierarchical Bayesian MMM for NAVER and OY channels.

This module builds a variant of the MMM with hierarchical (monthly) intercepts.

Rationale
---------
In the basic MMM, each channel has a single intercept (`intercept_n`, `intercept_o`)
representing the brand's baseline (기초 체력). However, baseline sales often vary
by season or month. To allow this flexibility without overfitting, we model
monthly intercepts as draws from a common hyper-prior. This introduces a
hierarchical structure:

    mu_intercept ~ Normal(0, sigma_mu)
    sigma_intercept ~ HalfNormal(1.0)
    intercept_month[m] ~ Normal(mu_intercept, sigma_intercept)

Each day's intercept for NAVER/OY is given by the intercept of its month.

The remainder of the model (adstock, saturation, promo effects, spillover,
weekday seasonality) is identical to the base model in `model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from .transforms import geometric_adstock, hill_saturation


@dataclass
class MMMData:
    date: pd.Series
    naver_spend: np.ndarray
    oy_spend: np.ndarray
    log_naver_sales: np.ndarray
    log_oy_sales: np.ndarray
    month_idx: np.ndarray  # 0..11
    weekday: np.ndarray  # 0..6
    promo_cols: list[str]
    promos: np.ndarray


def load_dataset(csv_path: str) -> MMMData:
    """Load dataset and produce month & weekday indices for hierarchical model."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"일자": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # month index: 0=Jan,...,11=Dec
    month_idx = df["date"].dt.month.values - 1
    weekday_idx = df["date"].dt.weekday.values.astype("int64")
    # promo columns: all starting with 'promo_' plus optional 'naver_promo_any'
    promo_cols = [c for c in df.columns if c.startswith("promo_")]
    if "naver_promo_any" in df.columns:
        promo_cols = ["naver_promo_any"] + promo_cols
    promos = df[promo_cols].fillna(0.0).astype(float).values if promo_cols else np.zeros((len(df), 0))
    return MMMData(
        date=df["date"],
        naver_spend=df["naver_spend"].fillna(0.0).astype(float).values,
        oy_spend=df["oy_spend"].fillna(0.0).astype(float).values,
        log_naver_sales=df["log_naver_sales"].astype(float).values,
        log_oy_sales=df["log_oy_sales"].astype(float).values,
        month_idx=month_idx,
        weekday=weekday_idx,
        promo_cols=promo_cols,
        promos=promos,
    )


def build_model(
    data: MMMData,
    *,
    max_spend_scale: float | None = None,
    weekday_seasonality: bool = True,
    allow_promo_spillover: bool = True,
) -> pm.Model:
    """
    Build a PyMC hierarchical MMM for NAVER & OY.

    See base model in `model.py` for details. The key differences are:
    - Each channel's intercept is indexed by month with a shared hyper prior.
    """
    T = len(data.naver_spend)
    K = data.promos.shape[1]
    M = 12  # months
    n_spend = data.naver_spend
    o_spend = data.oy_spend
    # scale factors
    n_scale = float(np.nanmax(n_spend)) if max_spend_scale is None else float(max_spend_scale)
    o_scale = float(np.nanmax(o_spend)) if max_spend_scale is None else float(max_spend_scale)
    n_scaled = np.clip(n_spend / (n_scale + 1e-12), 0, None)
    o_scaled = np.clip(o_spend / (o_scale + 1e-12), 0, None)
    weekday_idx = data.weekday
    month_idx = data.month_idx

    with pm.Model() as model:
        # Hyper priors for intercepts
        mu_int_n = pm.Normal("mu_intercept_n", mu=0.0, sigma=5.0)
        mu_int_o = pm.Normal("mu_intercept_o", mu=0.0, sigma=5.0)
        sigma_int_n = pm.HalfNormal("sigma_intercept_n", sigma=2.0)
        sigma_int_o = pm.HalfNormal("sigma_intercept_o", sigma=2.0)
        # monthly intercepts
        intercept_n_month = pm.Normal("intercept_n_month", mu=mu_int_n, sigma=sigma_int_n, shape=M)
        intercept_o_month = pm.Normal("intercept_o_month", mu=mu_int_o, sigma=sigma_int_o, shape=M)
        # Adstock parameters
        theta_n = pm.Beta("theta_n", alpha=3, beta=3)
        theta_o = pm.Beta("theta_o", alpha=3, beta=3)
        # Saturation parameters
        alpha_n = pm.LogNormal("alpha_n", mu=0.0, sigma=0.5)
        alpha_o = pm.LogNormal("alpha_o", mu=0.0, sigma=0.5)
        gamma_n = pm.HalfNormal("gamma_n", sigma=1.0)
        gamma_o = pm.HalfNormal("gamma_o", sigma=1.0)
        # Media coefficients
        beta_n = pm.HalfNormal("beta_n", sigma=1.0)
        beta_o = pm.HalfNormal("beta_o", sigma=1.0)
        # Cross-channel spillover
        cross_no = pm.Normal("cross_no", mu=0.0, sigma=0.2)
        cross_on = pm.Normal("cross_on", mu=0.0, sigma=0.2)
        # Transform spends
        x_n = pt.as_tensor_variable(n_scaled.astype("float64"))
        x_o = pt.as_tensor_variable(o_scaled.astype("float64"))
        ad_n = geometric_adstock(x_n, theta_n)
        ad_o = geometric_adstock(x_o, theta_o)
        sat_n = hill_saturation(ad_n, alpha_n, gamma_n)
        sat_o = hill_saturation(ad_o, alpha_o, gamma_o)
        # Promotions
        if K > 0:
            promo = pt.as_tensor_variable(data.promos.astype("float64"))
            promo_n = pm.Normal("promo_n", mu=0.0, sigma=0.3, shape=K)
            promo_o = pm.Normal("promo_o", mu=0.0, sigma=0.3, shape=K)
            promo_term_n = pt.dot(promo, promo_n)
            promo_term_o = pt.dot(promo, promo_o)
            if allow_promo_spillover:
                promo_spill_n = pm.Normal("promo_spill_n", mu=0.0, sigma=0.1, shape=K)
                promo_spill_o = pm.Normal("promo_spill_o", mu=0.0, sigma=0.1, shape=K)
                promo_term_n = promo_term_n + pt.dot(promo, promo_spill_n)
                promo_term_o = promo_term_o + pt.dot(promo, promo_spill_o)
        else:
            promo_term_n = 0.0
            promo_term_o = 0.0
        # Weekday seasonality
        if weekday_seasonality:
            wd_n_raw = pm.Normal("weekday_n_raw", mu=0.0, sigma=0.2, shape=7)
            wd_o_raw = pm.Normal("weekday_o_raw", mu=0.0, sigma=0.2, shape=7)
            wd_n_centered = wd_n_raw - pt.mean(wd_n_raw)
            wd_o_centered = wd_o_raw - pt.mean(wd_o_raw)
            wd_term_n = wd_n_centered[weekday_idx]
            wd_term_o = wd_o_centered[weekday_idx]
        else:
            wd_term_n = 0.0
            wd_term_o = 0.0
        # Linear predictors
        mu_n = intercept_n_month[month_idx] + beta_n * sat_n + cross_no * sat_o + promo_term_n + wd_term_n
        mu_o = intercept_o_month[month_idx] + beta_o * sat_o + cross_on * sat_n + promo_term_o + wd_term_o
        # Observation noise
        sigma_n = pm.HalfNormal("sigma_n", sigma=0.5)
        sigma_o = pm.HalfNormal("sigma_o", sigma=0.5)
        pm.Normal("log_naver_sales", mu=mu_n, sigma=sigma_n, observed=data.log_naver_sales)
        pm.Normal("log_oy_sales", mu=mu_o, sigma=sigma_o, observed=data.log_oy_sales)
    return model