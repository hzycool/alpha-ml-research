"""Factor dictionary with economic interpretation and formulas."""

from __future__ import annotations

import pandas as pd


FACTOR_DICTIONARY: dict[str, dict[str, str]] = {
    "ret_1": {"category": "Momentum/Reversal", "formula": "close / close.shift(1) - 1", "meaning": "1-day return, often used as short-term reversal signal."},
    "ret_3": {"category": "Momentum/Reversal", "formula": "close / close.shift(3) - 1", "meaning": "3-day return."},
    "ret_5": {"category": "Momentum/Reversal", "formula": "close / close.shift(5) - 1", "meaning": "1-week return."},
    "ret_10": {"category": "Momentum/Reversal", "formula": "close / close.shift(10) - 1", "meaning": "2-week return."},
    "ret_20": {"category": "Momentum/Reversal", "formula": "close / close.shift(20) - 1", "meaning": "1-month return."},
    "ret_60": {"category": "Momentum/Reversal", "formula": "close / close.shift(60) - 1", "meaning": "Quarterly return."},
    "ret_120": {"category": "Momentum/Reversal", "formula": "close / close.shift(120) - 1", "meaning": "Half-year return."},
    "volatility_5": {"category": "Volatility", "formula": "std(daily_return, 5)", "meaning": "Recent realized volatility."},
    "volatility_10": {"category": "Volatility", "formula": "std(daily_return, 10)", "meaning": "2-week realized volatility."},
    "volatility_20": {"category": "Volatility", "formula": "std(daily_return, 20)", "meaning": "1-month realized volatility."},
    "volatility_60": {"category": "Volatility", "formula": "std(daily_return, 60)", "meaning": "Quarterly realized volatility."},
    "amplitude_5": {"category": "Volatility", "formula": "rolling_max(high, 5) / rolling_min(low, 5) - 1", "meaning": "5-day price range."},
    "amplitude_10": {"category": "Volatility", "formula": "rolling_max(high, 10) / rolling_min(low, 10) - 1", "meaning": "10-day price range."},
    "amplitude_20": {"category": "Volatility", "formula": "rolling_max(high, 20) / rolling_min(low, 20) - 1", "meaning": "20-day price range."},
    "volume_mean_5": {"category": "Liquidity", "formula": "mean(volume, 5)", "meaning": "Recent average trading volume."},
    "volume_mean_20": {"category": "Liquidity", "formula": "mean(volume, 20)", "meaning": "Monthly average trading volume."},
    "volume_mean_60": {"category": "Liquidity", "formula": "mean(volume, 60)", "meaning": "Quarterly average trading volume."},
    "amount_mean_5": {"category": "Liquidity", "formula": "mean(amount, 5)", "meaning": "Recent average traded value."},
    "amount_mean_20": {"category": "Liquidity", "formula": "mean(amount, 20)", "meaning": "Monthly average traded value."},
    "amount_mean_60": {"category": "Liquidity", "formula": "mean(amount, 60)", "meaning": "Quarterly average traded value."},
    "turnover_mean_5": {"category": "Liquidity", "formula": "mean(turnover_rate, 5)", "meaning": "Recent trading activity."},
    "turnover_mean_20": {"category": "Liquidity", "formula": "mean(turnover_rate, 20)", "meaning": "Monthly trading activity."},
    "turnover_mean_60": {"category": "Liquidity", "formula": "mean(turnover_rate, 60)", "meaning": "Quarterly trading activity."},
    "corr_ret_volume_10": {"category": "Price-Volume", "formula": "corr(daily_return, volume, 10)", "meaning": "Short-term relation between price moves and volume."},
    "corr_ret_volume_20": {"category": "Price-Volume", "formula": "corr(daily_return, volume, 20)", "meaning": "Monthly return-volume relation."},
    "corr_close_volume_20": {"category": "Price-Volume", "formula": "corr(close, volume, 20)", "meaning": "Price-volume co-movement."},
    "corr_ret_amount_20": {"category": "Price-Volume", "formula": "corr(daily_return, amount, 20)", "meaning": "Return-traded value co-movement."},
    "volume_chg_5_20": {"category": "Activity Change", "formula": "volume_mean_5 / volume_mean_20 - 1", "meaning": "Recent volume acceleration."},
    "amount_chg_5_20": {"category": "Activity Change", "formula": "amount_mean_5 / amount_mean_20 - 1", "meaning": "Recent traded-value acceleration."},
    "turnover_chg_5_20": {"category": "Activity Change", "formula": "turnover_mean_5 / turnover_mean_20 - 1", "meaning": "Recent turnover acceleration."},
    "close_to_high_20": {"category": "Price Location", "formula": "close / rolling_max(high, 20) - 1", "meaning": "Distance to 20-day high."},
    "close_to_low_20": {"category": "Price Location", "formula": "close / rolling_min(low, 20) - 1", "meaning": "Distance to 20-day low."},
    "close_position_20": {"category": "Price Location", "formula": "(close - rolling_min(low,20)) / (rolling_max(high,20)-rolling_min(low,20))", "meaning": "Where close sits in the 20-day range."},
    "close_position_60": {"category": "Price Location", "formula": "(close - rolling_min(low,60)) / (rolling_max(high,60)-rolling_min(low,60))", "meaning": "Where close sits in the 60-day range."},
    "ret_20_div_vol_20": {"category": "Risk-adjusted Return", "formula": "ret_20 / volatility_20", "meaning": "Monthly return adjusted by realized volatility."},
    "ret_60_div_vol_60": {"category": "Risk-adjusted Return", "formula": "ret_60 / volatility_60", "meaning": "Quarterly return adjusted by realized volatility."},
}


def get_factor_list() -> list[str]:
    """Return factor names in dictionary order."""
    return list(FACTOR_DICTIONARY.keys())


def factor_dictionary_frame() -> pd.DataFrame:
    """Return the factor dictionary as a DataFrame."""
    return pd.DataFrame.from_dict(FACTOR_DICTIONARY, orient="index").reset_index(names="factor")
