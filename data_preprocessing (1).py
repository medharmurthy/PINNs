"""
data_preprocessing.py

Data loading, cleaning, feature engineering, and sequence-windowing for the
BTC/ETH volatility + direction forecasting pipeline.

This module owns everything upstream of the models: reading the raw OHLCV
and investor-attention CSVs, aligning BTC/ETH, building the chronological
train/val/test split, engineering the TA-Lib-based alpha feature matrix, and
turning a feature-engineered dataframe into PyTorch-ready tensors.

Import from this module in the training script, e.g.:

    from data_preprocessing import (
        load_raw_data, build_alpha_feature_matrix, prepare_pytorch_data,
    )
    df_train, df_val, df_test = load_raw_data()
    df_train_alpha, feature_cols, feature_cols_ret = build_alpha_feature_matrix(df_train)

Requires:
    pip install ccxt pandas
    pip install --upgrade torchao peft transformers
    pip install -q TA-Lib
    pip install shap xgboost tqdm

TA-Lib requires the underlying C library in addition to the Python wrapper,
e.g. on Colab: `!apt-get install -y ta-lib` (or build from source) before
`pip install TA-Lib` -- see TA-Lib docs for the current recommended install
path on your platform.
"""

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ==========================================
# 1. LOAD & CLEAN RAW OHLCV DATA
# ==========================================

def to_naive_utc(series):
    """
    Parse a datetime-like column and normalize it to timezone-NAIVE UTC.

    Source files are inconsistent about timezones -- some export naive
    timestamps (already implicitly UTC), others export tz-aware ones
    (e.g. '...+00:00'). merge_asof requires BOTH sides of a join to share
    the exact same dtype, so every datetime column in this pipeline is
    routed through this function before being used as a join/sort key.
    `utc=True` converts tz-aware values to UTC and treats naive values as
    already-UTC; `.dt.tz_localize(None)` then drops the tz label so the
    dtype matches plain `datetime64[ns]` columns.
    """
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def load_and_clean(path, prefix=None):
    """Load a raw OHLCV csv, standardize columns, set datetime index."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[['open time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['open time'] = to_naive_utc(df['open time'])
    df.set_index('open time', inplace=True)
    if prefix:
        df = df.rename(columns={c: f'{prefix}_{c}' for c in df.columns})
    return df


# ==========================================
# 2. INVESTOR ATTENTION FEATURES (BTC + ETH, joined by calendar day)
# ==========================================
# Each attention dataset has columns:
#   datetime, score_combined_pct, score_combined_pct_7D_rolling,
#   score_combined_pct_30D_rolling, score_ratio_7D_30D
# These are DAILY readings, while the main dataset is 15-minute candles.
# So each row is anchored to its calendar day and joined on that day: every
# 15m candle on a given date gets that date's attention reading, applied
# uniformly across all ~96 candles in that day (see load_attention_data's
# docstring for why day-level join is used instead of an exact-timestamp
# asof join).

def load_attention_data(path, prefix):
    """
    Load a DAILY investor-attention csv and prefix its score columns.

    The attention data has one row per calendar day, while the OHLCV data
    has one row per 15-minute candle. Rather than aligning on the exact
    timestamp in the file (which could sit at any hour and risks bleeding
    into the wrong day near midnight), we normalize each row down to its
    calendar day (`attn_date`, time-of-day dropped) and join on that date.
    Every 15m candle that falls within a given calendar day then gets that
    day's attention reading, regardless of what hour the source file used.
    """
    att = pd.read_csv(path)
    att.columns = att.columns.str.strip().str.lower()
    att['datetime'] = to_naive_utc(att['datetime'])
    att['attn_date'] = att['datetime'].dt.normalize()

    score_cols = [
        'score_combined_pct',
        'score_combined_pct_7d_rolling',
        'score_combined_pct_30d_rolling',
        'score_ratio_7d_30d',
    ]
    att = att[['attn_date'] + score_cols].copy()
    # Guard against duplicate rows for the same day (keep the last one)
    att = att.drop_duplicates(subset='attn_date', keep='last').sort_values('attn_date')
    att = att.rename(columns={c: f'{prefix}_{c}' for c in score_cols})
    return att


BTC_ATTENTION_COLS = [
    'attn_score_combined_pct',
    'attn_score_combined_pct_7d_rolling',
    'attn_score_combined_pct_30d_rolling',
    'attn_score_ratio_7d_30d',
]
ETH_ATTENTION_COLS = [
    'eth_attn_score_combined_pct',
    'eth_attn_score_combined_pct_7d_rolling',
    'eth_attn_score_combined_pct_30d_rolling',
    'eth_attn_score_ratio_7d_30d',
]


# ==========================================
# 3. TOP-LEVEL DATA LOADING / ALIGNMENT / SPLIT
# ==========================================

def load_raw_data(
    btc_path='/content/btc_15m_data_2018_to_2025.csv',
    eth_path='/content/ETHUSDT 15M.csv',
    btc_attention_path='/content/df_cropped_processed_BTC.csv',
    eth_attention_path='/content/df_cropped_processed_ETH.csv',
    test_size=10_000,
    val_size=10_000,
    verbose=True,
):
    """
    End-to-end raw-data pipeline: load BTC + ETH OHLCV, align on shared
    timestamps, engineer returns/temporal encodings, attach daily investor
    attention scores, and perform the chronological train/val/test split.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
        Chronologically ordered splits (oldest -> train, then val, then
        test), each with its datetime index reset to a plain integer index.
        `df_train` additionally excludes any rows from calendar year 2018
        ("more recent trends"); val/test are left untouched.
    """
    # ---- 3.1 Load & align BTC/ETH OHLCV ----
    df_btc = load_and_clean(btc_path)
    df_eth = load_and_clean(eth_path, prefix='eth')

    if verbose:
        print(f"BTC rows: {len(df_btc)} | ETH rows: {len(df_eth)}")

    # Inner join: keep only timestamps present in both
    df = df_btc.join(df_eth, how='inner')
    if verbose:
        print(f"Aligned dataset shape: {df.shape}")
        print(f"Aligned range: {df.index.min()} -> {df.index.max()}")

    # ---- 3.2 Returns + temporal encoding ----
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['eth_log_returns'] = np.log(df['eth_close'] / df['eth_close'].shift(1))

    df['year'] = df.index.year
    month = df.index.month
    day = df.index.day
    hour = df.index.hour + df.index.minute / 60.0

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * day / 31)
    df['day_cos'] = np.cos(2 * np.pi * day / 31)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # ---- 3.3 Target-source volatility (NOT a model feature) ----
    # `trailing_volatility` is only used inside build_alpha_feature_matrix to
    # build `target_vol` (= trailing_volatility.shift(-1)). It is deliberately
    # never added to `optimized_features`. ATR(14) from TA-Lib is what the
    # model actually sees as its volatility-style input.
    log_ret_sq = df['log_returns'] ** 2
    df['trailing_volatility'] = np.sqrt(log_ret_sq.rolling(window=6).sum())

    # ---- 3.4 Investor attention features (BTC + ETH, joined by calendar day) ----
    btc_attention = load_attention_data(btc_attention_path, prefix='attn')
    eth_attention = load_attention_data(eth_attention_path, prefix='eth_attn')

    # Temporarily move the datetime index into a column and derive its
    # calendar day, then join the daily attention tables on that day (plain
    # equality merge, not asof -- see load_attention_data's docstring for
    # why). Restore the datetime index afterwards.
    df = df.reset_index()
    df['attn_date'] = df['open time'].dt.normalize()

    df = df.merge(btc_attention, on='attn_date', how='left')
    df = df.merge(eth_attention, on='attn_date', how='left')

    df = df.drop(columns=['attn_date']).set_index('open time')

    if verbose:
        n_missing_btc = df[BTC_ATTENTION_COLS].isna().all(axis=1).sum()
        print(f"Attention features merged. Candles with no BTC attention data yet: {n_missing_btc}")
    # Any remaining NaNs (e.g. candles before the attention series starts)
    # are handled later by the .ffill() + dropna(subset=...) inside
    # build_alpha_feature_matrix.

    # ---- 3.5 Chronological split (before any normalization / feature scaling) ----
    total_rows = len(df)
    test_start_idx = total_rows - test_size
    val_start_idx = total_rows - test_size - val_size

    df_train = df.iloc[:val_start_idx].copy()
    df_val = df.iloc[val_start_idx:test_start_idx].copy()
    df_test = df.iloc[test_start_idx:].copy()

    if verbose:
        print(f"Training set shape: {df_train.shape}")
        print(f"Validation set shape: {df_val.shape}")
        print(f"Test set shape: {df_test.shape}")

    # Drop 2018 from training only, to focus on more recent trends
    df_train = df_train[df_train['year'] >= 2019]
    if verbose:
        print(f"Training set shape after dropping 2018: {df_train.shape}")

    for d in (df_train, df_val, df_test):
        d.reset_index(drop=True, inplace=True)

    return df_train, df_val, df_test


# ==========================================
# 4. ALPHA FEATURE MATRIX (TA-Lib indicators, built from RAW prices,
#    consistent across splits)
# ==========================================

def build_alpha_feature_matrix(df_raw):
    """
    Builds model-ready features + targets from a RAW (non-normalized) OHLCV
    dataframe. Replaces raw technicals with rolling means, relative
    distances, and engineered momentum/attention signals.

    Returns
    -------
    d_clean : pd.DataFrame
        Feature-engineered, target-appended dataframe with all warmup/NaN
        rows dropped.
    optimized_features : list[str]
        Feature columns used by the volatility model (VolatilityModel).
    optimized_features_ret : list[str]
        Feature columns used by the returns/direction model (XGBoost).
    """
    d = df_raw.copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)

    for col in ['close', 'open', 'high', 'low', 'eth_close', 'eth_open']:
        d[col] = d[col].clip(lower=1e-6)

    close = d['close'].to_numpy(dtype=np.float64)
    high = d['high'].to_numpy(dtype=np.float64)
    low = d['low'].to_numpy(dtype=np.float64)
    volume = d['volume'].to_numpy(dtype=np.float64)

    # ---------------------------------------------------------
    # 4.1 Base TA-Lib Calculations (used as intermediate components)
    # ---------------------------------------------------------
    rsi_14 = talib.RSI(close, timeperiod=14)
    stochrsi_k, stochrsi_d = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    adx_14 = talib.ADX(high, low, close, timeperiod=14)
    atr_14 = talib.ATR(high, low, close, timeperiod=14)
    obv = talib.OBV(close, volume)
    ad = talib.AD(high, low, close, volume)

    # Existing features required by user directly
    d['ema_12'] = talib.EMA(close, timeperiod=12)
    d['ema_26'] = talib.EMA(close, timeperiod=26)
    d['ht_trendmode'] = talib.HT_TRENDMODE(close)
    d['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)
    d['macd_signal'] = pd.Series(macd_signal, index=d.index)
    d['macd_hist'] = pd.Series(macd_hist, index=d.index)

    # ---------------------------------------------------------
    # 4.2 Engineered Features
    # ---------------------------------------------------------
    # Volume indicator rolling means
    d['ad_rolling_mean_30d'] = pd.Series(ad, index=d.index).rolling(30).mean()
    d['ad_rolling_mean_15d'] = pd.Series(ad, index=d.index).rolling(15).mean()
    d['obv_rolling_mean_30d'] = pd.Series(obv, index=d.index).rolling(30).mean()
    d['obv_rolling_mean_15d'] = pd.Series(obv, index=d.index).rolling(15).mean()

    # Attention features
    attn_combined = d['attn_score_combined_pct']
    d['z_score_attn_score_combined_pct'] = (
        (attn_combined - attn_combined.rolling(30).mean())
        / (attn_combined.rolling(30).std() + 1e-9)
    )

    # Distance from Bollinger Bands
    d['bb_upper_distance'] = (pd.Series(bb_upper, index=d.index) - d['close']) / d['close']
    d['bb_middle_distance'] = (pd.Series(bb_middle, index=d.index) - d['close']) / d['close']
    d['bb_lower_distance'] = (pd.Series(bb_lower, index=d.index) - d['close']) / d['close']

    # Percentage Normalizations
    d['atr_14_percentage'] = pd.Series(atr_14, index=d.index) / d['close']
    d['adx_14_percentage'] = pd.Series(adx_14, index=d.index) / d['close']

    # Momentum Rolling Means & Differences
    d['macd_rolling_mean_30d'] = pd.Series(macd, index=d.index).rolling(30).mean()
    d['rsi_14_rolling_mean_30d'] = pd.Series(rsi_14, index=d.index).rolling(30).mean()
    d['rsi_14_rolling_mean_7d'] = pd.Series(rsi_14, index=d.index).rolling(7).mean()
    d['stochrsi_k_rolling_mean_30d'] = pd.Series(stochrsi_k, index=d.index).rolling(30).mean()
    d['stochrsi_k_rolling_mean_15d'] = pd.Series(stochrsi_k, index=d.index).rolling(15).mean()
    d['stochrsi_d_rolling_mean_30d'] = pd.Series(stochrsi_d, index=d.index).rolling(30).mean()
    d['stochrsi_d_rolling_mean_15d'] = pd.Series(stochrsi_d, index=d.index).rolling(15).mean()
    d['macd_hist_diff'] = d['macd_hist'].diff(1)
    d['rsi_velocity'] = talib.EMA(rsi_14, timeperiod=5)  # EMA of RSI for velocity

    # Extreme Threshold Flags
    d['is_oversold'] = (pd.Series(rsi_14, index=d.index) < 30).astype(int)
    d['is_overbought'] = (pd.Series(rsi_14, index=d.index) > 70).astype(int)

    # Custom Divergence & Filter Features
    close_rolling_30d = d['close'].rolling(30).mean()
    close_rolling_7d = d['close'].rolling(7).mean()

    d['price_momentum_divergence_30day'] = close_rolling_30d / (d['rsi_14_rolling_mean_30d'] + 1e-9)
    d['price_momentum_divergence_7day'] = close_rolling_7d / (d['rsi_14_rolling_mean_7d'] + 1e-9)
    d['hype_filter'] = d['attn_score_combined_pct_7d_rolling'] / (d['obv_rolling_mean_15d'] + 1e-9)

    # Signed volume: raw volume signed by the direction of that candle's
    # return. This is a RETURNS-MODEL-ONLY feature -- it is added to
    # `optimized_features_ret` below but deliberately left OUT of
    # `optimized_features` (the volatility model's feature set).
    d['signed_volume'] = d['volume'] * np.sign(d['log_returns'])

    # ---------------------------------------------------------
    # 4.3 Assemble Final Feature Sets
    # ---------------------------------------------------------
    optimized_features = [
        'ad_rolling_mean_30d',
        'ad_rolling_mean_15d',
        'obv_rolling_mean_30d',
        'obv_rolling_mean_15d',
        'attn_score_combined_pct_30d_rolling',
        'z_score_attn_score_combined_pct',
        'attn_score_ratio_7d_30d',
        'attn_score_combined_pct_7d_rolling',
        'ema_12',
        'ema_26',
        'atr_14_percentage',
        'ht_trendmode',
        'bb_width',
        'macd_rolling_mean_30d',
        'macd_signal',
        'rsi_14_rolling_mean_30d',
        'rsi_14_rolling_mean_7d',
        'adx_14_percentage',
        'macd_hist',
        'is_oversold',
        'is_overbought',
        'rsi_velocity',
        'price_momentum_divergence_30day',
        'price_momentum_divergence_7day',
    ]

    # Returns-model-only feature set: started as `optimized_features +
    # ['signed_volume']`, but per updated instructions the following
    # features are EXCLUDED from the returns model (they remain in
    # `optimized_features`, i.e. the volatility model's feature set, which
    # is untouched):
    ret_features_to_exclude = [
        'ht_trendmode',
        'is_overbought',
        'is_oversold',
        'price_momentum_divergence_30day',
        'macd_rolling_mean_30d',
        'z_score_attn_score_combined_pct',
        'rsi_14_rolling_mean_30d',
        'price_momentum_divergence_7day',
        'atr_14_percentage',
        'adx_14_percentage',
        'macd_signal',
        'bb_width',
        'rsi_14_rolling_mean_7d',
        'rsi_velocity',
        'macd_hist',
        'signed_volume',
    ]
    optimized_features_ret = [
        f for f in (optimized_features + ['signed_volume'])
        if f not in ret_features_to_exclude
    ]

    # Targets (built from forward-looking columns; NEVER included in optimized_features)
    d['target_vol'] = d['trailing_volatility'].shift(-1)
    future_ret = d['log_returns'].shift(-1)

    # 3-class target system (using a +/- 0.1% threshold)
    d['target_ret'] = np.where(
        future_ret > 0.001, 2.0,                # Class 2: Strong Up (> +0.1%)
        np.where(future_ret < -0.001, 0.0, 1.0)  # Class 0: Strong Down (< -0.1%); Class 1: Chop/Noise
    )

    # Union of BOTH feature sets -- optimized_features_ret is a strict subset
    # of optimized_features (several columns are excluded from the returns
    # set), so checking only the ret subset lets NaNs in vol-only columns
    # (rolling-window warmup NaNs, HT_TRENDMODE, etc.) leak into the rows
    # used to train VolatilityModel, which is what previously produced NaN
    # loss.
    cols_needed = sorted(
        set(optimized_features) | set(optimized_features_ret) | {'target_vol', 'target_ret'}
    )

    # Ensure no NaNs remain after forward/rolling window creation
    d_clean = d.dropna(subset=cols_needed)

    return d_clean, optimized_features, optimized_features_ret


# ==========================================
# 5. SEQUENCE GENERATOR (pandas -> pytorch tensors)
# ==========================================

def prepare_pytorch_data(df_clean, features, seq_length=10, fit_scaler=None):
    """
    Scales features and rolls them into 3D (batch, seq, features) tensors.

    Pass `fit_scaler=None` to fit a new StandardScaler (training set), or
    pass an already-fitted scaler to transform val/test sets without
    leakage.

    Returns
    -------
    X : np.ndarray, shape (n_windows, seq_length, len(features))
    y_vol : np.ndarray, shape (n_windows,)
    y_ret : np.ndarray, shape (n_windows,)
    scaler : fitted sklearn StandardScaler
    """
    if fit_scaler is None:
        # Fit fresh on this split (should only ever be the TRAINING split) --
        # every column in `features` (all TA-Lib indicators, the regime
        # flag, the cyclical time encodings, and the BTC attention scores)
        # is normalized to mean 0 / std 1.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_clean[features].values)
        print(f"Fit StandardScaler on {len(features)} features: {features}")
        print(f"  post-scale mean (should be ~0): {scaled.mean(axis=0).round(3)}")
        print(f"  post-scale std  (should be ~1): {scaled.std(axis=0).round(3)}")
    else:
        # Val/test/inference: reuse the scaler fitted on train, never refit,
        # to avoid leaking val/test statistics into the normalization.
        scaler = fit_scaler
        scaled = scaler.transform(df_clean[features].values)

    vol_targets = df_clean['target_vol'].values
    ret_targets = df_clean['target_ret'].values

    n_windows = len(scaled) - seq_length
    X, y_vol, y_ret = [], [], []
    # This loop can be a couple hundred thousand iterations for the full
    # training split -- wrap it in tqdm so it's visible that work is
    # actually happening instead of looking like a hang.
    for i in tqdm(range(n_windows), desc="Building sequence windows", leave=False):
        X.append(scaled[i:i + seq_length])
        y_vol.append(vol_targets[i + seq_length])
        y_ret.append(ret_targets[i + seq_length])

    return (np.array(X, dtype=np.float32),
            np.array(y_vol, dtype=np.float32),
            np.array(y_ret, dtype=np.float32),
            scaler)
