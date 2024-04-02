import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights(d, size):
    # thres>0 drops insignificant weights
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_standard(series, d, thres):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1]."""
    # 1) Compute weights for the longest series
    w = get_weights(d, series.shape[0])
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    print("--- debug standard frac diff ---")
    df = {}
    for name in series.columns:
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            print(f"processing iloc={iloc}, loc={loc}")
            if not np.isfinite(series.loc[loc, name]):
                print("continuining")
                continue  # exclude NAs
            df_[loc] = np.dot(w[-(iloc + 1) :, :].T, seriesF.loc[:loc])[0, 0]
        df[name] = df_.copy(deep=True)

    df = pd.concat(df, axis=1)
    print("--- end of code for standard frac diff ---")
    return df


    """
    Compute the weights of individual data points for fractional
    differentiation with fixed window:
    Args:
    * d (float): Fractional differentiation value.
    * threshold (float): Minimum weight to calculate.

    Returns:
    * pd.DataFrame: Dataframe containing the weights for each point.
    """
    w = [1.0]
    k = 1
    while True:
        v = -w[-1] / k * (d - k + 1)
        if abs(v) < threshold:
            break
        w.append(v)
        k += 1

    w = np.array(w[::-1]).reshape(-1, 1)
    return pd.DataFrame(w)


def fixed_window_fracc_diff(
    df: pd.DataFrame, d: float, threshold: float = 1e-5
) -> pd.DataFrame:
    """
    Compute the d fractional difference of the series with a fixed
    width window.

    Args:
    * df (pd.DataFrame): Dataframe with series to be differentiated.
    * d (float): Order of differentiation.
    * threshold (float): threshold value to drop non-significant
      weights.

    Returns:
    * pd.DataFrame: Dataframe containing differentiated series.
    """

    w = compute_weights_fixed_window(d, threshold)
    width = len(w)
    results = {}
    names = df.columns
    for name in names:
        series_f = df[name].ffill().dropna()

        if width > series_f.shape[0]:
            print(
                "width > series.shape[0]. Doing standard frac diff with full window l"
            )
            std_fracdiff = frac_diff_standard(df, d, 1)
            print(f"returning standard fracdiff: \n{std_fracdiff}")

            return std_fracdiff
        r = range(width, series_f.shape[0])
        # df_ = pd.Series(index=r)

        for idx in r:
            if not np.isfinite(df[name].iloc[idx]):
                continue
            results[idx] = np.dot(
                w.iloc[-(idx):, :].T, series_f.iloc[idx - width : idx]
            )[0]

    result = pd.DataFrame(pd.Series(results), columns=["Frac_diff"])
    result.set_index(df[width:].index, inplace=True)

    return result


def find_stat_series(
    df: pd.DataFrame,
    threshold: float = 0.01,
    diffs: np.ndarray = np.linspace(0.05, 0.95, 19),
    p_value: float = 0.05,
) -> pd.DataFrame:
    """
    Find the series that passes the adf test at the given p_value.
    The time series must be a single column dataframe.

    Args:

    * df (pd.DataFrame): Dataframe with series to be differentiated.
    * threshold (float): threshold value to drop non-significant weights.
    * diffs (np.linspace): Space for candidate d values.
    * p_value (float): ADF test p-value limit for rejection of null
    hypothesis.

    Returns:
    * pd.DataFrame: Dataframe containing differentiated series. This
    series is stationary and maintains maximum memory information.
    """
    for diff in diffs:
        if diff == 0:
            continue
        s = fixed_window_fracc_diff(df, diff, threshold)
        adf_stat = adfuller(s, maxlag=1, regression="c", autolag=None)[1]
        if adf_stat < p_value:
            s.columns = ["d=" + str(diff)]
            return s

    raise Exception("No series found with p-value < 0.05.")
