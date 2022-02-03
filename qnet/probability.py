import logging

import numpy as np
import plotly.graph_objs as go
import scipy.interpolate as interpolate
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__file__)

def brute_force_monotonicity(scaled_predictions):
    """
    The output of isotonic regression is strictly "not decreasing" as opposed to strictly increasing. This
    function bumps two predictions that are predicted as the same value consecutively so that the results are
    strictly increasing

    Args:
        quantile_predictions (list): list of predictions at each quantile in order, going from smaller quantiles to
        larger quantiles

    Returns: list
    The quantile_predictions list but with modifications to make it strictly increasing

    """
    scaled_predictions = scaled_predictions.flatten()

    idx = 1
    while idx<len(scaled_predictions):
        if (scaled_predictions[idx] - 1e-3) >= scaled_predictions[idx - 1]:
            idx += 1
        else:
            scaled_predictions[idx - 1] = scaled_predictions[idx - 1] - 1e-3
            idx = 1

    return scaled_predictions


def enforce_monotonicity(quantile_predictions, quantiles):

    """
    Fits an isotonic regression on the predicted quantiles to take the first and most technical step in
    enforcing the monotonicity of what would be the estimated cumulative density function

    Args:
        quantile_predictions: the predictions at each of the quantiles from a machine learning algorithm
        quantiles (list): a list of the predicted quantiles

    Returns: numpy.ndarray
    The output of a fit isotonic regression of the original predictions

    """

    iso = IsotonicRegression()
    model = iso.fit(quantiles, quantile_predictions)
    monotonic_quantile_predictions = model.predict(quantiles)
    scaler = MinMaxScaler()
    scaled_predictions = scaler.fit_transform(np.array(monotonic_quantile_predictions).reshape(-1, 1))
    scaled_monotonic_quantile_predictions = brute_force_monotonicity(
        scaled_predictions.flatten()
    )
    monotonic_quantile_predictions = scaler.inverse_transform(scaled_monotonic_quantile_predictions.reshape(-1,1))
    return monotonic_quantile_predictions.flatten()

def enforce_monotonicity_penalized(quantile_predictions, quantiles, kp=1e10, la=100):

    dd = 3
    E = np.eye(len(quantile_predictions))
    D3 = np.diff(E, n=dd, axis=0)
    D1 = np.diff(E, n=1, axis=0)

    # Monotone smoothing
    ws = np.zeros(len(quantile_predictions) - 1)

    for it in range(30):
        Ws = np.diag(ws * kp)
        mon_cof = np.linalg.solve(E + la * D3.T @ D3 + D1.T @ Ws @ D1, quantile_predictions)
        ws_new = (D1 @ mon_cof < 0.0) * 1
        dw = np.sum(ws != ws_new)
        ws = ws_new
        if (dw == 0): break

    # Monotonic and non monotonic fits
    z = mon_cof
    scaler = MinMaxScaler()
    scaled_predictions = scaler.fit_transform(z.reshape(-1, 1))
    scaled_monotonic_quantile_predictions = brute_force_monotonicity(
        scaled_predictions.flatten()
    )
    monotonic_quantile_predictions = scaler.inverse_transform(scaled_monotonic_quantile_predictions.reshape(-1, 1))

    return monotonic_quantile_predictions.flatten()


def generate_cdf_pdf_plot(quantiles, quantile_predictions):
    """
    Generate the CDF and PDF plotly charts with a list of quantiles and a list of predictions for those quantiles
    as inputs

    Args:
        quantiles: list of quantiles
        quantile_predictions: list of quantile predictions

    Returns: two plotly graphs

    """

    quantile_predictions = enforce_monotonicity(quantile_predictions, quantiles)

    xmin, xmax = quantile_predictions.min(), quantile_predictions.max()
    N = 2000
    xx = np.linspace(xmin, xmax, N)
    spline = interpolate.PchipInterpolator(quantile_predictions, quantiles)
    deriv_spline = spline.derivative()
    yy = spline(xx)

    cdf = go.Figure()
    cdf.add_trace(go.Scatter(x=xx, y=yy, showlegend=False, fill="tozeroy"))
    cdf.update_layout(
        title="Estimated Cumulative Probability Density",
        xaxis_title="Cost per MHz per Month",
        yaxis_title="Probability",
    )

    pp = deriv_spline(xx)
    pdf = go.Figure()
    pdf.add_trace(go.Scatter(x=xx, y=pp, showlegend=False, fill="tozeroy"))
    pdf.update_layout(
        title="Estimated Probability Density",
        xaxis_title="Cost per MHz per Month",
        yaxis_title="Probability",
    )

    return cdf, pdf


def generate_cdf_function(quantiles, quantile_predictions, kp=1e10, la=1e-1):
    """
    Generates the spline representing the CDF given a list of quantiles and a list of predictions for
    those quantiles as inputs

    Args:
        quantiles: list of quantiles
        quantile_predictions: list of quantile predictions

    Returns: scipy spline, minimum value, maximum value

    """
    quantile_predictions = enforce_monotonicity_penalized(quantile_predictions, quantiles, kp=kp, la=la)
    xmin, xmax = quantile_predictions.min(), quantile_predictions.max()

    try:
        spline = interpolate.PchipInterpolator(quantile_predictions, quantiles)
    except:
        logger.info("Error: enforce monotonicity didnt work right. {}".format(quantile_predictions))
        raise ValueError

    return spline, xmin, xmax


if __name__ == "__main__":
    preds = [-17.57417827890519, -18.111178356306326, -4.162798259133957,
             -1.5150036888537803, 0.5537992458004614, -0.8252836210034833,
             1.2276416191973365, 0.4944095231667302, 1.2528415392796122,
             2.183402386366614, 1.101750030312852, 1.8777380738355094,
             0.15956548121625472, 1.423051088506615, 2.5022441051787894,
             1.1512691611042676, 2.402308517064839, 1.6686194491896813,
             3.946432996962857, 4.160935580795402, 3.0553960601400774, 4.421160274820089,
             4.163677411862713, 5.565259910576582, 10.951736596728765,
             15.449128862121253, 31.970322758461865]
    quantiles = [0.0001, 0.001, 0.005, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25,
                 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
                 0.8, 0.85, 0.9, 0.95, 0.975, 0.995, 0.999, 0.9999]
    enforce_monotonicity(preds, quantiles)