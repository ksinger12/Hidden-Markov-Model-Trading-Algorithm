import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def test_stationarity(time_series):
    crit_val = "1%"

    # perform ADF
    dftest = adfuller(time_series)
    dftest_crit = dftest[4]
    dftest_test_stat = dftest[0]

    # compare crit_val dict input with test stat
    if (dftest_crit[crit_val] > dftest_test_stat):
        dftest_result = True

    else:
        dftest_result = False

    # perform KPSS
    kpsstest = kpss(time_series)
    kpsstest_crit = kpsstest[3]
    kpsstest_test_stat = dftest[0]

    # compare crit_val dict input with test stat
    if (kpsstest_crit[crit_val] < kpsstest_test_stat):
        kpsstest_result = False

    else:
        kpsstest_result = True

    return dftest_result and kpsstest_result


def differencing_s(value_series, trend):
    # shift before doing difference
    rolled_series = np.roll(value_series, trend)
    if (trend > 0):
        con_cat_series = [rolled_series, value_series]  # combine the list for differencing
        value_series = np.diff(con_cat_series, axis=0)  # difference shifted list
        cut_roll = value_series[0, trend:]  # remove front overlap values (slice from front)
        return cut_roll
    elif (trend < 0):
        con_cat_series = [value_series, rolled_series]
        value_series = np.diff(con_cat_series, axis=0)
        cut_roll = value_series[0, :trend]  # remove rear overlap values (slice from rear)
        return cut_roll
    else:
        raise ValueError("This shit can't be zero")


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def auto_stationary(time_series, method='auto1', window=20):
    """
    Auto Stationarization of Data

    :param time_series: 0 by x ndarray of time series data
    :param method: preferred method of stationarization
    auto#: attempts automatic stationarization with a differencing/seasonality of #
    :param window: window size for rolling windwo transformation

    methods can be designed with the following:

    'n' normal differencing
    'l' logarithmic tranformation
    's' square root transformation
    'c' cube root transformation
    'r' rolling window (size 20)
    '#' the differencing/seasonality value, can be +/-

    example: method = 'ls2', log transform, then square transform, than seasonality differencing by 2

    :return: stationary_time_series, method
    """
    method = method.lower()

    stationary_data = []
    method_used = ''

    if 'auto' in method:
        # automatic stationary analysis

        # Strip number for differencing/seasonality
        roll = re.findall("\d+", method)[0]
        roll = int(roll[0])

        if (test_stationarity(time_series)):
            method_used = None
            return time_series, method_used

        # take log
        log = np.log(time_series)  # Perform trasnformation
        log_dif = differencing_s(log, roll)  # Take difference/seasonality
        if (test_stationarity(log_dif)):  # Test for stationarity
            method_used = str(roll) + 'l'  # If pass, return new series + method used
            return log_dif, method_used

        # take square root
        sqrt = np.sqrt(time_series)
        sqrt_dif = differencing_s(sqrt, roll)
        if (test_stationarity(sqrt_dif)):
            method_used = str(roll) + 's'
            return sqrt_dif, method_used

        # take cube root
        cube = np.cbrt(time_series)
        cube_dif = differencing_s(time_series, roll)
        if (test_stationarity(cube_dif)):
            method_used = str(roll) + 'c'
            return cube_dif, method_used

        # normal differencing
        normal_dif = differencing_s(time_series, roll)
        if (test_stationarity(normal_dif)):
            method_used = str(roll) + 'n'
            return normal_dif, method_used

        return time_series, 'error'  # could potentially raise ValueError here

    else:
        # split method entry into its steps
        order = list(method)
        try:
            roll = int(order.pop())  # pop off differencing/seasonality value
        except:
            roll = 1

        mod_series = time_series

        # for each step in the stationarization sequence perform the action
        for step in order:
            if 'n' in step:
                pass
            elif 'l' in step:
                mod_series = np.log(mod_series)
            elif 's' in step:
                mod_series = np.sqrt(mod_series)
            elif 'c' in step:
                mod_series = np.cbrt(mod_series)
            elif 'r' in step:
                mod_series = rolling_window(mod_series, window).mean(1)
            else:
                raise ValueError("The step: " + step + " is not an acceptable stationarization method")

        # perform differencing/seasonality
        if 'n' in order:
            mod_series = differencing_s(mod_series, roll)
            
        # replace zeroes with extremely small numbers 10^-5
        # mod_series = np.where(mod_series==0, 0.00001 ,mod_series)

        return mod_series, method  # return sequence
