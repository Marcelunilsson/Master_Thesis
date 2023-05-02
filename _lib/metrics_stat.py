import numpy as np
from scipy.stats import ttest_rel


def cohens_d(df, col_1, col_2):
    """_summary_

    Args:
        df (_type_): _description_
        col_1 (_type_): _description_
        col_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t1, t2 = df[col_1], df[col_2]
    stdev1, stdev2 = np.std(t1), np.std(t2)
    mean1, mean2 = np.mean(t1), np.mean(t2)
    n1, n2 = len(t1), len(t2)
    pooled_stdev = np.sqrt(
        ((n1-1)*stdev1**2 + (n2-1)*stdev2**2) / (n1 + n2 - 2)
    )
    return (mean1 - mean2) / pooled_stdev


def ttest_p_val(df, col_1, col_2):
    t1, t2 = df[col_1], df[col_2]
    return ttest_rel(t1, t2).pvalue


def RCI(df, col_1, col_2):
    """_summary_

    Args:
        df (_type_): _description_
        col_1 (_type_): _description_
        col_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t1, t2 = df[col_1], df[col_2]
    r = np.corrcoef(t1, t2)[1, 0]
    stdev1, stdev2 = np.std(t1), np.std(t2)
    deltas = [v2 - v1 for v1, v2 in zip(t1, t2)]
    b = (np.sqrt((stdev1*np.sqrt(1-r))**2 + (stdev2*np.sqrt(1-r))**2))
    RCIs = [delta/b for delta in deltas]
    return RCIs, [abs(rci) >= 1.96 for rci in RCIs], b*1.96, r


def SID(df, col_1, col_2):
    """_summary_

    Args:
        df (_type_): _description_
        col_1 (_type_): _description_
        col_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t1, t2 = df[col_1], df[col_2]
    deltas = [v2 - v1 for v1, v2 in zip(t1, t2)]
    s_dif = np.std(deltas)
    return [delta / s_dif for delta in deltas]
