# Courtesy of Cédric Colas
# https://github.com/ccolas/rl_stats

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, rankdata, median_test, wilcoxon

tests_list = ['t-test', "Welch t-test", 'Mann-Whitney', 'Ranked t-test', 'permutation']


def run_permutation_test(all_data, n1, n2):
    np.random.shuffle(all_data)
    data_a = all_data[:n1]
    data_b = all_data[-n2:]
    return data_a.mean() - data_b.mean()


def run_test(test_id, data1, data2, alpha=0.05):
    """
    Compute tests comparing data1 and data2 with confidence level alpha
    :param test_id: (str) refers to what test should be used
    :param data1: (np.ndarray) sample 1
    :param data2: (np.ndarray) sample 2
    :param alpha: (float) confidence level of the test
    :return: (bool) if True, the null hypothesis is rejected
    """
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    n1 = data1.size
    n2 = data2.size

    if all(data1 == data2):
        return False

    if test_id == 't-test':
        _, p = ttest_ind(data1, data2, equal_var=True)
        return p < alpha

    elif test_id == "Welch t-test":
        _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha

    elif test_id == 'Mann-Whitney':
        _, p = mannwhitneyu(data1, data2, alternative='two-sided')
        return p < alpha

    elif test_id == 'Wilcoxon':
        _, p = wilcoxon(data1, data2, correction=True, alternative='two-sided', zero_method="pratt")
        return p < alpha

    elif test_id == 'Ranked t-test':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        ranks = rankdata(all_data)
        ranks1 = ranks[: n1]
        ranks2 = ranks[n1:n1 + n2]
        assert ranks2.size == n2
        _, p = ttest_ind(ranks1, ranks2, equal_var=True)
        return p < alpha

    elif test_id == 'permutation':
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        delta = np.abs(data1.mean() - data2.mean())
        num_samples = 1000
        estimates = []
        for _ in range(num_samples):
            estimates.append(run_permutation_test(all_data.copy(), n1, n2))
        estimates = np.abs(np.array(estimates))
        diff_count = len(np.where(estimates <= delta)[0])
        return (1.0 - (float(diff_count) / float(num_samples))) < alpha

    else:
        raise NotImplementedError
