import numpy as np

def exp_weights(length, alpha=0.9):
    idx = np.arange(length)
    weights = alpha ** (length - 1 - idx)
    # return weights / weights.sum()
    return weights

print("exp:", exp_weights(5, alpha=0.8))
## [0.12184674 0.15230842 0.19038553 0.23798191 0.29747739]


def linear_weights(length):
    """
    线性衰减权重
    length: 时间序列长度
    """
    weights = np.arange(1, length + 1)  # 1, 2, ..., length
    # weights = weights / weights.sum()
    return weights

print("Linear:", linear_weights(5))
## [0.06666667 0.13333333 0.2   0.26666667 0.33333333]

def gaussian_weights(length, sigma=1.0):
    """
    高斯衰减权重
    length: 时间序列长度
    sigma: 标准差，越大衰减越慢
    """
    idx = np.arange(length)
    # 以最后一个点 (length-1) 为中心
    weights = np.exp(-0.5 * ((idx - (length - 1)) / sigma) ** 2)
    # weights = weights / weights.sum()
    return weights

# 示例
print("Gaussian:", gaussian_weights(2, sigma=1.0))
# 输出:[1.91330997e-04 6.33601245e-03 7.71884334e-02 3.45934558e-01, 5.70349665e-01]