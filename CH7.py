# coding=utf-8
from __future__ import division
import math, random


if __name__ == "__main__":
    # coding=utf-8
    #from __future__ import division
    #import math, random
    #第一題 假設檢定
    #設定mu = 98 , sigma = 10
    mu_0, sigma_0 = 98,10
    print "mu_0", mu_0
    print "sigma_0", sigma_0


    # 常態函數分布累積method
    def normal_cdf(x, mu=0, sigma=1):
        return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

    #逼近Z值method
    def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
        """find approximate inverse using binary search"""
        # if not standard, compute standard and rescale
        if mu != 0 or sigma != 1:
            return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

        low_z, low_p = -10.0, 0  # normal_cdf(-10) is (very close to) 0
        hi_z, hi_p = 10.0, 1  # normal_cdf(10)  is (very close to) 1
        while hi_z - low_z > tolerance:
            mid_z = (low_z + hi_z) / 2  # consider the midpoint
            mid_p = normal_cdf(mid_z)  # and the cdf's value there
            if mid_p < p:
                # midpoint is still too low, search above it
                low_z, low_p = mid_z, mid_p
            elif mid_p > p:
                # midpoint is still too high, search below it
                hi_z, hi_p = mid_z, mid_p
            else:
                break

        return mid_z

    #求Z值的method
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)

    #第一個問題
    # a = 5% = 0.95機率
    print "normal_lower_bound(0.95, mu_0, sigma_0)", normal_lower_bound(0.95, mu_0, sigma_0)
    print

    #第一題第二個問題
    print "normal_lower_bound(0.9, mu_0, sigma_0) = ", normal_lower_bound(0.9, mu_0, sigma_0)

    #第一題第三個問題，求顯著性
    #normal_probability_below = normal_cdf
    print "normal_probability_belowx,mu,sigma) = ", normal_cdf(81.55,98, 10)
    print ("")

    #第二題 信賴區間

    #右尾檢定
    def normal_upper_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z <= z) = probability"""
        return inverse_normal_cdf(probability, mu, sigma)

    #左尾檢定
    def normal_lower_bound(probability, mu=0, sigma=1):
        """returns the z for which P(Z >= z) = probability"""
        return inverse_normal_cdf(1 - probability, mu, sigma)

    #雙尾檢定
    def normal_two_sided_bounds(probability, mu=0, sigma=1):
        """returns the symmetric (about the mean) bounds
        that contain the specified probability"""
        tail_probability = (1 - probability) / 2

        # upper bound should have tail_probability above it
        upper_bound = normal_lower_bound(tail_probability, mu, sigma)

        # lower bound should have tail_probability below it
        lower_bound = normal_upper_bound(tail_probability, mu, sigma)

        return lower_bound, upper_bound

    print "Confidence Intervals"
    print "normal_two_sided_bounds(信賴水準,平均數,標準差) = ",normal_two_sided_bounds(0.95,4.015,0.02)

    #第三題 A/B Testing

    #計算p(期望值/平均數) sigma(標準差)
    def estimated_parameters(N, n):
        p = n / N
        sigma = math.sqrt(p * (1 - p) / N)
        return p, sigma

    #計算兩者差距
    def a_b_test_statistic(N_A, n_A, N_B, n_B):
        p_A, sigma_A = estimated_parameters(N_A, n_A)
        p_B, sigma_B = estimated_parameters(N_B, n_B)
        return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

    z = a_b_test_statistic(1500, 400, 1400, 350)
    print "PB-PA間的差距",(z)

    #計算p-value值

    normal_probability_below = normal_cdf


    # it's above the threshold if it's not below the threshold
    def normal_probability_above(lo, mu=0, sigma=1):
        return 1 - normal_cdf(lo, mu, sigma)


    def two_sided_p_value(x, mu=0, sigma=1):
        if x >= mu:
            # if x is greater than the mean, the tail is above x
            return 2 * normal_probability_above(x, mu, sigma)
        else:
            # if x is less than the mean, the tail is below x
            return 2 * normal_probability_below(x, mu, sigma)

    print "檢定兩者之間是否有差異",(two_sided_p_value(z))