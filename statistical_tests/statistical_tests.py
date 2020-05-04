import statsmodels as sm
from statsmodels.stats.stattools import durbin_watson


# =============================================================================
# 
# In statistics, the Durbin–Watson statistic is a test statistic used to detect the 
# presence of autocorrelation at lag 1 in the residuals (prediction errors) from a 
# regression analysis. It is named after James Durbin and Geoffrey Watson. The small 
# sample distribution of this ratio was derived by John von Neumann (von Neumann, 1941). 
# Durbin and Watson (1950, 1951) applied this statistic to the residuals 
# from least squares regressions, and developed bounds tests for the 
# null hypothesis that the errors are serially uncorrelated against the alternative 
# that they follow a first order autoregressive process. Later, John Denis Sargan 
# and Alok Bhargava developed several von Neumann–Durbin–Watson type test statistics 
# for the null hypothesis that the errors on a regression model follow a process 
# with a unit root against the alternative hypothesis that the errors follow a 
# stationary first order autoregression (Sargan and Bhargava, 1983). Note that the 
# distribution of this test statistic does not depend on the estimated regression 
# coefficients and the variance of the errors.
# =============================================================================

db = durbin_watson(train_df.SK_ID_CURR)

