from statistics import linear_regression, correlation
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_suitable_regression(dataframe, min_var=10, n_col_crit=5):
    '''
    Check Suitable Regression
    -------------------------
    Checks whether a dataframe contains enough numeric variables with
    sufficient variance to perform regression analysis.
    '''

    df_num = dataframe.select_dtypes(include='number', exclude='bool')

    suitable_columns = []
    for col in df_num.columns:
        if len(set(df_num[col])) >= min_var:
            suitable_columns.append(col)

    if len(suitable_columns) >= n_col_crit:
        print('Enough suitable columns')
        return suitable_columns
    else:
        print('Insufficient suitable columns, choose another dataset')
        return []

def bivariate_regression(x, y):
    '''
    Bivariate Regression
    --------------------
    Calculates a simple linear regression between x and y.
    '''

    model = linear_regression(x, y)

    b1 = model[0]
    b0 = model[1]

    r2 = correlation(x, y)**2

    results = {
        'constant': b0,
        'gradient': b1,
        'det_coeff': r2
    }

    return results

# ----------------------------
# Gaussian Elimination
# ----------------------------
# Improvements in this implementation:
# - Forward elimination uses list comprehensions for clarity.
# - Back substitution is simplified using sum + zip.
# - Makes a copy of the input matrix to avoid modifying the original.
# - Raises a clear error if the matrix is singular.
# - Docstring clearly explains purpose and output.

def gaussian_elimination(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian elimination.
    Returns the solution vector x.
    """
    n = len(A)
    # Make a copy to avoid modifying the original
    M = [row[:] + [b_val] for row, b_val in zip(A, b)]

    # Forward elimination
    for i in range(n):
        pivot = M[i][i]
        if pivot == 0:
            raise ValueError("Singular matrix")
        M[i] = [elem / pivot for elem in M[i]]  # normalize pivot row
        for k in range(i + 1, n):
            factor = M[k][i]
            M[k] = [M[k][j] - factor * M[i][j] for j in range(n + 1)]

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i + 1, n))

    return x

# ----------------------------
# Multivariate Regression
# ----------------------------
# Improvements in this implementation:
# - Covariance calculation is more readable.
# - Avoids modifying input lists directly (safer).
# - Predicted values are computed using a clean list comprehension.
# - Docstrings are more descriptive, explaining inputs and outputs.
# - Works seamlessly with gaussian_elimination() to calculate regression coefficients.
# - R-squared calculation included for model evaluation.

def multivariate_regression(multi_x, y):
    """
    Performs multivariate regression with multiple predictors.
    Returns a dictionary with:
        - 'constant': intercept
        - 'gradients': list of coefficients for predictors
        - 'det_coeff': R-squared
    """
    k = len(multi_x)          # number of predictors
    n = len(multi_x[0])       # number of observations
    y_bar = sum(y) / n

    # Means of predictors
    x_bars = [sum(x) / n for x in multi_x]

    # Covariance matrix
    cov_matrix = [
        [
            sum((multi_x[i][m] - x_bars[i]) * (multi_x[j][m] - x_bars[j]) for m in range(n)) / n
            for j in range(k)
        ]
        for i in range(k)
    ]

    # Constant terms
    const_list = [
        sum((multi_x[i][m] - x_bars[i]) * (y[m] - y_bar) for m in range(n)) / n
        for i in range(k)
    ]

    # Solve for coefficients
    coeffs = gaussian_elimination(cov_matrix, const_list)

    # Intercept
    b0 = y_bar - sum(c * xb for c, xb in zip(coeffs, x_bars))

    # Predicted values
    y_hat = [b0 + sum(coeffs[j] * multi_x[j][i] for j in range(k)) for i in range(n)]

    # R-squared
    SS_t = sum((y_i - y_bar) ** 2 for y_i in y)
    SS_r = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    r2 = 1 - SS_r / SS_t

    return {
        'constant': b0,
        'gradients': coeffs,
        'det_coeff': r2
    }

def calculate_vif(dataframe):
    """
    Variance Inflation Factor
    -------------------------
    Detects multicollinearity between independent variables.
    Returns a list of dicts with variable names and VIF values.
    """
    
    X = dataframe.select_dtypes(include='number')
    vif_list = []

    for i, col in enumerate(X.columns):
        vif_value = variance_inflation_factor(X.values, i)
        vif_list.append({'variable': col, 'VIF': vif_value})

    return vif_list

