from statistics import linear_regression, correlation

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

def gaussian_elimination(A, b):
    '''
    Gaussian Elimination
    --------------------
    Solves a system of linear equations.
    '''

    n = len(A)

    for i in range(n):
        A[i].append(b[i])

    for i in range(n):

        pivot = A[i][i]

        if pivot == 0:
            raise ValueError("Singular matrix")

        for j in range(i, n + 1):
            A[i][j] /= pivot

        for k in range(i + 1, n):
            factor = A[k][i]

            for j in range(i, n + 1):
                A[k][j] -= factor * A[i][j]

    x = [0] * n

    for i in range(n - 1, -1, -1):

        x[i] = A[i][n]

        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]

    return x

def multivariate_regression(multi_x, y):
    '''
    Multivariate Regression
    -----------------------
    Calculates regression coefficients for multiple predictors.
    '''

    k = len(multi_x)
    n = len(multi_x[0])

    y_bar = sum(y) / n

    x_bars = []
    cov_matrix = [[0] * k for _ in range(k)]

    for i in range(k):

        x_bar = sum(multi_x[i]) / n
        x_bars.append(x_bar)

        for j in range(i, k):

            x_bar_j = sum(multi_x[j]) / n

            cov = sum(
                (multi_x[i][m] - x_bar) *
                (multi_x[j][m] - x_bar_j)
                for m in range(n)
            ) / n

            cov_matrix[i][j] = cov
            cov_matrix[j][i] = cov

    coeff_matrix = []
    const_list = []

    for i in range(k):

        row = []

        for j in range(k):
            row.append(cov_matrix[j][i])

        coeff_matrix.append(row)

        const = sum(
            (multi_x[i][m] - x_bars[i]) *
            (y[m] - y_bar)
            for m in range(n)
        ) / n

        const_list.append(const)

    coeffs = gaussian_elimination(coeff_matrix, const_list)

    b0 = y_bar - sum(coeffs[i] * x_bars[i] for i in range(k))

    y_hat = []

    for i in range(n):

        prediction = b0

        for j in range(k):
            prediction += coeffs[j] * multi_x[j][i]

        y_hat.append(prediction)

    SS_t = sum((y_i - y_bar) ** 2 for y_i in y)

    SS_r = sum((y[i] - y_hat[i]) ** 2 for i in range(n))

    r2 = 1 - SS_r / SS_t

    results = {
        'constant': b0,
        'gradients': coeffs,
        'det_coeff': r2
    }

    return results

def calculate_vif(dataframe):
    '''
    Variance Inflation Factor
    -------------------------
    Detects multicollinearity between independent variables.
    '''

    X = dataframe.select_dtypes(include='number')

    vif_data = pd.DataFrame()

    vif_data["variable"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(len(X.columns))
    ]

    return vif_data

