from statistics import linear_regression, correlation

def check_suitable_regression(dataframe, min_var=10, n_col_crit=5):
    '''
    Check Suitable Regression
    -------------------------
    This function checks if a given dataframe might be suitable for regression analysis. 
    With 'suitable' it is meant that there are at least a given number (n_col_crit) columns 
    that have numeric values, and in each there are at least a given number (min_var) 
    of different values.
    '''
    
    # select only the numeric fields
    df_num = dataframe.select_dtypes(include='number', exclude='bool')

    # check for each if it has at least min_var different scores
    suitable_columns = []
    for col in df_num.columns:
        if len(set(df_num[col])) >= min_var:
            suitable_columns.append(col)

    if len(suitable_columns) >= n_col_crit:
        print('enough suitable columns')
        return suitable_columns
    else:
        print('insufficient suitable columns, need different dataset')
        return []

def bivariate_regression(x, y):
    '''
    Peter Hates Libraries - Bivariate Regression

    input must be a Python list without missing values and equal length
    '''
    model = linear_regression(x, y)
    b0 = model[1]
    b1 = model[0]

    r2 = correlation(x, y)**2
    
    results = {'constant':b0, 'gradient':b1, 'det coeff':r2}
    return  results

def gaussian_elimination(A, b):
    n = len(A)
    
    # Augment the matrix with the constants
    for i in range(n):
        A[i].append(b[i])
    
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Singular matrix detected")
        for j in range(i, n + 1):
            A[i][j] /= pivot
        
        # Make the elements below the diagonal 0
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n + 1):
                A[k][j] -= factor * A[i][j]
    
    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
    
    return x

def multivariate_regression(multi_x, y):
    '''
    Peter Hates Libraries - Multivariate Regression Function

    input must be a Python list without missing values and equal length
    '''
    #some basic measures (number of input features, size and average of y)
    k = len(multi_x)
    n = len(multi_x[0])
    y_bar = sum(y)/n

    # determine all possible covariances and average of each x
    x_bars = []
    cov_matrix = [[0] * k for _ in range(k)]
    for i in range(0, k):
        x1_bar = sum(multi_x[i])/n
        x_bars.append(x1_bar)
        for j in range(i, k):
            x2_bar = sum(multi_x[j])/n 
            cov_matrix[i][j] = sum([(multi_x[i][m] - x1_bar)*(multi_x[j][m] - x2_bar) for m in range(n)])/n
            cov_matrix[j][i]= cov_matrix[i][j]

    #setup the system of equations
    coeff_matrix = []
    const_list = []
    for i in range(0, k):
        new_row = []
        for j in range(0, k):
            new_row.append(cov_matrix[j][i])
        coeff_matrix.append(new_row)
    
        const_list.append(sum([(multi_x[i][m] - x1_bar)*(y[m] - y_bar) for m in range(n)])/n)

    #solve the system
    coeffs = gaussian_elimination(coeff_matrix, const_list)

    #find the constant
    b0 = y_bar - sum([coeffs[i]*x_bars[i] for i in range(k)])

    #calculate predicted values
    y_hat = []
    for i in range(n):
        y_hat.append(b0 + sum([coeffs[j]*multi_x[j][i] for j in range(k)]))

    #determine determination coefficient
    SS_t = sum([(y_i - y_bar)**2 for y_i in y])
    SS_r = sum([(y[i] - y_hat[i])**2 for i in range(n)])
    r2 = 1 - SS_r/SS_t
    #results
    results = {'constant':b0, 'gradients':coeffs, 'det coeff':r2}
    
    return  results

