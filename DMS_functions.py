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
