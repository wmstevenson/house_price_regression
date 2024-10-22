import numpy as np
import pandas as pd


def compute_training_mean(train_df: pd.DataFrame) -> np.ndarray:
    """
    Computes the mean of each column in the DataFrame and returns it as an array.

    Parameters:
        train_df (pd.DataFrame): The input training DataFrame.

    Returns:
        np.ndarray: An array containing the mean of each numeric column.
    """

    return train_df.mean().values


def compute_training_standard_deviation(train_df: pd.DataFrame) -> np.ndarray:
    """
    Computes the standard deviation of each column and returns it as an array.

    Parameters:
        train_df (pd.DataFrame): The input training DataFrame.

    Returns:
        np.ndarray: An array containing the standard deviation of each numeric column.
    """

    return train_df.std().values


def z_score_scaling(
    X: pd.DataFrame, standard_deviation: np.ndarray, mean: np.ndarray) -> pd.DataFrame:
    """

    Scales the DataFrame columns using z-score normalization.

    Parameters:
        X (pd.DataFrame): The input DataFrame to scale.
        standard_deviation (np.ndarray): The standard deviation of each column.
        mean (np.ndarray): The mean of each column.

    Returns:
        pd.DataFrame: A new DataFrame with z-score normalized values.

    """

    #  mean and standard_deviation are 1D array with one value for each column in X therefore pandas will excecute the operations column-wise
    return (X - mean) / standard_deviation


def compute_cost(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Computes the average squared error based on all data using vectorized operations.
    
    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target values.
        w (np.ndarray): Weight vector.
        b (float): Bias term.
    
    Returns:
        float: The computed cost (mean squared error).
    """
    m = X.shape[0] 
    
    # predictions will be an NumPy array with a prediction for each row in the dataframe
    predictions = (X.values @ w) + b
    
    # predictions and y are both 1 D NumPy arrays and subtraction will occur element wise
    errors = predictions - y

    # np.square() will square each element in a NumPy array
    squared_errors = np.square(errors)
    
    # final cost is the sum of errors divided by 2 times the number of examples
    cost = 1 / (2 * m) * np.sum(squared_errors)
    
    return cost


import numpy as np
import pandas as pd

def compute_gradient(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """
    Computes the gradients of the cost function with respect to weights and bias.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target values.
        w (np.ndarray): Weight vector.
        b (float): Bias term.
    Returns:
    tuple[np.ndarray, float]
        A tuple containing:
            - Gradient with respect to bias (dj_db)
            - Gradient with respect to weights (dj_dw)
    """
    m = X.shape[0]

    # predictions and errors are both 1 D NumPy arrays with an entry for each row in the DataFrame
    predictions = X.values @ w + b 
    errors = predictions - y  

    # dj_dw is a 1 D NumPy array with an entry for each weight parameter in the model
    dj_dw = (1/m) * (X.values.T @ errors)  
    dj_db = (1/m) * np.sum(errors) 

    return dj_db, dj_dw




def gradient_descent(
    X: pd.DataFrame,
    y: np.ndarray,
    b_initial: float,
    w_initial: np.ndarray,
    iterations: int,
    learning_rate: float,
    cost_function,
    gradient_function,
) -> tuple[float, np.array, np.array]:
    """
    Uses gradient descent to find parameter values that minimize the cost function.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target values.
        b_initial float: Initial bias value.
        w_initial np.ndarray: Initial weights (n x 1).
        iterations int: Number of iterations for gradient descent.
        learning_rate float: Learning rate for gradient descent.
        cost_function function: Function to calculate the cost.
    gradient_function function: Function to compute the gradients.

    Returns:
    tuple[float, np.ndarray, np.ndarray]
        A tuple containing:
            - Final bias value (b)
            - Final weights (w)
            - History of cost values over iterations (cost_history)
    """

    m, n = X.shape
    cost_history = np.zeros(iterations)
    b = b_initial
    w = w_initial

    for iteration in range(iterations):
        cost_history[iteration] = cost_function(X, y, w, b)
        dj_db, dj_dw = gradient_function(X, y, w, b)
        b = b - learning_rate * dj_db        
        w = w - learning_rate * dj_dw

    return b, w, cost_history



def compute_predictions(
    X: pd.DataFrame, b_final: float, w_final: np.ndarray) -> np.ndarray:
    """
    Uses final parameter values to calculate model predictions.

    Parameters:
    X : pd.DataFrame
        Input feature data (m x n), where m is the number of samples and n is the number of features.
    b_final : float
        Final bias value.
    w_final : np.ndarray
        Final weights (n x 1).

    Returns:
    np.ndarray: Predictions for each input sample (m x 1).
    """
    # Calculate predictions using vectorized operations
    predictions = X.values @ w_final + b_final  # Matrix multiplication and addition

    return predictions



def compute_r_squared(
    y_true: np.ndarray, y_pred: np.ndarray, y_mean: np.ndarray) -> float:
    """
    Computes the R-squared score of the model based on its predictions.

    Parameters:
    y_true np.ndarray: The true target values.
    y_pred np.ndarray: The predicted target values.

    Returns:
        float: The R-squared score, a measure of how well the predicted values approximate the true values.
    """
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)

    return 1 - (rss / tss)





#------------------------------------------------------------------------------------------------------------------------------





"""
These functions were itterated on many times and it is only for reference that I below include my initial implementation of each of these function:


def compute_training_mean(train_df: pd.DataFrame):

    m, n = train_df.shape
    column_means = np.zeros(n)
    for j in range(n):
        column_name = train_df.columns[j]
        tmp_array = train_df[column_name].values
        column_means[j] = np.mean(tmp_array)
        
    return column_means

   
def compute_training_standard_deviation(train_df: pd.DataFrame):

    m, n = train_df.shape
    column_means = np.zeros(n)
    for j in range(n):
        column_name = train_df.columns[j]
        tmp_array = train_df[column_name].values
        column_means[j] = np.std(tmp_array)
        
    return column_means


 def z_score_scaling(X: pd.DataFrame, standard_deviation: np.ndarray, mean: np.ndarray) -> pd.DataFrame:
    
    df_copy = X.copy()
    m, n = df_copy.shape
    for j in range(n):
        column_name = df_copy.columns[j]
        df_copy[column_name] = (
            df_copy[column_name].values - mean[j]
        ) / standard_deviation[j]

    return df_copy

 
def compute_cost(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, b: float) -> float:

    m, n = X.shape
    temp_sum = 0
    for i in range(m):
        temp_sum += (np.dot(w, X.iloc[i].values) + b - y[i]) ** 2

    return 1 / (2 * m) * temp_sum


def compute_gradient(
    X: pd.DataFrame, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:

    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        error = np.dot(w, X.iloc[i].values) + b - y[i]
        dj_db += error
        for j in range(n):
            dj_dw[j] += error * X.iloc[i, j]

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_db, dj_dw

    
def gradient_descent(
    X: pd.DataFrame,
    y: np.ndarray,
    b_initial: float,
    w_initial: np.ndarray,
    iterations: int,
    learning_rate: float,
    cost_function,
    gradient_function,
) -> tuple[float, np.array, np.array]:


    m, n = X.shape
    cost_history = np.zeros(iterations)
    b = b_initial
    w = w_initial

    for iteration in range(iterations):
        cost_history[iteration] = cost_function(X, y, w, b)
        dj_db, dj_dw = gradient_function(X, y, w, b)
        b = b - learning_rate * dj_db        
        for j in range(n):
            w[j] = w[j] - learning_rate * dj_dw[j]

    return b, w, cost_history


def compute_predictions(
    X: pd.DataFrame, b_final: float, w_final: np.ndarray) -> np.ndarray:
 
    m, n = X.shape
    predictions = np.zeros(m)
    for i in range(m):
        f_wb = np.dot(w_final, X.iloc[i]) + b_final
        predictions[i] = f_wb

    return predictions


"""
