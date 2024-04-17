

from statsmodels.tsa.arima_model import SARIMAX
from sklearn.metrics import mean_squared_error
from datetime import datetime

def SARIMAX_hyperparameter_tune(X_train, X_test, y_train, y_test, Pmax=3, Qmax=3, Dmax=3, pmax=5, qmax=5, dmax=5, S1 = 0, S2 = 3, S3 = 12):
    """
    This function attempts to find the best hyperparameters for a SARIMAX model by tuning the parameters P, D, Q, p, q, d, and S.

    Parameters:
    X_train (DataFrame): The training data for the exogenous variables.
    X_test (DataFrame): The testing data for the exogenous variables.
    y_train (Series): The training data for the response variable.
    y_test (Series): The testing data for the response variable.
    Pmax (int, optional): The maximum number of autoregressive terms to consider for the seasonal component. Defaults to 3.
    Qmax (int, optional): The maximum number of moving average terms to consider for the seasonal component. Defaults to 3.
    Dmax (int, optional): The maximum number of differences terms to consider for the seasonal component. Defaults to 3.
    pmax (int, optional): The maximum number of autoregressive terms to consider for the non-seasonal component. Defaults to 5.
    qmax (int, optional): The maximum number of moving average terms to consider for the non-seasonal component. Defaults to 5.
    dmax (int, optional): The maximum number of differences terms to consider for the non-seasonal component. Defaults to 5.
    S1 (int, optional): The starting value for S in the range of values to try for the seasonal component. Defaults to 0.
    S2 (int, optional): The ending value for S in the range of values to try for the seasonal component. Defaults to 3.
    S3 (int, optional): The step size for S in the range of values to try for the seasonal component. Defaults to 12.

    Base on code by: Joseph Nelson, Justin Pounders, Matt Brems in General Assembly DSI curriculm.

    Returns:
    None

    """
    best_mse = 99 * (10 ** 16)
    final_P = 0
    final_D = 0
    final_Q = 0
    final_S = 0
    final_p = 0
    final_d = 0
    final_q = 0


    for P in range(Pmax):
        for Q in range(Qmax):
            for D in range(Dmax):
                for p in range(pmax): 
                    for q in range(qmax):
                        for d in range(dmax):                            
                            for S in range(S1,S2,S3):
                                try:
                                    print(f'Attempting to fit SARIMAX({p},{d},{q})x({P},{D},{Q},{S})')
                                    # Instantiate SARIMAX model.
                                    sarimax = SARIMAX(endog = y_train[1:],
                                                    order = (p, d, q),              # (p, d, q)
                                                    seasonal_order = (P, D, Q, S),  # (P, D, Q, S)
                                                    exog = X_train['precip_lag_1'].dropna()) 
                
                                    # Fit SARIMAX model.
                                    model = sarimax.fit(disp=False)

                                    # Generate predictions based on test set.
                                    preds = model.predict(start=datetime(1979,9,30),
                                                        end=datetime(1982,5,31),
                                                        exog=X_test[['precip_lag_1']].dropna())

                                    # Evaluate predictions.
                                    print(f'The MSE for ({p},{d},{q})x({P},{D},{Q},{S}) is: {mean_squared_error(y_test[1:len(y_test)-1], preds)}')

                                    # Save for final report.
                                    if best_mse > mean_squared_error(y_test[1:len(y_test)-1], preds):
                                        best_mse = mean_squared_error(y_test[1:len(y_test)-1], preds)
                                        final_P = P
                                        final_D = D
                                        final_Q = Q
                                        final_S = S
                                        final_p = p
                                        final_d = d
                                        final_q = q

                                except:
                                    pass
                print(f'Our model that minimizes MSE on the testing data is the SARIMAX({final_p}, {final_d}, {final_q})x({final_P},{final_D},{final_Q},{final_S}).')
                print(f'This model has an MSE of {best_mse}.')
    return preds
