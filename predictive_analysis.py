#Linear Regression
from sklearn.linear_model import LinearRegression

def lr(X_train, X_test, y_train, y_test, pos, posX):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    pos['PredictedSalary(LR)'] = lr.predict(posX)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)
    print('\n', 'R2 Score:', lr.score(X_test, y_test))
    
#Random Forest
from sklearn.ensemble import RandomForestRegressor

def rf(X_train, X_test, y_train, y_test, i, pos, posX):
    sc = StandardScaler()
    model = RandomForestRegressor(n_jobs=-1, random_state=20)
    estimators = np.arange(1, 200, 10)
    scores = []
    for n in estimators:
        model.set_params(n_estimators=n)
        model.fit(X_train, y_train) #random_state to keep stable results
        scores.append(model.score(X_test, y_test))
    plt.title("Effect of n_estimators")
    plt.xlabel("n_estimator")
    plt.ylabel("score")
    plt.plot(estimators, scores)
    
    reg = RandomForestRegressor(n_estimators=i, random_state=0)
    reg.fit(X_train, y_train)
    
    pos['PredictedSalary(RF)'] = reg.predict(posX)
    
    y_pred = reg.predict(X_test)
    
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('\n', 'R2 Score:', reg.score(X_test, y_test))
    
#SVR
from sklearn.svm import SVR

def svr_sal(X_train, X_test, y_train, y_test, pos, posX):
    svr_lin = SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e8, degree = 1, gamma = 0.01)
    svr_rbf = SVR(kernel = 'rbf', C=1e8, gamma= 0.01)
    
    svr_lin.fit(X_train, y_train)
    svr_poly.fit(X_train, y_train)
    svr_rbf.fit(X_train, y_train)
    
    svrbf_pred = pd.DataFrame(svr_rbf.predict(X_test))
    svlin_pred = pd.DataFrame(svr_lin.predict(X_test))
    svpoly_pred = pd.DataFrame(svr_poly.predict(X_test))
    
    pos['PredictedSalary(SVR-RBF)'] = svr_rbf.predict(posX)
    pos['PredictedSalary(SVR-Linear)'] = svr_lin.predict(posX)
    pos['PredictedSalary(SVR-Polynomial)'] = svr_poly.predict(posX)
    
    print('RBF Model')
    print('\n', 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, svrbf_pred)
    print('\n', 'R2 Score:', svr_rbf.score(X_test, y_test))
          
    print('\n')
    
    print('Linear Model')
    print('\n', 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, svlin_pred))
    print('\n', 'R2 Score:', svr_lin.score(X_test, y_test))
    
    print('\n')
    
    print('Polynomial Model')
    print('\n', 'Mean Absolute Error:', metrics.mean_absolute_error(y_test, svpoly_pred))
    print('\n', 'R2 Score:', svr_poly.score(X_test, y_test))
