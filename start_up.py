import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start_up=pd.read_csv('C:/Users/USER/Desktop/multiple liner regression/50_Startups.csv')
dum=pd.get_dummies(start_up.State)
merged=pd.concat([start_up,dum],axis=1)
merged.columns='R_Spend', 'Administration', 'Marketing Spend', 'State', 'Profit',\
    'California', 'Florida', 'NewYork'
start_up1=merged.drop(['State','NewYork'],axis=1)
start_up1=start_up1.iloc[:,[3,0,1,2,4,5,]]

########lasso
x=start_up1.iloc[:,1:]
y=start_up1.iloc[:,0]    
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.13,normalize= True)
lasso.fit(x,y )

lasso.coef_
lasso.intercept_

plt.bar(height=pd.Series(lasso.coef_),x=pd.Series(start_up1.columns[1:]))
pred_lasso=lasso.predict(x)
lasso.score(x,y)
np.sqrt(np.mean((pred_lasso-y)**2))######r2=0.95

######grid search in lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
lasso_reg=GridSearchCV(lasso,parameters,scoring='r2',cv=5)
lasso_reg.fit(x,y)
lasso_reg.best_params_
lasso_reg.best_score_

######ridge
from sklearn.linear_model import Ridge
ridge=Lasso(alpha=0.13,normalize= True)
ridge.fit(x,y )

ridge.coef_
ridge.intercept_

plt.bar(height=pd.Series(ridge.coef_),x=pd.Series(start_up1.columns[1:]))
pred_ridge=ridge.predict(x)
ridge.score(x,y)####0.95
np.sqrt(np.mean((pred_ridge-y)**2))

######grid search in lasso
from sklearn.model_selection import GridSearchCV
ridge=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
ridge_reg=GridSearchCV(lasso,parameters,scoring='r2',cv=5)
ridge_reg.fit(x,y)
ridge_reg.best_params_
ridge_reg.best_score_

