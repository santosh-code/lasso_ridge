import pandas as pd
import matplotlib.pylab as plt
import numpy as np
toyota=pd.read_csv('C:/Users/USER/Desktop/ToyotaCorolla.csv',encoding= 'unicode_escape')
toyota1=toyota.iloc[:,[2,3,6,8,12,13,14,15,16,17]]
x=toyota1.iloc[:,1:]
y=toyota1.iloc[:,0]
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=1,normalize=True)
lasso.fit(x,y)
lasso.coef_
lasso.intercept_
plt.bar(height=pd.Series(lasso.coef_),x=pd.Series(toyota1.columns[1:]))
pred_lasso=lasso.predict(x)
lasso.score(x,y)
np.sqrt(np.mean((pred_lasso-y)**2))
######
from sklearn.model_selection import GridSearchCV
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
cv_lasso=GridSearchCV(lasso,parameters,scoring='r2',cv=5)
cv_lasso.fit(x,y)
cv_lasso.best_params_
cv_lasso.best_score_

###########ridge_reg
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=1,normalize=True)
ridge.fit(x,y)
ridge.coef_
ridge.intercept_

plt.bar(height=pd.Series(ridge.coef_),x=pd.Series(toyota1.columns[1:]))
pred_ridge=ridge.predict(x)
ridge.score(x,y)
np.sqrt(np.mean((pred_ridge-y)**2))
########optimal lambda value
from sklearn.model_selection import GridSearchCV
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
cv_ridge=GridSearchCV(ridge,parameters,scoring='r2',cv=5)
cv_ridge.fit(x,y)
cv_lasso.best_params_
cv_lasso.best_score_
