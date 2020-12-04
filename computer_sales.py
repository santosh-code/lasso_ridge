import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder

comp_sales=pd.read_csv('C:/Users/USER/Desktop/multiple liner regression/Computer_Data.csv')
comp_sales1=comp_sales.iloc[:,1:]
label=LabelEncoder()
comp_sales1.cd=label.fit_transform(comp_sales1.cd)
comp_sales1.multi=label.fit_transform(comp_sales1.multi)
comp_sales1.premium=label.fit_transform(comp_sales1.premium)

x=comp_sales1.iloc[:,1:]
y=comp_sales1.iloc[:,0]

from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.13,normalize=True)
lsso_reg=lasso.fit(x,y)
lasso.coef_
lasso.intercept_

plt.bar(height=pd.Series(lasso.coef_),x=pd.Series(comp_sales1.columns[1:]))
pred_lasso=lasso.predict(x)
lasso.score(x,y) ###r2=0.77
np.sqrt(np.mean((pred_lasso-y)**2))
#####with multiple lambda values
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
lasso_reg=GridSearchCV(lasso,parameters,scoring='r2',cv=5)
lasso_reg.fit(x,y)
lasso_reg.best_params_
lasso_reg.best_score_###0.68

#######ridge
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=0.13,normalize=True)
ridge_reg.fit(x,y)
ridge_reg.coef_
ridge_reg.intercept_
#######with different lambda values
from sklearn.model_selection import GridSearchCV
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
cv_ridge_reg=GridSearchCV(ridge_reg,parameters,scoring='r2',cv=5)
cv_ridge_reg.fit(x,y)
cv_ridge_reg.best_params_
cv_ridge_reg.best_score_####r2=0.68
