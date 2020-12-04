library(readr)
library(fastDummies)
library(plyr)
computer_sales<-read.csv(file.choose())

computer_sales$cd<-as.factor(revalue(computer_sales$cd,c("yes"=1, "no"=0)))
computer_sales$multi<-as.factor(revalue(computer_sales$multi,c("yes"=1, "no"=0)))
computer_sales$premium<-as.factor(revalue(computer_sales$premium,c("yes"=1, "no"=0)))
computer_sales1<-computer_sales[,-1]
library(glmnet)
x=model.matrix(price~.,data =computer_sales1 )[,-1]
y=computer_sales1$price
####
grid=10^seq(10,-2,length=100)

######lasso
lasso_reg<-glmnet(x,y,alpha = 1,lambda = grid)
summary(lasso_reg)
#######lasso with different values of lambda
cv_lasso<-cv.glmnet(x,y,alpha=1,lambda = grid)
summary(cv_lasso)
plot(cv_lasso)
cv_lasso$lambda.min

y_a<-predict(lasso_reg,s=cv_lasso$lambda.min,newx = x)
sse<-sum((y_a-y)^2)
sst<-sum((y-mean(y))^2)
rsq<-1-sse/sst##r2=0.77
predict(lasso_reg,s=cv_lasso$lambda.min,newx = x,type = "coefficients")

########ridge
ridge_reg<-glmnet(x,y,alpha=0,lambda = grid)
summary(ridge_reg)
#####ridge_cv
cv_ridge<-cv.glmnet(x,y,alpha=0,lambda =grid )
plot(cv_ridge)
cv_ridge$lambda.min
#####r2
pred<-predict(ridge_reg,s=cv_ridge$lambda.min,newx =x)
sse<-sum((pred-y)^2)
sst<-sum((y-mean(y))^2)
r2<-1-sse/sst
r2
####r2=0.77
predict(ridge_reg,s=cv_lasso$lambda.min,newx = x,type = "coefficients")
