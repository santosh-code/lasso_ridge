library(readr)
library(fastDummies)
start_up<-read.csv(file.choose())
dum1<-dummy_cols(start_up$State)
start_up1<-cbind(start_up[,-4],dum1[,-c(1,2)])
start_up1<-start_up1[,c(4,1,2,3,5,6)]
colnames(start_up1)

library(glmnet)
x<-model.matrix(Profit~.,data =start_up1)[,-1]
y<-start_up1$Profit

grid<-10^seq(10,-2,length=100)
grid

###ridge regression
model_ridge<-glmnet(x,y,alpha = 0,lambda = grid)
summary(model_ridge)

###cross validation
cv_fit<-cv.glmnet(x,y,alpha=0,lambda = grid)
plot(cv_fit)
optimumlambda<-cv_fit$lambda.min
optimumlambda

y_a<-predict(model_ridge,s=optimumlambda,newx = x)
sse<-sum((y_a-y)^2)
sst<-sum((y-mean(y))^2)
rsq<-1-sse/sst
rsq
predict(model_ridge,s=optimumlambda,newx = x,type = "coefficients")

##lasso regrssion
model_lasso<-glmnet(x,y,alpha = 1,lambda = grid)
summary(model_lasso)

cv_fit<-cv.glmnet(x,y,alpha=1,lambda = grid)
plot(cv_fit)
optimumlambda<-cv_fit$lambda.min
optimumlambda
log(optimumlambda)

y_a<-predict(model_ridge,s=log(optimumlambda),newx = x)
sse<-sum((y_a-y)^2)
sst<-sum((y-mean(y))^2)
rsq<-1-sse/sst
rsq
predict(model_ridge,s=optimumlambda,newx = x,type = "coefficients")


