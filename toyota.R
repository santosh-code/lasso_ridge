library(readr)
toyota_c<-read.csv(file.choose())
toyota_c1<-toyota_c[,-c(1,2)]
x=model.matrix(Price~.,data =toyota_c1 )[,-1]
y=toyota_c1$Price
library(glmnet)
grid=10^seq(10,-2,length=100)
####lasso_model
lasso_reg<-glmnet(x,y,apha=1,lambda = grid)
summary(lasso_reg)
###cv_model
cv_model<-cv.glmnet(x,y,alpha=1,lambda = grid)
plot(cv_model)
log(cv_model$lambda.min)

pred<-predict(lasso_reg,s=log(cv_model$lambda.min),newx = x)
sse<-sum((pred-y)^2)
sst<-sum((y-mean(y))^2)
rsq<-1-sse/sst
rsq ####r2=0.91
#####ridge_model
ridge_model<-glmnet(x,y,alpha = 0,lambda = grid)

cv_glmnt<-cv.glmnet(x,y,alpha=0,lambda = grid)
plot(cv_glmnt)
cv_glmnt$lambda.min
log(cv_glmnt$lambda.min)
pred<-predict(ridge_model,s=log(cv_glmnt$lambda.min),newx = x)
sse<-sum((pred-y)^2)
sst<-sum((y-mean(y))^2)
rsq<-1-sse/sst
rsq

#######r2=0.91