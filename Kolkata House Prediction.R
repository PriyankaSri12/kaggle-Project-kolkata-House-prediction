#--------------------------------------------------
#======================================================

library(tidyr)
library(dplyr)
library(ggplot2)
library(mltools)
library(data.table)
library(caret)
library(pROC)
library(tibble)
library(purrr)
library(stringr)

#========================================================
#-------------------------------------------------------

train_1<-read.csv("D:\\R Programme Folder\\Kaggle\\Kolkata_House_Price_Train_v2.csv")
test_1<-read.csv("D:\\R Programme Folder\\Kaggle\\Kolkata_House_Price_Test.csv")
sample_1<-read.csv("D:\\R Programme Folder\\Kaggle\\Kolkata_House_Price_Sample.csv")


head(train_1)
dim(train_1)

head(test_1)
dim(test_1)

summary(train_1)
dim(train_1)

summary(test_1)
dim(test_1)

#Creating copy of original dataset
h=train_1
ht=test_1

#24 value seems to be erraneous entry in the test data of RAD, which has only 1 to 8 entries in train data.
#Hence substituting 24 with the maximum entry of 8 in train data

#Replacing 24 value with 8
ht['RAD'][ht['RAD']==24]<-8
head(ht$RAD,102)

#Checking for Missing or Null values in Data
sum(is.na(h))
sum(is.na(ht))

#-----------Exploratory Data Analysis--------------------------
#==============================================================

str(h)

#Outlier treatment forfew variables

#Variable LSTAT
Q<-quantile(h$LSTAT,probs = c(.25, .75),na.rm = FALSE)
iqr<-IQR(h$LSTAT)
up<-Q[2]+1.5*iqr  #Upper Range
low<-Q[1]-1.5*iqr  #Lower Range

#Replacing value with Uppar Quartile
h['LSTAT'][h['LSTAT']>up]<-up

#Variable INDUS
Q<-quantile(h$INDUS,probs =c (.25, .75),na.rm =FALSE)
iqr<-IQR(h$INDUS)
up<-Q[2]+1.5*iqr  #Upper Range

#Replacing value with Upper Quartile
h['INDUS'][h['INDUS']>up]<-up


#----Bivariate analysis of data between dependent and independent variables
#==========================================================================
install.packages("ggpubr")
library(ggpubr)
ggscatter (data=h,
          x = "CRIM",
          y = "MEDV",
          cor.coef = TRUE,
          
        title ="MEDV VS CRIM",
          xlab="CRIM",
          ylab="MEDV",
          col='red')



ggscatter(data = h,
          x="ZN",
          y="MEDV",
          cor.coef = TRUE,
          title="MEDV VD ZN",
          xlab ="ZN",
          ylab ="MEDV",
          col='red')



ggscatter(data = h,
          x="TAX",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS TAX",
          xlab = "TAX",
          ylab = "MEDV",
          col='red')


ggscatter(data = h,
          x="nitric.oxides.concentration",
          y="MEDV",
          cor.coef = TRUE,
          title = "nitric.oxides.concentration VS MEDV",
          xlab = "nitric.oxides.concentration",
          ylab = "MEDV",
          col='red')


ggscatter(data = h,
          x="INDUS",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS INDUS",
          xlab = "INDUS",
          ylab = "MEDV",
          col='red')


ggscatter(data = h,
          x="AGE",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS AGE",
          xlab = "AGE",
          ylab = "MEDV",
          col='red')

ggscatter(data = h,
          x="DIS",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS DIS",
          xlab = "DIS",
          ylab = "MEDV",
          col='red')      


ggscatter(data = h,
          x="PTRATIO",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS PTRATIO",
          xlab = "PTRATIO",
          ylab = "MEDV",
          col='red')

ggscatter(data=h,
          x="B",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS B",
          xlab = "B",
          ylab = "MEDV",
          col='red')

ggscatter(data=h,
          x="LSTAT",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS LSTAT",
          xlab = "LSTAT",
          ylab = "MEDV",
          col='red')


ggscatter(data=h,
          x="X.rooms.dwelling",
          y="MEDV",
          cor.coef = TRUE,
          title = "MEDV VS X.rooms.dwelling",
          xlab = "X.rooms.dwelling",
          ylab = "MEDV",
          col='red')

#There are independent variables with significant linear correlation values
#Hence multilinear model can be used to predict MEDV.


h %>% ggplot(aes(x=MEDV))+
               geom_histogram(bins = 5)+
               facet_wrap(~RAD,ncol = 4) #there are more count of RADS in data.


h %>% ggplot(aes(x=MEDV))+
  geom_histogram(bins = 5)+
  facet_wrap(~RIVER_FLG,ncol = 1) #More than 90% houses don't have river as their bounds.


head(h)


#----------Performing transformations to skewed data------------
#===============================================================

h['logCRIM']<-log(h$CRIM)
ht['logCRIM']<-log(ht$CRIM)

h['logB']<-log(max(h$B+1)-h$B)
ht['logB']<-log(max(ht$B+1)-ht$B)

h['logINDUS']<-log(h$INDUS)
ht['logINDUS']<-log(ht$INDUS)

h['logZN']<-log(h$ZN+1)
ht['logZN']<-log(ht$ZN+1) #Performing transfomation to ZN was not changing the distribution ofZN.

#-----------Feature Engineering----------------------------------
#=================================================================

#Creating variable that has combined effect of CRIM and B values
h$CRIM_B=h$CRIM+sqrt(h$B)
ht$CRIM_B=ht$CRIM+sqrt(ht$B)

hist(h$CRIM_B)

head(h)
head(ht)

#Standard Normalization of variables using Min and Max function
#define Min-Max normalization function
min_max_norm<-function(x){
  (x-min(x))/(max(x)-min(x))
  }


h_scale = select(h,-c('RAD','RIVER_FLG', 'ID'))
h_factor= select(h,c('RAD','RIVER_FLG'))

ht_scale = select(ht,-c('RAD','RIVER_FLG','ID'))
ht_factor = select(ht,c('RAD','RIVER_FLG'))

#apply Min-Max normalization to first four columns in dataset

h_scale<-as.data.frame(lapply(h_scale,min_max_norm))
ht_scale<-as.data.frame(lapply(ht_scale,min_max_norm))

h_scale=scale(h_scale)
ht_scale=scale(ht_scale)

head(h_scale)
head(ht_scale)

hf<-cbind(h_scale,h_factor)
hft<-cbind(ht_scale,ht_factor)


#Generating final dataframe of test and train for modelling
hf=h
hft=ht

head(hf)
head(hft)

install.packages("corrplot")
library(corrplot)
corrplot(cor(train_1))

cor(h) #MEDV has highest correlation with X.rooms.dwellings.


install.packages("Hmisc")
library(Hmisc)
res<-rcorr(as.matrix(hf))

#Printing the correlation matrix
signif(res$r,2)

#Printing the p-values of the correlations
signif(res$P,2)

#separate the Y variable to be predicted
y<-hf[,'MEDV']
head(y)


#Removing the variables MEDV and ID from training and ID from testing
hf<-select(hf,-c('MEDV','ID'))
hft<-select(hft,-ID)

#Removing variables CRIM,B and ZN as they are transformed by applying log transformation
hf<-select(hf,-c('CRIM','B'))
hft<-select(hft,-c('CRIM','B'))

hf<-select(hf,-c('ZN'))
hft<-select(hft,-c('ZN'))

head(hf)
head(hft)


#Base model of linear Regression
model=lm(y~.,data = hf)
summary(mode1)
model    #This model is the base model.The variables of RAD and River Flag to be converted to factor, as they are like rating values.


#Converting RAD and RIVER_flag variables to factor variables.

hf$RAD=as.factor(hf$RAD)
hf$RIVER_FLG=as.factor(hf$RIVER_FLG)


hft$RAD=as.factor(hft$RAD)
hft$RIVER_FLG=as.factor(ht$RIVER_FLG)


str(hf)

#Create vector of VIF values
install.packages("car")
library(car)
vif_values<-vif(model)
vif_values

#Create horizental bar chart to display each vif value
barplot(vif_values,main = "VIF Values",horiz = TRUE,col = "steelblue")


#add vertical line at 5
abline(v=5,lwd=3,lty=2)

#Model after converting RAD and RIVER_FLG to factor variables
model1=lm(y~.,data = hf)
summary(model1)
model1


#Creating Dummy Variables
hf$RIVER_FLG<-ifelse(hf$RIVER_FLG==0,1,0)
h$RIVER_FLG<-ifelse(h$RIVER_FLG==1,1,0)


h$RAD1<-ifelse(h$RAD==1,1,0)
hf$RAD2<-ifelse(hf$RAD==2,1,0)
hf$RAD3<-ifelse(hf$RAD==3,1,0)
hf$RAD4<-ifelse(hf$RAD==4,1,0)
hf$RAD5<-ifelse(hf$RAD==5,1,0)
hf$RAD6<-ifelse(hf$RAD==6,1,0)
hf$RAD7<-ifelse(hf$RAD==7,1,0)
hf$RAD8<-ifelse(hf$RAD==8,1,0)

str(hf)

#Creating Dummy Variables
hft$RIVER_FLG0<-ifelse(hft$RIVER_FLG==0,1,0)
hft$RIVER_FLG1<-ifelse(hft$RIVER_FLG==1,1,0)

ht$RAD1<-ifelse(ht$RAD==1,1,0)
hft$RAD2<-ifelse(hft$RAD==2,1,0)
hft$RAD3<-ifelse(hft$RAD==3,1,0)
hft$RAD4<-ifelse(hft$RAD==4,1,0)
hft$RAD5<-ifelse(hft$RAD==5,1,0)
hft$RAD6<-ifelse(hft$RAD==6,1,0)
hft$RAD7<-ifelse(hft$RAD==7,1,0)
hft$RAD8<-ifelse(hft$RAD==8,1,0)

str(hft)

#Removing original RAD and RIVER_FLG
hf<-select(hf,-c('RAD','RIVER_FLG'))
hft<-select(hft,-c('RAD','RIVER_FLG'))

hf<-select(hf,-'RIVER_FLG0')
hft<-select(hft,-'RIVER_FLG0')

#After removing insignificant RiverFlag and RAD original variables

model2=lm(y~.,data = hf)
summary(model2)
model2

#Applying log transformation to y variables
model3=lm(log(y)~.,data = hf)
summary(model3)

head(hf)


#Applying log transformation to y and pass it through step function
model4=step(model3,direction = "backward")
summary(model4)

install.packages("randomForest")

model_final=randomForest(y~.,data=hf,ntrees=100)

model_final
summary(model_final)
importance(model_final)

y_train_1=predict(model_final,newdata=hf)
install.packages("Metrics")
library(Metrics)
rms=rmse(train_1$MEDV,y_train_1)
rms

#log model
MEDV_pred=exp(model3$fitted.values)
head(MEDV_pred)


rms=rmse(train_1$MEDV,MEDV_pred)
rms



#Assumption checking with Random forest model

res=y-y_train_1
hist(res)

install.packages("qqplotr")
library(qqplotr)
qqplot(res)
#constant variance check
plot(y_train_1)

#Checking of assumption with log model
hist(model3$residuals)

#constant variance check
plot(model3$fitted.values,model3$residuals)

#-------Prediction of test data--------------
#===========================================







