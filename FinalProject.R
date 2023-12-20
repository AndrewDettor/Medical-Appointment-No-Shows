---
title: "Final Project"
author: "Andrew Dettor"
date: "5/2/2021"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r}
data = read.csv("noshows.csv")
dim(data)
```

\section EDA

```{r}
# missing columns as percent
san = function(x) sum(is.na(x))
percent = round(apply(data,2,FUN=san)/nrow(data),4) * 100 #pers of na's in columns
percent[percent!=0]
```
```{r}
# see balance of classes
prop.table(table(data$No.show))
barplot(prop.table(table(data$No.show)), main="Balance of classes for No Show")
```
 Imbalanced classes
 
```{r}
# See number of unique values

uniqueList = rep(0, length(names(data)))
names(uniqueList) = names(data)

for (i in 1:length(names(data))) {
  name = names(data)[i]
  uniqueList[i] = length(unique(data[,name]))
}

sort(uniqueList, decreasing = TRUE)
```

\section Data Cleaning / Feature Engineering

```{r}
library(lubridate)
scheduledday = ymd_hms(data$ScheduledDay)
appointmentday = ymd_hms(data$AppointmentDay)
```
```{r} 
x = scheduledday 
data[,"scheduledWeekDay"] = wday(x) 
data[,"scheduledMonth"]= month(x)
data[,"scheduledDayofMonth"] = day(x)
data[,"scheduledHourofDay"] = hour(x)
data[,"scheduledDayofYear"] = yday(x)
data[,"scheduledAM"] = am(x)
data[,"scheduledWeekofYear"] = week(x)
data[,"scheduledQuarter"] = quarter(x)
```

```{r} 
x = appointmentday
data[,"appointmentWeekDay"] = wday(x) 
data[,"appointmentMonth"]= month(x)
data[,"appointmentDayofMonth"] = day(x)
data[,"appointmentHourofDay"] = hour(x)
data[,"appointmentDayofYear"] = yday(x)
data[,"appointmentAM"] = am(x)
data[,"appointmentWeekofYear"] = week(x)
data[,"appointmentQuarter"] = quarter(x)
```

```{r}
# can now drop those columns
# what to do with patientID? treat each entry as a new patient
data = subset(data, select=-c(AppointmentID, ScheduledDay, AppointmentDay, PatientId))
```

```{r}
# replace with 0/1

data$appointmentAM = ifelse(data$appointmentAM==TRUE, 1, 0)
data$scheduledAM = ifelse(data$scheduledAM==TRUE, 1, 0)
data$Gender = ifelse(data$Gender=="M", 1, 0)
data$No.show = ifelse(data$No.show=="Yes", 1, 0)
```

```{r}
# One-Hot encode this column
library(dummies)
data = dummy.data.frame(data, sep="_", names="Neighbourhood")
```
```{r}
# see 0 variance columns and drop them

for (name in names(data)) {
  if (var(data[,name]) == 0) {
    cat(name, "\n")
    data = data[,!(names(data) %in% c(name))]
  }
}
```

```{r}
# scale data for clustering
scaled_data = scale(data)
```

\section Unsupervised Learning (Clustering)

Can do clustering without train/test splitting because it's unsupervised.

```{r}
trainX = subset(scaled_data, select=-No.show)
```

```{r}
# fit the clustering algorithm
set.seed(99)
fitK = kmeans(trainX, centers=10, nstart = 1)
data[,"cluster"] = fitK$cluster
```

```{r}
# see balance of clusters (2 real clusters)
# chose 10 clusters because there might be subsets in the data
barplot(prop.table(table(data$cluster)), main="Balance of classes for Clusters")
```
3 main groups: 4, 7 and not clustered

\section Supervised Learning (Gradient Boosting Machines)

```{r}
# split into train and test
# Source: https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function

set.seed(99)

train_size = floor(.75 * nrow(data))
train_indices = sample(seq_len(nrow(data)), size = train_size)

train = data[train_indices,]
test = data[-train_indices,]
```

```{r}
formula = No.show ~ .
```

```{r}
set.seed(99)
library(gbm)

shrinkage_grid = c(.1/3, .1, .1*3) # default .1
ntree_grid = c(75, 100, 125) # default 100

best_shrinkage = 0
best_ntree = 0
best_acc = 0.0


for (shrinkage_value in shrinkage_grid) {
  for (ntree_value in ntree_grid) {
    # fit the model
    boost.fit = gbm(formula,data=train, distribution="bernoulli",
                    n.trees=ntree_value, shrinkage=shrinkage_value)

    # get test prediction probabilities
    boost.probs = predict(boost.fit, newdata=train, 
                         n.trees=ntree_value, type="response")
    
    # get exact predictions
    boost.pred=rep(0, length(boost.probs))
    boost.pred[boost.probs>.5] = 1
    
    # get accuracy
    acc = 100*(mean(boost.pred == train$No.show))
    
    # save best values
    if (acc >= best_acc) {
      best_shrinkage = shrinkage_value
      best_ntree = ntree_value
    }
  }
}

best_shrinkage
best_ntree
```


```{r}
set.seed(99)

# fit boost with the best found hyperparameters
boost.fit = gbm(formula,data=train, distribution="bernoulli",
                    n.trees=best_ntree, shrinkage=best_shrinkage)

# get train and test prediction probabilities
boost.probs = predict(boost.fit, newdata=test, 
                     n.trees=best_ntree, type="response")
boost.probs.train = predict(boost.fit, newdata=train,
                           n.trees=best_ntree, type="response")

# get exact predictions
boost.pred=rep(0, length(boost.probs))
boost.pred[boost.probs>.5] = 1

boost.pred.train=rep(0, length(boost.probs.train))
boost.pred.train[boost.probs.train>.5] = 1

# get train and test accuracies
boost.test.acc = 100*(mean(boost.pred == test$No.show))
boost.train.acc = 100*(mean(boost.pred.train == train$No.show))

# print results
cat("Boosting Test Accuracy:", boost.test.acc, "\n")
cat("Boosting Train Accuracy:", boost.train.acc, "\n")
```
```{r}
perfcheck <- function(ct) {
  Accuracy <- (ct[1]+ct[4])/sum(ct)
  Recall <- ct[4]/sum((ct[2]+ct[4]))      #TP/P   or Power, Sensitivity, TPR 
  Type1 <- ct[3]/sum((ct[1]+ct[3]))       #FP/N   or 1 - Specificity , FPR
  Precision <- ct[4]/sum((ct[3]+ct[4]))   #TP/P*
  Type2 <- ct[2]/sum((ct[2]+ct[4]))       #FN/P
  F1 <- 2/(1/Recall+1/Precision)
  Values <- as.vector(round(c(Accuracy, Recall, Precision, F1),4)) *100
  Metrics = c("Accuracy", "Recall", "Precision", "F1")
  cbind(Metrics, Values)
  #list(Performance=round(Performance, 4))
}

test_allfeatures = perfcheck(table(test$No.show, boost.pred, dnn=c("True", "Predicted")))[,2]
train_allfeatures = perfcheck(table(train$No.show, boost.pred.train, dnn=c("True", "Predicted")))[,2]

```

\section Feature Selection

```{r}
# Boost Top 10 + plot

impList = head(summary(boost.fit)[2], n=10)$rel.inf

names(impList) = rownames(head(summary(boost.fit)[2], n=10))

par(mar=c(3, 15, 2, 2))
barplot(rev(impList), horiz=TRUE, names.arg=names(impList),las=1, main="Feature Importance")
```

```{r}
names(impList)
```

```{r}
keep = names(impList)[names(impList) != "`Neighbourhood_SANTOS DUMONT`"]
keep = append(keep, "No.show")
keep = append(keep, "Neighbourhood_SANTOS DUMONT")

reduced_data = data[,names(data) %in% keep]


train = reduced_data[train_indices,]
test = reduced_data[-train_indices,]
```

```{r}
# refit and see results

set.seed(99)
library(gbm)

shrinkage_grid = c(.1/3, .1, .1*3) # default .1
ntree_grid = c(75, 100, 125) # default 100

best_shrinkage = 0
best_ntree = 0
best_acc = 0.0


for (shrinkage_value in shrinkage_grid) {
  for (ntree_value in ntree_grid) {
    # fit the model
    boost.fit = gbm(formula,data=train, distribution="bernoulli",
                    n.trees=ntree_value, shrinkage=shrinkage_value)

    # get test prediction probabilities
    boost.probs = predict(boost.fit, newdata=train, 
                         n.trees=ntree_value, type="response")
    
    # get exact predictions
    boost.pred=rep(0, length(boost.probs))
    boost.pred[boost.probs>.5] = 1
    
    # get accuracy
    acc = 100*(mean(boost.pred == train$No.show))
    
    # save best values
    if (acc >= best_acc) {
      best_shrinkage = shrinkage_value
      best_ntree = ntree_value
    }
  }
}

best_shrinkage
best_ntree

# fit boost with the best found hyperparameters
boost.fit = gbm(formula,data=train, distribution="bernoulli",
                    n.trees=best_ntree, shrinkage=best_shrinkage)

# get train and test prediction probabilities
boost.probs = predict(boost.fit, newdata=test, 
                     n.trees=best_ntree, type="response")
boost.probs.train = predict(boost.fit, newdata=train,
                           n.trees=best_ntree, type="response")

# get exact predictions
boost.pred=rep(0, length(boost.probs))
boost.pred[boost.probs>.5] = 1

boost.pred.train=rep(0, length(boost.probs.train))
boost.pred.train[boost.probs.train>.5] = 1

# get train and test accuracies
boost.test.acc = 100*(mean(boost.pred == test$No.show))
boost.train.acc = 100*(mean(boost.pred.train == train$No.show))

# print results
cat("Boosting Test Accuracy:", boost.test.acc, "\n")
cat("Boosting Train Accuracy:", boost.train.acc, "\n")
```
```{r}
test_10features = perfcheck(table(test$No.show, boost.pred, dnn=c("True", "Predicted")))[,2]
train_10features = perfcheck(table(train$No.show, boost.pred.train, dnn=c("True", "Predicted")))[,2]
```

\section Results and Clarification

```{r}
A = as.matrix(cbind(test_allfeatures, test_10features, train_allfeatures, train_10features), nrow=4, ncol=4)
rownames(A)=c("Accuracy", "Recall", "Precision", "F1")
colnames(A)=c("Test (all features)", "Test (best 10 features)", "Train (all features)", "Train (best 10 features)")

# Lets use a better format from kable
knitr::kable(A, caption = "GBM Models Performance")

```

```{r}
dim(data)
dim(reduced_data)
```

```{r}

barplot(table(data$No.show, data[,"scheduledDayofMonth"]), main="Response vs. Scheduled Day of Month",
        xlab="Day of Month", ylab="Absolute Frequency", legend=c("No show", "Show"))
```




 