## install package if it is not already.
if(!require("data.table")){
  install.packages("data.table")
}

## attach all functions provided by these packages.
library(data.table)
library(ggplot2)

## download spam data set to local directory, if it is not present.
if(!file.exists("spam.data")){
  download.file("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data", "spam.data")
}

## Read spam data set and conver to X matrix and y vector we need for
## gradient descent.
spam.dt <- data.table::fread("spam.data")
N.obs <- nrow(spam.dt)
X.raw <- as.matrix(spam.dt[, -ncol(spam.dt), with=FALSE]) 
y.vec <- spam.dt[[ncol(spam.dt)]]
yt.vec <- ifelse(spam.dt[[ncol(spam.dt)]]==1, 1, -1)
X.sc <- scale(X.raw) #scaled X/feature/input matrix.

## split into train and test data
n.folds <- 5
set.seed(1)
fold.vec <- sample(rep(1:n.folds, l=length(yt.vec)))

test.fold <- 5
is.test <- fold.vec == test.fold
is.train <- !is.test
table(is.train)

X.test <- X.sc[is.test,]
yt.test <- yt.vec[is.test]

X.train <- X.sc[is.train,]
yt.train <- yt.vec[is.train]

# split into subtrain and validation data
n.subfolds <- 5
set.seed(1)
subfold.vec <- sample(rep(1:n.subfolds, l=length(yt.train)))

subtrain.fold.vec <- c(5,4)
is.subtrain <- subfold.vec == subtrain.fold.vec[1] | subfold.vec == subtrain.fold.vec[2]
is.val <- !is.subtrain
table(is.subtrain)

fit <- NNetOneSplit(X.train , yt.train , 200 , .01 , 20 , is.subtrain)


## plot log loss of subtrian and validation split


  ## find min log loss of validation
split.vec <- fit$loss.values$split
subtrain <- "subtrain"
is.split.subtrain <- split.vec == subtrain
min.dt <- fit$loss.values[!is.split.subtrain][ which.min(mean.loss)]


ggplot()+
  geom_line( aes( x = epoch ,
                  y = mean.loss ,
                  color = split, 
                  group = split),
             data = fit$loss.values)+
  geom_point( aes( x= epoch ,
                   y= mean.loss) ,
              color = "black",
              size = 1.5 ,
              data = min.dt)

## best number of epochs for minimizing mean log loss
 best_epochs <- min.dt$epoch

## using weight matrix list on test set to see prediction accuracy
 
 test.fit <- NNetOneSplit(X.train , yt.train , 77 , .01 , 20 , is.subtrain)
 
 test.var <- X.test %*% test.fit$V.mat
 test.pred <- test.var %*% test.fit$w.vec
 
 thresh <- 0 
 is.spam.pred <- test.pred > thresh
 is.spam.test <- yt.test > thresh
 
 spam.correct <- is.spam.pred == is.spam.test
 acc = 100 * sum(spam.correct) / nrow(spam.correct)





#### Funcitons:
NNetOneSplit<- function(X.mat, y.vec, max.epochs, step.size, n.hidden.units, is.subtrain){
  ## subtrain set is for computing gradients
  ## validation set is for choosing the optimal number of steps (with minimal loss)
  X.subtrain <- X.mat[is.subtrain, ]
  X.validation <- X.mat[!is.subtrain, ]
  y.subtrain <- y.vec[is.subtrain]
  y.validation <- y.vec[!is.subtrain]
  
  ## V.mat used to predict hidden units
  V.mat <- matrix(rnorm(ncol(X.mat) * n.hidden.units), ncol(X.mat),  n.hidden.units)
  ## w.vec used to predict output given hidden units
  w.vec <- matrix(rnorm(n.hidden.units) ,  n.hidden.units)
  
  weight.matrix.list <- list(V.mat, w.vec)
  epoch.output.list <- list()
  output.list <-list()
  
  for(k in 1:max.epochs){
    log.loss.vec <- vector()
    
    ## find gradients and take step in the negative direction
     for(data in 1:length(y.subtrain)){
      x<- X.subtrain[data ,]
      y<- y.subtrain[data]
      
      hidden_layer_list <- ForwardPropagation(x, weight.matrix.list)
      gradient.list <- list()
      gradient.list <- BackPropagation(hidden_layer_list, weight.matrix.list, y)
      
      weight.matrix.list[[1]] <- weight.matrix.list[[1]] - step.size * t(gradient.list[[1]])
      weight.matrix.list[[2]] <- weight.matrix.list[[2]] - step.size * t(gradient.list[[2]])
    }
    
    ## create data frame of log loss for each epoch by split
    
    pred.layer.list <- ForwardPropagation( X.mat , weight.matrix.list)
    pred.vec <- pred.layer.list[[3]]
  
    log.loss.vec <- LogisticLoss( pred.vec , yt.train)
    subtrian.log.loss <- log.loss.vec[is.subtrain]
    validation.log.loss <- log.loss.vec[!is.subtrain]
    
    index <- 2 * k
    epoch.output.list[[index-1]] <- data.table(
      epoch = k,
      split = "subtrain",
      mean.loss = mean(subtrian.log.loss))
    epoch.output.list[[index]] <- data.table(
      epoch = k,
      split = "validation",
      mean.loss = mean(validation.log.loss))
  }
  
  
  epoch.output.dt <- do.call( rbind ,epoch.output.list )

  output.list$loss.values <- epoch.output.dt
  output.list$V.mat <- weight.matrix.list[[1]]
  output.list$w.vec <- weight.matrix.list[[2]]
  
  return(output.list)
}


ForwardPropagation <- function(input, weights){
  hidden_layer <- list(input)
  
  for(i in 1:length(weights)){
    
    if(i == length(weights)){
      activation <-  hidden_layer[[i]] %*% weights[[i]]
      hidden_layer[[i+1]] <- activation
    }
    
    else{
      activation <-  hidden_layer[[i]] %*% weights[[i]]
      hidden_layer[[i+1]] <- 1/(1+exp(-activation))
    }
  }
  
  return(hidden_layer)
}

 BackPropagation <- function(hidden.layer, weight.matrix.list, y.train){
  gradient_list <- list()
  for(i in length(weight.matrix.list):1){
    if(i == length(weight.matrix.list)){
      grad_w <- -y.train / (1+exp(y.train*hidden.layer[[length(hidden.layer)]])) 
    }
    else{
      grad_hidden <-  weight.matrix.list[[i+1]]  %*% grad_w 
      grad_w <- t(grad_hidden) * hidden.layer[[i+1]] * (1 - hidden.layer[[i+1]])
      grad_w <- t(grad_w)
    }
    gradient_list[[i]] <- grad_w  %*% hidden.layer[[i]]
  }
  
  return(gradient_list)
}
LogisticLoss <- function( pred.vec , label.vec){
  log(1+exp(-label.vec*pred.vec))
}
