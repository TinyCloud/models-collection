Understanding K-Nearest Neighbors Regression
================
Anumeha Dwivedi
19 October 2017

``` r
library(dplyr)    #For Data manipulation 
library(class)    #For KNN Regression
library(ggplot2)  #For making plots
```

``` r
df <- read.csv("/Users/tinycloud/Documents/Spring 18/Data Mining 1/Data/auto_mpg.csv",stringsAsFactors = FALSE)
df1 <- as.data.frame(sapply(df,as.numeric))
```

### K Nearest Neighbors Regression (Unnormalized)

``` r
df2 <- df1 %>% 
  select(cylinders, displacement, horsepower,weight, acceleration, model.year,mpg)

df3 <- df2[complete.cases(df2),]

set.seed(1)
# Split train and test
train_idx <- sample(1:nrow(df3),nrow(df3)*0.8 )
train <- df3[train_idx, ]
test <- df3[-train_idx, ]

#  Select the feature variables
train.X=train[,1:6]
# Set the target for training
train.Y=train[,7]

# Do the same for test set
test.X=test[,1:6]
test.Y=test[,7]
```

### Trying different number of neighbors

``` r
error <- c()
set.seed(1)
# Create a list of neighbors
neighbors <-c(1,2,4,5,8,10,15,20)
for(i in seq_along(neighbors))
{
  # Perform a KNN regression fit
  knn_res <- knn(train.X, test.X, train.Y, k = neighbors[i])
  # Compute R squared
  error[i] <- sqrt(sum((test.Y - as.numeric(knn_res))^2))
}
```

``` r
# Make a dataframe for plotting
dfx <- data.frame(neighbors,error = error)

# Plot the number of neighors vs the R squared
ggplot(dfx,aes(x = neighbors,y = error)) + 
  geom_point() +
  geom_line(color="blue") +
  xlab("Number of neighbors") + 
  ylab("Error") +
  ggtitle("KNN regression - Error vs No. of Neighors (Unnormalized)")
```

![](KNN_Regression_Auto_MPG_data_files/figure-markdown_github/Plotting%20error%20with%20value%20of%20K-1.png)

### K Nearest Neighbors Regression (Normalized)

``` r
#  Select the feature variables
train.X.scaled=scale(train[,1:6])
# Set the target for training
train.Y=train[,7]

# Do the same for test set
test.X.scaled <- scale(test[,1:6])
test.Y=test[,7]
```

### Trying different number of neighbors

``` r
error_normal <- c()
set.seed(1)
# Create a list of neighbors
neighbors <-c(1,2,4,5,8,10,15,20)
for(i in seq_along(neighbors))
{
  # Perform a KNN regression fit
  knn_res <- knn(train.X.scaled, test.X.scaled, train.Y, k = neighbors[i])
  # Compute R squared
  error_normal[i] <- sqrt(sum((test.Y - as.numeric(knn_res))^2))
}
```

``` r
# Make a dataframe for plotting
dfx <- data.frame(neighbors, error = error_normal)

# Plot the number of neighors vs the R squared
ggplot(dfx,aes(x = neighbors,y = error_normal)) + 
  geom_point() +
  geom_line(color="blue") +
  xlab("Number of neighbors") + 
  ylab("Error") +
  ggtitle("KNN regression - Error vs No. of Neighors (Normalized)")
```

![](KNN_Regression_Auto_MPG_data_files/figure-markdown_github/Plotting%20error%20with%20value%20of%20K%20(Normalised)-1.png)
