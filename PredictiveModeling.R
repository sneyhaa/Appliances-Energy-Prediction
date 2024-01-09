library(ggplot2)
library(caret)
library(e1071)
library(MASS)


app <- read.csv('/Users/snehakarki/Downloads/KAG_energydata_complete.csv')
#app <- read.csv("/Users/karenfernandes/Documents/Predictive modelling/KAG_energydata_complete.csv")
head(app)
summary(app)
str(app)
dim(app)

#Data preprocessing
ggplot(app, aes(Appliances)) +
  geom_histogram(binwidth = 30,color = "#000000", fill = "#0099F8") +
  ggtitle("Response distribution") +
  theme_classic() +
  theme(plot.title = element_text(size = 18))

#Checking missing values
# Check for missing values in the entire dataset
missing_values <- sum(is.na(app))
cat("Total missing values in the dataset:", missing_values, "\n")


# Check for missing values in each column
missing_values_per_column <- colSums(is.na(app))
missing_values_per_column



# Define the list of variables for which you want to create histograms
variables_to_plot <- c(names(app)[-1])


# Set up the layout for the plots
par(mfrow = c(3, 3))  # Adjust the rows and columns as needed

# Create histograms in a loop
for (variable in variables_to_plot) {
  hist(app[, variable], main = paste("Histogram of", variable), xlab = variable, col = "lightblue", border = "black")
}
# Reset the layout to the default
par(mfrow = c(1, 1))

cont_vars <- app[c(names(app)[-1])]
numeric_vars <- app[, sapply(app, is.numeric)]
dim(numeric_vars)


# Create individual boxplots for each numeric predictor
par(mfrow = c(3, 3)) # Set the layout for multiple plots
for (var_name in names(cont_vars)) {
  boxplot(numeric_vars[[var_name]], main = var_name, col = "#ad1538", border = "black")
}
# Reset the plotting layout
par(mfrow = c(1, 1))


#Skewness values
skewness_values <- sapply(cont_vars, skewness)
skewness_summary <- data.frame(Predictor = names(skewness_values), Skewness =skewness_values)
print(skewness_summary)

#Transforming the data

trans <- preProcess(cont_vars, method = c("BoxCox", "center", "scale")) 
app_transformed <- predict(trans, cont_vars)


#Checking skewness after transformation
skewness_trans<- sapply(app_transformed, skewness)
skewness_summ <- data.frame(Predictor = names(skewness_trans), Skewness =skewness_trans)
print(skewness_summ)

#Checking distribution after transformation
par(mfrow = c(3, 3))  # Adjust the rows and columns as needed

# Create histograms in a loop
for (i in names(app_transformed)) {
  hist(app_transformed[[i]], main = paste("Histogram of", i), xlab = variable, col = "lightblue", border = "black")
}
# Reset the layout to the default
par(mfrow = c(1, 1))



#Spatial sign transformation to remove outliers
# Spatial sign transformation
spatial_sign <- function(x) {
  norms <- rowSums(x^2)^0.5
  return(x / norms)
}
transformed_data <- spatial_sign(app_transformed)

# Create individual boxplots for each numeric predictor
par(mfrow = c(3, 3)) # Set the layout for multiple plots
for (k in names(transformed_data)) {
  boxplot(transformed_data[[k]], main = k, col = "#15ad4f", border = "black")
}
# Reset the plotting layout
par(mfrow = c(1, 1))

#No attributes with Near zero variance
near_zero_var <- nearZeroVar(transformed_data, saveMetrics = TRUE)
print(near_zero_var)


preds <- transformed_data[, -c(1, 2)]
preds
dim(preds)

resp <- transformed_data$Appliances
resp

#PCA
pca_result <- prcomp(preds, scale. = TRUE)

# Summary of PCA
summary(pca_result)

# Scree plot to visualize the variance explained by each principal component
screeplot(pca_result, type = "line", main = "Scree Plot")

# Variance explained by each principal component
cumsum_var <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
cumsum_var

#number of PCs based on the threshold
desired_variance <- 0.95  
num_components <- which(cumsum_var >= desired_variance)[1]

# final number of principal components
pca_result_selected <- prcomp(preds, scale. = TRUE, rank. = num_components)

# data transformation - getting desired number of components
data_pca <- predict(pca_result_selected, newdata = preds)

# The transformed data contains the scores for each principal component
head(data_pca)
dim(data_pca)

#As we can see, we have reduced the number of columns/predictor variables
#from 26 to 13 while still capturing 95% of the data

library(caret)
library(e1071)
library(class)
library(nnet)
library(glmnet)


# Split the data
set.seed(123)
train_indices <- createDataPartition(resp, p = 0.8, list = FALSE)
train_data <- preds[train_indices, ]
test_data <- preds[-train_indices, ]
train_resp <- resp[train_indices]
test_resp <- resp[-train_indices]


#######Linear models
#SVM model
svm_model <- train(train_data, train_resp,
                   method = "svmRadial", 
                   trControl = trainControl(method = "cv", number = 5))
svm_model
svm_model$results[3,]
plot(svm_model, main = "Tuning parameter plot for SVM ")
#KNN model
knn_model <- train(train_data, train_resp, method = "knn", trControl = trainControl(method = "cv", number = 5))
knn_model$results[1,]
knn_model
plot(knn_model, main = "Tuning parameter plot for KNN" )
# Train a MARS model
mars_model <- train(train_data,
                    train_resp,
                    method = "earth")
mars_model$results[1,]
mars_model
plot(mars_model, main = "Tuning parameter plot for MARS")


####if we have time for this#####

#Neural Network model
nn_model <- train(train_data, train_resp, method = "nnet", 
                  trControl = trainControl(method = "cv", number = 5),
                  preProc = c("center", "scale"))
nn_model$results[1,]
plot(nn_model, main = "Tuning parameter plot for Neural Network")

###Enet model
set.seed(476)
glmnTuned <- train(
  x = train_data,
  y = train_resp,
  method = "glmnet",
  # preProc = c("center", "scale"), 
  # metric = "Accuracy",  
  tuneGrid = expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0.1, 1, by = 0.1)),
  trControl = trainControl(method = "cv", number = 5)
)

glmnTuned
summary(glmnTuned)
glmnTuned$results[1,]

glmpred <- predict(glmnTuned, newdata = test_data)
glm_pred_factor <- factor(glmpred, levels = levels(test_resp))

glmpred_output <- postResample(glmpred, test_data[ ,1])
glmpred_output


plot(glmnTuned)

#Lasso regression model
lasso_model <- cv.glmnet(train_data, train_resp, alpha = 1)

#Ridge regression model
ridge_model <- cv.glmnet(train_data, train_resp, alpha = 0)

#PCR model
# Train the PCR model
pcr_model <- train(
  x = train_data,
  y = train_resp,
  method = "pcr",
  #tuneLength = 5,  # Number of principal components to consider (you can adjust this)
  trControl =  trainControl(method = "cv", number = 5)
)

#PLS model
# Train the PLS model
pls_model <- train(
  x = train_data,
  y = train_resp,
  method = "pls",
  #tuneLength = 5,  # Number of components to consider (you can adjust this)
  trControl = ctrl
)

# Print the summary of the trained PLS model
print(pls_model)

#Prediction
svm_pred <- predict(svm_model, test_data)
knn_pred <- predict(knn_model, test_data)
marsPred <- predict(mars_model, newdata = test_data)
nn_pred <- predict(nn_model, test_data)
lasso_pred <- predict(lasso_model, s = lasso_model$lambda.min, newx = test_data)
ridge_pred <- predict(ridge_model, s = ridge_model$lambda.min, newx = test_data)
pcr_pred <- predict(pcr_model, newdata = test_data)
pls_pred <- predict(pls_model, newdata = test_data)


# Evaluate the models using RMSE and R-squared
svm_perf<- postResample(svm_pred, test_data[ ,1])
svm_perf
knn_perf<- postResample(knn_pred, test_data[ ,1])
knn_perf
mars_perf<- postResample(marsPred, test_data[ ,1])
mars_perf
nn_perf<- postResample(nn_pred, test_data[ ,1])
nn_perf


pcr_perf<- postResample(pcr_pred, test_data[ ,1])
pcr_perf


marsPerf <- postResample(pred = marsPred, obs = testData$y)



#########Non-Linear models
ctrl <-trainControl(summaryFunction = defaultSummary, classProbs = TRUE,
                    method = "repeatedcv", number = 10, repeats = 5,
                    savePredictions = TRUE)



######FDA
#Train
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:20)
fda_mod <- train(train_data,
                 train_resp,
                 method = "fda",
                 preProc = c("spatialSign"),
                 metric ="RMSE",
                 tuneGrid = marsGrid,
                 trControl = ctrl)
fda_mo #wrong model type for regression!
plot(fda_mod)
#Test
fda_pred <- predict(fda_mod, test_data_bio)
fda_mat <- confusionMatrix(fda_pred, factor(injury_test))
fda_mat

####SVM
trl <- trainControl(summaryFunction = twoClassSummary,
                    classProbs = TRUE)
sigmaRangeReduced <- sigest(as.matrix(simulatedTrain[,1:4]))


svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))
set.seed(476)
svmRModel <- train(x = simulatedTrain[,1:4], 
                   y = simulatedTrain$class,
                   method = "svmRadial",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)
#took forever to load
#Train
library(kernlab)
set.seed(476)
sigmaRangeReduced <- sigest(as.matrix(train_data))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced,
                               .C = 2^(seq(-4, 6)))

svm_mod <- train(
  x = train_data,
  y = train_resp,
  method = "svmRadial",
  #metric = "ROC",
  tuneGrid = svmRGridReduced,
  fit = FALSE,
  trControl = ctrl
)



svm_mod
plot(svm_mod)

svm_mod$results[3,]

#Test
svm_pred <- predict(svm_mod, test_data_bio)
svm_mat <- confusionMatrix(svm_pred, factor(injury_test))
svm_mat




#####NB
#Train
nb_mod <- train(train_data,
               train_resp,
                method = "nb",
                metric = "RMSE",
                trControl = ctrl,
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE)
)
nb_mod

#Test
nb_pred <- predict(nb_mod, test_data_bio)
nb_mat <- confusionMatrix(nb_pred, factor(injury_test))
nb_mat
#wrong model type for regression



# Neural Network

nnetGrid <- expand.grid(size = 3, decay = 0.1)

# Calculate the maximum size for the neural network
maxSize <- max(nnetGrid$size)
numWts <- (maxSize * (ncol(train_data) + 1) + (maxSize + 1) * 2)

# Train the neural network model
nnet_mod <- train(
  x = train_data,
  y = train_resp,
  method = "nnet",
  metric = "RMSE",  # Use "RMSE" for regression tasks
  tuneGrid = nnetGrid,
  preProc = c("center", "scale", "spatialSign"),
  trace = FALSE,
  maxit = 2000,
  MaxNWts = numWts,
  trControl = trainControl(method = "cv", number = 5)  # Use cross-validation
)

nnet_mod

print(nnet_mod)
plot(nnet_mod)

# Test the model
nnet_pred <- predict(nnet_mod, newdata = test_data)
nn_output<- postResample(nnet_pred, test_data[ ,1])
nn_output



# SVM (Support Vector Machines)
svm_mod <- train(
  x = train_data,
  y = train_resp,
  method = "svmRadial",
  metric = "RMSE",  # Use "RMSE" for regression tasks
  trControl = trainControl(method = "cv", number = 5)  # Use cross-validation
)

#trained SVM model and plot
print(svm_mod)
plot(svm_mod, main = "SVM plot")

#Predictions with SVM model
svm_pred <- predict(svm_mod, newdata = test_data)


svm_output<- postResample(svm_pred, test_data[ ,1])
svm_output


# KNN (K-Nearest Neighbors)
knn_mod <- train(
  x = train_data,
  y = train_resp,
  method = "knn",
  metric = "RMSE",  # Use "RMSE" for regression tasks
  trControl = trainControl(method = "cv", number = 5)  # Use cross-validation
)

#trained KNN model and plot
print(knn_mod)
knn_mod$results[2,]
plot(knn_mod, main ="KNN plot")

#Predictions with KNN model
knn_pred <- predict(knn_mod, newdata = test_data)

knn_output<- postResample(knn_pred, test_data[ ,1])
knn_output

#MDA

mdaFit <- train(x = train_data,
                y = train_resp,
                metric = "RMSE",
                method = "mda",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
mdaFit

#For RDA, use method = "rda", tuning parameters are .gamma (Gamma) and .lambda (Lambda)
## For QDA, use method = "qda"

#RDA
rdaFit <- train(x = train_data,
                y = train_resp,
                metric = "RMSE",
                method = "rda",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
rdaFit


#QDA

qdaFit <- train(x = train_data,
                y = train_resp,
                metric = "RMSE",
                method = "qda",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
qdaFit


