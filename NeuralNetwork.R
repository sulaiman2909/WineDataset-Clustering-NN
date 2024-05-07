install.packages("readxl")
install.packages("nnet")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("neuralnet")

# Load required libraries
library(readxl)
library(nnet)
library(caret)
library(dplyr)
library(ggplot2)
library(neuralnet)

normalize <- function(x) {
  if (is.numeric(x)) {
    return((x - min(x)) / (max(x) - min(x)))
  } else {
    return(x)  # Return non-numeric columns as they are
  }
}

unnormalize <- function(x, min_val, max_val) {
  if (is.numeric(x)) {
    return((max_val - min_val) * x + min_val)
  } else {
    return(x)  # Return non-numeric columns as they are
  }
}

# Load the dataset
data <- read_excel("C:/Users/Mohamed Sulaiman/Desktop/ML Exam/ExchangeUSD.xlsx")

# Normalize the dataset, excluding non-numeric columns
dataset_normalized <- as.data.frame(lapply(data, normalize))


# Check the normalized dataset
head(dataset_normalized)


#Select only the third column
exchange_rate <- data[[3]]

# Split dataset into training and testing sets
train_data <- exchange_rate[1:400]
test_data <- exchange_rate[401:length(exchange_rate)]

# Define Input Variables for MLP Models (Autoregressive Approach)
create_input <- function(data, lag){
  if (!is.vector(data)) {
    stop("Input data must be a vector.")
  }
  lagged_data <- embed(data, lag + 1)
  input <- lagged_data[, -1]
  output <- lagged_data[, 1]
  return(list(input = input, output = output))
}

# Experiment with four input vectors
lag_values <- c(1, 4, 7, 10)  # Choose lag values
input_vectors <- lapply(lag_values, function(lag) create_input(as.vector(train_data), lag))

# Construct Input/Output Matrices for Training and Testing
train_input <- lapply(input_vectors, function(input) input$input)
train_output <- lapply(input_vectors, function(input) input$output)

# Print Input/Output Matrices
print("Input/Output Matrix for t1_train")
t1_train <- data.frame(previous_Day1 = train_input[[1]], t1_expected = train_output[[1]])
print(t1_train)

print("Input/Output Matrix for t1_test")
t1_test <- data.frame(previous_Day1 = train_input[[2]], t1_expected = train_output[[2]])
print(t1_test)

print("Input/Output Matrix for t2_train")
t2_train <- data.frame(previous_Day2 = train_input[[3]], previous_Day1 = train_input[[2]], t1_expected = train_output[[3]])
print(t2_train)

print("Input/Output Matrix for t2_test")
t2_test <- data.frame(previous_Day2 = train_input[[4]], previous_Day1 = train_input[[3]], t1_expected = train_output[[4]])
print(t2_test)

print("Input/Output Matrix for t3_train")
t3_train <- data.frame(previous_Day3 = train_input[[5]], previous_Day2 = train_input[[4]], previous_Day1 = train_input[[3]], t1_expected = train_output[[5]])
print(t3_train)

print("Input/Output Matrix for t3_test")
t3_test <- data.frame(previous_Day3 = train_input[[6]], previous_Day2 = train_input[[5]], previous_Day1 = train_input[[4]], t1_expected = train_output[[6]])
print(t3_test)

# Train MLP Models
models <- lapply(input_vectors, function(input) {
  lapply(c(5, 10, 15), function(size) {
    nnet(train_input[[1]], train_output[[1]], size = size, decay = 1e-5, maxit = 1000, linout = TRUE)
  })
})

# Flatten the list of models
models <- unlist(models, recursive = FALSE)

# Evaluate MLP Models
# Function to evaluate model performance
evaluate_model <- function(model, input, output) {
  predicted_values <- predict(model, input)
  rmse <- sqrt(mean((predicted_values - output)^2))
  mae <- mean(abs(predicted_values - output))
  mape <- mean(abs((predicted_values - output) / output)) * 100
  smape <- mean(200 * abs(predicted_values - output) / (abs(predicted_values) + abs(output)))
  return(list(rmse = rmse, mae = mae, mape = mape, smape = smape))
}
model_evaluation <- lapply(models, function(model) evaluate_model(model, train_input[[1]], train_output[[1]]))

# Performance table
performance_table <- data.frame(
  Model = paste("Model", 1:length(model_evaluation)),
  Neurons = rep(c(5, 10, 15), each = 4),
  RMSE = sapply(model_evaluation, function(x) x$rmse),
  MAE = sapply(model_evaluation, function(x) x$mae),
  MAPE = sapply(model_evaluation, function(x) x$mape),
  sMAPE = sapply(model_evaluation, function(x) x$smape)
)

# Print performance table
print(performance_table)

# Plot Actual vs. Predicted Exchange Rates
predicted_values <- predict(models[[1]], as.matrix(train_input[[1]]))
actual_vs_predicted <- data.frame(Actual = train_output[[1]], Predicted = predicted_values)

ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(x = "Actual Exchange Rate", y = "Predicted Exchange Rate", title = "Comparison of Actual vs. Predicted Exchange Rates",
       subtitle = "Scatter plot showing the correlation between actual and MLP-predicted exchange rates")

# Plot RMSE vs. Number of Neurons
rmse_vs_neurons <- data.frame(Neurons = rep(c(5, 10, 15), each = 4), RMSE = performance_table$RMSE)

ggplot(rmse_vs_neurons, aes(x = factor(Neurons), y = RMSE)) +
  geom_boxplot() +
  labs(x = "Number of Neurons", y = "RMSE", title = "Impact of Neuron Count on RMSE",
       subtitle = "Boxplot showing the distribution of RMSE values across different neuron counts in MLP models")

# Density Plot of Residuals
residuals <- actual_vs_predicted$Actual - actual_vs_predicted$Predicted
ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_density(fill = "pink", color = "blue") +
  labs(x = "Residuals", y = "Density", title = "Density Plot of Prediction Residuals")

# Time Series Plot of Actual vs. Predicted Exchange Rates
exchange_ts <- data.frame(
  Date = seq(as.Date("2020-01-01"), by = "day", length.out = length(train_output[[1]])),
  Actual = train_output[[1]],
  Predicted = predicted_values
)

ggplot(exchange_ts, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed") +
  labs(x = "Date", y = "Exchange Rate", color = "Series", title = "Time Series Analysis of Exchange Rates")

# Print test performance metrics
test_rmse <- sapply(models, function(model) {
  predicted_values <- predict(model, as.matrix(train_input[[1]]))
  sqrt(mean((predicted_values - train_output[[1]])^2))
})
test_mae <- sapply(models, function(model) {
  predicted_values <- predict(model, as.matrix(train_input[[1]]))
  mean(abs(predicted_values - train_output[[1]]))
})
test_mape <- sapply(models, function(model) {
  predicted_values <- predict(model, as.matrix(train_input[[1]]))
  mean(abs((predicted_values - train_output[[1]]) / train_output[[1]])) * 100
})
test_smape <- sapply(models, function(model) {
  predicted_values <- predict(model, as.matrix(train_input[[1]]))
  mean(2 * abs(predicted_values - train_output[[1]]) / (abs(predicted_values) + abs(train_output[[1]]))) * 100
})


# Print test performance metrics
cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")
cat("Test MAPE:", test_mape, "\n")
cat("Test sMAPE:", test_smape, "\n")
