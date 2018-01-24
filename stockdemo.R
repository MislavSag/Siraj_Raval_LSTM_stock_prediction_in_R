url <- "https://raw.githubusercontent.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/master/sp500.csv"
sp500 <- read.csv(url, header = FALSE, stringsAsFactors = FALSE)
colnames(sp500) <- "closingPrice"

# choose sequence length
seq_length <- 50
sequence_length <- seq_length + 1
result <- list()
for (i in 1:(nrow(sp500) - seq_length)){
  result[[i]] <- sp500[i : (i + seq_length),1]
}

# normalised data
normalised_data <- list()
for (i in 1:length(result)){
  normalised_window <- ((result[[i]] / result[[i]][[1]]) - 1)
  normalised_data[[i]] <- normalised_window
}
result <- normalised_data

# test <- do.call(rbind, result)
# define train and test datasets
row <- round(0.9 * length(result))
train <- result[1:as.integer(row)]
# train <- sample(train)
x_train <- lapply(train, '[', -length(train[[1]]))
y_train <- lapply(train, '[', length(train[[1]]))
y_train <- unlist(y_train)
test = result[(as.integer(row)+1):length(result)]
x_test <- lapply(test, '[', -length(test[[1]]))
y_test <- lapply(test, '[', length(test[[1]]))

# x_train <- array(as.numeric(unlist(x_train)), dim = c(3709, 50, 1))
# x_test <- array(as.numeric(unlist(x_test)), dim = c(412, 50, 1))
x_train <- array_reshape(as.numeric(unlist(x_train)), dim = c(3709, 50, 1))
x_test <- array_reshape(as.numeric(unlist(x_test)), dim = c(412, 50, 1))


#########################
# Step 2: Build a model #
#########################

library(keras)

model <- keras_model_sequential()
model %>% layer_lstm(units = 50L, return_sequences = TRUE, input_shape = list(NULL, 1)) %>%
  layer_dropout(0.2) %>%
  layer_lstm(units = 100L, return_sequences = FALSE) %>%
  layer_dropout(0.2) %>%
  layer_dense(1L) %>%
  layer_activation('linear')
summary(model)

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'mse'
)

###########################
# Step 2: Train the model #
###########################

model %>% fit(x_train, y_train, epochs=5, batch_size=512, validation_split = 0.05)


################################
# Step 2: Plot the predictions #
################################

predict_sequences_multiple <- function(model, data, window_size, prediction_len){
  #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
  prediction_seqs = list()
  for (i in 1:as.integer(nrow(data)/prediction_len)){
    curr_frame = array(data[i*prediction_len,,], dim = c(1,prediction_len,1))
    predicted = list()
    for (j in 1:prediction_len){
      predicted[[j]] <- predict_on_batch(model, curr_frame)[1]
      curr_frame <- array_reshape(curr_frame[,2:50,], dim = c(1,49,1))
      curr_frame <- array(c(curr_frame, predicted[[j]]), dim = c(1,prediction_len,1))
    }
    prediction_seqs[[i]] <- unlist(as.numeric(predicted))
  }
  return(prediction_seqs)
}
predictions <- predict_sequences_multiple(model, x_test, 50, 50)
predictions <- data.frame(pred = unlist(predictions), stringsAsFactors = FALSE)


library(ggplot2)
library(tidyr)
library(rowr)
library(dplyr)
library(optmach)

# fr <- as.data.frame(unlist(predictions))
plot_data <- data.frame(y_test = unlist(y_test), stringsAsFactors = FALSE)
plot_data <- cbind.fill(plot_data, predictions, fill = NA)
number_of_predictions <- nrow(plot_data) %/% 50
cols <- paste0("Prediction ", 1:number_of_predictions)
help_vector <- c(1, seq(50, number_of_predictions*50, by = 50))
for (i in 1:number_of_predictions){
  if(i == 1){
    plot_data[,cols[i]] <- NA
    plot_data[help_vector[i]:help_vector[i+1],cols[i]] <- c(plot_data[(help_vector[i]):help_vector[i+1],"pred"])
  }else{
    plot_data[,cols[i]] <- NA
    x <- plot_data[help_vector[i]+1,"pred"] - plot_data[help_vector[i]+1,"y_test"]
    plot_data[(help_vector[i]+1):(help_vector[i+1]),cols[i]] <- c(plot_data[(help_vector[i]+1):help_vector[i+1],"pred"]) - x
  }
}

plot_data[,"pred"] <- NULL
plot_data <- gather(plot_data, key = "key", value = "value")
plot_data <- plot_data %>% dplyr::group_by(key) %>% dplyr::mutate(n = 1:n())

ggplot(plot_data, aes(x = n, y = value, col = key)) + geom_line()


