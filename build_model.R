input_data <- read.csv('train_input_data.csv')
input_data$X <- NULL
output_data <- read.csv('train_output_data.csv')
output_data$X <- NULL
colnames(output_data) <- c('output')
data <- cbind(input_data,output_data)

test_input_data <- read.csv('test_input_data2.csv')
test_input_data$X <- NULL

train_data <- data

myNtree  =	100
myMtry  =	5
myImportance  =	TRUE
fit  <- randomForest(output	~	.,	data=train_data,	ntree=myNtree,	mtry=myMtry,	
                         importance=myImportance)

ans2 <- predict(fit,newdata = test_input_data)
write(ans2,file="output_prediction.txt",sep = "\n")
