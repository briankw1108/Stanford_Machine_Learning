setwd("C:/Users/u213493/Desktop/Stanford Machine Learning/machine-learning-ex1/ex1")

#load dataset
suppressMessages(library(ggplot2))
data = read.table(file = "ex1data1.txt", sep = ",")
data[, 3] = 1
data = data[, c(3, 1, 2)]
names(data) = c("x_0", "x_1", "y")

#set up X matrix with x0 and x1, y matrix with output y
X = as.matrix(data[, 1:2]); colnames(X) = NULL
y = as.matrix(data[, 3]); colnames(y) = NULL
m = nrow(X)
#set up a matrix for theta0 and theta1
theta = matrix(data = c(0, 0), nrow = 2, ncol = 1)

# Cost funtion J(theta) = 1/2m * sum(H(theta) - y)^2
J = sum((1/(2*m) * ((X %*% theta) - y)^2))
print(paste("J = ", round(J, 3), sep = ""))

# Gradient Descent
alpha = 0.01 # learning rate
alltheta = matrix(data = c(0, 0), nrow = 2, ncol = 1)
#get all theta values calculated by gredient descent
for (i in 1:3000) { 
        theta = theta - ((t(X) %*% ((X %*% theta) - y)) * (alpha / m))
        alltheta = cbind(alltheta, theta)
}

alltheta = t(alltheta)
colnames(alltheta) = c("theta_0", "theta_1")
#final optimum thetas (global optimal)
print(tail(alltheta, 1))

windows()
plot(alltheta[, 1], alltheta[, 2])

#get all J from all thetas
all_J = numeric()
for (i in 1:ncol(t(alltheta))) {
        J = sum((1/(2*m) * ((X %*% t(alltheta))[, i] - y)^2))
        all_J = c(all_J, J)
}
#plot scatter plot and fitted lines
g = ggplot(data = data, aes(x = x_1, y = y))
g = g + geom_point(colour = "red", size = 3) +
        xlab("Population of City in 10,000s") +
        ylab("Profit in $10,000s") +
        geom_abline(intercept = alltheta[, 1], slope = alltheta[, 2], colour = "blue", alpha = 0.01) 
windows()
print(g)
