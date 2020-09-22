library(forecast)
library(tseries)

data("AirPassengers")
class(AirPassengers)
start(AirPassengers)
end(AirPassengers)
frequency(AirPassengers)
sum(is.na(AirPassengers))
summary(AirPassengers)

AirPassengers
tsdata <- ts(AirPassengers, frequency = 12)
ddata <- decompose(tsdata, "multiplicative")

plot(ddata)
plot(AirPassengers)
abline(reg = lm(AirPassengers~time(AirPassengers)))
cycle(AirPassengers)
boxplot(AirPassengers~cycle(AirPassengers))

mymodel <- auto.arima(AirPassengers)
mymodel
auto.arima(AirPassengers,ic = "aic", trace = TRUE)


plot.ts(mymodel$residuals)
acf(ts(mymodel$residuals))
pacf(ts(mymodel$residuals))


futurecast <- forecast(mymodel, level = c(95), h = 10*12)
plot(futurecast)

Box.test(mymodel$residuals, lag = 5, type = "Ljung-Box")
Box.test(mymodel$residuals, lag = 10, type = "Ljung-Box")
Box.test(mymodel$residuals, lag = 15, type = "Ljung-Box")
