rm(list=ls())
# Main goal of this script is the preparation of data for cluster analysis and 
# portfolio selection

# Loading Libraries and Functions -----------------------------------------

libraries = c("data.table","moments","PerformanceAnalytics")
lapply(libraries, function(x) if (!(x %in% installed.packages())){
  install.packages(x)})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Loading Data ------------------------------------------------------------

stoxx = read.csv("STOXX_USA_19980101_20160101_with_Names.csv",
                 header = T,sep = ";",dec = ",")
stoxx = stoxx[1:1040, c(1:10, dim(stoxx)[2])]
# using Date-format instead of integers
stoxx[, "Date"] = as.Date(stoxx[, "Date"], origin = "1899-12-30")

# Functions ---------------------------------------------------------------

colDateFunc = function(obj){
  # find column named 'Date'
  colN = colnames(obj)
  if (any(is.na(colN)))
    colN[is.na(colN)] = "NA"
  colDate = (colN == "Date" | colN == "date" | colN == "DATE")
  colDate
}
price2ret = function(obj){
  colDate = colDateFunc(obj)
  # stopping if data contain Neg. Prices, column 'Date' are missing or if there 
  # are more than one column named Date, plus if there is at least once no time step
  if(any(obj < 0 & !is.na(obj)))
    stop("NegativePrices")
  if (!any(colDate))
    stop("Date as column name is missing!")
  if (sum(colDate) > 1)
    stop("Date as column name is more than once there!")
  if (all(colDate))
    stop("Only Date is in prices!")
  if (any(diff(as.integer()) <= 0))
    stop("No time step at one point")
  # computing date differences
  dateDiff = as.integer(diff(obj[, colDate]))
  # computing log returns 
  obj[, !colDate] = apply(X = obj[, !colDate], MARGIN = 2, FUN = log)
  obj[-1, !colDate] = apply(X = obj[, !colDate], MARGIN = 2, FUN = diff)
  obj = obj[-1, ]
  # standardizing log returns by difference of days
  for (i in grep(pattern = TRUE,x = !colDate))
    obj[,i] = obj[,i]/dateDiff
  return(obj)
}
stoxxAttr = function(obj,yyyy,alpha = 0.05){
  # goal: computing risk measures of year 'yyyy' and returning it
  colDate = colDateFunc(obj)
  obj2 = obj[year(obj[, colDate]) == yyyy, ]
  # leaving all time series out with missing data (NA)
  obj3 = obj2[, !apply(X = is.na(obj2), MARGIN = 2, FUN = any)]
  colDate = colDateFunc(obj3)
  # assigning stoxx values and all risk measures to list 'ans'
  ans       = list()
  ans$stoxx = obj3
  ans$sigma = sqrt(apply(X = obj3[, !colDate], MARGIN = 2, FUN = var))
  ans$skew  = apply(obj3[, !colDate], MARGIN = 2, FUN = skewness)
  ans$kurt  = apply(obj3[, !colDate], MARGIN = 2, FUN = kurtosis)
  ans$VaR   = apply(X = obj3[, !colDate], MARGIN = 2, FUN = quantile, probs = alpha)
  ans$ES    = ans$VaR
  for (nam in colnames(obj3[,!colDate])){
    i  = colnames(obj3) == nam
    i2 = names(ans$VaR) == nam
    ans$ES[i2] = sum(obj3[obj3[,i] < ans$VaR[i2],i])/(dim(obj3)[1]*alpha)
  }
  market      = colnames(obj)[dim(obj)[2]]
  CAPM.beta2  = function(x)
    CAPM.beta(Ra = as.ts(x), Rb = as.ts(obj3[, market]))
  ans$beta    = apply(X = obj3[, !colDate], MARGIN = 2, FUN = CAPM.beta2)
  sharpeRatio = function(x) mean(x) / sd(x)
  ans$sharpe  = apply(X = obj3[, !colDate], MARGIN = 2, FUN = sharpeRatio)
  return(ans)
}
addYear = function(y){
  # assigning year 'y's risk measures to environment 'stoxxYear'
  assign(x = paste0("stoxx",y),value = stoxxAttr(obj = stoxxLog,yyyy = y),
         envir = stoxxYear)
} 

# Compuatation and saving data --------------------------------------------

# all years with more than 1 data point
years = table(year(stoxx[, 1]))
years = names(years[years > 1])
# Computing log returns of the prices
stoxxLog  = price2ret(stoxx)
stoxxYear = new.env()
# function 'addYear' is called for every year. 'addYear' splits log-returns
# yearly and computes risk measures.
sapply(X = years, FUN = addYear)
# transforming riskmeasures into dataframe
stoxxProp = new.env()
for (itime in years) {
  tmp = get(x = paste0("stoxx", itime), envir = stoxxYear)
  tmp = tmp[!grepl(pattern = "stoxx", x = names(tmp))]
  tmp = list(riskMeasures = data.frame(tmp))
  class(tmp) = "hwmsCluster_riskMeasures"
  assign(x = paste0("stoxx", itime),value = tmp,envir = stoxxProp)
}

# coersion of environements into list and saving the data
stoxxProp = as.list(stoxxProp)
save(stoxxProp, file = "stoxxYearProperties.Rdata")
stoxxYear = as.list(stoxxYear)
save(stoxxYear, file = "stoxxYearAttr_with_data.Rdata")