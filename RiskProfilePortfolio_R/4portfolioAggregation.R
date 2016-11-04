rm(list=ls())

# installing and activating libraries -------------------------------------

libraries = c("foreach","doParallel")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {install.packages(x)})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Functions ---------------------------------------------------------------

isHwmsPort    = function(x) return("hwmsPort" %in% class(x))
portCalc      = function(obj,nam,clName,clNum,...){
  # goal: calculate portfolio according to selected stocks
  # selecting year before
  namBef   = paste0("stoxx",as.integer(substr(nam,6,9))-1) # Using indexes of year before
  cluster  = obj[[namBef]][[clName]]
  clustOne = lapply(as.list(cluster[clNum,]),FUN = sort)
  # finding names of stocks in portfolio
  tmp = lapply(clustOne, FUN = function (x) rownames(obj[[namBef]]$riskMeasures)[x])
  # short cut if all portfolios are the same
  if (all(sapply(X = tmp, FUN = identical, y = tmp[[1]]))) {
    tmp = tmp[[1]]
    warning(paste0("All simulated Portfolios of '", clName,
                   "' are the same such that only one is used!")
            )
  }
  # portfolio composition, 1/n means equally weighted portfolios, add other options here!
  portComp = function(x, method = "1/n") {
    if (method == "1/n"){
      return(rowSums(x) / dim(x)[2])
    }
  }
  # getting log returns of stocks 
  stoxxSmall = stoxx[[nam]][["stoxx"]][, ]
  if (is.list(tmp)) {
    logi = lapply(tmp, FUN = function(x) colnames(stoxxSmall) %in% x)
    portSimu = sapply(logi, FUN = function(x) portComp(stoxxSmall[, x]))
  } else{
    logi = colnames(stoxxSmall) %in% tmp
    portSimu = portComp(stoxxSmall[, logi])
  }
  # adding date and columnname
  port = as.data.frame(portSimu, )
  rownames(port) = stoxxSmall[, "Date"]
  colnames(port) = paste0(clNum, "port", 1:dim(port)[2])
  # computing portfolio mean!
  portMean = NULL
  if (!is.vector(portSimu)) {
    portMean = as.data.frame(rowSums(port) / dim(portSimu)[2])
    rownames(portMean) = stoxxSmall[, "Date"]
    colnames(portMean) = paste0(clNum)
  } else{
    portMean = port
    colnames(portMean) = paste0(clNum)
  }
  # returning data 
  class(port) = c(class(port), "hwmsPortValueSimu")
  class(portMean) = c(class(portMean), "hwmsPortValueMean")
  return(list(port = port,portMean = portMean))
}
portMean      = function(obj,clName,nam,...){
  library("doParallel")
  clNum = dimnames(obj[[nam]][[clName]])[[1]]
  # calculating portfolios for each cluster
  res = foreach(K = clNum, .export = c("portCalc", "stoxx")) %do% {
    portCalc(obj = obj, clName = clName, clNum = K, nam = nam)
  }
  # restoring names of clusters, since foreach does not return them
  clusterNames = sapply(X = 1:length(res), function(x) names(res[[x]]$portMean))
  names(res)   = clusterNames
  # saving mean portfolio and the simulated ones in variables for each cluster
  portMean     = as.data.frame(lapply(names(res), FUN = function(x) res[[x]]$portMean))
  portSimu     = lapply(names(res), FUN = function(x) res[[x]]$port)
  names(portSimu) = clusterNames
  class(portMean) = c(class(portMean),"hwmsPortMean")
  class(portSimu) = c(class(portSimu),"hwmsPortSimu")
  # returning mean portfolios and simulated 
  result = list("portMean" = portMean)#,"portSimu" = portSimu)
  class(result) = c(class(result),"hwmsPortCluster")
  attr(result,"tmpName") = paste0("portCluster",substr(clName,start = 5,stop = nchar(clName)))
  return(result)
}
portfoliosOne = function(obj,nam,...){
  # computing mean portfolios for one year
  library("doParallel")
  ports  = sapply(X = obj[[nam]], FUN = isHwmsPort)
  clName = names(ports[ports])
  res    = foreach(i = clName,.export = c("portMean", "portCalc", "stoxx")) %do% {
    portMean(obj = obj,nam = nam,clName = i)
  }
  names(res) = sapply(X = res, FUN = function(x) {
      tmp = attributes(x)$tmpName
      attributes(x)$tmpName = NULL
      return(tmp)
    }
  )
  obj = res
  return(obj)
}
portfolios    = function(obj){
  # computing mean portfolios for all years, using 'portfoliosOne'
  res = list()
  res = foreach(nam = sort(names(obj))[-1],
                .export = c("portfoliosOne","isHwmsPort","CORES","portMean",
                            "portCalc","stoxx")) %dopar%{
    portfoliosOne(obj = obj,nam = nam)
  }
  for (i in 1:length(res))
    names(res)[i] = paste0("stoxx", substr(rownames(res[[c(i, 1)]]$portMean)[1], 1, 4))
  return(res)
}
cumPortfolios = function(obj){
  # computing cumulated performance of portfolios for all years and cluster types
  port = list()
  for (cl in names(obj[[1]])){#}
    port[[cl]] = data.frame()
    for (nam in sort(names(obj))){
      port[[cl]] = rbind(port[[cl]],as.data.frame(obj[[nam]][[cl]]))
    }
    port[[cl]] = exp(port[[cl]])
  }
  port2 = port
  for (cl in names(obj[[1]])){
    port2[[cl]] = cumprod(port2[[cl]])
  }
  
  return(port2)
}

# Computation and saving --------------------------------------------------

# setting parrallel computing
CORES = detectCores()/2
no_cores = floor(CORES)
cl1 = makeCluster(no_cores)
registerDoParallel(cl1)

dat = list.files()[grepl(pattern = "stoxxPort",x=list.files())
                   &!grepl(pattern = "stoxxPortfolioMean",x=list.files())]
for (indDat in dat){
  # loading data
  namDist   = load(indDat)
  namStoxx  = load("stoxxYearAttr_with_data.Rdata")
  stoxxPort = get(namDist)
  stoxx     = get(namStoxx)
  # removing data from memory
  rm(list = namDist)
  rm(list = namStoxx)
  
  # computing mean portfolios and cumulated ones
  stoxxPort2 = portfolios(stoxxPort)
  port       = cumPortfolios(stoxxPort2)
  # naming and saving data
  name = gsub(pattern = "Port", "PortfolioMean_", namDist)
  assign(name, port)
  do.call(save, list(name, file = paste0(name, ".Rdata")))
}
stopCluster(cl1)
