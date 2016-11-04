rm(list=ls())
# Parameter selection -----------------------------------------------------
num = 100 # number of potfolios
selection = c("Sharpe","random") # portfolio selection "random" or by highest "sharpe" ratio. 
# If there are more than one stocks with maximal Sharpe ratio one of them is 
# selected randomly.

# installing and activating libraries -------------------------------------

libraries = c("foreach","doParallel")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {install.packages(x)})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

# Functions ---------------------------------------------------------------

select = function (log,selection,obj,...){
  # goal: selection of stocks out of cluster
  ind = apply(X = log, MARGIN = 2,grepl,pattern = TRUE)
  # random picking function
  selRandom = function(x){
    x = grep(pattern = T,x = x)
    x[floor(runif(1,1,length(x)+1))]
  }
  if(selection == "random"){
    ind = apply(X = ind,MARGIN = 2,FUN = selRandom)
  }else if (selection == "sharpe"){
    # function to select stocks out of cluster via highest sharpe ratio
    selSharpe = function(x){
      if (!any(x))
        return(NULL)
      m = max(obj$riskMeasures[x, "sharpe"])
      ans = grepl(pattern = m, x = obj$riskMeasures[, "sharpe"])
      if (sum(ans) > 1)
        ans = ans & 1:dim(obj$riskMeasures)[1] & x
      ans2 = grep(pattern = T, x = ans)
      # if by chance 2 or more stocks have the same max sharpe ratio, one of them
      # is picked randomly
      if (length(ans2) > 1)
        ans2 = selRandom(1:length(ans) %in% ans2)
      return(ans2)
    }
    ind = apply(X = ind, MARGIN = 2, FUN = selSharpe)
  }
  return(as.list(ind))
}
listUnion = function(x,y){
  # function to union two lists according to list elements names
  tmp = as.list(lapply(names(x), FUN = function(z) c(x[[z]], y[[z]])))
  tmp = lapply(tmp, FUN = function(z) 
    z[!is.na(z)]
  )
  names(tmp) = names(x)
  return(tmp)
}
portSelOne = function(obj,clusterType, selection = "random",...){
  # 'select'ing one stock per cluster
  colNam = colnames(obj[[clusterType]])
  clustK = as.integer(gsub(pattern = "cluster", replacement = "", x = colNam))
  port   = select(log = obj[[clusterType]] == 1, selection, obj)
  for (i in 2:max(clustK))
    port = listUnion(port, select(log = obj[[clusterType]] == i, selection, obj))
  return(port)
}
addPort = function(obj,n,selection = "random",clusterType = "all",...){
  # goal: chosing  portfolios based on  cluster choice made before
  isHwmsCluster = function(x)  "hwmsCluster" %in% class(x)
  # if no special clusterType is choosen, all cluster types are identified and 
  # portfolios are computed for them
  if (clusterType == "all"){
    clTypeSearch = function(nam,obj){
      nam2 = names(obj[[nam]])
      lapply(X = nam2,FUN = function(x,...) ifelse(isHwmsCluster(obj[[nam]][[x]]),x,NA))
    }
    clType = unique(unlist(lapply(X = names(obj),FUN = function(nam) clTypeSearch(nam,obj = obj))))
    clType = clType[!is.na(clType)]
  }else{
    clType = clusterType
  }
  if (tolower(selection) %in% c("random","sharpe")){
    selection = tolower(selection)
  }else{ 
    stop("selection=c('random','sharpe')")
  }
  
  portMult = function(obj,clType,selection,...){
    # computing n portfolios for all clTypes
    for (cl in clType) {
      tmp = replicate(n, expr = portSelOne(obj, clusterType = cl, selection))
      class(tmp) = c(class(tmp), "hwmsPort")
      selection2 = paste0(toupper(substr(selection, 1, 1)), substr(selection, 2, nchar(selection)))
      obj[[paste0("port", selection2, substr(cl, 3, nchar(cl)))]] = tmp
    }
    return(obj)
  }
  obj2 = obj
  # computing portfolios for years parrallel
  res  = foreach(nam = sort(names(obj2)),.export = c("isHwmsCluster", "portSelOne","select", "listUnion")) %dopar% {
    tmp = portMult(obj = obj2[[nam]], clType, selection)
    attr(tmp, "year") = nam
    return(tmp)
  }
  # restoring names (foreach does not return names of list elements)
  for (i in 1:length(res)) {
    nam = attributes(res[[i]])$year
    attr(res[[i]], "year") = NULL
    obj2[[nam]] = res[[i]]
  }
  return(obj2)
}

# Computation and saving --------------------------------------------------

# setting parrallel computing
no_cores = floor(detectCores()/2)
cl       = makeCluster(no_cores)
registerDoParallel(cl)

# computing portfolios for all files in 'files'
files = list.files()[grepl(pattern = "stoxxPropCluster",x = list.files())]
for (xxx in files){
  nam1 = load(xxx)
  name = gsub(pattern = "Prop","Port", nam1)
  stoxxProp = get(nam1)
  
  # Adding portfolios for different clusters
  if ("sharpe" %in% tolower(selection)) {
    stoxxProp = addPort(obj = stoxxProp, n = 1, selection = "sharpe")
  } 
  if ("random" %in% tolower(selection)) {
    stoxxProp = addPort(obj = stoxxProp, n = num, selection = "random")
  }
  # saving data in file 'name'
  assign(name, stoxxProp)
  do.call(save, list(name, file = paste0(name, ".Rdata")))
}
# stopping multicore computing
stopCluster(cl)