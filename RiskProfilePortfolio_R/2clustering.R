rm(list = ls())
# This script creates for all distance measures a file, in which a list for all 
# years, number of clusters and all agglomeration methods is added
###########################################################################
# Setting parameters ------------------------------------------------------
###########################################################################
#determine parameters:
K            = 2:5 # number of clusters
distMethods  = c("euclidean", "maximum", "manhattan", "canberra", "binary", 
                "minkowski") # all hier. clustering distance measure
aggloMethods = c("ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid") 
# agglomeration method! Options: "ward.D", "ward.D2", "single", "complete", 
# "average", "mcquitty", "median", "centroid"
###########################################################################


# Activating libraries and load data --------------------------------------

libraries = c("gdata", "cluster", "fpc", "R.matlab")
lapply(libraries, function(x)if (!(x %in% installed.packages())) {install.packages(x)})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)
load("stoxxYearProperties.Rdata") 

# Functions for Hierarchical Clustering -----------------------------------
clHi = function(obj, ...) {
  pmatrix = scale(obj$riskMeasures) # scale data
  clustering_hierarchical = function(x, K, distancemeasure, agglo_meth){
    # goal: returning matrix with cluster assignments
    # computation of distance matrix
    if (distancemeasure == "minkowski") {
      d = dist(x, method = distancemeasure, p = 10 ^ (-2))
    } else{
      d = dist(x, method = distancemeasure)
    }
    # computing hclust obj 'pfit' 
    pfit   = hclust(d, method = agglo_meth)
    # cutting dendrogram (tree) at point with K clusters
    groups = cutree(pfit, k = K)
    # naming columnames meaningfull
    if (is.vector(groups)) {
      groups = matrix(groups, dimnames = list(names(groups), paste0("cluster", K)))
    } else{
      colnames(groups) = paste0("cluster", K)
    }
    class(groups) = c(class(groups),"hwmsCluster")
    return(groups)
  }
  
  # adding hier. cluster with agglomeration method 'agglo_method'
  obj[[paste0("clHi_DIST", distancemeasure, "_AGGMETH", agglo_meth)]] = 
    clustering_hierarchical(x = pmatrix, K = K, distancemeasure, agglo_meth)
  # adding attribute
  if (!("hierarchicalCluster" %in% attr(obj, "hwmsCluster"))) {
    attr(obj, "hwmsCluster") = c(attr(obj, "hwmsCluster"), "hierarchicalCluster")
  }
  return(obj)
}
addClusterHierarchical = function(obj, K , distancemeasure = distancemeasure, 
                                  agglo_meth = agglo_meth){
  addClHi = function(obj, ...) {
    # error handling
    if (class(obj) != "hwmsCluster_riskMeasures")
      stop("obj has wrong class! should be 'hwmsCluster_riskMeasures'")
    # cumputation of hier. clusters
    obj = clHi(obj = obj, ...)
    return(obj)
  }
  # adding for all years hierarchical cluster
  for (i in ls(obj))
    obj[[i]] = addClHi(obj = obj[[i]])
  return(obj)
}

# Functions for K-Means clustering ----------------------------------------
addClusterKMeans_one = function(obj, K){
  addClKMe = function(x,K){
    # creating data frame with cluster choice
    tmp = function(x,K){
      # returning cluster choice for one number of clusters
      if(any(is.na(x))){
        # error handling: if time series contains NA value
        colLog = apply(X = is.na(x), MARGIN = 2, FUN = any)
        rowLog = apply(X = is.na(x), MARGIN = 1, FUN = any)
        war    = paste0("There has been a NA in riskMeasure '", colnames(x)[colLog], "' of ",
                     " and it has been left out. This occured for ", rownames(x)[rowLog])
        warning(war)
        x = x[,!colLog]
      }
      fit = kmeans(x, K)  # 5 cluster solution
      return(fit$cluster)
    }
    tmp2           = sapply(X = K, FUN = tmp, x = x)
    colnames(tmp2) = paste0("cluster", K)
    class(tmp2)    = c(class(tmp2), "hwmsCluster")
    return(tmp2)
  }
  # adding K-Means cluster  
  obj$clKm = addClKMe(x = obj$riskMeasures, K = K)
  # adding attribute to object
  if (!("KMeansCluster" %in% attr(obj, "hwmsCluster")))
    attr(obj, "hwmsCluster") = c(attr(obj, "hwmsCluster"), "KMeansCluster")
  return(obj)
}
addClusterKMeans = function(obj, K) {
  # adding K-Means cluster for all years  
  for (i in names(obj))
    obj[[i]] = addClusterKMeans_one(obj = obj[[i]], K)
  return(obj)
}


# Computing and saving cluster choices ------------------------------------

for (distancemeasure in distMethods){
  stoxxPropTMP = stoxxProp
  # adding K-Means clusters
  stoxxPropTMP = addClusterKMeans(obj = stoxxPropTMP, K)
  # adding hierarchical clusters for all aggloMethods
  for (agglo_meth in  aggloMethods)
    stoxxPropTMP = addClusterHierarchical(obj = stoxxPropTMP, K, distancemeasure, agglo_meth)
  # name of distance out
  dist = paste0(toupper(substr(distancemeasure, 1, 1)), substr(distancemeasure, 2, nchar(distancemeasure)))
  name = paste0("stoxxProp", dist)
  # saving list named as variable 'name' says
  assign(name, value = stoxxPropTMP)
  do.call(save, list(name, file = paste0("stoxxPropCluster", dist, ".Rdata")))
}
