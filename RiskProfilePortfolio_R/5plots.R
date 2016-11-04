rm(list = ls())


# setting figure parameter ------------------------------------------------

logPNG = TRUE # TRUE: PNG,  FALSE: PDF are returned
res    = 200 # resolution of png
width  = 8 # figure width in inch
height = 5 # figure height in inch


# producing PDF/PNG  ------------------------------------------------------

for(dat in list.files(pattern = "stoxxPortfolioMean_")){
  # loading data
  xxx = load(dat)
  name = gsub(pattern = "stoxxPortfolioMean_", replacement = "" , xxx)
  assign(x = "stoxx", value = get(xxx))
  # computing tables of cumulative end performance
  for (x in c("Random","Sharpe")){
    log = grep(pattern = paste0("portCluster", x) , names(stoxx))
    names = names(stoxx)[log]
    nam = gsub(pattern = paste0("portCluster", x), "", names)
    for (i in nam){
      nam2 = strsplit(i, "_AGGMETH")[[1]][2]
      nam[i == nam] = ifelse(is.na(nam2), i, nam2)
    }
    tab = matrix(0, 
                 nrow = length(names), 
                 ncol = dim(stoxx[[1]])[2], 
                 dimnames = list(nam, gsub(pattern = "portMean.","",names(stoxx[[1]])))
    )
    tab = as.data.frame(tab)
    names=cbind(names,nam)
    for (i2 in 1:dim(names)[1]){
      cl = names[i2,1]
      d=dim(stoxx[[cl]])
      nam = names[i2,2]
      tab[nam,] = stoxx[[cl]][d[1],]
    }
    assign(paste0("tab",x),tab)
  }
  
  # figure 1: K-Means plots -------------------------------------------------
  for (i in c("Sharpe","Random")){#}
    nam = paste0("portCluster",i,"Km")
    if(logPNG){png(file = paste0("figures\\portfolioKMeans",i,".png"), width = width, height = height, units = "in",res = res)
    }else{pdf(file = paste0("figures\\portfolioKMeans",i,".pdf"),width = width,height = height)}
      plot(rep(1,dim(stoxx[[nam]])[1]),type="n",col="white",xaxt = "n",
           main = paste0("KMeans with Portfolios selected ",ifelse(i=="Random","randomly","by Sharpe ratio")),
           ylab = "rel. Portfolio Performance",
           xlab = NA,
           ylim = range(stoxx[[nam]])
           )
      numDim = dim(stoxx[[nam]])[2]
      for (i2 in 1:numDim){
        lines(stoxx[[nam]][,i2],col=rainbow(numDim, end=0.75,alpha = 0.75)[i2],lwd=0.5)
      }
      num = as.integer(gsub("portMean.cluster","",names(stoxx[[nam]])))
      if (length(num)>6){
        ausw = ceiling(seq(min(num),max(num),(length(num)/5)))
      }else {
        ausw = num
      }
      
      legend("topleft", legend = c(ausw), col = rainbow(length(num), end=0.75,alpha = 0.75)[ausw-min(ausw)+1],
             lty = c(1),title = "Cluster")
  
      
      lab = as.integer(substr(rownames(stoxx[[nam]]),1,4))
      at=c(1,grep(pattern = 1,x=diff(lab) )+1)
      lab2 = lab[at]
      at=c(at,length(lab))
      lab2 = c(lab2,2016)
      axis(side = 1,at = at,labels = lab2)
    dev.off()
  }
  
  # figure 2: Logo ----------------------------------------------------------
  if(logPNG){png(file = paste0("figures\\portfolioKMeansSharpeLogo.png"), width = width/4, height = height/4, units = "in",res = res)
  }else{pdf(file = paste0("figures\\portfolioKMeansSharpeLogo.pdf"),width = width/4,height = height/4)}
    par(mar=c(0,0,0,0))
    nam = "portClusterSharpeKm"
    plot(rep(1,dim(stoxx[[nam]])[1]),type="n",col="white",xaxt = "n",yaxt="n",
         main = NA, #paste0("KMeans with Portfolios selected ",ifelse(i=="Random","randomly","by Sharpe ratio")),
         ylab = NA, #"rel. Portfolio Performance",
         xlab = NA, #"Time",
         ylim = range(stoxx[[nam]])
    )
    for (i2 in 1:dim(stoxx[[nam]])[2]){
      lines(stoxx[[nam]][,i2],col=rainbow(dim(stoxx[[nam]])[2], end=0.75,alpha = 0.75)[i2],lwd=0.5)
    }
  dev.off()
  
  # figure 3: End cumulative Performance Hier Cluster -----------------------
  if(logPNG){png(file = paste0("figures\\portfolioPerformanceHier_",name,".png"), width = width, height = height, units = "in",res = res)
  }else{pdf(file = paste0("figures\\portfolioPerformanceHier_",name,".pdf"), width = width, height = height)}
    start  = c(0,0,0)
    end    = c(0,1,1)
    numDim = as.integer(gsub(pattern = "portMean.cluster", "", names(stoxx[[nam]])))
    plot(numDim, rep(1, length(numDim)), type = "n",
         ylim = range(c(tabRandom[-1, ], tabSharpe[-1, ])),
         main = "Cumulative Portfolio Performance Hierarchical Cluster Methods",
         ylab = "rel. Portfolio Performance",
         xlab = "No. of Cluster"
         )
    for (x in c("Random","Sharpe")){
      tab = get(paste0("tab", x))
      d   = dim(tab)
      colorMap = function(n,start,end,...){
        colors = lapply(X = seq(from = 0,to = (n-1))/(n-1),function(x) start + (end-start)*x)
        colors = sapply(colors, function(x) rgb(x[1],x[2],x[3],...))
        rainbow(n = n,start = 0.7,end = 0.1)
      }
      pchList = c(0,1,2,5,6)
      for (i in 2:d[1]){
        lines(x = numDim, y = tab[i, ], 
              lty = ifelse(x == "Random", 2, 1),
              col = colorMap(length(2:d[1]), start, end, alpha = 0.75)[i - 1]
        )
      }
    }
    legend("topright", cex = 0.5, 
           legend = c(rownames(tab)[-1], "Random", "Sharpe"),
           col = c(colorMap(length(2:d[1]), start, end, alpha = 1)[2:d[1] - 1], rep(rgb(0, 0, 0), 2)),
           lty = c(rep(1,5),2,1),title = "Hier. Cluster")
  dev.off()
} 
# figure 4: End cumulative Performance K-Means ----------------------------
if(logPNG){png(file = paste0("figures\\portfolioPerformanceKMeans.png"), width = width, height = height, units = "in",res = res)
}else{pdf(file = paste0("figures\\portfolioPerformanceKMeans.pdf"), width = width, height = height)}
start  = c(0,0,0)
end    = c(0,1,1)
numDim = as.integer(gsub(pattern = "portMean.cluster", "", names(stoxx[[nam]])))
plot(numDim, rep(1, length(numDim)), type = "n",
     ylim = range(c(tabRandom[1,],tabSharpe[1,])),
     main = "Cumulative Portfolio Performance KMeans Cluster Methods",
     ylab = "rel. Portfolio Performance",
     xlab = "No. of Cluster"
)
for (x in c("Random","Sharpe")){
  tab = get(paste0("tab",x))
  d = dim(tab)
  colorMap = function(n,start,end,...){
    colors = lapply(X = seq(from = 0,to = (n-1))/(n-1),function(x) start + (end-start)*x)
    colors = sapply(colors, function(x) rgb(x[1],x[2],x[3],...))
    rainbow(n = n,start = 0.7,end = 0.1)
  }
  pchList = c(0,1,2,5,6)
  lines(x = numDim,y = tab[1,],
        lty = ifelse(x=="Random",2,1),
        col = colorMap(length(2:d[1]),start,end,alpha=0.75)[1]
  )
}
legend("topright", legend = c("Random","Sharpe"),lty = c(2,1),title = "KMeans Cluster")
dev.off()  