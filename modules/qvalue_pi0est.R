pi0est <- function(p, lambda = seq(0.05,0.95,0.05), pi0.method = c("smoother", "bootstrap"),
                   smooth.df = 3, smooth.log.pi0 = FALSE, ...) {
  # Check input arguments
  rm_na <- !is.na(p)
  p <- p[rm_na]
  pi0.method = match.arg(pi0.method)
  m <- length(p)
  lambda <- sort(lambda) # guard against user input
  
  ll <- length(lambda)
  if (min(p) < 0 || max(p) > 1) {
    stop("ERROR: p-values not in valid range [0, 1].")
  } else if (ll > 1 && ll < 4) {
    stop(sprintf(paste("ERROR:", paste("length(lambda)=", ll, ".", sep=""),
                       "If length of lambda greater than 1,",
                       "you need at least 4 values.")))
  } else if (min(lambda) < 0 || max(lambda) >= 1) {
    stop("ERROR: Lambda must be within [0, 1).")
  }
  
  if (max(p) < max(lambda)) {
    stop("ERROR: maximum p-value is smaller than lambda range. Change the range of lambda or use qvalue_truncp() for truncated p-values.") 
  }
  
  # Determines pi0
  if (ll == 1) {
    pi0 <- mean(p >= lambda)/(1 - lambda)
    pi0.lambda <- pi0
    pi0 <- min(pi0, 1)
    pi0Smooth <- NULL
  } else {
    ind <- length(lambda):1
    #print(tabulate(findInterval(p, vec=lambda))[ind])
    #print((length(p) * (1-lambda[ind])))
    #print(findInterval(p, vec=lambda))
    pi0 <- cumsum(tabulate(findInterval(p, vec=lambda))[ind]) / (length(p) * (1-lambda[ind]))
    pi0 <- pi0[ind]
    pi0.lambda <- pi0
    # Smoother method approximation
    if (pi0.method == "smoother") {
      if (smooth.log.pi0) {
        pi0 <- log(pi0)
        spi0 <- smooth.spline(lambda, pi0, df = smooth.df)
        pi0Smooth <- exp(predict(spi0, x = lambda)$y)
        pi0 <- min(pi0Smooth[ll], 1)
      } else {
        spi0 <- smooth.spline(lambda, pi0, df = smooth.df, nknots=length(lambda))
        print(spi0)
        pi0Smooth <- predict(spi0, x = lambda)$y
        print(pi0Smooth)
        pi0 <- min(pi0Smooth[ll], 1)
      }
    } else if (pi0.method == "bootstrap") {
      # Bootstrap method closed form solution by David Robinson
      minpi0 <- quantile(pi0, prob = 0.1)
      W <- sapply(lambda, function(l) sum(p >= l))
      mse <- (W / (m ^ 2 * (1 - lambda) ^ 2)) * (1 - W / m) + (pi0 - minpi0) ^ 2
      pi0 <- min(pi0[mse == min(mse)], 1)
      pi0Smooth <- NULL
    } else {
      stop('ERROR: pi0.method must be one of "smoother" or "bootstrap".')
    }
  }
  if (pi0 <= 0) {
    warning("The estimated pi0 <= 0. Setting the pi0 estimate to be 1. Check that you have valid p-values or use a different range of lambda.")
    pi0 <- pi0.lambda <- 1
    pi0Smooth <- lambda <- 0
  }
  return(list(pi0 = pi0, pi0.lambda = pi0.lambda,
              lambda = lambda, pi0.smooth = pi0Smooth))
}

## -----------------------------------------------------------------------------
## Name: AverageEstimate.R
## R code for the (Cancer Informatics) paper by Hongmei Jiang and R.W. Doerge:
## "Estimating the proportion of true null hypotheses for multiple comparisons"
## Date: December 2007
## Contact: Hongmei Jiang  hongmei@northwestern.edu
##          R.W. Doerge    doerge@purdue.edu

## Example: 
##   y <- runif(1000)
##   Result <- AverageEstimate(y)
##   sum(Result$significant)
##   Result$pi0
## -----------------------------------------------------------------------------


############### Subfunction ############################
## Compute the estimate of pi0 for a fixed value of B ##
########################################################
FixedB <- function(p, B)
{
  ## Input:
  #p: a vector of p-values
  #B: an integer, the interval [0,1] is divided into B equal-length intervals
  
  ## Output:
  #pi0: an estimate of the proportion of true null hypotheses
  
  m <- length(p) 
  t <- seq(0,1,length=B+1)   # equally spaced points in the interval [0,1]
  
  NB <- rep(0,B)		    # number of p-values greater than t_i	
  NBaverage <- rep(0,B)      # average number of p-values in each of the (B-i+1) small intervals on [t_i,1]
  NS <- rep(0,B)             # number of p-values in the interval [t_i, t_(i+1)]
  pi <- rep(0,B)		    # estimates of pi0 	
  for(i in 1:B)
  {
    NB[i] <- length(p[p>=t[i]])
    NBaverage[i] <- NB[i]/(B-(i-1))
    NS[i] <- length(p[p>=t[i]]) - length(p[p>=t[i+1]])
    pi[i] <- NB[i]/(1-t[i])/m
  }
  #print(pi)
  i <- min(which(NS <= NBaverage))  # Find change point i
  print(i)
  print(pi[(i-1):B])
  pi0 <- min(1, mean(pi[(i-1):B]))          # average estiamte of pi0
  return(pi0)
}


############## main function ###########################
## (1) Compute the estimate of pi0                    ##
## (2) Apply the adaptive FDR controlling procedure   ##
########################################################
AverageEstimate <- function(p=NULL, Bvector=c(5, 10, 20, 50, 100), alpha=0.05)
{
  ## Input:
  #p: a vector of p-values
  #Bvector: a vector of integer values where the interval [0,1] is divided into B equal-length intervals
  #         When Bvector is an integer, number of intervals is consider as fixed. For example Bvector = 10;
  #         When Bvector is a vector, bootstrap method is used to choose an optimal value of B
  #alpha: FDR signficance level so that the FDR is controlled below alpha
  #Numboot: number of bootstrap samples
  
  ## Output:
  #pi0: an estimate of the proportion of true null hypotheses
  #significant: a vector of indicator variables; 
  #             it is 1 if the corresponding p-value is significant
  #	       it is 0 if the corresponding p-value is not significant
  
  # check if the p-values are valid
  if (min(p)<0 || max(p)>1) print("Error: p-values are not in the interval of [0,1]")
  m <- length(p) 		# Total number p-values
  
  Bvector <- as.integer(Bvector)  # Make sure Bvector is a vector of integers
  
  #Bvector has to be bigger than 1
  if(min(Bvector) <=1) print ("Error: B has to be bigger than 1")
  
  ######## Estimate pi0 ########
  if(length(Bvector) == 1)        # fixed number of numbers, i.e., B is fixed
  {
    pi0 <- AverageEstimateFixedB(p, Bvector)
  }
  else
  {
    Numboot <- 100
    OrigPi0Est <- rep(0, length(Bvector))
    for (Bloop in 1:length(Bvector))
    {
      OrigPi0Est[Bloop] <- FixedB(p, Bvector[Bloop])
    }
    
    BootResult <- matrix(0, nrow=Numboot, ncol=length(Bvector)) # Contains the bootstrap results
    
    for(k in 1:Numboot)
    {
      p.boot <- sample(p, m, replace=TRUE)    # bootstrap sample 
      for (Bloop in 1:length(Bvector))
      {
        BootResult[k,Bloop] <- FixedB(p.boot, Bvector[Bloop])
      }	
    }
    
    MeanPi0Est <- mean(OrigPi0Est)             # avearge of pi0 estimates over the range of Bvector
    MSEestimate <- rep(0, length(Bvector))     # compute mean-squared error
    for (i in 1:length(Bvector))
    {
      MSEestimate[i] <- (OrigPi0Est[i]- MeanPi0Est)^2
      for (k in 1:Numboot)
      {
        MSEestimate[i] <- MSEestimate[i]+1/Numboot*(BootResult[k,i] - OrigPi0Est[i])^2	
      }
    }
    pi0 <- OrigPi0Est[MSEestimate==min(MSEestimate)]
  }  # end of else           
  
  
  ######## Apply the adaptive FDR controlling procedure ########
  
  sorted.p <- sort(p) 		# sorted p-values
  order.p <- order(p)	 	# order of the p-values
  m0 <- pi0*m                    # estimate of the number of true null       
  i <- m
  
  crit <- i/m0*alpha
  
  while (sorted.p[i] > crit )
  { 
    i <- i-1
    crit <- i/m0*alpha
    if (i==1) break
  }
  K <- i
  if (K==1 & sorted.p[K] <= 1/m0*alpha) K <- 1
  if (K==1 & sorted.p[K] > 1/m0*alpha)  K <- 0
  
  significant <- rep(0,m)       # indicator of significance of the p-values
  if (K > 0) significant[order.p[1:K]] <- 1
  
  result <- list(pi0=pi0, significant=significant)
  result
}

pvals1 = c(0.2382444719108452,0.26279886806672315,0.8699015393957243,0.56188029422673,0.1943236010455187,0.37901293333507124,0.4477404985808746,0.7238081283411658,0.56188029422673,0.3913744655929684,0.6707198865659609,0.34724722722482404,0.4125376824265483,0.7207420181182407,0.9472130598426465,0.8369556595478901,0.2382444719108452,0.247278294021404,0.17484349328997395,0.6834120426994836,0.14049813558457036,0.4477404985808746,0.436747054936364,0.6299533188143605,0.14049813558457036,0.6834120426994836,0.4477404985808746,0.14049813558457036,0.4985913889378877,0.4125376824265483,0.60200001091326,0.9472130598426465,0.6426468421463174,0.60200001091326,0.6557124831605461,0.2891372543426396,0.6426468421463174,0.4125376824265483,0.4020588772590912,0.2154365661110459,0.6426468421463174,0.6426468421463174,0.6707198865659609,0.37901293333507124,0.6577149529048896,0.7238081283411658,0.8341343842357212,0.2382444719108452,0.60200001091326,0.2382444719108452,0.56188029422673,0.4844948134723742,0.6426468421463174,0.17484349328997395,0.7364831024946765,0.60200001091326,0.5226210332986871,0.60200001091326,0.37901293333507124,0.26279886806672315,0.7238081283411658,0.56188029422673,0.37901293333507124,0.2382444719108452,0.9779002318651105,0.8341343842357212,0.6834120426994836,0.6426468421463174,0.6707198865659609,0.7238081283411658,0.6426468421463174,0.37901293333507124,0.4477404985808746,0.9254555492637617,0.3172840066683832,0.2154365661110459,0.6426468421463174,0.7238081283411658,0.56188029422673,0.3172840066683832,0.3172840066683832,0.7238081283411658,0.37901293333507124,0.3378429008786611,0.9969649701982914,0.8699015393957243,0.56188029422673,0.4844948134723742,0.6426468421463174,0.8995616723379164,0.2891372543426396,0.3172840066683832,0.4477404985808746,0.56188029422673,0.8995616723379164,0.4477404985808746,0.34724722722482404,0.8369556595478901,0.9472130598426465,0.3172840066683832)


