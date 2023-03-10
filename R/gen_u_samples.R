#' Returns the file name and type for MCNR function
#' 
#' Returns the file name and type for MCNR function. Used internally.
#' 
#' @param family family object
#' @return list with filename and type
mcnr_family <- function(family){
  f1 <- family[[1]]
  link <- family[[2]]
  gaussian_list <- c("identity")
  binomial_list <- c("logit","log","identity","probit")
  poisson_list <- c("log")
  type <- which(get(paste0(f1,"_list"))==link)
  return(list(file = paste0("mcml_",f1,".stan"),type=type))
}

#' Generate samples of random effects using MCMC
#' 
#' Generate samples of random effects using MCMC
#' 
#' @details 
#' Calls Stan through `cmdstanr` to generate `m` samples of the random effects from
#' a GLMM conditional on fixed values of the model parameters. To make use of 
#' parallelisation, the model is parameterised in terms of the Cholesky decomposition
#' of the covariance matrix of the random effects. Only a single chain is run to generate 
#' the samples.
#' @param y Vector of outcomes
#' @param X Matrix of covariates
#' @param Z Matrix, the design matrix of the random effects
#' @param L Matrix, the Cholesky decomposition of the covariance matrix of the random effects
#' @param beta Vector of mean function parameters
#' @param family A family function, e.g. gaussian()
#' @param sigma Numeric, the scale parameter of the distribution
#' @param warmup_iter Numeric, the number of warmup iterations
#' @param m Numeric, the number of sampling iterations
#' @return A matrix in which each column is a sample of the random effects
#' @examples 
#' \dontrun{
#' ## small example with simulated data
#' df <- nelder(~(j(10) * t(3)) > i(5))
#' des <- ModelMCML$new(
#'  covariance = list(
#'   formula =  ~(1|gr(j)*ar1(t)),
#'   parameters = c(0.25,0.7)
#' ),
#' mean = list(
#'   formula = ~factor(t)-1,
#'   parameters = rnorm(3)
#' ),
#' data=df,
#' family=gaussian()
#' )
#' ## simulate data
#' y <- des$sim_data()
#' ## get covariance definition matrix
#' ddata <- des$covariance$get_D_data()
#' ## simulate some values of the random effects
#' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
#' L <- des$covariance$get_chol_D()
#' samp <- gen_u_samples(y=y,
#'   Z = as.matrix(des$covariance$Z),
#'   L = as.matrix(L),
#'   X = as.matrix(des$mean_function$X),
#'   beta = des$mean_function$parameters,,
#'   family = des$mean_function$family
#'  )
#' }
#' @export
gen_u_samples <- function(y,X,Z,L,beta,family,sigma=1,warmup_iter=100,m=100){
  if(!requireNamespace("cmdstanr")){
    stop("cmdstanr not available")
  } else {
    file_type <- mcnr_family(family)
    model_file <- system.file("stan",
                              file_type$file,
                              package = "glmmrMCML",
                              mustWork = TRUE)
    mod <- suppressMessages(cmdstanr::cmdstan_model(model_file))
    data <- list(
      N = nrow(X),
      Q = ncol(Z),
      Xb = drop(as.matrix(X)%*%beta),
      Z = as.matrix(Z)%*%L,
      y = y,
      sigma = sigma,
      type=as.numeric(file_type$type)
    )
    
    capture.output(fit <- mod$sample(data = data,
                                     chains = 1,
                                     iter_warmup = warmup_iter,
                                     iter_sampling = m,
                                     refresh = 0),
                   file=tempfile())
    dsamps <- fit$draws("gamma",format = "matrix")
    dsamps <- L%*%t(dsamps)
    return(dsamps)
    }
 
}
