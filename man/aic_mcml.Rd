% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{aic_mcml}
\alias{aic_mcml}
\title{Calculates the conditional Akaike Information Criterion for the GLMM}
\usage{
aic_mcml(cov, data, eff_range, Z, X, y, u, family, link, beta_par, cov_par)
}
\arguments{
\item{cov}{An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.}

\item{data}{Vector of data. Created by flattening the matrices in column-major order of the data used in each block.}

\item{eff_range}{Vector of values with the effective range parameters of the covariance functions, where required.}

\item{Z}{Matrix Z of the GLMM}

\item{X}{Matrix X of the GLMM}

\item{y}{Vector of observations}

\item{u}{Matrix of samples of the random effects. Each column is a sample.}

\item{family}{Character specifying the family}

\item{link}{Character specifying the link function}

\item{beta_par}{Vector specifying the values of the mean function parameters to estimate the AIC at}

\item{cov_par}{Vector specifying the values of the covariance function parameters to estimate the AIC at}
}
\value{
Estimated conditional AIC
}
\description{
Calculates the conditional Akaike Information Criterion for the GLMM. These functions are not intended to be used by the general user since 
the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML}  class. 
These functions are exported for users wishing to use the modular components or each step 
separately of the MCML algorithm.
}
\examples{
\donttest{
## small example with simulated data
## create data and model object with 
## parameters to simulate data and 
## act as starting values
df <- nelder(~(j(10) * t(3)) > i(5))
des <- ModelMCML$new(
 covariance = list(
  formula =  ~(1|gr(j)*ar1(t)),
  parameters = c(0.25,0.7)
),
mean = list(
  formula = ~factor(t)-1,
  parameters = rnorm(3)
),
data=df,
family=gaussian()
)
## simulate data
y <- des$sim_data()
## get covariance definition matrix
ddata <- des$covariance$get_D_data()
## get parameter estimates using Laplace approximation
## this function also returns estimates of the random 
## effects
out <- mcml_la_nr(cov=ddata$cov,
   data=ddata$data,
   eff_range = rep(0,30),
   Z = as.matrix(des$covariance$Z),
   X = as.matrix(des$mean_function$X),
   y = y,
   family = des$mean_function$family[[1]],
   link=des$mean_function$family[[2]],
   start = c(rep(0.01,5),0.25,0.7,1.0),
   usehess = FALSE,
   tol=1e-2,verbose=FALSE,trace=0)
## get AIC
aic_mcml(cov=ddata$cov,
  data=ddata$data,
  eff_range = rep(0,30),
  Z = as.matrix(des$covariance$Z),
  X = as.matrix(des$mean_function$X),
  y = y,
  u = out$u,
  family = des$mean_function$family[[1]],
  link=des$mean_function$family[[2]],
  beta = c(out$beta,out$sigma),
  cov_par = out$theta)
  }
}
