#include <glmmr.h>
#include "../inst/include/glmmrMCML.h"
#include <RcppEigen.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppEigen)]]

//' Likelihood maximisation for the GLMM 
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters. These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML} 
//' class. These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' @details
//' Member function `$get_D_data()` of the covariance class will provide the necessary objects to specify the covariance matrix
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @return A vector of the parameters that maximise the likelihood functions
//' @examples
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(8) * t(3)) > i(4))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' out <- mcml_optim(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0,
//'    mcnr = TRUE)
// [[Rcpp::export]]
Rcpp::List mcml_optim(const Eigen::ArrayXXi &cov,
                      const Eigen::ArrayXd &data,
                      const Eigen::ArrayXd &eff_range,
                      const Eigen::MatrixXd &Z, 
                      const Eigen::MatrixXd &X,
                      const Eigen::VectorXd &y, 
                      Eigen::MatrixXd u,
                      std::string family, 
                      std::string link,
                      Eigen::ArrayXd start,
                      int trace,
                      bool mcnr = false){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  glmmr::MCMLDmatrix dmat(&dat, thetapars);
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model, start,trace);
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();
  
  beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List out = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return out;
}

//' Simulated likelihood optimisation step for MCML
//' 
//' Conditional on values of the random effects terms, this function will return the maximum simulated likelihood 
//' parameter estimates. These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML} 
//' class. These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
//' @examples
//' \donttest{
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(8) * t(3)) > i(4))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' out <- mcml_simlik(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0)
//' }
// [[Rcpp::export]]
Rcpp::List mcml_simlik(const Eigen::ArrayXXi &cov,
                       const Eigen::ArrayXd &data,
                       const Eigen::ArrayXd &eff_range,
                       const Eigen::MatrixXd &Z, 
                       const Eigen::MatrixXd &X,
                       const Eigen::VectorXd &y, 
                       Eigen::MatrixXd u,
                       std::string family, 
                       std::string link,
                       Eigen::ArrayXd start,
                       int trace){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::MCMLDmatrix dmat(&dat, thetapars);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model,start,trace);
  
  mc.f_optim();
  
  beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Likelihood maximisation for the GLMM using sparse matrix methods
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters. This version of the function uses sparse matrix methods for the operations involving
//' the random effects covariance matrix, see \link[Matrix]{dsCMatrix-class} for specification. 
//' These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML} 
//' class. These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' Likelihood maximisation for the GLMM
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Ap Integer vector of pointers, one for each column, specifying the initial (zero-based) index of elements in the column. Slot `p`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Ai Integer vector specifying the row indices of the non-zero elements of the matrix. Slot `i`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @param mcnr Logical indicating whether to use Newton-Raphson (TRUE) or Expectation Maximisation (FALSE)
//' @return A vector of the parameters that maximise the simulated likelihood
//' @examples
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(8) * t(3)) > i(4))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC - note this step does not use the sparse matrix
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' ## the specification of the covariance above results in a sparse covariance matrix,
//' ## so we can just extract the components as below.
//' out <- mcml_optim_sparse(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Ap = des$covariance$D@p,
//'    Ai = des$covariance$D@i,
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0,
//'    mcnr = TRUE)
// [[Rcpp::export]]
Rcpp::List mcml_optim_sparse(const Eigen::ArrayXXi &cov,
                             const Eigen::ArrayXd &data,
                             const Eigen::ArrayXd &eff_range,
                             const Eigen::ArrayXi &Ap,
                             const Eigen::ArrayXi &Ai,
                             const Eigen::MatrixXd &Z, 
                             const Eigen::MatrixXd &X,
                             const Eigen::VectorXd &y, 
                             Eigen::MatrixXd u,
                             std::string family, 
                             std::string link,
                             Eigen::ArrayXd start,
                             int trace,
                             bool mcnr = false){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::SparseDMatrix dmat(&dat, thetapars,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,&model, start,trace);
  
  if(!mcnr){
    mc.l_optim();
  } else {
    mc.mcnr();
  }
  mc.d_optim();

  beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
  double sigma = mc.get_sigma();

  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma,
                                    _["Ap"] = dmat.chol_->L->Ap,_["Ai"] = dmat.chol_->L->Ai,
                                    _["Ax"] = dmat.chol_->L->Ax,_["D"] = dmat.chol_->D);
  return L;
}

//' Simulated likelihood optimisation step for MCML using sparse matrix methods
//' 
//' Given model data and random effects samples `u`, this function will run the MCML steps to generate new estimates of 
//' mean function and covariance parameters that maximise the simulated likelihood. This version of the function uses sparse matrix methods for the operations involving 
//' the random effects covariance matrix, see \link[Matrix]{dsCMatrix-class} for specification. 
//' These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML} 
//' class. These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' Likelihood maximisation for the GLMM
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Ap Integer vector of pointers, one for each column, specifying the initial (zero-based) index of elements in the column. Slot `p`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Ai Integer vector specifying the row indices of the non-zero elements of the matrix. Slot `i`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return A vector of the parameters that maximise the simulated likelihood
//' @examples
//' \donttest{
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(10) * t(3)) > i(5))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC - note this step does not use the sparse matrix
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' ## the specification of the covariance above results in a sparse covariance matrix,
//' ## so we can just extract the components as below.
//' out <- mcml_simlik_sparse(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Ap = des$covariance$D@p,
//'    Ai = des$covariance$D@i,
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0)
//' }
 // [[Rcpp::export]]
Rcpp::List mcml_simlik_sparse(const Eigen::ArrayXXi &cov,
                              const Eigen::ArrayXd &data,
                              const Eigen::ArrayXd &eff_range,
                              const Eigen::ArrayXi &Ap,
                              const Eigen::ArrayXi &Ai,
                              const Eigen::MatrixXd &Z, 
                              const Eigen::MatrixXd &X,
                              const Eigen::VectorXd &y, 
                              Eigen::MatrixXd u,
                              std::string family, 
                              std::string link,
                              Eigen::ArrayXd start,
                              int trace){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd thetapars = start.segment(X.cols(),dat.n_cov_pars());
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::SparseDMatrix dmat(&dat, thetapars,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,&model, start,trace);
  
  mc.f_optim();
  
  beta = mc.get_beta();
  Eigen::VectorXd theta = mc.get_theta();
  double sigma = mc.get_sigma();
  
  Rcpp::List L = Rcpp::List::create(_["beta"] = beta, _["theta"] = theta,  _["sigma"] = sigma);
  return L;
}

//' Generate Hessian matrix of GLMM
//' 
//' Generate Hessian matrix of a GLMM using numerical differentiation. These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML}  class. 
//' These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param tol The tolerance of the numerical differentiation routine
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return The estimated Hessian matrix
//' @examples
//' \donttest{
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(10) * t(3)) > i(5))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' ## fit model to get parameter estimates
//' out <- mcml_optim(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0,
//'    mcnr = TRUE)
//' ## get hessian matrix
//' hess <- mcml_hess(cov=ddata$cov,
//'   data=ddata$data,
//'   eff_range = rep(0,30),
//'   Z = as.matrix(des$covariance$Z),
//'   X = as.matrix(des$mean_function$X),
//'   y = y,
//'   u = mat,
//'   family = des$mean_function$family[[1]],
//'   link=des$mean_function$family[[2]],
//'   start = c(out$beta,out$theta,out$sigma))
//' }
// [[Rcpp::export]]
Eigen::MatrixXd mcml_hess(const Eigen::ArrayXXi &cov,
                    const Eigen::ArrayXd &data,
                    const Eigen::ArrayXd &eff_range,
                    const Eigen::MatrixXd &Z, 
                    const Eigen::MatrixXd &X,
                    const Eigen::VectorXd &y, 
                    Eigen::MatrixXd u,
                    std::string family, 
                    std::string link,
                    Eigen::ArrayXd start,
                      double tol = 1e-5,
                      int trace = 0){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd theta = start.segment(X.cols(),dat.n_cov_pars());
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::MCMLDmatrix dmat(&dat, theta);
  glmmr::mcmloptim<glmmr::MCMLDmatrix> mc(&dmat,&model, start,trace);
  
  Eigen::MatrixXd hess = mc.f_hess(tol);
  return hess;
}

//' Generate Hessian matrix of GLMM using sparse matrix methods
//' 
//' Generate Hessian matrix of a GLMM using numerical differentiation. Operations with the random effects covariance matrix
//' use sparse matrix methods,see \link[Matrix]{dsCMatrix-class} for specification. 
//' These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML}  class. 
//' These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//'  
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Ap Integer vector of pointers, one for each column, specifying the initial (zero-based) index of elements in the column. Slot `p`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Ai Integer vector specifying the row indices of the non-zero elements of the matrix. Slot `i`
//' of a matrix of a class defined in \link[Matrix]{sparseMatrix}
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param start Vector of starting values for the optimisation
//' @param tol The tolerance of the numerical differentiation routine
//' @param trace Integer indicating what to report to the console, 0= nothing, 1-3=detailed output
//' @return The estimated Hessian matrix
//' @examples
//' \donttest{
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(10) * t(3)) > i(5))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC - note that this function does not use sparse methods
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' ## fit model to get parameter estimates using sparse matrix methods
//' ## the covariance matrix with this specification is a sparse matrix so
//' ## we can extract the components as below
//' out <- mcml_optim_sparse(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Ap = des$covariance$D@p,
//'    Ai = des$covariance$D@i,
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    u = mat,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,3),0.25,0.7,1.0),
//'    trace=0,
//'    mcnr = TRUE)
//' ## get hessian matrix
//' hess <- mcml_hess_sparse(cov=ddata$cov,
//'   data=ddata$data,
//'   eff_range = rep(0,30),
//'   Ap = des$covariance$D@p,
//'   Ai = des$covariance$D@i,
//'   Z = as.matrix(des$covariance$Z),
//'   X = as.matrix(des$mean_function$X),
//'   y = y,
//'   u = mat,
//'   family = des$mean_function$family[[1]],
//'   link=des$mean_function$family[[2]],
//'   start = c(out$beta,out$theta,out$sigma))
//' }
// [[Rcpp::export]]
Eigen::MatrixXd mcml_hess_sparse(const Eigen::ArrayXXi &cov,
                          const Eigen::ArrayXd &data,
                          const Eigen::ArrayXd &eff_range,
                          const Eigen::ArrayXi &Ap,
                          const Eigen::ArrayXi &Ai,
                          const Eigen::MatrixXd &Z, 
                          const Eigen::MatrixXd &X,
                          const Eigen::VectorXd &y, 
                          Eigen::MatrixXd u,
                          std::string family, 
                          std::string link,
                          Eigen::ArrayXd start,
                          double tol = 1e-5,
                          int trace = 0){
  
  glmmr::DData dat(cov,data,eff_range);
  Eigen::ArrayXd theta = start.segment(X.cols(),dat.n_cov_pars());
  Eigen::VectorXd beta = start.segment(0,X.cols());
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,1,family,link);
  glmmr::SparseDMatrix dmat(&dat, theta,Ap,Ai);
  glmmr::mcmloptim<glmmr::SparseDMatrix> mc(&dmat,&model, start,trace);
  
  Eigen::MatrixXd hess = mc.f_hess(tol);
  return hess;
}

//' Calculates the conditional Akaike Information Criterion for the GLMM
//' 
//' Calculates the conditional Akaike Information Criterion for the GLMM. These functions are not intended to be used by the general user since 
//' the complete model fitting algorithm can be accessed through the member functions of the \link[glmmrMCML]{ModelMCML}  class. 
//' These functions are exported for users wishing to use the modular components or each step 
//' separately of the MCML algorithm.
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param Z Matrix Z of the GLMM
//' @param X Matrix X of the GLMM
//' @param y Vector of observations
//' @param u Matrix of samples of the random effects. Each column is a sample.
//' @param family Character specifying the family
//' @param link Character specifying the link function
//' @param beta_par Vector specifying the values of the mean function parameters to estimate the AIC at
//' @param cov_par Vector specifying the values of the covariance function parameters to estimate the AIC at
//' @return Estimated conditional AIC
//' @examples
//' \donttest{
//' ## small example with simulated data
//' ## create data and model object with 
//' ## parameters to simulate data and 
//' ## act as starting values
//' df <- nelder(~(j(10) * t(3)) > i(5))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## simulate data
//' y <- des$sim_data()
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## get parameter estimates using Laplace approximation
//' ## this function also returns estimates of the random 
//' ## effects
//' out <- mcml_la_nr(cov=ddata$cov,
//'    data=ddata$data,
//'    eff_range = rep(0,30),
//'    Z = as.matrix(des$covariance$Z),
//'    X = as.matrix(des$mean_function$X),
//'    y = y,
//'    family = des$mean_function$family[[1]],
//'    link=des$mean_function$family[[2]],
//'    start = c(rep(0.01,5),0.25,0.7,1.0),
//'    usehess = FALSE,
//'    tol=1e-2,verbose=FALSE,trace=0)
//' ## get AIC
//' aic_mcml(cov=ddata$cov,
//'   data=ddata$data,
//'   eff_range = rep(0,30),
//'   Z = as.matrix(des$covariance$Z),
//'   X = as.matrix(des$mean_function$X),
//'   y = y,
//'   u = out$u,
//'   family = des$mean_function$family[[1]],
//'   link=des$mean_function$family[[2]],
//'   beta = c(out$beta,out$sigma),
//'   cov_par = out$theta)
//'   }
// [[Rcpp::export]]
double aic_mcml(const Eigen::ArrayXXi &cov,
                const Eigen::ArrayXd &data,
                const Eigen::ArrayXd &eff_range,
                const Eigen::MatrixXd &Z, 
                const Eigen::MatrixXd &X,
                const Eigen::VectorXd &y, 
                Eigen::MatrixXd u, 
                std::string family, 
                std::string link,
                const Eigen::VectorXd& beta_par,
                const Eigen::VectorXd& cov_par){
  
  int niter = u.cols();
  int n = y.size();
  int P = X.cols();
  double var_par;
  int dof = beta_par.size() + cov_par.size();
  Eigen::VectorXd beta;
  
  if(family=="gaussian" || family=="Gamma" || family=="beta"){
    var_par = beta_par(P);
    //xb = X*beta_par.segment(0,P-1);
    beta = beta_par.segment(0,P);
  } else {
    var_par = 0;
    beta = beta_par;
  }
  glmmr::DData dat(cov,data,eff_range);
  glmmr::MCMLDmatrix dmat(&dat, cov_par);
  glmmr::mcmlModel model(Z,nullptr,X,y,&u,beta,var_par,family,link);
  double dmvvec = dmat.loglik(u);
  double ll = model.log_likelihood();
  
  return (-2*( ll + dmvvec ) + 2*dof); 
  
}

//' Multivariate normal log likelihood
//' 
//' Calculates the log likelihood of the multivariate normal distribution using `glmmr` covariance representation.
//' 
//' @param cov An integer matrix with columns of block identifier, dimension of block, function definition, number of variables
//' in the argument to the funciton, and index of the parameters, respectively. Rows are specific functions of each block.
//' @param data Vector of data. Created by flattening the matrices in column-major order of the data used in each block.
//' @param eff_range Vector of values with the effective range parameters of the covariance functions, where required.
//' @param gamma Vector specifying the values of the covariance function parameters
//' @param u Matrix (or vector) of observed values
//' @return Scalar value 
//' ## small example with simulated random effects
//' ## first, create data and model object with parameters
//' df <- nelder(~(j(10) * t(3)) > i(5))
//' des <- ModelMCML$new(
//'  covariance = list(
//'   formula =  ~(1|gr(j)*ar1(t)),
//'   parameters = c(0.25,0.7)
//' ),
//' mean = list(
//'   formula = ~factor(t)-1,
//'   parameters = rnorm(3)
//' ),
//' data=df,
//' family=gaussian()
//' )
//' ## get covariance definition matrix
//' ddata <- des$covariance$get_D_data()
//' ## simulate some values of the random effects
//' ## first, we need to extract the Cholesky decomposition of the covariance matrix D
//' L <- des$covariance$get_chol_D()
//' ## generate samples using HMC
//' mat <- mcmc_sample(Z = as.matrix(des$covariance$Z),
//'    L = as.matrix(L),
//'    X = as.matrix(des$mean_function$X),
//'    y = as.vector(y),
//'    beta = des$mean_function$parameters,
//'    var_par = 1,
//'    family = des$mean_function$family[[1]],
//'    link = des$mean_function$family[[2]],
//'    warmup = 250, 
//'    nsamp = 250,
//'    lambda = 5,
//'    maxsteps = 100,
//'    trace=1,
//'    target_accept = 0.95)
//' ## calculate the log likelihood for the simulated random effects
//' mvn_ll(cov=ddata$cov,
//'   data=ddata$data,
//'   eff_range = rep(0,30),
//'   gamma = c(0.25,0.7),
//'   u = mat) 
// [[Rcpp::export]]
double mvn_ll(const Eigen::ArrayXXi &cov,
              const Eigen::ArrayXd &data,
              const Eigen::ArrayXd &eff_range,
              const Eigen::ArrayXd &gamma,
              const Eigen::MatrixXd &u){
  glmmr::DData dat(cov,data,eff_range);
  glmmr::MCMLDmatrix dmat(&dat, gamma);
  return dmat.loglik(u);
}
