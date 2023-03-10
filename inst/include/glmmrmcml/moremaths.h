#ifndef MOREMATHS_H
#define MOREMATHS_H

#include <cmath> 
#include <unordered_map>
#include <RcppEigen.h>
#include <Rcpp.h>
#include <glmmr/maths.h>

// [[Rcpp::depends(RcppEigen)]]

namespace glmmr {
  namespace maths {
  
  //ramanujans approximation
  inline double log_factorial_approx(double n){
    double ans;
    if(n==0){
      ans = 0;
    } else {
      ans = n*log(n) - n + log(n*(1+4*n*(1+2*n)))/6 + log(3.141593)/2;
    }
    return ans;
  }
  
  inline double log_likelihood(double y,
                               double mu,
                               double var_par,
                               int flink) {
    double logl;
    
    switch (flink){
    case 1:
      {
      
      double lf1 = glmmr::maths::log_factorial_approx(y);
      logl = y * mu - exp(mu) - lf1;
      //Rcpp::Rcout << "\n lf: " << lf1 << " ymu " << y*mu - exp(mu) << " y " << y << " mu " << mu << " logl " << logl;
      break;
      }
    case 2:
      {
        double lf1 = log_factorial_approx(y);
        logl = y*log(mu) - mu-lf1;
        break;
      }
    case 3:
      if(y==1){
        logl = log(1/(1+exp(-1.0*mu)));
      } else if(y==0){
        logl = log(1 - 1/(1+exp(-1.0*mu)));
      }
      break;
    case 4:
      if(y==1){
        logl = mu;
      } else if(y==0){
        logl = log(1 - exp(mu));
      }
      break;
    case 5:
      if(y==1){
        logl = log(mu);
      } else if(y==0){
        logl = log(1 - mu);
      }
      break;
    case 6:
      if(y==1){
        logl = (double)R::pnorm(mu,0,1,true,true);
      } else if(y==0){
        logl = log(1 - (double)R::pnorm(mu,0,1,true,false));
      }
      break;
    case 7:
      logl = -1*log(var_par) -0.5*log(2*3.141593) -
        0.5*((y - mu)/var_par)*((y - mu)/var_par);
      break;
    case 8:
      logl = -1*log(var_par) -0.5*log(2*3.141593) -
        0.5*((log(y) - mu)/var_par)*((log(y) - mu)/var_par);
      break;
    case 9:
      {
          double ymu = var_par*y/exp(mu);
          logl = log(1/(tgamma(var_par)*y)) + var_par*log(ymu) - ymu;
          break;
      }
    case 10:
      {
        double ymu = var_par*y*mu;
        logl = log(1/(tgamma(var_par)*y)) + var_par*log(ymu) - ymu;
        break;
      }
    case 11:
      logl = log(1/(tgamma(var_par)*y)) + var_par*log(var_par*y/mu) - var_par*y/mu;
      break;
    case 12:
      logl = (mu*var_par - 1)*log(y) + ((1-mu)*var_par - 1)*log(1-y) - lgamma(mu*var_par) - lgamma((1-mu)*var_par) + lgamma(var_par);
    }
    return logl;
  }
  
  template <typename MatrixType>
  inline typename MatrixType::Scalar logdet(const MatrixType& M) {
    using namespace Eigen;
    using std::log;
    typedef typename MatrixType::Scalar Scalar;
    Scalar ld = 0;
    LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
    auto& U = chol.matrixL();
    for (unsigned i = 0; i < M.rows(); ++i)
      ld += log(U(i,i));
    ld *= 2;
    return ld;
  }
  
  inline Eigen::VectorXd detadmu(const Eigen::VectorXd& xb,
                                std::string link) {
    
    Eigen::VectorXd wdiag(xb.size());
    Eigen::VectorXd p(xb.size());
    const static std::unordered_map<std::string, int> string_to_case{
      {"log",1},
      {"identity",2},
      {"logit",3},
      {"probit",4},
      {"inverse",5}
    };
    
    switch (string_to_case.at(link)) {
    case 1:
      wdiag = glmmr::maths::exp_vec(-1.0 * xb);
      break;
    case 2:
      for(int i =0; i< xb.size(); i++){
        wdiag(i) = 1.0;
      }
      break;
    case 3:
      p = glmmr::maths::mod_inv_func(xb, "logit");
      for(int i =0; i< xb.size(); i++){
        wdiag(i) = 1/(p(i)*(1.0 - p(i)));
      }
      break;
    case 4:
      {
        Eigen::ArrayXd pinv = gaussian_pdf_vec(xb);
        wdiag = (pinv.inverse()).matrix();
        break;
      }
    case 5:
      for(int i =0; i< xb.size(); i++){
        wdiag(i) = -1.0 * xb(i) * xb(i);
      }
      break;
    
    }
    
    return wdiag;
  }
  
  }

namespace algo {
inline Eigen::VectorXd forward_sub(Eigen::MatrixXd* U,
                                   Eigen::VectorXd* u,
                                   int n)
{
  Eigen::VectorXd y(n);
  for (int i = 0; i < n; i++) {
    double lsum = 0;
    for (int j = 0; j < i; j++) {
      lsum += (*U)(i,j) * y(j);
    }
    y(i) = ((*u)(i) - lsum) / (*U)(i,i);
  }
  return y;
}
}



}





#endif
