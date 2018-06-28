//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

/*
Written by Dong-Uk Kim, cmskdu@gmail.com
Date: June 05, 2018
Strong form: L*div(df/d(grad_aeta)), f = 0.5*kappa(grad_aeta)*grad_aeta^2 + w(grad_aeta)*g(aeta)
Integrand of the weak form: L*df/d(grad_aeta) = L*(kappa*grad_aeta + 0.5*grad_aeta^2*dkappa_dgrad_aeta + dw_dgrad_aeta*g(aeta))
*/


#ifndef MYDPMTRAINING_H
#define MYDPMTRAINING_H

#include "Kernel.h"
//#include "KernelGrad.h"
#include "JvarMapInterface.h"   /// For the off-diagonal Jacobian terms
#include "DerivativeMaterialInterface.h"

class myDPMtraining;
//class RankTwoTensor;

template <>
InputParameters validParams<myDPMtraining>();

class myDPMtraining : public DerivativeMaterialInterface<JvarMapKernelInterface<Kernel>>
{
public:
  myDPMtraining(const InputParameters & parameters);
  virtual void initialSetup();

protected:
  RealGradient get_dkappa_darg(unsigned int qp);
  RealGradient get_d2kappa_darg2(unsigned int cvar, unsigned int qp);
  RealGradient get_dargv_darg(unsigned int cvar);
  virtual Real computeQpResidual();
  virtual Real computeQpJacobian();
  virtual Real computeQpOffDiagJacobian(unsigned int jvar);

  const unsigned int _nvar;
  /// Phase-field mobility (assumed to be constant)
  const MaterialProperty<Real> & _L;
  /// Gradient energy coefficient; 0th order derivative from Material data
  const MaterialProperty<Real> & _kappa;
  /// Gradient energy coefficient; 1st order derivatives from Material data
  std::vector<const MaterialProperty<Real> *> _dkappa_darg;
  /// Gradient energy coefficient; 2nd order derivatives from Material data
  std::vector<std::vector<const MaterialProperty<Real> *>> _d2kappa_darg2;
};

#endif // MYDPMTRAINING_H
