//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#ifndef TAUTIMEDERIVATIVE_H
#define TAUTIMEDERIVATIVE_H

#include "TimeKernel.h"
//#include "Kernel.h"
#include "JvarMapInterface.h"   /// For the off-diagonal Jacobian terms
#include "DerivativeMaterialInterface.h"
#include "Assembly.h"
#include "libmesh/quadrature.h"

// Forward Declaration
class TauTimeDerivative;

template <>
InputParameters validParams<TauTimeDerivative>();

/**
 * This calculates the time derivative * orientation dependent scaling function
 */
class TauTimeDerivative : public DerivativeMaterialInterface<JvarMapKernelInterface<TimeKernel>>
{
public:
  TauTimeDerivative(const InputParameters & parameters);
  virtual void initialSetup() override;
  virtual void computeJacobian() override;

protected:
  virtual Real computeQpResidual() override;
  virtual Real computeQpJacobian() override;
  virtual Real computeQpOffDiagJacobian(unsigned int jvar) override;

  bool _lumping;

  const unsigned int _nvar;
  const MaterialProperty<Real> & _tau;
  std::vector<const MaterialProperty<Real> *> _dtau_darg;
};

#endif // TAUTIMEDERIVATIVE_H
