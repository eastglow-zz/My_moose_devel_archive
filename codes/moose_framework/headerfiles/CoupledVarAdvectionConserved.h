//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#ifndef COUPLEDVARADVECTIONCONSERVED_H
#define COUPLEDVARADVECTIONCONSERVED_H

#include "Kernel.h"

// Forward Declaration
class CoupledVarAdvectionConserved;

template <>
InputParameters validParams<CoupledVarAdvectionConserved>();

class CoupledVarAdvectionConserved : public Kernel
{
public:
  CoupledVarAdvectionConserved(const InputParameters & parameters);

protected:
  virtual Real computeQpResidual();
  virtual Real computeQpJacobian();
  virtual Real computeQpOffDiagJacobian(unsigned int jvar);

  const VariableValue & _v_x;
  unsigned int _v_x_var;
  const VariableValue & _v_y;
  unsigned int _v_y_var;
  const VariableValue & _v_z;
  unsigned int _v_z_var;
  const VariableValue & _arg;
  unsigned int _arg_var;
};

#endif // COUPLEDVARADVECTIONCONSERVED_H
