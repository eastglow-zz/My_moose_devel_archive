//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "TauTimeDerivative.h"

registerMooseObject("PhaseFieldApp", TauTimeDerivative);

template <>
InputParameters
validParams<TauTimeDerivative>()
{
  InputParameters params = validParams<TimeKernel>();
  params.addClassDescription("Orientation dependent scaled time derivative Kernel");
  params.addParam<MaterialPropertyName>("tau_name","tau_op","Orientation dependent scaling function in terms of the coupled variables");
  params.addCoupledVar("coupled_variables", "Vector of nonlinear (Aux)Variable arguments tau_op depends on");
  params.addParam<bool>("lumping", false, "True for mass matrix lumping, false otherwise");
  return params;
}

TauTimeDerivative::TauTimeDerivative(const InputParameters & parameters)
  : DerivativeMaterialInterface<JvarMapKernelInterface<TimeKernel>>(parameters),
  _nvar(_coupled_moose_vars.size()),
  _tau(getMaterialProperty<Real>("tau_name")),
  _dtau_darg(_nvar)
{
  /// Get derivative data
  for (unsigned int i = 0; i < _nvar; ++i)
  {
    MooseVariable * ivar = _coupled_standard_moose_vars[i];
    const VariableName iname = ivar->name();
    if (iname == _var.name())
      paramError("coupled_variables",\
                 "The kernel variable should not be specified in the coupled `args` parameter.");

    /// The 1st derivatives
    _dtau_darg[i] = &getMaterialPropertyDerivative<Real>("tau_name", iname);
  }
}

void
TauTimeDerivative::initialSetup()
{
  validateCoupling<Real>("tau_name");
}

Real
TauTimeDerivative::computeQpResidual()
{
  return _test[_i][_qp] * _u_dot[_qp] * _tau[_qp];
}

Real
TauTimeDerivative::computeQpJacobian()
{
  Real dtau_du = 0;
  for (unsigned int i = 0; i < _nvar; i++)
  {
    dtau_du += (*_dtau_darg[i])[_qp] * _grad_phi[_j][_qp](i);
  }
  return (dtau_du * _u_dot[_qp] + _du_dot_du[_qp] * _tau[_qp]) * _test[_i][_qp];
}

Real
TauTimeDerivative::computeQpOffDiagJacobian(unsigned int jvar)
{
  if (0)
  {
    const unsigned int cvar = mapJvarToCvar(jvar);
    return _test[_i][_qp] * _u_dot[_qp] * (*_dtau_darg[cvar])[_qp] * _phi[_j][_qp];
  }else{
    return 0;
  }
}

void
TauTimeDerivative::computeJacobian()
{
  if (_lumping)
  {
    DenseMatrix<Number> & ke = _assembly.jacobianBlock(_var.number(), _var.number());

    for (_i = 0; _i < _test.size(); _i++)
      for (_j = 0; _j < _phi.size(); _j++)
        for (_qp = 0; _qp < _qrule->n_points(); _qp++)
          ke(_i, _i) += _JxW[_qp] * _coord[_qp] * computeQpJacobian();
  }
  else
    TimeKernel::computeJacobian();
}
