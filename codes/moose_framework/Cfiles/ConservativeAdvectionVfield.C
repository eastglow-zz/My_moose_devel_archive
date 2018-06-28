//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

#include "ConservativeAdvectionVfield.h"

registerMooseObject("MooseApp", ConservativeAdvectionVfield);

template <>
InputParameters
validParams<ConservativeAdvectionVfield>()
{
  InputParameters params = validParams<Kernel>();
  params.addClassDescription("Conservative form of $\\nabla \\cdot \\vec{v} u$ which in its weak "
                             "form is given by: $(-\\nabla \\psi_i, \\vec{v} u)$.");
  //params.addRequiredParam<RealVectorValue>("velocity", "Velocity vector");
  params.addRequiredCoupledVar("vel_x","x_velocity_component");
  params.addCoupledVar("vel_y","y_velocity_component");
  params.addCoupledVar("vel_z","z_velocity_component");
  return params;
}

ConservativeAdvectionVfield::ConservativeAdvectionVfield(const InputParameters & parameters)
  : Kernel(parameters),
    _v_x(coupledValue("vel_x")), _v_x_var(coupled("vel_x")),
    _v_y(_mesh.dimension() >= 2 ? coupledValue("vel_y") : _zero), _v_y_var(_mesh.dimension() >= 2 ? coupled("vel_y") : _mesh.dimension()+1),
    _v_z(_mesh.dimension() >= 3 ? coupledValue("vel_z") : _zero), _v_z_var(_mesh.dimension() >= 2 ? coupled("vel_z") : _mesh.dimension()+1)
{
}

Real
ConservativeAdvectionVfield::computeQpResidual()
{
  const RealGradient _v(_v_x[_qp], _v_y[_qp], _v_z[_qp]);
  return -_u[_qp] * (_v * _grad_test[_i][_qp]);
}

Real
ConservativeAdvectionVfield::computeQpJacobian()
{
  const RealGradient _v(_v_x[_qp], _v_y[_qp], _v_z[_qp]);
  return -_phi[_j][_qp] * (_v * _grad_test[_i][_qp]);
}

Real
ConservativeAdvectionVfield::computeQpOffDiagJacobian(unsigned int jvar)
{
  const RealGradient _v(jvar == _v_x_var ? _phi[_j][_qp] : _v_x[_qp], jvar == _v_y_var ? _phi[_j][_qp] : _v_y[_qp], jvar == _v_z_var ? _phi[_j][_qp] : _v_z[_qp]);
  return -_u[_qp] * (_v * _grad_test[_i][_qp]);
}
