//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html



#include "myDPMtraining2.h"

registerMooseObject("PhaseFieldApp", myDPMtraining2);

template <>
InputParameters
validParams<myDPMtraining2>()
{
  InputParameters params = validParams<Kernel>();
  params.addClassDescription("Getting derivative data from DerivativeParsedMaterial");
  params.addParam<MaterialPropertyName>("mob_name","L","The mobility used with the kernel, assumed as a constant");
  params.addParam<MaterialPropertyName>("kappa_name","kappa_op","The kappa used with the kernel, may be a function of grad_op");
  params.addCoupledVar("grad_aeta_x","Vector component of nonlinear (Aux)Variable arguments kappa_op depends on");
  params.addCoupledVar("grad_aeta_y","Vector component of nonlinear (Aux)Variable arguments kappa_op depends on");
  params.addCoupledVar("grad_aeta_z","Vector component of nonlinear (Aux)Variable arguments kappa_op depends on");
  return params;
}

myDPMtraining2::myDPMtraining2(const InputParameters & parameters)
  : DerivativeMaterialInterface<JvarMapKernelInterface<Kernel>>(parameters),
  _nvar(_coupled_moose_vars.size()),
  _L(getMaterialProperty<Real>("mob_name")),
  _kappa(getMaterialProperty<Real>("kappa_name")),
  _dkappa_darg(_nvar),  // If I do this, I don't need to resize the variables in the routine
  _d2kappa_darg2(_nvar),
  _vx(_nvar >= 1 ? coupledValue("grad_aeta_x") : _zero),
  _vy(_nvar >= 2 ? coupledValue("grad_aeta_y") : _zero),
  _vz(_nvar >= 3 ? coupledValue("grad_aeta_z") : _zero)
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
    _dkappa_darg[i] = &getMaterialPropertyDerivative<Real>("kappa_name", iname);

    /// The 2nd derivatives
    _d2kappa_darg2[i].resize(_nvar);
    for (unsigned int j = 0; j < _nvar; ++j)
    {
      const VariableName jname = _coupled_moose_vars[j]->name();
      if (jname == _var.name())
        paramError("coupled_variables",\
                   "The kernel variable should not be specified in the coupled `args` parameter.");
      _d2kappa_darg2[i][j] = &getMaterialPropertyDerivative<Real>("kappa_name", iname, jname);
    }
  }
}

void
myDPMtraining2::initialSetup()
{
  validateCoupling<Real>("kappa_name");
}

RealGradient
myDPMtraining2::get_dkappa_darg(unsigned int qp) // This function must be called in computeQp* functions
{
  RealGradient v0(0.0, 0.0, 0.0);
  switch (_nvar) {
    case 1:
      {
        RealGradient v1((*_dkappa_darg[0])[qp], 0.0, 0.0);
        //printf("debug:get_dkappa_darg:_nvar=%d, v1 has made @qp = %d, v1(%lf, 0, 0)\n", _nvar, qp, (*_dkappa_darg[0])[qp]);
        return v1;
      }
      break;
    case 2:
      {
        RealGradient v2((*_dkappa_darg[0])[qp], (*_dkappa_darg[1])[qp], 0.0);
        //printf("debug:get_dkappa_darg:_nvar=%d, v2 has made @qp = %d, v2(%lf, %lf, 0)\n", _nvar, qp, (*_dkappa_darg[0])[qp], (*_dkappa_darg[1])[qp]);
        return v2;
      }
      break;
    case 3:
      {
        RealGradient v3((*_dkappa_darg[0])[qp], (*_dkappa_darg[1])[qp], (*_dkappa_darg[2])[qp]);
        //printf("debug:get_dkappa_darg:_nvar=%d, v3 has made @qp = %d, v3(%lf, %lf, %lf)\n", _nvar, qp, (*_dkappa_darg[0])[qp], (*_dkappa_darg[1])[qp], (*_dkappa_darg[2])[qp]);
        return v3;
      }
      break;
    default:
      return v0;
  }
}

RealGradient
myDPMtraining2::get_d2kappa_darg2(unsigned int i, unsigned int qp)  // This function must be called in computeQp* functions
{
  RealGradient v0(0.0, 0.0, 0.0);
  switch (_nvar) {
    case 1:
      {
        RealGradient v1((*_d2kappa_darg2[i][0])[qp], 0.0, 0.0);
        return v1;
      }
      break;
    case 2:
      {
        RealGradient v2((*_d2kappa_darg2[i][0])[qp], (*_d2kappa_darg2[i][1])[qp], 0.0);
        return v2;
        break;
      }
      break;
    case 3:
      {
        RealGradient v3((*_d2kappa_darg2[i][0])[qp], (*_d2kappa_darg2[i][1])[qp], (*_d2kappa_darg2[i][2])[qp]);
        return v3;
      }
      break;
    default:
      return v0;
  }
}

RealGradient
myDPMtraining2::get_dargv_darg(unsigned int i)
{
  RealGradient v0(0.0, 0.0, 0.0);
  switch (i) {
    case 0:
      {
        RealGradient v1(1.0, 0.0, 0.0);
        return v1;
      }
      break;
    case 1:
      {
        RealGradient v2(0.0, 1.0, 0.0);
        return v2;
        break;
      }
      break;
    case 2:
      {
        RealGradient v3(0.0, 0.0, 1.0);
        return v3;
      }
      break;
    default:
      return v0;
  }
}

Real
myDPMtraining2::computeQpResidual()
{
  //RealGradient _v_vec(_vx[_qp], _vy[_qp], _vz[_qp]);
  RealGradient _v_vec = _grad_u[_qp];
  Real grad_aeta_sq = _v_vec * _v_vec;  /// may be this is inner product for the two vector operands
  Real grad_aeta_dot_grad_test = _v_vec * _grad_test[_i][_qp];
  //if (grad_u_sq > 1e-2/1e-4)
  if (1)
  {
    RealGradient dkappa_dgradaeta = get_dkappa_darg(_qp);
    Real dkappa_dgradaeta_dot_grad_test = dkappa_dgradaeta * _grad_test[_i][_qp];
    return _L[_qp] * (_kappa[_qp] * grad_aeta_dot_grad_test\
                      + 0.5 * grad_aeta_sq * dkappa_dgradaeta_dot_grad_test);
  }else{
    RealGradient dkappa_dgradaeta = get_dkappa_darg(_qp);
    return _L[_qp] * _kappa[_qp] * grad_aeta_dot_grad_test;
  }
}

Real
myDPMtraining2::computeQpJacobian()
{

  //RealGradient _v_vec(_vx[_qp], _vy[_qp], _vz[_qp]);
  RealGradient _v_vec = _grad_u[_qp];
  if (1)
  {
    RealGradient dkappa_dgradaeta = get_dkappa_darg(_qp);
    Real dkappa_dgradaeta_dot_grad_phi = dkappa_dgradaeta * _grad_phi[_j][_qp];
    Real grad_aeta_dot_grad_test = _v_vec * _grad_test[_i][_qp];
    return _L[_qp] * dkappa_dgradaeta_dot_grad_phi * grad_aeta_dot_grad_test;
  }else{
    return 0;
  }
}


Real
myDPMtraining2::computeQpOffDiagJacobian(unsigned int jvar)
{
  // get the coupled variable jvar is referring to
  const unsigned int cvar = mapJvarToCvar(jvar);
  //RealGradient _v_vec(_vx[_qp], _vy[_qp], _vz[_qp]);
  RealGradient _v_vec = _grad_u[_qp];
  Real grad_aeta_sq = _v_vec * _v_vec;
  Real xcvar_dot_grad_test = get_dargv_darg(cvar) * _grad_test[_i][_qp];
  Real xcvar_dot_grad_aeta = get_dargv_darg(cvar) * _v_vec;
  //if (grad_u_sq > 1e-2/1e-4)
  if (1)
  {
    Real grad_aeta_dot_grad_test = _v_vec * _grad_test[_i][_qp];
    Real dkappa_dgradaeta_dot_grad_test = get_dkappa_darg(_qp) * _grad_test[_i][_qp];
    Real d2kappa_dgradaeta_dcvar_dot_grad_test = get_d2kappa_darg2(cvar, _qp) * _grad_test[_i][_qp];
    return _L[_qp] * _phi[_j][_qp] * ((*_dkappa_darg[cvar])[_qp] * grad_aeta_dot_grad_test \
                      + _kappa[_qp] * xcvar_dot_grad_test \
                      + xcvar_dot_grad_aeta * dkappa_dgradaeta_dot_grad_test \
                      + 0.5 * grad_aeta_sq * d2kappa_dgradaeta_dcvar_dot_grad_test);
  }else{
    return _L[_qp] * _phi[_j][_qp] * _kappa[_qp] * xcvar_dot_grad_test;
  }
}
