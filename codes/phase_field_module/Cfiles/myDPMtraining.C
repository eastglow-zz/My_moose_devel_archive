//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html



#include "myDPMtraining.h"

registerMooseObject("PhaseFieldApp", myDPMtraining);

template <>
InputParameters
validParams<myDPMtraining>()
{
  InputParameters params = validParams<Kernel>();
  params.addClassDescription("Getting derivative data from DerivativeParsedMaterial");
  params.addParam<MaterialPropertyName>("mob_name","L","The mobility used with the kernel, assumed as a constant");
  params.addParam<MaterialPropertyName>("kappa_name","kappa_op","The kappa used with the kernel, may be a function of grad_op");
  params.addCoupledVar("coupled_variables","Vector of nonlinear (Aux)Variable arguments kappa_op depends on");
  return params;
}

myDPMtraining::myDPMtraining(const InputParameters & parameters)
  : DerivativeMaterialInterface<JvarMapKernelInterface<Kernel>>(parameters),
  _nvar(_coupled_moose_vars.size()),
  _L(getMaterialProperty<Real>("mob_name")),
  _kappa(getMaterialProperty<Real>("kappa_name")),
  _dkappa_darg(_nvar),  // If I do this, I don't need to resize the variables in the routine
  _d2kappa_darg2(_nvar)
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
myDPMtraining::initialSetup()
{
  validateCoupling<Real>("kappa_name");
}

RealGradient
myDPMtraining::get_dkappa_darg(unsigned int qp) // This function must be called in computeQp* functions
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
myDPMtraining::get_d2kappa_darg2(unsigned int i, unsigned int qp)  // This function must be called in computeQp* functions
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
myDPMtraining::get_dargv_darg(unsigned int i)
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
myDPMtraining::computeQpResidual()
{
  Real grad_u_sq = _grad_u[_qp] * _grad_u[_qp];  /// may be this is inner product for the two vector operands
  Real grad_u_dot_grad_test = _grad_u[_qp] * _grad_test[_i][_qp];
  if (1)
  {
    RealGradient dkappa_dgradaeta = get_dkappa_darg(_qp);
    Real dkappa_dgradaeta_dot_grad_test = dkappa_dgradaeta * _grad_test[_i][_qp];
    return _L[_qp] * (_kappa[_qp] * grad_u_dot_grad_test\
                      + 0.5 * grad_u_sq * dkappa_dgradaeta_dot_grad_test);
  }else{
    return _L[_qp] * _kappa[_qp] * grad_u_dot_grad_test;
  }
}

Real
myDPMtraining::computeQpJacobian()
{

  Real grad_u_sq = _grad_u[_qp] * _grad_u[_qp];
  Real kappa_gradphi_dot_grad_test = _kappa[_qp] * _grad_phi[_j][_qp] * _grad_test[_i][_qp];
  if (1)
  {
    RealGradient dkappa_dgradaeta = get_dkappa_darg(_qp);
    Real dkappa_dgradaeta_dot_grad_phi = dkappa_dgradaeta * _grad_phi[_j][_qp];
    Real grad_u_dot_grad_test = _grad_u[_qp] * _grad_test[_i][_qp];
    Real grad_u_dot_grad_phi = _grad_u[_qp] * _grad_phi[_j][_qp];
    Real dkappa_dgradaeta_dot_grad_test = dkappa_dgradaeta * _grad_test[_i][_qp];
    Real d2kappa_dgradaeta2_dot_grad_phi_dot_grad_test = 0.0;
    for (unsigned int i = 0; i < _nvar; ++i)
    {
      for (unsigned int j = 0; j < _nvar; ++j)
      {
        d2kappa_dgradaeta2_dot_grad_phi_dot_grad_test += \
          _grad_test[_i][_qp](i) * (*_d2kappa_darg2[i][j])[_qp] * _grad_phi[_j][_qp](j);
      }
    }
    return _L[_qp] * (kappa_gradphi_dot_grad_test \
                     + dkappa_dgradaeta_dot_grad_phi * grad_u_dot_grad_test \
                     + grad_u_dot_grad_phi * dkappa_dgradaeta_dot_grad_test \
                     + 0.5 * grad_u_sq * d2kappa_dgradaeta2_dot_grad_phi_dot_grad_test);
  }else{
    return _L[_qp] * kappa_gradphi_dot_grad_test;
  }
}


Real
myDPMtraining::computeQpOffDiagJacobian(unsigned int jvar)
{
  //if (grad_u_sq > 1e-2/1e-4)
  if (0)
  {
    // get the coupled variable jvar is referring to
    const unsigned int cvar = mapJvarToCvar(jvar);
    Real grad_u_sq = _grad_u[_qp] * _grad_u[_qp];
    Real xcvar_dot_grad_test = get_dargv_darg(cvar) * _grad_test[_i][_qp];
    Real grad_u_dot_grad_test = _grad_u[_qp] * _grad_test[_i][_qp];
    Real dkappa_dgradaeta_dot_grad_test = get_dkappa_darg(_qp) * _grad_test[_i][_qp];
    Real d2kappa_dgradaeta_dcvar_dot_grad_test = get_d2kappa_darg2(cvar, _qp) * _grad_test[_i][_qp];
    return _L[_qp] * _phi[_j][_qp] * ((*_dkappa_darg[cvar])[_qp] * grad_u_dot_grad_test \
                      + _kappa[_qp] * xcvar_dot_grad_test \
                      + _grad_u[_qp](cvar) * dkappa_dgradaeta_dot_grad_test \
                      + 0.5 * grad_u_sq * d2kappa_dgradaeta_dcvar_dot_grad_test);
  }else{
    //return _L[_qp] * _phi[_j][_qp] * _kappa[_qp] * xcvar_dot_grad_test;
    return 0;
  }
}
