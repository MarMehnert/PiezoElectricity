#ifndef NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_2_H
#define NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_2_H





#include <map>
#include <vector>

#include "Material.h"
#include "HyperelasticMaterial.h"
#include "Standard_Tensors.h"
#include <errno.h>



using namespace dealii;

template<int dim>
class NeoHookean_compressible_one_field_2 : private HyperelasticMaterial<dim>
{

    public:
        NeoHookean_compressible_one_field_2(double mu, double nu);
        ~NeoHookean_compressible_one_field_2();

        void update_material_data(const Tensor<2, dim> &F);
        SymmetricTensor<2, dim> get_tau() ;
        SymmetricTensor<2, dim> get_sigma() ;
        Tensor<2, dim> get_PKS() ;
        SymmetricTensor<4, dim> get_Jc() ;
        double get_det_F() ;
        double get_mu();
        double get_lambda();
		double get_nu();
		SymmetricTensor<4,dim> get_c_0();


    protected:

        const double kappa;
        const double lambda;
        const double mu;
        const double nu;
        double det_F;
        Tensor<2,dim> F;
        //SETUP Standard_Tensors object
        const Standard_Tensors<dim> Tensors;
        SymmetricTensor<2,dim> compute_sigma() const;
        Tensor<2,dim> compute_PKS() const;
        SymmetricTensor<4,dim> compute_tangent() const;
        SymmetricTensor<2,dim> compute_tau()const ;
        SymmetricTensor<2,dim> get_b()const;

		DeclException1 (ExcNegativeJacobian, double, <<"Jacobian of deformation gradient: "<<arg1);


    private:

};

//CSTR
//-------------------------------
//-------------------------------
template <int dim>
NeoHookean_compressible_one_field_2<dim>::NeoHookean_compressible_one_field_2(double mu, double nu)
:
HyperelasticMaterial<dim>::HyperelasticMaterial(mu, nu),
kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
lambda((2.0*mu*nu)/(1-(2*nu))),
mu(mu),
nu(nu),
det_F(1.0)
{
    Assert(kappa > 0, ExcInternalError());
}
//-------------------------------
//-------------------------------

//DCSTR
//-------------------------------
//-------------------------------
template <int dim>
NeoHookean_compressible_one_field_2<dim>::~NeoHookean_compressible_one_field_2()
{

}
//-------------------------------
//-------------------------------

//PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
void NeoHookean_compressible_one_field_2<dim>::update_material_data(const Tensor<2, dim> &F)
{
    det_F = determinant(F);
    this->F = F;
    if(det_F <= 0)
    {
		deallog<<"detF: "<<det_F<<std::endl;
        deallog<<"F:\n"<<F<<std::endl;
		errno=1;
// 		throw det_F;
//         throw std::runtime_error("det_F !> 0");
// 		throw std::bad_exception();
    }
//     AssertNothrow(det_F > 0, ExcNegativeJacobian(det_F));
}

//------------------------------------------

template<int dim>
double NeoHookean_compressible_one_field_2<dim>::get_mu()
{
    return this->mu;
}
//------------------------------------------

template<int dim>
double NeoHookean_compressible_one_field_2<dim>::get_lambda()
{
    return this->lambda;
}
//------------------------------------------
template<int dim>
double NeoHookean_compressible_one_field_2<dim>::get_nu()
{
	return this->nu;
}
//------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_2<dim>::get_tau()
{
    return compute_tau();
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_2<dim>::get_sigma()
{
    return (compute_sigma() );
}

//------------------------------------------

template <int dim>
Tensor<2, dim> NeoHookean_compressible_one_field_2<dim>::get_PKS()
{
    return (compute_PKS() );
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_2<dim>::get_Jc()
{
    return (compute_tangent()  * det_F);
}

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_2<dim>::get_c_0()
{
    return ( (lambda*Tensors.IxI) + (2*mu*Tensors.II) );
}

template <int dim>
double NeoHookean_compressible_one_field_2<dim>::get_det_F()
{
  return det_F;
}


//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------




//PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_2<dim>::compute_tau() const
{
    return (compute_sigma() * det_F);
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_2<dim>::get_b() const
{
  return symmetrize((F * transpose(F)));
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_2<dim>::compute_sigma() const
{
  return (   (mu/det_F)*(get_b()-Tensors.I) + ( ((lambda * std::log(det_F))/det_F ) * Tensors.I)  );
}

//------------------------------------------

template <int dim>
Tensor<2, dim> NeoHookean_compressible_one_field_2<dim>::compute_PKS() const
{
    Tensor<2,dim> tau = static_cast<Tensor<2,dim> > ( compute_tau() );
    Tensor<2,dim> F_inv = invert(this->F);
    return (( tau ) * (transpose(F_inv)) );
}

//------------------------------------------


template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_2<dim>::compute_tangent() const
{
    return ( ((lambda/det_F)*Tensors.IxI + 2*( (mu-(lambda*std::log(det_F)))/ det_F  )*Tensors.II ) );
}

//------------------------------------------


//END PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------



#endif // NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_2_H
