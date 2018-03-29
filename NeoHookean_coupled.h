#ifndef NEOHOOKEAN_COUPLED_H
#define NEOHOOKEAN_COUPLED_H





#include <map>
#include <vector>

#include "Material.h"
#include "HyperelasticMaterial.h"
#include "Standard_Tensors.h"
#include <errno.h>



using namespace dealii;

template<int dim>
class NeoHookean_coupled : private HyperelasticMaterial<dim>
{

public:
	NeoHookean_coupled(const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61);
	~NeoHookean_coupled();

	void update_material_data(const Tensor<2, dim> &F,const Tensor<2, dim> &grad_u,const Tensor<1, dim> &grad_V);
	SymmetricTensor<2, dim> get_tau_tot() ;
	SymmetricTensor<2, dim> get_tau_mech() ;
	SymmetricTensor<2, dim> get_tau_piezo() ;
	SymmetricTensor<2, dim> get_sigma_tot() ;
	SymmetricTensor<2, dim> get_sigma_mech() ;
	SymmetricTensor<2, dim> get_sigma_piezo() ;
	Tensor<1,dim> get_D();
	Tensor<2, dim> get_PKS() ;
	SymmetricTensor<4, dim> get_Jc() ;
	Tensor<3, dim> get_P() ;
	double get_det_F() ;
	double get_mu();
	double get_lambda();
	double get_nu();
	double get_epsilon_x();
	double get_epsilon_y();
	double get_epsilon_z();
	double get_e_13();
	double get_e_33();
	double get_e_61();
	SymmetricTensor<4,dim> get_c_0();


protected:

	const double kappa;
	const double lambda;
	const double mu;
	const double nu;
	const double epsilon_x;
	const double epsilon_y;
	const double epsilon_z;
	const double e_13;
	const double e_33;
	const double e_61;
	double det_F;
	Tensor<2,dim> F;
	Tensor<2,dim> grad_u;
	Tensor<1,dim> E;
	Tensor<1,dim> grad_V;
	//SETUP Standard_Tensors object
	const Standard_Tensors<dim> Tensors;
	SymmetricTensor<2,dim> compute_sigma_tot() const;
	SymmetricTensor<2,dim> compute_sigma_mech() const;
	SymmetricTensor<2,dim> compute_sigma_piezo() const;
	Tensor<1,dim> compute_D() const;
	Tensor<2,dim> compute_PKS() const;
	SymmetricTensor<4,dim> compute_tangent() const;
	Tensor<3,dim>compute_P() const;
	SymmetricTensor<2,dim> compute_tau_tot()const ;
	SymmetricTensor<2,dim> compute_tau_mech()const ;
	SymmetricTensor<2,dim> compute_tau_piezo()const ;
	SymmetricTensor<2,dim> get_b()const;

	DeclException1 (ExcNegativeJacobian, double, <<"Jacobian of deformation gradient: "<<arg1);


private:

};

//CSTR
//-------------------------------
//-------------------------------
template <int dim>
NeoHookean_coupled<dim>::NeoHookean_coupled(double mu, double nu, double epsilon_x, double epsilon_y, double epsilon_z, double e_13, double e_33, double e_61)
:
HyperelasticMaterial<dim>::HyperelasticMaterial(mu, nu, epsilon_x, epsilon_y, epsilon_z, e_13, e_33, e_61),
kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
lambda((2.0*mu*nu)/(1-(2*nu))),
mu(mu),
nu(nu),
epsilon_x(epsilon_x),
epsilon_y(epsilon_y),
epsilon_z(epsilon_z),
e_13(e_13),
e_33(e_33),
e_61(e_61),
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
NeoHookean_coupled<dim>::~NeoHookean_coupled()
{

}
//-------------------------------
//-------------------------------

//PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
void NeoHookean_coupled<dim>::update_material_data(const Tensor<2, dim> &F,const Tensor<2, dim> &Grad_u_n, const Tensor<1,dim>&Grad_V_n)
{
	det_F = determinant(F);
	this->grad_u=Grad_u_n;
	this->grad_V=Grad_V_n;
	this->F = F;
	this->E = -Grad_V_n;
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
double NeoHookean_coupled<dim>::get_mu()
{
	return this->mu;
}
//------------------------------------------

template<int dim>
double NeoHookean_coupled<dim>::get_lambda()
{
	return this->lambda;
}
//------------------------------------------
template<int dim>
double NeoHookean_coupled<dim>::get_nu()
{
	return this->nu;
}

template<int dim>
double NeoHookean_coupled<dim>::get_epsilon_x()
{
	return this->epsilon_x;
}

template<int dim>
double NeoHookean_coupled<dim>::get_epsilon_y()
{
	return this->epsilon_y;
}

template<int dim>
double NeoHookean_coupled<dim>::get_epsilon_z()
{
	return this->epsilon_z;
}

template<int dim>
double NeoHookean_coupled<dim>::get_e_13()
{
	return this->e_13;
}

template<int dim>
double NeoHookean_coupled<dim>::get_e_33()
{
	return this->e_33;
}

template<int dim>
double NeoHookean_coupled<dim>::get_e_61()
{
	return this->e_61;
}
//------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_tau_tot()
{
	return compute_tau_tot();
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_tau_mech()
{
	return compute_tau_mech();
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_tau_piezo()
{
	return compute_tau_piezo();
}
//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_sigma_tot()
{
	return (compute_sigma_tot() );
}
template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_sigma_mech()
{
	return (compute_sigma_mech() );
}
template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_sigma_piezo()
{
	return (compute_sigma_piezo() );
}

template <int dim>
Tensor<1, dim> NeoHookean_coupled<dim>::get_D()
{
	return (compute_D() );
}
//------------------------------------------

template <int dim>
Tensor<2, dim> NeoHookean_coupled<dim>::get_PKS()
{
	return (compute_PKS() );
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_coupled<dim>::get_Jc()
{
	return (compute_tangent()  * det_F);
}

template <int dim>
Tensor<3, dim> NeoHookean_coupled<dim>::get_P()
{
	return (compute_P()  * det_F);
}

template <int dim>
SymmetricTensor<4, dim> NeoHookean_coupled<dim>::get_c_0()
{
	return ( (lambda*Tensors.IxI) + (2*mu*Tensors.II) );
}

template <int dim>
double NeoHookean_coupled<dim>::get_det_F()
{
	return det_F;
}


//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------




//PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_tau_tot() const
{
	return (compute_sigma_tot() * det_F);
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_tau_mech() const
{
	return (compute_sigma_mech() * det_F);
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_tau_piezo() const
{
	return (compute_sigma_piezo() * det_F);
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::get_b() const
{
	return symmetrize((F * transpose(F)));
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_sigma_tot() const
{
	return compute_sigma_mech()+compute_sigma_piezo();
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_sigma_mech() const
{
	return (   (mu/det_F)*(get_b()-Tensors.I) + ( ((lambda * std::log(det_F))/det_F ) * Tensors.I)  );
}

template <int dim>
SymmetricTensor<2, dim> NeoHookean_coupled<dim>::compute_sigma_piezo() const
{
	Tensor<3,dim> P;

	P[0][0][2]=e_13;
	P[1][1][2]=e_13;
	P[2][2][2]=e_33;

	return symmetrize(P*E);
}

template <int dim>
Tensor<1, dim> NeoHookean_coupled<dim>::compute_D() const
{
	SymmetricTensor<2,dim> DD = unit_symmetric_tensor<dim>();
	SymmetricTensor<2,dim> strain =symmetrize(0.5*(transpose(grad_u)+grad_u));

	DD[0][0]= epsilon_x;
	DD[1][1]= epsilon_y;
	DD[2][2]= epsilon_z;

	Tensor<3,dim> P;

	P[0][0][2]=e_13;
	P[1][1][2]=e_13;
	P[2][2][2]=e_33;
	Tensor<1,dim>help;
	for(unsigned int i=0; i<3;i++)
		for(unsigned int j=0; j<3;j++)
			for(unsigned int k=0; k<3;k++)
		{
			help[i]=P[i][j][k]*strain[j][k];
		}

	return (help+DD*E);
}

//------------------------------------------

template <int dim>
Tensor<2, dim> NeoHookean_coupled<dim>::compute_PKS() const
{
	Tensor<2,dim> tau_tot = static_cast<Tensor<2,dim> > ( compute_tau_tot() );
	Tensor<2,dim> F_inv = invert(this->F);
	return (( tau_tot ) * (transpose(F_inv)) );
}

//------------------------------------------


template <int dim>
SymmetricTensor<4, dim> NeoHookean_coupled<dim>::compute_tangent() const
{
	return ( ((lambda/det_F)*Tensors.IxI + 2*( (mu-(lambda*std::log(det_F)))/ det_F  )*Tensors.II ) );
}

template <int dim>
Tensor<3, dim> NeoHookean_coupled<dim>::compute_P() const
{
	Tensor<3,dim> P;

	P[0][0][2]=e_13;
	P[1][1][2]=e_13;
	P[2][2][2]=e_33;

	return P;
}

//------------------------------------------


//END PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------



#endif // NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_2_H
