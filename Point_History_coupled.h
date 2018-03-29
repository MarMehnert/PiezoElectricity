#ifndef POINT_HISTORY_COUPLED_H
#define POINT_HISTORY_COUPLED_H

#include "Standard_Tensors.h"
#include "NeoHookean_compressible_one_field_old.h"
#include "NeoHookean_compressible_one_field_2.h"
#include "NeoHookean_coupled.h"

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>



template<int dim>
class Point_History_coupled
{
    public:
        Point_History_coupled();
        virtual ~Point_History_coupled();
        void setup_lqp (const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61);
        void update_values (const Tensor<2, dim> &Grad_u_n, const Tensor<1, dim> &Grad_V_n);
        double get_det_F() const;
        const Tensor<2, dim> &get_F_inv() const;
        const Tensor<2, dim> &get_F() const;
        const Tensor<2, dim> &get_grad_u() const;
        const Tensor<1, dim> &get_E() const;
        const SymmetricTensor<2, dim> &get_tau_tot() const;
        const SymmetricTensor<2, dim> &get_tau_mech() const;
        const SymmetricTensor<2, dim> &get_tau_piezo() const;
        const Tensor<2, dim> &get_PKS() const;
        const SymmetricTensor<2, dim> &get_sigma_tot() const;
        const SymmetricTensor<2, dim> &get_sigma_mech() const;
        const SymmetricTensor<2, dim> &get_sigma_piezo() const;
        const Tensor<1,dim> &get_D() const;
        const SymmetricTensor<4, dim> &get_Jc() const;
        const Tensor<3, dim> &get_P() const;
		const SymmetricTensor<4, dim> &get_c_0() const;
        double get_mu();
        double get_lambda();
		double get_nu();
		double get_epsilon_x();
		double get_epsilon_y();
		double get_epsilon_z();
		double get_e_13();
		double get_e_33();
		double get_e_61();
    protected:
    private:
        const Standard_Tensors<dim> Tensors;
        //NeoHookean_coupled<dim> *material;
        NeoHookean_coupled<dim> *material;
        Tensor<1,dim> E;
        Tensor<1,dim> grad_V;
        Tensor<2, dim> F_inv;
        Tensor<2, dim> F;
        Tensor<2, dim> grad_u;
        SymmetricTensor<2, dim> tau_mech;
        SymmetricTensor<2, dim> tau_piezo;
        SymmetricTensor<2, dim> tau_tot;
        Tensor<2, dim> PKS;
        SymmetricTensor<2, dim> sigma_tot;
        SymmetricTensor<2, dim> sigma_mech;
        SymmetricTensor<2, dim> sigma_piezo;
        Tensor<1,dim> D;
        SymmetricTensor<4, dim> Jc;
        Tensor<3,dim> P;
		SymmetricTensor<4, dim> c_0;
};



//CSTR
//-------------------------------
//-------------------------------
template<int dim>
Point_History_coupled<dim>::Point_History_coupled()
:
material(NULL),
F_inv(Tensors.I),
F(Tensors.I),
tau_tot(SymmetricTensor<2, dim>()),
PKS(Tensor<2,dim>()),
Jc(SymmetricTensor<4, dim>())
{}
//-------------------------------
//-------------------------------


//DCSTR
//-------------------------------
//-------------------------------
template<int dim>
Point_History_coupled<dim>::~Point_History_coupled()
{
    delete material;
    material = NULL;
}
//-------------------------------
//-------------------------------

//PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template<int dim>
void Point_History_coupled<dim>::setup_lqp  (const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61)
{
    material = new NeoHookean_coupled<dim>(mu, nu, epsilon_x, epsilon_y, epsilon_z, e_13, e_33, e_61);
    update_values(grad_u,grad_V);
}

//------------------------------------------

template<int dim>
double Point_History_coupled<dim>::get_mu()
{
    double tmp;
    tmp = material->get_mu();
    return tmp;
}

//------------------------------------------

template<int dim>
double Point_History_coupled<dim>::get_lambda()
{
    double tmp;
    tmp = material->get_lambda();
    return tmp;
}

//------------------------------------------

template<int dim>
double Point_History_coupled<dim>::get_nu()
{
    double tmp;
    tmp = material->get_nu();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_epsilon_x()
{
    double tmp;
    tmp = material->get_epsilon_x();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_epsilon_y()
{
    double tmp;
    tmp = material->get_epsilon_x();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_epsilon_z()
{
    double tmp;
    tmp = material->get_epsilon_x();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_e_13()
{
    double tmp;
    tmp = material->get_e_13();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_e_33()
{
    double tmp;
    tmp = material->get_e_33();
    return tmp;
}

template<int dim>
double Point_History_coupled<dim>::get_e_61()
{
    double tmp;
    tmp = material->get_e_61();
    return tmp;
}
//------------------------------------------

template<int dim>
void Point_History_coupled<dim>::update_values (const Tensor<2, dim> &Grad_u_n, const Tensor<1,dim> &Grad_V_n)
{
//    const Tensor<2, dim> F = (Tensor<2, dim>(Tensors.I) + Grad_u_n);
    F = (Tensor<2, dim>(Tensors.I) + Grad_u_n);
    grad_u=Grad_u_n;
    E = Grad_V_n;
    grad_V=Grad_V_n;
    material->update_material_data(F,grad_u,grad_V);

    F_inv = invert(F);
    tau_tot = material->get_tau_tot();
    tau_mech = material->get_tau_mech();
    tau_piezo = material->get_tau_piezo();
    sigma_tot = material->get_sigma_tot();
    sigma_mech = material->get_sigma_mech();
    sigma_piezo = material->get_sigma_piezo();
    D = material->get_D();
    PKS = material->get_PKS();
    Jc = material->get_Jc();
    P = material->get_P();
	c_0 = material->get_c_0();
}

//------------------------------------------

template<int dim>
double Point_History_coupled<dim>::get_det_F() const
{
    return material->get_det_F();
}

//------------------------------------------

template<int dim>
const Tensor<2, dim> & Point_History_coupled<dim>::get_F_inv() const
{
    return F_inv;
}

//------------------------------------------


template<int dim>
const Tensor<2, dim> & Point_History_coupled<dim>::get_F() const
{
    return F;
}

template<int dim>
const Tensor<2, dim> & Point_History_coupled<dim>::get_grad_u() const
{
    return grad_u;
}

template<int dim>
const Tensor<1, dim> & Point_History_coupled<dim>::get_E() const
{
    return E;
}

//------------------------------------------

template<int dim>
const SymmetricTensor<2, dim> & Point_History_coupled<dim>::get_tau_tot() const
{
    return tau_tot;
}

//------------------------------------------


template<int dim>
const Tensor<2, dim> & Point_History_coupled<dim>::get_PKS() const
{
    return PKS;
}

//------------------------------------------

template<int dim>
const SymmetricTensor<2, dim> & Point_History_coupled<dim>::get_sigma_tot() const
{
    return sigma_tot;
}

template<int dim>
const SymmetricTensor<2, dim> & Point_History_coupled<dim>::get_sigma_mech() const
{
    return sigma_mech;
}

template<int dim>
const SymmetricTensor<2, dim> & Point_History_coupled<dim>::get_sigma_piezo() const
{
    return sigma_piezo;
}

template<int dim>
const Tensor<1, dim> & Point_History_coupled<dim>::get_D() const
{
    return D;
}
//------------------------------------------

//tangent:
template<int dim>
const SymmetricTensor<4, dim> & Point_History_coupled<dim>::get_Jc() const
{
    return Jc;
}

template<int dim>
const Tensor<3, dim> & Point_History_coupled<dim>::get_P() const
{
    return P;
}

//initial tangent:
template<int dim>
const SymmetricTensor<4, dim> & Point_History_coupled<dim>::get_c_0() const
{
    return c_0;
}


//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------

#endif // POINT_HISTORY_H
