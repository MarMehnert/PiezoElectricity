#ifndef POINT_HISTORY_H
#define POINT_HISTORY_H

#include "Standard_Tensors.h"
#include "NeoHookean_compressible_one_field_old.h"
#include "NeoHookean_compressible_one_field_2.h"

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>



template<int dim>
class Point_History
{
    public:
        Point_History();
        virtual ~Point_History();
        void setup_lqp (const double mu, const double nu);
        void update_values (const Tensor<2, dim> &Grad_u_n);
        double get_det_F() const;
        const Tensor<2, dim> &get_F_inv() const;
        const Tensor<2, dim> &get_F() const;
        const Tensor<2, dim> &get_grad_u() const;
        const SymmetricTensor<2, dim> &get_tau() const;
        const Tensor<2, dim> &get_PKS() const;
        const SymmetricTensor<2, dim> &get_sigma() const;
        const SymmetricTensor<4, dim> &get_Jc() const;
		const SymmetricTensor<4, dim> &get_c_0() const;
        double get_mu();
        double get_lambda();
		double get_nu();
    protected:
    private:
        const Standard_Tensors<dim> Tensors;
//        NeoHookean_compressible_one_field_old<dim> *material;
        NeoHookean_compressible_one_field_2<dim> *material;
        Tensor<2, dim> F_inv;
        Tensor<2, dim> F;
        Tensor<2, dim> grad_u;
        SymmetricTensor<2, dim> tau;
        Tensor<2, dim> PKS;
        SymmetricTensor<2, dim> sigma;
        SymmetricTensor<4, dim> Jc;
		SymmetricTensor<4, dim> c_0;
};



//CSTR
//-------------------------------
//-------------------------------
template<int dim>
Point_History<dim>::Point_History()
:
material(NULL),
F_inv(Tensors.I),
F(Tensors.I),
grad_u(Tensors.I),
tau(SymmetricTensor<2, dim>()),
PKS(Tensor<2,dim>()),
Jc(SymmetricTensor<4, dim>())
{}
//-------------------------------
//-------------------------------


//DCSTR
//-------------------------------
//-------------------------------
template<int dim>
Point_History<dim>::~Point_History()
{
    delete material;
    material = NULL;
}
//-------------------------------
//-------------------------------

//PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template<int dim>
void Point_History<dim>::setup_lqp (const double mu, const double nu)
{
//    material = new NeoHookean_compressible_one_field_old<dim>(mu, nu);
    material = new NeoHookean_compressible_one_field_2<dim>(mu, nu);
    update_values(Tensor<2, dim>());
}

//------------------------------------------

template<int dim>
double Point_History<dim>::get_mu()
{
    double tmp;
    tmp = material->get_mu();
    return tmp;
}

//------------------------------------------

template<int dim>
double Point_History<dim>::get_lambda()
{
    double tmp;
    tmp = material->get_lambda();
    return tmp;
}

//------------------------------------------

template<int dim>
double Point_History<dim>::get_nu()
{
    double tmp;
    tmp = material->get_nu();
    return tmp;
}

//------------------------------------------

template<int dim>
void Point_History<dim>::update_values (const Tensor<2, dim> &Grad_u_n)
{
//    const Tensor<2, dim> F = (Tensor<2, dim>(Tensors.I) + Grad_u_n);
	grad_u=Grad_u_n;
    F = (Tensor<2, dim>(Tensors.I) + Grad_u_n);
    material->update_material_data(F);

    F_inv = invert(F);
    tau = material->get_tau();
    sigma = material->get_sigma();
    PKS = material->get_PKS();
    Jc = material->get_Jc();
	c_0 = material->get_c_0();
}

//------------------------------------------

template<int dim>
double Point_History<dim>::get_det_F() const
{
    return material->get_det_F();
}

//------------------------------------------

template<int dim>
const Tensor<2, dim> & Point_History<dim>::get_F_inv() const
{
    return F_inv;
}

//------------------------------------------


template<int dim>
const Tensor<2, dim> & Point_History<dim>::get_F() const
{
    return F;
}

template<int dim>
const Tensor<2, dim> & Point_History<dim>::get_grad_u() const
{
    return grad_u;
}
//------------------------------------------

template<int dim>
const SymmetricTensor<2, dim> & Point_History<dim>::get_tau() const
{
    return tau;
}

//------------------------------------------


template<int dim>
const Tensor<2, dim> & Point_History<dim>::get_PKS() const
{
    return PKS;
}

//------------------------------------------

template<int dim>
const SymmetricTensor<2, dim> & Point_History<dim>::get_sigma() const
{
    return sigma;
}

//------------------------------------------

//tangent:
template<int dim>
const SymmetricTensor<4, dim> & Point_History<dim>::get_Jc() const
{
    return Jc;
}

//initial tangent:
template<int dim>
const SymmetricTensor<4, dim> & Point_History<dim>::get_c_0() const
{
    return c_0;
}


//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------

#endif // POINT_HISTORY_H
