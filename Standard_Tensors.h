#ifndef STANDARD_TENSORS_H
#define STANDARD_TENSORS_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>


using namespace dealii;

template<int dim>
class Standard_Tensors
{
    public:
        Standard_Tensors() {}
        virtual ~Standard_Tensors() {}
        //Second-order unit tensor
        static const SymmetricTensor<2,dim> I;
        //Used for NeoHookian material
        static const SymmetricTensor<4,dim> IxI;
        //Fourth-order symmetric identity tensor
        static const SymmetricTensor<4,dim> II;
        //Fourth-order deviatoric operator
        static const SymmetricTensor<4,dim> dev_P;
    protected:
    private:
};

//-------------------------------------------
//-------------------------------------------
template <int dim>
const SymmetricTensor<2, dim>
Standard_Tensors<dim>::I = unit_symmetric_tensor<dim>();
//-------------------------------------------
//-------------------------------------------

//-------------------------------------------
//-------------------------------------------
template <int dim>
const SymmetricTensor<4, dim>
Standard_Tensors<dim>::IxI = outer_product(I, I);
//-------------------------------------------
//-------------------------------------------

//-------------------------------------------
//-------------------------------------------
template <int dim>
const SymmetricTensor<4, dim>
Standard_Tensors<dim>::II = identity_tensor<dim>();
//-------------------------------------------
//-------------------------------------------

//-------------------------------------------
//-------------------------------------------
template <int dim>
const SymmetricTensor<4, dim>
Standard_Tensors<dim>::dev_P = deviator_tensor<dim>();
//-------------------------------------------
//-------------------------------------------


#endif // STANDARD_TENSORS_H
