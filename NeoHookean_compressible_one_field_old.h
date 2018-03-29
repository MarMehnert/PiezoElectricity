#ifndef NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_OLD_H
#define NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_OLD_H


#include <map>
#include <vector>

#include "Material.h"
#include "HyperelasticMaterial.h"
#include "Standard_Tensors.h"



using namespace dealii;

template<int dim>
class NeoHookean_compressible_one_field_old : private HyperelasticMaterial<dim>
{

    public:
        NeoHookean_compressible_one_field_old(double mu, double nu);
        ~NeoHookean_compressible_one_field_old();

        void update_material_data(const Tensor<2, dim> &F);
        SymmetricTensor<2, dim> get_tau();
        SymmetricTensor<2, dim> get_sigma();
        SymmetricTensor<4, dim> get_Jc() const;
        double get_dPsi_vol_dJ() const;
        double get_d2Psi_vol_dJ2() const;
        double get_det_F() const;


    protected:

        SymmetricTensor<2, dim> get_tau_vol() const;
        SymmetricTensor<2, dim> get_tau_iso() const;
        SymmetricTensor<2, dim> get_tau_bar() const;
        SymmetricTensor<4, dim> get_Jc_vol() const;
        SymmetricTensor<4, dim> get_Jc_iso() const;
        SymmetricTensor<4, dim> get_c_bar() const;

        const double kappa;
        const double c_1;
        double det_F;
        SymmetricTensor<2, dim> b_bar;
        //SETUP Standard_Tensors object
        const Standard_Tensors<dim> Tensors;


    private:

};

//CSTR
//-------------------------------
//-------------------------------
template <int dim>
NeoHookean_compressible_one_field_old<dim>::NeoHookean_compressible_one_field_old(double mu, double nu)
:
HyperelasticMaterial<dim>::HyperelasticMaterial(mu, nu),
kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
c_1(mu / 2.0),
det_F(1.0),
b_bar(Tensors.I)
{
    Assert(kappa > 0, ExcInternalError());
}
//-------------------------------
//-------------------------------

//DCSTR
//-------------------------------
//-------------------------------
template <int dim>
NeoHookean_compressible_one_field_old<dim>::~NeoHookean_compressible_one_field_old()
{

}
//-------------------------------
//-------------------------------

//PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
void NeoHookean_compressible_one_field_old<dim>::update_material_data(const Tensor<2, dim> &F)
{
    det_F = determinant(F);
    b_bar = std::pow(det_F, -2.0 / 3.0) * symmetrize(F * transpose(F));

    Assert(det_F > 0, ExcInternalError());
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_old<dim>::get_tau()
{
    return get_tau_iso() + get_tau_vol();
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_old<dim>::get_sigma()
{
    return ((get_tau_iso() + get_tau_vol()) * (1/det_F) );
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_old<dim>::get_Jc() const
{
    return get_Jc_vol() + get_Jc_iso();
}

//------------------------------------------

template <int dim>
double NeoHookean_compressible_one_field_old<dim>::get_dPsi_vol_dJ() const
{
    return (kappa / 2.0) * (det_F - 1.0 / det_F);
}

//------------------------------------------

template <int dim>
double NeoHookean_compressible_one_field_old<dim>::get_d2Psi_vol_dJ2() const
{
    return ( (kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
}

//------------------------------------------

template <int dim>
double NeoHookean_compressible_one_field_old<dim>::get_det_F() const
{
  return det_F;
}


//END PUBLIC MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------




//PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------
template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_old<dim>::get_tau_vol() const
{
    return get_dPsi_vol_dJ() * det_F * Tensors.I;
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_old<dim>::get_tau_iso() const
{
  return Tensors.dev_P * get_tau_bar();
}

//------------------------------------------

template <int dim>
SymmetricTensor<2, dim> NeoHookean_compressible_one_field_old<dim>::get_tau_bar() const
{
  return 2.0 * c_1 * b_bar;
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_old<dim>::get_Jc_vol() const
{
    return det_F
    * ( (get_dPsi_vol_dJ() + det_F * get_d2Psi_vol_dJ2())*Tensors.IxI
       - (2.0 * get_dPsi_vol_dJ())*Tensors.II );
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_old<dim>::get_Jc_iso() const
{
  const SymmetricTensor<2, dim> tau_bar = get_tau_bar();
  const SymmetricTensor<2, dim> tau_iso = get_tau_iso();
  const SymmetricTensor<4, dim> tau_iso_x_I
    = outer_product(tau_iso,
                    Tensors.I);
  const SymmetricTensor<4, dim> I_x_tau_iso
    = outer_product(Tensors.I,
                    tau_iso);
  const SymmetricTensor<4, dim> c_bar = get_c_bar();

  return (2.0 / 3.0) * trace(tau_bar)
         * Tensors.dev_P
         - (2.0 / 3.0) * (tau_iso_x_I + I_x_tau_iso)
         + Tensors.dev_P * c_bar
         * Tensors.dev_P;
}

//------------------------------------------

template <int dim>
SymmetricTensor<4, dim> NeoHookean_compressible_one_field_old<dim>::get_c_bar() const
{
  return SymmetricTensor<4, dim>();
}

//END PRIVATE MEMBER FUNCTIONS
//----------------------------------------------------------------------------------------

#endif // NEOHOOKEAN_COMPRESSIBLE_ONE_FIELD_OLD_H
