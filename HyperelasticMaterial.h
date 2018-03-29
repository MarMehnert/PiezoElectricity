#ifndef HYPERELASTICMATERIAL_H
#define HYPERELASTICMATERIAL_H

#include "Material.h"



template<int dim>
class HyperelasticMaterial : private Material<dim>
{
    public:
        HyperelasticMaterial(const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61);
        virtual ~HyperelasticMaterial(){}
    protected:
    private:
};



//CSTR
//--------------------------------------
//--------------------------------------
template <int dim>
HyperelasticMaterial<dim>::HyperelasticMaterial(const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61)
:
Material<dim>::Material(mu, nu, epsilon_x, epsilon_y, epsilon_z, e_13, e_33, e_61)
{}
//--------------------------------------
//--------------------------------------



#endif // HYPERELASTICMATERIAL_H
