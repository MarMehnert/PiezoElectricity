#ifndef HYPERELASTICMATERIAL_H
#define HYPERELASTICMATERIAL_H

#include "Material.h"



template<int dim>
class HyperelasticMaterial : private Material<dim>
{
    public:
        HyperelasticMaterial(const double mu, const double nu);
        virtual ~HyperelasticMaterial(){}
    protected:
    private:
};



//CSTR
//--------------------------------------
//--------------------------------------
template <int dim>
HyperelasticMaterial<dim>::HyperelasticMaterial(const double mu, const double nu)
:
Material<dim>::Material(mu, nu)
{}
//--------------------------------------
//--------------------------------------



#endif // HYPERELASTICMATERIAL_H
