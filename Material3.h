#ifndef MATERIAL_H
#define MATERIAL_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


template<int dim>
class Material
{
    public:
        Material(const double mu, const double nu);
        virtual ~Material();
    protected:
        double mu;
        double nu;
    private:
};

//-------------------------------------------
//-------------------------------------------
//CSTR
template <int dim>
Material<dim>::Material(const double mu, const double nu)
:
mu(mu),
nu(nu)
{}
//-------------------------------------------
//-------------------------------------------


//-------------------------------------------
//-------------------------------------------
//DCSTR
template <int dim>
Material<dim>::~Material()
{

}
//-------------------------------------------
//-------------------------------------------

#endif // MATERIAL_H
