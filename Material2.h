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
        Material(const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61);
        virtual ~Material();
    protected:
        double mu;
        double nu;
        double epsilon_x;
        double epsilon_y;
        double epsilon_z;
        double e_13;
        double e_33;
        double e_61;
    private:
};

//-------------------------------------------
//-------------------------------------------
//CSTR
template <int dim>
Material<dim>::Material(const double mu, const double nu, const double epsilon_x, const double epsilon_y, const double epsilon_z, const double e_13, const double e_33, const double e_61)
:
mu(mu),
nu(nu),
epsilon_x(epsilon_x),
epsilon_y(epsilon_y),
epsilon_z(epsilon_z),
e_13(e_13),
e_33(e_33),
e_61(e_61)
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
