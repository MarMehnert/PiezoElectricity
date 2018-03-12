/* =========================
 * Linear Piezoelectricity
 * =========================
 * Problem description:
 *   Deformation of a cube leads to a voltage difference
 *
 * Initial implementation
 *   Author: Markus Mehnert (2015)
 *           Friedrich-Alexander University Erlangen-Nuremberg
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/block_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>


#include <deal.II/physics/notation.h>


#include <deal.II/lac/generic_linear_algebra.h>
#define USE_TRILINOS_LA
namespace LA
{
#ifdef USE_TRILINOS_LA
using namespace dealii::LinearAlgebraTrilinos;
#else
using namespace dealii::LinearAlgebraPETSc;
#endif
}

#include <mpi.h>
#include <fstream>
#include <iostream>

namespace Coupled_TEE
{
using namespace dealii;

template <int dim>
class Microscopic_point
{
public:
	Microscopic_point()
:
	C(2*dim,2*dim),
	P(),
	DD(unit_symmetric_tensor<dim>())
{}

	FullMatrix<double> C;
	Tensor<3, dim> P;
	Tensor<2, dim> DD;

	virtual ~Microscopic_point()
	{

	}

	void setup_Microscopic_point ()
	{
		update_values(FullMatrix<double>(), Tensor<3, dim>(), Tensor<2, dim>());
	}

	void update_Microscopic_point  (FullMatrix<double> &C_update, const Tensor<3, dim> &P_update, const Tensor<2, dim> &DD_update)
	{
		C=C_update;
		P=P_update;
		DD=DD_update;
	}
};


template <int dim>
class BoundaryValues_XYZ : public Function<dim>
{
public:
	BoundaryValues_XYZ (const Tensor<2,dim> F, const unsigned int n_components)
: Function<dim>(n_components),
  F (F)
  { }

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;

private:

	const Tensor<2,dim> F;
};
template <int dim>
double
BoundaryValues_XYZ<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	const Tensor<1,dim> help = (F-unit_symmetric_tensor<dim>())*p;

	if (component < dim)
	{
		return help[component];
	}
	else
		return 0.0;
		}

template <int dim>
void
BoundaryValues_XYZ<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = BoundaryValues_XYZ<dim>::value (p, c);
		}

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
	BoundaryValues (const double neu, const unsigned int n_components)
: Function<dim>(n_components),
  lambda_i (neu)
  { }

	virtual double value (const Point<dim>   &p,
			const unsigned int  component = 0) const;
	virtual void vector_value (const Point<dim> &p,
			Vector<double>   &value) const;

private:

	const double lambda_i;
};
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>  &p,
		const unsigned int component) const
		{
	Assert (component < this->n_components,
			ExcIndexRange (component, 0, this->n_components));

	if (component < dim)
	{
		return (lambda_i)*p[component];
	}
	else
		return 0.0;
		}

template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
		Vector<double>   &values) const
		{
	for (unsigned int c=0; c<this->n_components; ++c)
		values(c) = BoundaryValues<dim>::value (p, c);
		}

struct Parameters
{
	// RVE Size Information in mm
	static constexpr double x_length=1;
	static constexpr double y_length=1;
	static constexpr double z_length=1;

	// Boundary ids
	static constexpr unsigned int boundary_id_bottom = 0;
	static constexpr unsigned int boundary_id_top = 1;
	static constexpr unsigned int boundary_id_front = 2;
	static constexpr unsigned int boundary_id_back = 3;
	static constexpr unsigned int boundary_id_left = 4;
	static constexpr unsigned int boundary_id_right = 5;
	static constexpr unsigned int boundary_id_force = 6;

	// Dirichlet Boundary Conditions

	// Prescribed Displacement in mm
	static constexpr double displacement = -0.01;

	// Precscribed Potential Difference in V
	static constexpr double potential_difference = 100;

	// Neumann Boundary Conditions

	// Prescribed Force on the Cantilever in N (?)
	static constexpr double applied_pressure = -0.0001;

	// Time
	static constexpr double dt = 0.1;
	static constexpr unsigned int n_timesteps = 1;

	//J-Ps additions

	static constexpr double time_end = 50.0e-3;
	static constexpr double time_delta = time_end/(static_cast<double>(n_timesteps));

	// Refinement
	static constexpr unsigned int n_global_refinements = 0;
	static constexpr bool perform_AMR = false;
	static constexpr unsigned int n_ts_per_refinement = 2;
	static constexpr unsigned int max_grid_level = 4;
	static constexpr double frac_refine = 0.3;
	static constexpr double frac_coarsen = 0.03;

	// Finite element
	static constexpr unsigned int poly_order =1;

	// Nonlinear solver
	static constexpr unsigned int max_newton_iterations = 1;

	// Linear solver: Electro-mechanical
	static const std::string solver_type_EM;
	static constexpr double tol_rel_EM = 1e-6;
};
const std::string Parameters::solver_type_EM = "Direct";

namespace Material
{
struct Coefficients
{
	static constexpr double length_scale = 1.0;

	struct material_1
	{
		// Parameters in N, mm, V

		// Elastic parameters
		static constexpr double E_mod = 2.0e3;// E-Modulus in MPa=N/mm^2
		static constexpr double nu = 0.29; // Poisson ratio
		static constexpr double mu = E_mod/(2*(1+nu)); // Small strain shear modulus

		static constexpr double lambda = 2.0*mu*nu/(1.0-2.0*nu); // Lame constant
		static constexpr double kappa = 2.0*mu*(1.0+nu)/(3.0*(1.0-2.0*nu)); // mu = 3*10^5 Pa

		// Electro parameters
		static constexpr double epsilon_0 =8.854187817e-12; // F/m = C/(V*m)= (A*s)/(V*m) = N/(V*V)
		static constexpr double epsilon_x = -1.062502538114400e-10;  //electrical permittivity in x-Direction
		static constexpr double epsilon_y = -1.062502538114400e-10;  //electrical permittivity in y-Direction
		static constexpr double epsilon_z = -1.040639889353618e-10; //electrical permittivity in z-Direction

		// Coupling parameters
		static constexpr double e_13 = 4.91e-5;  //piezoelectric coupling coefficient
		static constexpr double e_33 = -3.15e-5;  //piezoelectric coupling coefficient
		static constexpr double e_61 = 0; //piezoelectric coupling coefficient
	};


	struct material_2
	{
		// Parameters in N, mm, V

		// Elastic parameters
		static constexpr double E_mod = 2.0e3;// E-Modulus in MPa=N/mm^2
		static constexpr double nu = 0.29; // Poisson ratio
		static constexpr double mu = E_mod/(2*(1+nu)); // Small strain shear modulus

		static constexpr double lambda = 2.0*mu*nu/(1.0-2.0*nu); // Lame constant
		static constexpr double kappa = 2.0*mu*(1.0+nu)/(3.0*(1.0-2.0*nu)); // mu = 3*10^5 Pa

		// Electro parameters
		static constexpr double epsilon_0 =8.854187817e-12; // F/m = C/(V*m)= (A*s)/(V*m) = N/(V*V)
		static constexpr double epsilon_x = -1.062502538114400e-10;  //electrical permittivity in x-Direction
		static constexpr double epsilon_y = -1.062502538114400e-10;  //electrical permittivity in y-Direction
		static constexpr double epsilon_z = -1.040639889353618e-10; //electrical permittivity in z-Direction

		// Coupling parameters
		static constexpr double e_13 = -4.91e-5;  //piezoelectric coupling coefficient
		static constexpr double e_33 = -3.15e-5;  //piezoelectric coupling coefficient
		static constexpr double e_61 = 0; //piezoelectric coupling coefficient
	};
};

template<int dim>
struct Values
{
	Values (const Tensor<2,dim> & F,
			const Tensor<1,dim> & E)
	: F (F),
	  E (E),

	  C (symmetrize(transpose(F)*F)),
	  b (symmetrize(F*transpose(F))),
	  GLST (0.5*(C - unit_symmetric_tensor<dim>())),
	  C_inv (symmetrize(invert(static_cast<Tensor<2,dim> >(C)))),
	  J (determinant(F)),

	  I1 (first_invariant(C)), // tr(C)
	  I4 (E*E), // [ExE].I
	  I5 (E*(C_inv*E)) // [ExE].C^{-1}
	{}

	// Directly related to solution field
	const Tensor<2,dim> F; // Deformation gradient
	const Tensor<1,dim> E; // Electric field vector

	// Commonly used elastic quantities
	const SymmetricTensor<2,dim> C; // Right Cauchy-Green deformation tensor
	const SymmetricTensor<2,dim> b; // left Cauchy-Green deformation tensor
	const SymmetricTensor<2,dim> GLST; // Green-Lagrange strain tensor
	const SymmetricTensor<2,dim> C_inv;
	const double J;

	// Invariants
	const double I1;
	const double I4;
	const double I5;

	// === First derivatives ===
	// --- Mechanical ---
	SymmetricTensor<2,dim>
	dI1_dC () const
	{
		return unit_symmetric_tensor<dim>();
	}

	SymmetricTensor<2,dim>
	dI4_dC () const
	{
		return SymmetricTensor<2,dim>();
	}

	SymmetricTensor<2,dim>
	dI5_dC () const
	{
		const SymmetricTensor<2,dim> ExE = outer_product(E,E);
		return ExE*dC_inv_dC();
	}

	SymmetricTensor<2,dim>
	dJ_dC () const
	{
		return (0.5*J)*C_inv;
	}

	// --- Electric ---
	Tensor<1,dim>
	dI4_dE () const
	{
		return 2.0*E;
	}

	Tensor<1,dim>
	dI5_dE () const
	{
		return 2.0*(C_inv*E);
	}

	// === Second derivatives ===
	// --- Mechanical ---
	SymmetricTensor<4,dim>
	d2I1_dC_dC () const
	{
		return SymmetricTensor<4,dim>();
	}

	SymmetricTensor<4,dim>
	d2I4_dC_dC () const
	{
		return SymmetricTensor<4,dim>();
	}

	SymmetricTensor<4,dim>
	d2I5_dC_dC () const
	{
		const Tensor<1,dim> Y = C_inv*E;

		SymmetricTensor<4,dim> d2I5_dC_dC;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
					{
						// Need to ensure symmetries of (I,J) and (K,L)
						d2I5_dC_dC[A][B][C][D] += 0.25*( Y[A]*Y[C]*C_inv[B][D] + Y[B]*Y[C]*C_inv[A][D] + Y[A]*Y[D]*C_inv[B][C] + Y[B]*Y[D]*C_inv[A][C]
																																					+ Y[C]*Y[B]*C_inv[A][D] + Y[C]*Y[A]*C_inv[B][D] + Y[D]*Y[B]*C_inv[A][C] + Y[D]*Y[A]*C_inv[B][C] );
					}

		return d2I5_dC_dC;
	}

	SymmetricTensor<4,dim>
	dC_inv_dC () const
	{
		SymmetricTensor<4,dim> dC_inv_dC;

		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=A; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					for (unsigned int D=C; D<dim; ++D)
						dC_inv_dC[A][B][C][D] = -0.5*(C_inv[A][C]*C_inv[B][D]+ C_inv[A][D]*C_inv[B][C]);

		return dC_inv_dC;
	}

	// --- Electric ---
	SymmetricTensor<2,dim>
	d2I4_dE_dE () const
	{
		return 2.0*unit_symmetric_tensor<dim>();
	}

	SymmetricTensor<2,dim>
	d2I5_dE_dE () const
	{
		return 2.0*C_inv;
	}

	// --- Electro-mechanical ---
	Tensor<3,dim>
	d2I4_dE_dC () const
	{
		return Tensor<3,dim>();
	}

	Tensor<3,dim>
	d2I5_dE_dC () const
	{
		const Tensor<1,dim> Y = C_inv*E;

		Tensor<3,dim> d2I5_dE_dC;
		for (unsigned int A=0; A<dim; ++A)
			for (unsigned int B=0; B<dim; ++B)
				for (unsigned int C=0; C<dim; ++C)
					d2I5_dE_dC[A][B][C] -= C_inv[C][A]*Y[B] + C_inv[C][B]*Y[A];

		return d2I5_dE_dC;
	}

};

template<int dim>
class CM_Base
{
public: CM_Base (const Tensor<2,dim> & F,
		const Tensor<1,dim> & E)
: values (F,E)
{}

virtual ~CM_Base () {}

// --- Kinematic Quantities ---

const Values<dim> values;

// --- Kinetic Quantities ---

// Second Piola-Kirchhoff stress tensor
inline SymmetricTensor<2,dim>
get_S () const
{
	return 2.0*get_dPsi_dC();
}

// Referential electric displacement vector
inline Tensor<1,dim>
get_D () const
{
	return -get_dPsi_dE();
}

// --- Material tangents ---

// Referential elastic stiffness tensor
inline FullMatrix<double>
get_C () const
{
	return get_d2Psi_dC_dC();
}

// Referential pizeo-elasticity tensor
inline Tensor<3,dim>
get_P () const
{
	return get_d2Psi_dE_dC();
}

// Referential dielectric tensor
inline Tensor<2,dim>
get_DD () const
{
	return get_d2Psi_dE_dE();
}

protected:

// --- Total stored energy ---

virtual double
get_Psi (void) const = 0;

// --- Pure mechanical volumetric response ---

double
get_W_J (const double kappa,
		const double g_0) const
{
	const double &J = values.J;

	// See Wriggers p45 equ 3.118
	return (kappa/2)*(std::log(J)*std::log(J)) // This part is from Holzapfel p
			- g_0*std::log(J); // This part cancels out dilatory response of Neo-Hookean material
}

// Derivative of the volumetric free energy with respect to
// $\widetilde{J}$ return $\frac{\partial
// \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
double
get_dW_J_dJ (const double kappa,
		const double g_0) const
{
	const double tr_C=values.C[0][0]+values.C[1][1]+values.C[2][2];
	const double &J = values.J;
	return ((kappa)*(std::log(J))/J
			- g_0/J);
}


// Second derivative of the volumetric free energy wrt $\widetilde{J}$. We
// need the following computation explicitly in the tangent so we make it
// public.  We calculate $\frac{\partial^2
// \Psi_{\textrm{vol}}(\widetilde{J})}{\partial \widetilde{J} \partial
// \widetilde{J}}$
double
get_d2W_J_dJ_dJ (const double kappa,
		const double g_0) const
{
	const double &J = values.J;
	return (kappa)*(1-std::log(J))/(J*J)
			+ g_0/(J*J);
}


SymmetricTensor<2,dim>
get_dW_J_dC (const double kappa,
		const double g_0) const
		{
	// See Wriggers p46 eqs. 3.123, 3.124; Holzapfel p230
	return get_dW_J_dJ(kappa, g_0)*this->values.dJ_dC();
		}

SymmetricTensor<4,dim>
get_d2W_J_dC_dC (const double kappa,
		const double g_0) const
		{
	const double &J = values.J;
	const SymmetricTensor<2,dim> &C_inv = values.C_inv;

	// See Holzapfel p255, p265 ; Wriggers p75-76
	const double dW_J_dJ = get_dW_J_dJ(kappa, g_0);
	const double d2W_J_dJ_dJ = get_d2W_J_dJ_dJ(kappa, g_0);

	const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();

	SymmetricTensor<4,dim> d2W_J_dC_dC;
	const double  p_tilde = dW_J_dJ + J*d2W_J_dJ_dJ;
	const double &p = dW_J_dJ;
	for (unsigned int A=0; A<dim; ++A)
		for (unsigned int B=A; B<dim; ++B)
			for (unsigned int C=0; C<dim; ++C)
				for (unsigned int D=C; D<dim; ++D)
				{
					d2W_J_dC_dC[A][B][C][D]
										 = 0.25*J*( p_tilde*(C_inv[A][B]*C_inv[C][D])
												 + (2.0*p)*dC_inv_dC[A][B][C][D] );
				}

	return d2W_J_dC_dC;
		}

// --- Coupled mechanical response ---

virtual SymmetricTensor<2,dim>
get_dPsi_dC () const = 0;

virtual FullMatrix<double>
get_d2Psi_dC_dC () const = 0;


// --- Coupled electric response ---

virtual Tensor<1,dim>
get_dPsi_dE () const = 0;

virtual Tensor<2,dim>
get_d2Psi_dE_dE () const = 0;

// --- Electro-mechanical coupling ---

virtual Tensor<3,dim>
get_d2Psi_dE_dC () const = 0;
};

template<int dim>
class CM_Macroscopic : public CM_Base<dim>
{
public:
	CM_Macroscopic (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E)
: CM_Base<dim> (F,E)
  {}

	virtual ~CM_Macroscopic () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{

		return get_W_F(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	// === ELECTRO-MECHANICAL PROBLEM ===
	inline double
	get_W_F (const double lambda,
			const double mu) const
	{
		const double &I1 = this->values.I1;
		const double &J = this->values.J;
		return (mu/2)*(I1-3)-mu*std::log(J) + lambda/2*(std::log(J)*std::log(J));
	}

	// --- Mechanical contributions ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return get_dW_F_dC(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double lambda,
			const double mu) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;

		return mu*(unit_symmetric_tensor<dim>()-C_inv) + lambda*std::log(J)*C_inv;
			}

	virtual FullMatrix<double>//Tensor<2,dim>
	get_d2Psi_dC_dC () const
	{
		FullMatrix<double> C_kelvin(2*dim,2*dim);
		Tensor<2,dim>C;

		C [0][0] = C [1][1] =C [2][2] = Material::Coefficients::material_1::lambda+2*Material::Coefficients::material_1::mu;
		C [0][1] = C [0][2] =C [1][0] = C [1][2] = C [2][0] =C [2][1] = Material::Coefficients::material_1::lambda;
		C [3][3] = C [4][4] =C [5][5] = (Material::Coefficients::material_1::mu);

		C_kelvin [0][0] = C_kelvin [1][1] =C_kelvin [2][2] = Material::Coefficients::material_2::lambda+2*Material::Coefficients::material_2::mu;
		C_kelvin [0][1] = C_kelvin [0][2] =C_kelvin [1][0] = C_kelvin [1][2] = C_kelvin [2][0] =C_kelvin [2][1] = Material::Coefficients::material_2::lambda;
		C_kelvin [3][3] = C_kelvin [4][4] =C_kelvin [5][5] = (Material::Coefficients::material_2::mu);


		return C_kelvin;
	}

	//	inline SymmetricTensor<4,dim>
	//	get_d2W_F_dC_dC (const double lambda,
	//			const double mu) const
	//			{
	//		const double &J = this->values.J;
	//		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;
	//		static const SymmetricTensor<4,dim> C_invxC_inv = outer_product(C_inv,
	//				C_inv);
	//		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();
	//
	//		return (-2*mu+2*lambda*std::log(J))*dC_inv_dC + lambda*C_invxC_inv;
	//			}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return Tensor<1,dim> ();
	}

	virtual Tensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		Tensor<2,dim> DD;

		DD[0][0]= Material::Coefficients::material_1::epsilon_x;
		DD[1][1]= Material::Coefficients::material_1::epsilon_x;
		DD[2][2]= Material::Coefficients::material_1::epsilon_z;

		return DD;
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		Tensor<3,dim> P;

		P[0][0][2]=Material::Coefficients::material_1::e_13;
		P[1][1][2]=Material::Coefficients::material_1::e_13;
		P[2][2][2]=Material::Coefficients::material_1::e_33;

		return P;
	}

};

template<int dim>
class CM_Microscopic_Sheet : public CM_Base<dim>
{
public:
	CM_Microscopic_Sheet (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E)
: CM_Base<dim> (F,E)
  {}

	virtual ~CM_Microscopic_Sheet () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{

		return get_W_F(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	// === ELECTRO-MECHANICAL PROBLEM ===
	inline double
	get_W_F (const double lambda,
			const double mu) const
	{
		const double &I1 = this->values.I1;
		const double &J = this->values.J;
		return (mu/2)*(I1-3)-mu*std::log(J) + lambda/2*(std::log(J)*std::log(J));
	}

	// --- Mechanical contributions ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return get_dW_F_dC(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double lambda,
			const double mu) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;

		return mu*(unit_symmetric_tensor<dim>()-C_inv) + lambda*std::log(J)*C_inv;
			}

	virtual FullMatrix<double>//Tensor<2,dim>
	get_d2Psi_dC_dC () const
	{
		FullMatrix<double> C_kelvin(2*dim,2*dim);
		Tensor<2,dim>C;

		C [0][0] = C [1][1] =C [2][2] = Material::Coefficients::material_1::lambda+2*Material::Coefficients::material_1::mu;
		C [0][1] = C [0][2] =C [1][0] = C [1][2] = C [2][0] =C [2][1] = Material::Coefficients::material_1::lambda;
		C [3][3] = C [4][4] =C [5][5] = (Material::Coefficients::material_1::mu);

		C_kelvin [0][0] = C_kelvin [1][1] =C_kelvin [2][2] = Material::Coefficients::material_2::lambda+2*Material::Coefficients::material_2::mu;
		C_kelvin [0][1] = C_kelvin [0][2] =C_kelvin [1][0] = C_kelvin [1][2] = C_kelvin [2][0] =C_kelvin [2][1] = Material::Coefficients::material_2::lambda;
		C_kelvin [3][3] = C_kelvin [4][4] =C_kelvin [5][5] = (Material::Coefficients::material_2::mu);


		return C_kelvin;
	}

	//	inline SymmetricTensor<4,dim>
	//	get_d2W_F_dC_dC (const double lambda,
	//			const double mu) const
	//			{
	//		const double &J = this->values.J;
	//		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;
	//		static const SymmetricTensor<4,dim> C_invxC_inv = outer_product(C_inv,
	//				C_inv);
	//		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();
	//
	//		return (-2*mu+2*lambda*std::log(J))*dC_inv_dC + lambda*C_invxC_inv;
	//			}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return Tensor<1,dim> ();
	}

	virtual Tensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		Tensor<2,dim> DD;

		DD[0][0]= Material::Coefficients::material_1::epsilon_x;
		DD[1][1]= Material::Coefficients::material_1::epsilon_x;
		DD[2][2]= Material::Coefficients::material_1::epsilon_z;

		return DD;
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		Tensor<3,dim> P;

		P[0][0][2]=Material::Coefficients::material_1::e_13;
		P[1][1][2]=Material::Coefficients::material_1::e_13;
		P[2][2][2]=Material::Coefficients::material_1::e_33;

		return P;
	}

};

template<int dim>
class CM_Microscopic_Fabric : public CM_Base<dim>
{
public:
	CM_Microscopic_Fabric (const Tensor<2,dim> &F,
			const Tensor<1,dim> &E)
: CM_Base<dim> (F,E)
  {}

	virtual ~CM_Microscopic_Fabric () {}

protected:

	// --- Total stored energy ---
	virtual double
	get_Psi (void) const
	{

		return get_W_F(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	// === ELECTRO-MECHANICAL PROBLEM ===
	inline double
	get_W_F (const double lambda,
			const double mu) const
	{
		const double &I1 = this->values.I1;
		const double &J = this->values.J;
		return (mu/2)*(I1-3)-mu*std::log(J) + lambda/2*(std::log(J)*std::log(J));
	}

	// --- Mechanical contributions ---
	virtual SymmetricTensor<2,dim>
	get_dPsi_dC () const
	{
		return get_dW_F_dC(Coefficients::material_1::lambda,
				Coefficients::material_1::mu); // Pure mechanical response
	}

	inline SymmetricTensor<2,dim>
	get_dW_F_dC (const double lambda,
			const double mu) const
			{
		const double &J = this->values.J;
		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;

		return mu*(unit_symmetric_tensor<dim>()-C_inv) + lambda*std::log(J)*C_inv;
			}

	virtual FullMatrix<double>
	get_d2Psi_dC_dC () const
	{
		FullMatrix<double> C(2*dim,2*dim);

		C [0][0] = C [1][1] =C [2][2] = Material::Coefficients::material_1::lambda+2*Material::Coefficients::material_2::mu;
		C [0][1] = C [0][2] =C [1][0] = C [1][2] = C [2][0] =C [2][1] = Material::Coefficients::material_2::lambda;
		C [3][3] = C [4][4] =C [5][5] = (Material::Coefficients::material_2::mu);

		return C;
	}

	//	inline SymmetricTensor<4,dim>
	//	get_d2W_F_dC_dC (const double lambda,
	//			const double mu) const
	//			{
	//		const double &J = this->values.J;
	//		const SymmetricTensor<2,dim> &C_inv = this->values.C_inv;
	//		static const SymmetricTensor<4,dim> C_invxC_inv = outer_product(C_inv,
	//				C_inv);
	//		const SymmetricTensor<4,dim> dC_inv_dC = this->values.dC_inv_dC();
	//
	//		return (-2*mu+2*lambda*std::log(J))*dC_inv_dC + lambda*C_invxC_inv;
	//			}

	// --- Electric component ---
	virtual Tensor<1,dim>
	get_dPsi_dE () const
	{
		return Tensor<1,dim> ();
	}

	virtual Tensor<2,dim>
	get_d2Psi_dE_dE () const
	{
		Tensor<2,dim> DD;

		DD[0][0]= Material::Coefficients::material_2::epsilon_x;
		DD[1][1]= Material::Coefficients::material_2::epsilon_x;
		DD[2][2]= Material::Coefficients::material_2::epsilon_z;

		return DD;
	}

	// --- Electro-mechanical coupling ---
	virtual Tensor<3,dim>
	get_d2Psi_dE_dC () const
	{
		Tensor<3,dim> P;

		P[0][0][2]=Material::Coefficients::material_2::e_13;
		P[1][1][2]=Material::Coefficients::material_2::e_13;
		P[2][2][2]=Material::Coefficients::material_2::e_33;

		return P;
	}

};

}



template<int dim>
class CoupledProblem
{
	//      typedef Material::CM_St_Venant_Kirchhoff<dim> Continuum_Point;
	//      typedef Material::CM_Decoupled_NeoHookean<dim> Continuum_Point;
	typedef Material::CM_Microscopic_Sheet<dim> Continuum_Point_Sheet;
	typedef Material::CM_Microscopic_Fabric<dim> Continuum_Point_Fabric;
	typedef Material::CM_Macroscopic<dim> Continuum_Point_Macroscopic;

public:
	CoupledProblem ();
	~CoupledProblem ();
	void
	run ();

private:
	void
	run_microscopic (const Tensor<2,dim> F);
	void
	update_microscopic_data ();
	void
	make_grid ();
	void
	make_grid_microscopic ();
	void
	refine_grid ();
	void
	setup_system ();
	void
	make_constraints (const unsigned int newton_iteration, const unsigned int timestep);
	void
	make_microscopic_constraints (const unsigned int newton_iteration, const unsigned int timestep);
	void
	assemble_system_mech (const unsigned int ts);
	void
	solve_mech (LA::MPI::BlockVector & solution_update);
	void
	solve_timestep (const double time, const int ts);
	void
	solve_microscopic_timestep (const double time, const int ts);
	void
	output_results (const unsigned int timestep) const;

	const unsigned int n_blocks;
	const unsigned int first_u_component; // Displacement
	const unsigned int V_component; // Voltage / Potential difference
	const unsigned int n_components;

	enum
	{
		uV_block = 0
	};

	enum
	{
		u_dof = 0,
		V_dof = 1
	};

	const FEValuesExtractors::Vector displacement;
	const FEValuesExtractors::Scalar x_displacement;
	const FEValuesExtractors::Scalar y_displacement;
	const FEValuesExtractors::Scalar z_displacement;
	const FEValuesExtractors::Scalar voltage;

	MPI_Comm           mpi_communicator;
	const unsigned int n_mpi_processes;
	const unsigned int this_mpi_process;
	mutable ConditionalOStream pcout;
	mutable TimerOutput computing_timer;

	parallel::distributed::Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler;

	IndexSet locally_owned_dofs;
	IndexSet locally_relevant_dofs;
	std::vector<IndexSet> locally_owned_partitioning;
	std::vector<IndexSet> locally_relevant_partitioning;

	const unsigned int poly_order;
	FESystem<dim> fe_cell;
	FESystem<dim> fe_face;

	QGauss<dim> qf_cell;
	QGauss<dim-1> qf_face;

	ConstraintMatrix hanging_node_constraints;
	ConstraintMatrix dirichlet_constraints;
	ConstraintMatrix all_constraints;

	LA::MPI::BlockSparseMatrix system_matrix;
	LA::MPI::BlockVector       system_rhs;
	LA::MPI::BlockVector       solution;

	LA::MPI::BlockVector locally_relevant_solution;
	LA::MPI::BlockVector locally_relevant_solution_t1;

};

template<int dim>
CoupledProblem<dim>::CoupledProblem ()
:
n_blocks (1),
first_u_component (0), // Displacement
V_component (first_u_component + dim), // Voltage / Potential difference
n_components (V_component+1),

displacement(first_u_component),
x_displacement(first_u_component),
y_displacement(first_u_component+1),
z_displacement(dim==3 ? first_u_component+2 : first_u_component+1),
voltage(V_component),

mpi_communicator (MPI_COMM_WORLD),
n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
pcout(std::cout, this_mpi_process == 0),
computing_timer(mpi_communicator,
		pcout,
		TimerOutput::summary,
		TimerOutput::wall_times),

		triangulation (mpi_communicator,
				typename Triangulation<dim>::MeshSmoothing
				(Triangulation<dim>::smoothing_on_refinement |
						Triangulation<dim>::smoothing_on_coarsening)),
						dof_handler(triangulation),

						poly_order (Parameters::poly_order),
						fe_cell(FE_Q<dim> (poly_order), dim, // Displacement
								FE_Q<dim> (poly_order), 1), // Voltage
								fe_face(FE_Q<dim> (poly_order), dim, // Displacement
										FE_Q<dim> (poly_order), 1), // Voltage

										qf_cell(poly_order+1),
										qf_face(poly_order+1)
										{
										}

template<int dim>
CoupledProblem<dim>::~CoupledProblem ()
{
	dof_handler.clear();
}

template<int dim>
void
CoupledProblem<dim>::make_grid () //Generate thick walled cylinder
{
	TimerOutput::Scope timer_scope (computing_timer, "Make grid");


	GridIn<dim> grid_in;
	grid_in.attach_triangulation (triangulation);
	std::ifstream input_file("Cantilever.inp");
	//paofm
	grid_in.read_abaqus (input_file);

	triangulation.refine_global (Parameters::n_global_refinements);

	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[1] == 1)
				{ // Faces at cylinder bottom
					cell->face(f)->set_boundary_id(Parameters::boundary_id_back);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[1] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_front);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_left);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 100)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_right);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[2] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_bottom);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[2] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_top);
					cell->set_all_manifold_ids(0);
				}


			}
		}
	}

}

template<int dim>
void
CoupledProblem<dim>::make_grid_microscopic ()
{
	TimerOutput::Scope timer_scope (computing_timer, "Make grid");


	GridIn<dim> grid_in;
	grid_in.attach_triangulation (triangulation);
	std::ifstream input_file("Cube_mm.inp");

	grid_in.read_abaqus (input_file);

	typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
	for (; cell != endc; ++cell)
	{
		for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
		{
			if (cell->face(f)->at_boundary())
			{
				const Point<dim> face_center = cell->face(f)->center();
				if (face_center[2] == 0)
				{ // Faces at cylinder bottom
					cell->face(f)->set_boundary_id(Parameters::boundary_id_bottom);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[2] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_top);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[1] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_left);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[1] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_right);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 0)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_front);
					cell->set_all_manifold_ids(0);
				}
				else if (face_center[0] == 1)
				{ // Faces at cylinder top
					cell->face(f)->set_boundary_id(Parameters::boundary_id_back);
					cell->set_all_manifold_ids(0);
				}


			}
		}
	}

}

template<int dim>
void
CoupledProblem<dim>::update_microscopic_data ()
{
	FEValues<dim> fe_values(fe_cell,
			qf_cell,
			update_values |
			update_gradients);

	const unsigned int n_q_points_cell = qf_cell.size();
	std::vector<std::vector<Tensor<1,dim> > > displacement_increment_grads (qf_cell.size(), std::vector<Tensor<1,dim> >(dim));
	for (typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active();
			cell != dof_handler.end(); ++cell)
		if (cell->subdomain_id() == this_mpi_process)
		{
			Microscopic_point<dim> *Microscopic_point_data
			= reinterpret_cast<Microscopic_point<dim> *>(cell->user_pointer());
			Assert (Microscopic_point_data >=
					&Microscopic_point.front(),
					ExcInternalError());
			Assert (Microscopic_point_data <
					&Microscopic_point.back(),
					ExcInternalError());
			fe_values.reinit (cell);

			std::vector< Tensor<2,dim> > Grad_u(n_q_points_cell); // Material gradient of displacement
			std::vector< Tensor<1,dim> > Grad_V(n_q_points_cell); // Material gradient of voltage

			for (unsigned int q_point=0; q_point<n_q_points_cell; ++q_point)
			{

				fe_values[displacement].get_function_gradients(locally_relevant_solution, Grad_u);
				fe_values[voltage].get_function_gradients(locally_relevant_solution, Grad_V);


				//Deformation gradient at quadrature point
				const Tensor<2,dim> F_q_point = (static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u[q_point]);

				const Tensor<2,dim> A_1_old= local_quadrature_points_history[q_point].A_1;
				const Tensor<2,dim> A_2_old= local_quadrature_points_history[q_point].A_2;
				const Tensor<2,dim> A_3_old= local_quadrature_points_history[q_point].A_3;
				const Tensor<2,dim> A_4_old= local_quadrature_points_history[q_point].A_4;

				const Continuum_Point_Sheet cp (F_q_point,
						-Grad_V[q_point],
						-Grad_T[q_point],
						theta[q_point],
						A_1_old,
						A_2_old,
						A_3_old,
						A_4_old);


				const double tau_1 =  5.385*0.35;// in s^-1 from Ask
				const double tau_2 = 1.021*0.35;// in s^-1 from Ask
				const double tau_3 = 1.021*0.5;
				const double tau_4 = 1.769;


				const Tensor<2,dim> A_1_new= cp.get_A(tau_1, A_1_old);
				const Tensor<2,dim> A_2_new= cp.get_A(tau_2, A_2_old);
				const Tensor<2,dim> A_3_new= cp.get_A(tau_3, A_3_old);
				const Tensor<2,dim> A_4_new= cp.get_A(tau_4, A_4_old);

				local_quadrature_points_history[q_point].update_incremental(A_1_new,A_2_new,A_3_new,A_4_new);

			}
		}
}

template<int dim>
void
CoupledProblem<dim>::setup_system ()
{
	TimerOutput::Scope timer_scope (computing_timer, "System setup");
	pcout << "Setting up the electro-mechanical system..." << std::endl;

	dof_handler.distribute_dofs(fe_cell);

	std::vector<types::global_dof_index>  block_component(n_components, uV_block); // Displacement
	block_component[V_component] = uV_block; // Voltage

	DoFRenumbering::Cuthill_McKee(dof_handler);
	DoFRenumbering::component_wise(dof_handler, block_component);

	std::vector<types::global_dof_index> dofs_per_block(n_blocks);
	DoFTools::count_dofs_per_block(dof_handler, dofs_per_block, block_component);
	const types::global_dof_index &n_u_V = dofs_per_block[0];

	pcout
	<< "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl
	<< "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl
	<< "Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< " (" << n_u_V << ')'
	<< std::endl;

	locally_owned_partitioning.clear();
	locally_owned_dofs = dof_handler.locally_owned_dofs ();
	locally_owned_partitioning.push_back(locally_owned_dofs.get_view(0, n_u_V));

	DoFTools::extract_locally_relevant_dofs (dof_handler,
			locally_relevant_dofs);
	locally_relevant_partitioning.clear();
	locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(0, n_u_V));

	hanging_node_constraints.clear();
	hanging_node_constraints.reinit (locally_relevant_dofs);
	DoFTools::make_hanging_node_constraints(dof_handler,
			hanging_node_constraints);
	hanging_node_constraints.close();


	TrilinosWrappers::BlockSparsityPattern sp (locally_owned_partitioning,
			locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator);

	DoFTools::make_sparsity_pattern (dof_handler,
			sp,
			hanging_node_constraints,
			false,
			this_mpi_process);
	sp.compress();
	system_matrix.reinit (sp);


	system_rhs.reinit (locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator,
			true);
	solution.reinit (locally_owned_partitioning,
			locally_relevant_partitioning,
			mpi_communicator,
			true);
	locally_relevant_solution.reinit (locally_relevant_partitioning,
			mpi_communicator);


}

template<int dim>
void
CoupledProblem<dim>::make_constraints (const unsigned int newton_iteration, const unsigned int timestep)
{
	TimerOutput::Scope timer_scope (computing_timer, "Make constraints");

	if (newton_iteration == 0)
	{
		dirichlet_constraints.clear();
		dirichlet_constraints.reinit (locally_relevant_dofs);

		pcout << "  CST M" << std::flush;
		{
			const double displacement_per_ts = Parameters::displacement/(static_cast<double>(Parameters::n_timesteps));
			const double potential_difference_per_ts = Parameters::potential_difference/(static_cast<double>(Parameters::n_timesteps));

			//			// Bottom face
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_bottom,
			//					Functions::ZeroFunction<dim>(n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(z_displacement));

			// Back face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_front,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(y_displacement));

			// Back face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_back,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(y_displacement));

			// Left face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_left,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(x_displacement)|fe_cell.component_mask(y_displacement)|fe_cell.component_mask(z_displacement));


			//			// Prescribed voltage at lower surface
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_top,
			//					//ZeroFunction<dim>(n_components),
			//					Functions::ConstantFunction<dim>(+potential_difference_per_ts/2,n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(voltage));
			//
			// Prescribed voltage at lower surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					//ZeroFunction<dim>(n_components),
					Functions::ConstantFunction<dim>(+potential_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));

			// Prescribed voltage at upper surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					Functions::ZeroFunction<dim>(n_components),
					//Functions::ConstantFunction<dim>(-potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));


			//			// Radial displacement on internal surface
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_right,
			//					//ZeroFunction<dim>(n_components),
			//					Functions::ConstantFunction<dim>(displacement_per_ts,n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(z_displacement));


		}



		dirichlet_constraints.close();
	}
	else
	{
		pcout << "   CST ZERO   " << std::flush;
		// Remove inhomogenaities
		for (types::global_dof_index d=0; d<dof_handler.n_dofs(); ++d)
			if (dirichlet_constraints.can_store_line(d) == true)
				if (dirichlet_constraints.is_constrained(d) == true)
					if (dirichlet_constraints.is_inhomogeneously_constrained(d) == true)
						dirichlet_constraints.set_inhomogeneity(d,0.0);
	}

	// Combine constraint matrices
	all_constraints.clear();
	all_constraints.reinit (locally_relevant_dofs);
	all_constraints.merge(hanging_node_constraints);
	all_constraints.merge(dirichlet_constraints, ConstraintMatrix::left_object_wins);
	all_constraints.close();
}

template<int dim>
void
CoupledProblem<dim>::make_microscopic_constraints (const unsigned int newton_iteration, const unsigned int timestep)
{
	TimerOutput::Scope timer_scope (computing_timer, "Make constraints");

	if (newton_iteration == 0)
	{
		dirichlet_constraints.clear();
		dirichlet_constraints.reinit (locally_relevant_dofs);

		pcout << "  CST M" << std::flush;
		{
			const double displacement_per_ts = Parameters::displacement/(static_cast<double>(Parameters::n_timesteps));
			const double potential_difference_per_ts = Parameters::potential_difference/(static_cast<double>(Parameters::n_timesteps));

			//			// Bottom face
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_bottom,
			//					Functions::ZeroFunction<dim>(n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(z_displacement));

			// Back face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_front,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(y_displacement));

			// Back face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_back,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(y_displacement));

			// Left face
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_left,
					Functions::ZeroFunction<dim>(n_components),
					dirichlet_constraints,
					fe_cell.component_mask(x_displacement)|fe_cell.component_mask(y_displacement)|fe_cell.component_mask(z_displacement));


			//			// Prescribed voltage at lower surface
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_top,
			//					//ZeroFunction<dim>(n_components),
			//					Functions::ConstantFunction<dim>(+potential_difference_per_ts/2,n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(voltage));
			//
			// Prescribed voltage at lower surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_top,
					//ZeroFunction<dim>(n_components),
					Functions::ConstantFunction<dim>(+potential_difference_per_ts,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));

			// Prescribed voltage at upper surface
			VectorTools::interpolate_boundary_values(dof_handler,
					Parameters::boundary_id_bottom,
					Functions::ZeroFunction<dim>(n_components),
					//Functions::ConstantFunction<dim>(-potential_difference_per_ts/2,n_components),
					dirichlet_constraints,
					fe_cell.component_mask(voltage));


			//			// Radial displacement on internal surface
			//			VectorTools::interpolate_boundary_values(dof_handler,
			//					Parameters::boundary_id_right,
			//					//ZeroFunction<dim>(n_components),
			//					Functions::ConstantFunction<dim>(displacement_per_ts,n_components),
			//					dirichlet_constraints,
			//					fe_cell.component_mask(z_displacement));


		}



		dirichlet_constraints.close();
	}
	else
	{
		pcout << "   CST ZERO   " << std::flush;
		// Remove inhomogenaities
		for (types::global_dof_index d=0; d<dof_handler.n_dofs(); ++d)
			if (dirichlet_constraints.can_store_line(d) == true)
				if (dirichlet_constraints.is_constrained(d) == true)
					if (dirichlet_constraints.is_inhomogeneously_constrained(d) == true)
						dirichlet_constraints.set_inhomogeneity(d,0.0);
	}

	// Combine constraint matrices
	all_constraints.clear();
	all_constraints.reinit (locally_relevant_dofs);
	all_constraints.merge(hanging_node_constraints);
	all_constraints.merge(dirichlet_constraints, ConstraintMatrix::left_object_wins);
	all_constraints.close();
}


template<int dim>
void
CoupledProblem<dim>::assemble_system_mech (const unsigned int ts)
{
	TimerOutput::Scope timer_scope (computing_timer, "Assembly: Mechanical");
	pcout << "  ASM M" << std::flush;

	FEValues<dim> fe_values(fe_cell,
			qf_cell,
			update_values |
			update_gradients |
			update_quadrature_points |
			update_JxW_values);

	FEFaceValues<dim> fe_face_values(fe_face,
			qf_face,
			update_values |
			update_quadrature_points |
			update_normal_vectors |
			update_JxW_values);

	const unsigned int dofs_per_cell = fe_cell.dofs_per_cell;
	const unsigned int n_q_points = qf_cell.size();
	const unsigned int n_face_q_points = qf_face.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double> cell_rhs(dofs_per_cell);

	const double mu=Material::Coefficients::material_1::mu;
	const double lambda=Material::Coefficients::material_1::lambda;

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	// Values at integration points
	std::vector< Tensor<2,dim> > Grad_u(n_q_points); // Material gradient of displacement
	std::vector< Tensor<1,dim> > Grad_V(n_q_points); // Material gradient of voltage
	std::vector< Tensor<1,dim> > Grad_T(n_q_points); // Material gradient of temperature
	std::vector<double> theta(n_q_points); // Temperature
	unsigned int node_nr;
	typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
	for (; cell != endc; ++cell)
	{
		if (cell->is_locally_owned() == false) continue;

		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);

		fe_values[displacement].get_function_gradients(locally_relevant_solution, Grad_u);
		fe_values[voltage].get_function_gradients(locally_relevant_solution, Grad_V);



		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		{
			const double &JxW = fe_values.JxW(q_point);

			const Tensor<2,dim> F_q_point = static_cast< Tensor<2,dim> >(unit_symmetric_tensor<dim>()) + Grad_u[q_point];
			const Tensor<2,dim> F_q_transpose = transpose(F_q_point);
			const double J=(determinant(F_q_point));
			const SymmetricTensor<2,dim> C_ = (symmetrize(transpose(F_q_point)*F_q_point));
			const SymmetricTensor<2,dim>  C_inv = (symmetrize(invert(static_cast<Tensor<2,dim> >(C_))));
			const SymmetricTensor<4,dim> C_invxC_inv = outer_product(C_inv,C_inv);



			SymmetricTensor<4,dim> dC_inv_dC;

			for (unsigned int A=0; A<dim; ++A)
				for (unsigned int B=A; B<dim; ++B)
					for (unsigned int C=0; C<dim; ++C)
						for (unsigned int D=C; D<dim; ++D)
							dC_inv_dC[A][B][C][D] = -0.5*(C_inv[A][C]*C_inv[B][D]+ C_inv[A][D]*C_inv[B][C]);



			// Material tangents

			FullMatrix<double> C(2*dim,2*dim);
			Tensor<3,dim> P;
			Tensor<2,dim> DD = unit_symmetric_tensor<dim>();

			if (cell->material_id()==1000)
			{

				const Microscopic_point<dim> *local_microscopic_point_data
				= reinterpret_cast<Microscopic_point<dim>*>(cell->user_pointer());

				run_microscopic(F_q_point);

				C= local_microscopic_point_data[q_point].C;
				P= local_microscopic_point_data[q_point].P;
				DD= local_microscopic_point_data[q_point].DD;

			}
			else if (cell->material_id()==1)
			{
				const Continuum_Point_Sheet cp (F_q_point,
						-Grad_V[q_point]);

				C = cp.get_C();
				P = cp.get_P();
				DD = cp.get_DD();

			}
			else if (cell->material_id()==2)
			{
				const Continuum_Point_Fabric cp (F_q_point,
						-Grad_V[q_point]);

				C = cp.get_C();
				P = cp.get_P();
				DD = cp.get_DD();
			}


			//FullMatrix<double> C_kelvin = Physics::Notation::Kelvin::to_matrix(C);

			// Variation / Linearisation of Green-Lagrange strain tensor
			std::vector< SymmetricTensor<2,dim> > dE (dofs_per_cell);

			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				const unsigned int k_group = fe_cell.system_to_base_index(k).first.first;
				if (k_group == u_dof)
					dE[k] = symmetrize(F_q_transpose*fe_values[displacement].gradient(k, q_point));

			}

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				const unsigned int component_i = fe_cell.system_to_component_index(i).first;
				const unsigned int i_group     = fe_cell.system_to_base_index(i).first.first;

				const SymmetricTensor<2,dim> &Grad_Nx_i_u      = symmetrize(fe_values[displacement].gradient(i, q_point));
				const Vector<double> &Grad_Nx_i_u_kelvin = Physics::Notation::Kelvin::to_vector(Grad_Nx_i_u);

				const Tensor<1,dim> &Grad_Nx_i_V      = fe_values[voltage].gradient(i, q_point);


				for (unsigned int j = 0; j < dofs_per_cell; ++j)
				{
					const unsigned int component_j = fe_cell.system_to_component_index(j).first;
					const unsigned int j_group     = fe_cell.system_to_base_index(j).first.first;

					const SymmetricTensor<2,dim> &Grad_Nx_j_u      = symmetrize(fe_values[displacement].gradient(j, q_point));
					const Vector<double> &Grad_Nx_j_u_kelvin = Physics::Notation::Kelvin::to_vector(Grad_Nx_j_u);

					const Tensor<1,dim> &Grad_Nx_j_V      = fe_values[voltage].gradient(j, q_point);



					if ((i_group == u_dof) && (j_group == u_dof))
					{
						Vector<double> help= Physics::Notation::Kelvin::to_vector(Grad_Nx_j_u);
						C.vmult(help,Grad_Nx_j_u_kelvin);
						cell_matrix(i, j) += (Grad_Nx_i_u_kelvin*help) * JxW;
					}
					else if ((i_group == u_dof) && (j_group == V_dof))
					{
						// u-V terms
						cell_matrix(i, j) += contract3(Tensor<2,dim>(dE[i]), P, Grad_Nx_j_V) * JxW;
					}
					else if ((i_group == V_dof) && (j_group == u_dof))
					{
						// V-u terms
						cell_matrix(i, j) += contract3(Tensor<2,dim>(dE[j]), P, Grad_Nx_i_V ) * JxW;
					}
					else if ((i_group == V_dof) && (j_group == V_dof))
					{
						// V-V terms
						cell_matrix(i, j) -= (Grad_Nx_i_V * DD * Grad_Nx_j_V) * JxW;
					}

				}

				// RHS = -Residual
				if (i_group == u_dof)
				{
					// u-terms
					cell_rhs(i) += fe_values[displacement].value(i,q_point)[component_i] *
							0 * JxW;
				}
				else if (i_group == V_dof)
				{
					// V-terms
					cell_rhs(i) += fe_values[voltage].value(i,q_point) *
							0 * JxW;
				}
			}
		}

		//		for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell;f++)
		//			if (cell->face(f)->at_boundary())
		//			{
		//
		//				if (cell->face(f)->boundary_id() == Parameters::boundary_id_top || cell->face(f)->boundary_id() == Parameters::boundary_id_force)
		//				{
		//					fe_face_values.reinit(cell, f);
		//
		//					for (unsigned int q_point=0;q_point<n_face_q_points; ++q_point)
		//					{
		//						const Tensor<1, dim> N = fe_face_values.normal_vector(q_point);
		//						const double &JxW = fe_face_values.JxW(q_point);
		//						const Tensor<1, dim> traction = -0.0001 * N;
		//
		//						for (unsigned int i=0; i<dofs_per_cell; ++i)
		//						{
		//							const unsigned int component_i = fe_cell.system_to_component_index(i).first;
		//							const unsigned int i_group     = fe_cell.system_to_base_index(i).first.first;
		//
		//							{
		//								if (i_group == u_dof)
		//								{
		//									// u-terms
		//									cell_rhs(i) += fe_face_values[displacement].value(i,q_point)[component_i] *
		//											traction[component_i] * JxW;
		//								}
		//								else if (i_group == V_dof)
		//								{
		//									// V-terms
		//									cell_rhs(i) += fe_face_values[voltage].value(i,q_point) *
		//											0 * JxW;
		//								}
		//							}
		//						}
		//					}
		//				}
		//			}

		cell->get_dof_indices(local_dof_indices);
		all_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
				local_dof_indices,
				system_matrix, system_rhs);

	}

	system_matrix.compress (VectorOperation::add);
	system_rhs.compress (VectorOperation::add);
}

template<int dim>
void
CoupledProblem<dim>::solve_mech (LA::MPI::BlockVector &locally_relevant_solution_update)
{
	TimerOutput::Scope timer_scope (computing_timer, "Solve: Mechanical");
	pcout << "  SLV M" << std::flush;

	LA::MPI::BlockVector
	completely_distributed_solution_update (locally_owned_partitioning,
			mpi_communicator);

	if (Parameters::solver_type_EM == "Iterative")
	{
		SolverControl solver_control(system_matrix.block(uV_block,uV_block).m(),
				Parameters::tol_rel_EM*system_rhs.block(uV_block).l2_norm());
		//        LA::SolverCG solver(solver_control, mpi_communicator);
		LA::SolverCG solver(solver_control);

		LA::MPI::PreconditionAMG preconditioner;
		LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_TRILINOS_LA
		/* Trilinos defaults are good */
#else
		data.symmetric_operator = true;
#endif

		solver.solve(system_matrix.block(uV_block, uV_block),
				completely_distributed_solution_update.block(uV_block),
				system_rhs.block(uV_block),
				preconditioner);
	}
	else
	{ // Direct solver
#ifdef USE_TRILINOS_LA
		SolverControl solver_control(1, 1e-12);
		TrilinosWrappers::SolverDirect solver (solver_control);

		solver.solve(system_matrix.block(uV_block, uV_block),
				completely_distributed_solution_update.block(uV_block),
				system_rhs.block(uV_block));
#else
		AssertThrow(false, ExcNotImplemented());
#endif
	}

	all_constraints.distribute(completely_distributed_solution_update);
	locally_relevant_solution_update.block(uV_block) = completely_distributed_solution_update.block(uV_block);

}

template<int dim>
void
CoupledProblem<dim>::output_results (const unsigned int timestep) const
{
	TimerOutput::Scope timer_scope (computing_timer, "Post-processing");

	// Write out main data file
	struct Filename
	{
		static std::string get_filename_vtu (unsigned int process,
				unsigned int cycle,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string (process, n_digits)
			<< "."
			<< Utilities::int_to_string(cycle, n_digits)
			<< ".vtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvtu (unsigned int timestep,
				const unsigned int n_digits = 4)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< "."
			<< Utilities::int_to_string(timestep, n_digits)
			<< ".pvtu";
			return filename_vtu.str();
		}

		static std::string get_filename_pvd (void)
		{
			std::ostringstream filename_vtu;
			filename_vtu
			<< "solution-"
			<< (std::to_string(dim) + "d")
			<< ".pvd";
			return filename_vtu.str();
		}
	};

	DataOut<dim> data_out;
	data_out.attach_dof_handler (dof_handler);

	std::vector<std::string> solution_names (n_components, "displacement");
	solution_names[V_component] = "voltage";

	std::vector<std::string> residual_names (solution_names);

	std::vector<std::string> reaction_forces_names (solution_names);

	for (unsigned int i=0; i < solution_names.size(); ++i)
	{
		solution_names[i].insert(0, "soln_");
		residual_names[i].insert(0, "res_");
		reaction_forces_names[i].insert(0, "reactF_");
	}

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(dim,
			DataComponentInterpretation::component_is_part_of_vector);
	data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

	data_out.add_data_vector(locally_relevant_solution, solution_names,
			DataOut<dim>::type_dof_data,
			data_component_interpretation);

	LA::MPI::BlockVector locally_relevant_residual;
	locally_relevant_residual.reinit (locally_relevant_partitioning,
			mpi_communicator);
	locally_relevant_residual = system_rhs;
	locally_relevant_residual *= -1.0;
	data_out.add_data_vector(locally_relevant_residual, residual_names,
			DataOut<dim>::type_dof_data,
			data_component_interpretation);



	Vector<float> subdomain (triangulation.n_active_cells());
	for (unsigned int i=0; i<subdomain.size(); ++i)
		subdomain(i) = triangulation.locally_owned_subdomain();
	data_out.add_data_vector (subdomain, "subdomain");

	LA::MPI::BlockVector reaction_forces_bid_4;
	reaction_forces_bid_4.reinit(locally_relevant_partitioning,
			mpi_communicator);

	{
		FEFaceValues<dim> fe_face_values_ref (fe_cell, qf_face,
				update_values | update_gradients |
				update_JxW_values | update_normal_vectors);

		std::vector<Tensor<2, dim> > solution_grads_u_total (qf_face.size());
		std::vector< Tensor<1,dim> > Grad_V(qf_face.size()); // Material gradient of voltage
		std::vector< Tensor<1,dim> > Grad_T(qf_face.size()); // Material gradient of temperature
		std::vector<double> theta(qf_face.size()); // Temperature
		const unsigned int dofs_per_cell = fe_cell.dofs_per_cell;
		const unsigned int n_q_points_cell = qf_cell.size();
		const unsigned int n_q_points_face = qf_face.size();


		// Note that the reaction forces computed at the face support points
		// now only need to be summed together since they're the nodal
		// representation of the surface forces
		// Since I can't think of a more simple way to RELIABLY determine the
		// component index of a global DoF (in general, there might be some
		// global reordering to that they are not {[x,y,z]_1, [x,y,z]_2} etc),
		// I'm just going to extract the local components through a cell loop
		Tensor<1,dim> force_reaction_bid_4;
		std::set<types::global_dof_index> touched_dofs;
		for (typename DoFHandler<dim>::active_cell_iterator cell =
				dof_handler.begin_active(); cell != dof_handler.end();
				++cell)
		{
			std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
			cell->get_dof_indices(local_dof_indices);

			// Technically not required because of the lack of non-zero entries
			// in the vector, but I suppose this is the most correct approach
			for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
				if (cell->face(face)->at_boundary() == true
						&& cell->face(face)->boundary_id() == Parameters::boundary_id_left) // -Z face
				{
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
					{
						// Don't sum a DoF that we've visited more than once
						if (touched_dofs.find(local_dof_indices[i]) != touched_dofs.end())
							continue;

						const unsigned int i_group =
								fe_cell.system_to_base_index(i).first.first;

						if (i_group == u_dof)
						{
							const unsigned int component_i =
									fe_cell.system_to_component_index(i).first;

							force_reaction_bid_4[component_i] += reaction_forces_bid_4[local_dof_indices[i]];
							touched_dofs.insert(local_dof_indices[i]); // Mark as visited
						}
					}
				}
		}
	}
	data_out.add_data_vector(reaction_forces_bid_4,
			reaction_forces_names,
			DataOut<dim>::type_dof_data,
			data_component_interpretation);

	data_out.build_patches (poly_order);

	const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process,
			timestep);
	std::ofstream output(filename_vtu.c_str());
	data_out.write_vtu(output);

	// Collection of files written in parallel
	// This next set of steps should only be performed
	// by master process
	if (this_mpi_process == 0)
	{
		// List of all files written out at this timestep by all processors
		std::vector<std::string> parallel_filenames_vtu;
		for (unsigned int p=0; p < n_mpi_processes; ++p)
		{
			parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p,
					timestep));
		}

		const std::string filename_pvtu (Filename::get_filename_pvtu(timestep));
		std::ofstream pvtu_master(filename_pvtu.c_str());
		data_out.write_pvtu_record(pvtu_master,
				parallel_filenames_vtu);

		// Time dependent data master file
		static std::vector<std::pair<double,std::string> > time_and_name_history;
		time_and_name_history.push_back (std::make_pair (timestep,
				filename_pvtu));
		const std::string filename_pvd (Filename::get_filename_pvd());
		std::ofstream pvd_output (filename_pvd.c_str());
		DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
	}
}

template<int dim>
void
CoupledProblem<dim>::solve_timestep (const double time, const int ts)
{
	locally_relevant_solution_t1 = locally_relevant_solution;

	for (unsigned int n=0; n < Parameters::max_newton_iterations; ++n)
	{
		pcout << "IT " << n << std::flush;

		LA::MPI::BlockVector locally_relevant_solution_update;
		locally_relevant_solution_update.reinit (locally_relevant_partitioning,
				mpi_communicator);

		make_constraints(n, ts);

		// === ELECTRO-MECHANICAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_mech(ts);
		solve_mech(locally_relevant_solution_update);
		locally_relevant_solution.block(uV_block) += locally_relevant_solution_update.block(uV_block);

	}

}

template<int dim>
void
CoupledProblem<dim>::solve_microscopic_timestep (const double time, const int ts)
{
	locally_relevant_solution_t1 = locally_relevant_solution;

	for (unsigned int n=0; n < Parameters::max_newton_iterations; ++n)
	{
		pcout << "IT " << n << std::flush;

		LA::MPI::BlockVector locally_relevant_solution_update;
		locally_relevant_solution_update.reinit (locally_relevant_partitioning,
				mpi_communicator);

		make_microscopic_constraints(n, ts);

		// === ELECTRO-MECHANICAL PROBLEM ===

		system_matrix = 0;
		system_rhs = 0;
		locally_relevant_solution_update = 0;

		assemble_system_mech(ts);
		solve_mech(locally_relevant_solution_update);
		locally_relevant_solution.block(uV_block) += locally_relevant_solution_update.block(uV_block);

	}

}


template<int dim>
void
CoupledProblem<dim>::refine_grid ()
{
	std::vector<const TrilinosWrappers::MPI::BlockVector *> storage_soln (2);
	storage_soln[0] = &locally_relevant_solution;
	storage_soln[1] = &locally_relevant_solution_t1;

	parallel::distributed::SolutionTransfer<dim,LA::MPI::BlockVector>
	soln_trans(dof_handler);

	{
		TimerOutput::Scope timer_scope (computing_timer, "Grid refinement");

		pcout
		<< "Executing grid refinement..."
		<< std::endl;

		// Estimate solution error
		Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate (dof_handler,
				QGauss<dim-1>(poly_order+2),
				typename FunctionMap<dim>::type(),
				locally_relevant_solution,
				estimated_error_per_cell);

		// Perform grid marking
		parallel::distributed::GridRefinement::
		refine_and_coarsen_fixed_number (triangulation,
				estimated_error_per_cell,
				Parameters::frac_refine,
				Parameters::frac_coarsen);

		// Limit refinement level
		if (triangulation.n_levels() > Parameters::max_grid_level)
			for (typename Triangulation<dim>::active_cell_iterator
					cell = triangulation.begin_active(Parameters::max_grid_level);
					cell != triangulation.end(); ++cell)
				cell->clear_refine_flag ();

		// Prepare solution transfer for refinement
		triangulation.prepare_coarsening_and_refinement();
		soln_trans.prepare_for_coarsening_and_refinement(storage_soln);

		// Perform grid refinement
		triangulation.execute_coarsening_and_refinement ();
	}

	// Reconfigure system with new DoFs
	setup_system();

	{
		TimerOutput::Scope timer_scope (computing_timer, "Grid refinement");

		TrilinosWrappers::MPI::BlockVector distributed_solution (system_rhs);
		TrilinosWrappers::MPI::BlockVector distributed_solution_old (system_rhs);
		std::vector<TrilinosWrappers::MPI::BlockVector *> soln_tmp (2);
		soln_tmp[0] = &(distributed_solution);
		soln_tmp[1] = &(distributed_solution_old);

		// Perform solution transfer
		soln_trans.interpolate (soln_tmp);

		hanging_node_constraints.distribute(distributed_solution);
		hanging_node_constraints.distribute(distributed_solution_old);
		locally_relevant_solution     = distributed_solution;
		locally_relevant_solution_t1 = distributed_solution_old;

		pcout
		<< "Grid refinement done."
		<< std::endl;
	}
}



template<int dim>
void
CoupledProblem<dim>::run_microscopic (const Tensor<2,dim> F)
{
	make_grid_microscopic();
	setup_system();

	double time = Parameters::dt;
	for (unsigned int ts = 1;
			ts<=Parameters::n_timesteps;
			++ts, time += Parameters::dt)
	{
		if (Parameters::perform_AMR == true &&
				(ts % Parameters::n_ts_per_refinement) == 0)
		{
			pcout
			<< std::endl;
			refine_grid();
		}

		pcout
		<< std::endl
		<< std::string(100,'=')
		<< std::endl
		<< "Timestep: " << ts
		<< "\t Time: " << time
		<< std::endl
		<< std::string(100,'=')
		<< std::endl;
		solve_timestep(time, ts);
		update_microscopic_data(Microscopic_data_point);
	}

}


template<int dim>
void
CoupledProblem<dim>::run ()
{
	make_grid();
	setup_system();
	output_results(0);

	double time = Parameters::dt;
	for (unsigned int ts = 1;
			ts<=Parameters::n_timesteps;
			++ts, time += Parameters::dt)
	{
		if (Parameters::perform_AMR == true &&
				(ts % Parameters::n_ts_per_refinement) == 0)
		{
			pcout
			<< std::endl;
			refine_grid();
		}

		pcout
		<< std::endl
		<< std::string(100,'=')
		<< std::endl
		<< "Timestep: " << ts
		<< "\t Time: " << time
		<< std::endl
		<< std::string(100,'=')
		<< std::endl;
		solve_timestep(time, ts);
		output_results(ts);

		// Update solution at previous timestep
		locally_relevant_solution_t1 = locally_relevant_solution;
	}

}
}

int
main (int argc, char *argv[])
{
	try
	{
		using namespace dealii;
		Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		deallog.depth_console (0);

		Coupled_TEE::CoupledProblem<3> coupled_thermo_electro_elastic_problem_3d;
		coupled_thermo_electro_elastic_problem_3d.run();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Exception on processing: "
				<< std::endl
				<< exc.what()
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
		std::cerr << "Unknown exception!"
				<< std::endl
				<< "Aborting!"
				<< std::endl
				<< "----------------------------------------------------"
				<< std::endl;
		return 1;
	}

	return 0;
}
