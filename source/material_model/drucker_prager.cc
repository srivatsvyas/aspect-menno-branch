/*
  Copyright (C) 2015 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/material_model/drucker_prager.h>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {

  template <int dim>
  void
  DruckerPrager<dim>::
  evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
          MaterialModel::MaterialModelOutputs<dim> &out) const
	{
		//set up additional output for the derivatives
	      MaterialModelDerivatives<dim> *derivatives;
	      derivatives = out.template get_additional_output<MaterialModelDerivatives<dim> >();

  	for (unsigned int i=0; i < in.temperature.size(); ++i)
  	        {

  	          // as documented, if the strain rate array is empty, then do not compute the
  	          // viscosities
  	          if (in.strain_rate.size() > 0)
  	            out.viscosities[i]                  = viscosity                     (in.temperature[i], in.pressure[i], in.composition[i], in.strain_rate[i], in.position[i]);

  	          out.densities[i]                      = density                       (in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
  	          out.thermal_expansion_coefficients[i] = thermal_expansion_coefficient (in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
  	          out.specific_heat[i]                  = specific_heat                 (in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
  	          out.thermal_conductivities[i]         = thermal_conductivity          (in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
  	          out.compressibilities[i]              = compressibility               (in.temperature[i], in.pressure[i], in.composition[i], in.position[i]);
  	          //out.entropy_derivative_pressure[i]    = entropy_derivative            (in.temperature[i], in.pressure[i], in.composition[i], in.position[i], NonlinearDependence::pressure);
  	          //out.entropy_derivative_temperature[i] = entropy_derivative            (in.temperature[i], in.pressure[i], in.composition[i], in.position[i], NonlinearDependence::temperature);
  	          //for (unsigned int c=0; c<in.composition[i].size(); ++c)
  	           // out.reaction_terms[i][c]            = reaction_term                 (in.temperature[i], in.pressure[i], in.composition[i], in.position[i], c);

  	        if (derivatives != NULL)
  	        {
      		  // finite difference
      		  const double finite_difference_accuracy = 1e-7;
      		  SymmetricTensor<2,dim> zerozero = SymmetricTensor<2,dim>();
      		  SymmetricTensor<2,dim> onezero = SymmetricTensor<2,dim>();
      		  SymmetricTensor<2,dim> oneone = SymmetricTensor<2,dim>();

      		  zerozero[0][0] = 1;
      		  onezero[1][0]  = 0.5; // because symmetry doubles this entry
      		  oneone[1][1]   = 1;

      		  SymmetricTensor<2,dim> strain_rate_zero_zero = in.strain_rate[i] + std::fabs(in.strain_rate[i][0][0]) * finite_difference_accuracy * zerozero;
      		  SymmetricTensor<2,dim> strain_rate_one_zero = in.strain_rate[i] + std::fabs(in.strain_rate[i][1][0]) * finite_difference_accuracy * onezero;
      		  SymmetricTensor<2,dim> strain_rate_one_one = in.strain_rate[i] + std::fabs(in.strain_rate[i][1][1]) * finite_difference_accuracy * oneone;

      		  double edot_ii_fd;

      		  double eta_zero_zero = viscosity(in.temperature[i], in.pressure[i], in.composition[i], strain_rate_zero_zero, in.position[i]);
      		  double deriv_zero_zero = eta_zero_zero - out.viscosities[i];

      		  if(deriv_zero_zero != 0)
      		  {
      			  if(strain_rate_zero_zero[0][0] != 0)
      			  {
      				  deriv_zero_zero /= std::fabs(strain_rate_zero_zero[0][0]) * finite_difference_accuracy;
      			  }
      			  else
      			  {
      				  deriv_zero_zero = 0;
      			  }

      		  }

      		  double eta_one_zero = viscosity(in.temperature[i], in.pressure[i], in.composition[i], strain_rate_one_zero, in.position[i]);
      		  double deriv_one_zero = eta_one_zero - out.viscosities[i];

      		  if(deriv_one_zero != 0)
      		  {
      			  if(strain_rate_one_zero[1][0] != 0)
      			  {
      				  deriv_one_zero /= std::fabs(strain_rate_one_zero[1][0]) * finite_difference_accuracy;
      			  }
      			  else
      			  {
      				  deriv_one_zero = 0;
      			  }
      		  }

      		  double eta_one_one = viscosity(in.temperature[i], in.pressure[i], in.composition[i], strain_rate_one_one, in.position[i]);
      		  double deriv_one_one = eta_one_one - out.viscosities[i];

      		  if(eta_one_one != 0)
      		  {
      			  if(strain_rate_one_one[1][1] != 0)
      			  {
      				  deriv_one_one /= std::fabs(strain_rate_one_one[1][1]) * finite_difference_accuracy;
      			  }
      			  else
      			  {
      				  deriv_one_one = 0;
      			  }
      		  }
//std::cout << deriv_zero_zero << ", " << deriv_one_zero << ", " << deriv_one_one << std::endl;
      		double sr = ( (this->get_timestep_number() == 0 && in.strain_rate[i].norm() <= std::numeric_limits<double>::min())
                    ?
                    reference_strain_rate * reference_strain_rate
                    :
					0.5 * in.strain_rate[i]*in.strain_rate[i]);
      		if(out.viscosities[i] <= minimum_viscosity || out.viscosities[i] >= maximum_viscosity)
      			derivatives->dviscosities_dstrain_rate[i] = 0;
      		else
      		derivatives->dviscosities_dstrain_rate[i] = -1/(std::sqrt(sr)*sr)*in.strain_rate[i]*(cohesion*std::cos(phi)+in.pressure[i]*std::sin(phi));
      		//derivatives->dviscosities_dstrain_rate[i][0][0] = deriv_zero_zero;
      		//derivatives->dviscosities_dstrain_rate[i][1][0] = deriv_one_zero;
      		//derivatives->dviscosities_dstrain_rate[i][1][1] = deriv_one_one;

      		/**
      		 * Now compute the derivative of the viscoisty to the pressure
      		 */
      		double pressure_difference = in.pressure[i] + (std::fabs(in.pressure[i]) * 1e-7);

      		double pressure_difference_eta = viscosity(in.temperature[i], pressure_difference, in.composition[i], in.strain_rate[i], in.position[i]);
      		double deriv_pressure = pressure_difference_eta - out.viscosities[i];
      		if(in.pressure[i] > 0)
      		std::cout << "pd = " << pressure_difference << " = " << in.pressure[i] << " + " << std::fabs(in.pressure[i]) << " * " << finite_difference_accuracy << ", pda = " << pressure_difference_eta << ", dp = " << deriv_pressure << std::endl;
      		if(pressure_difference_eta != 0)
      		{
      			if(in.pressure[i] != 0)
      			{
      				deriv_pressure /= std::fabs(in.pressure[i]) * 1e-7;
      			}
      			else
      			{
      				deriv_pressure = 0;
      			}
      		}
      		derivatives->dviscosities_dpressure[i] = 0;//deriv_pressure;

  	        }
  	        }
	}

    template <int dim>
    double
    DruckerPrager<dim>::
    viscosity (const double /*temperature*/,
               const double pressure,
               const std::vector<double> &/*composition*/,
               const SymmetricTensor<2,dim> &strain_rate,
               const Point<dim> &/*position*/) const
    {
      // For the very first time this function is called
      // (the first iteration of the first timestep), this function is called
      // with a zero input strain rate. We provide a representative reference
      // strain rate for this case, which avoids division by zero and produces
      // a representative first guess of the viscosities.
      // In later iterations and timesteps we calculate the second moment
      // invariant of the deviatoric strain rate tensor.
      // This is equal to the negative of the second principle
      // invariant calculated with the function second_invariant.
      const double strain_rate_dev_inv2 = ( (this->get_timestep_number() == 0 && strain_rate.norm() <= std::numeric_limits<double>::min())
                                            ?
                                            reference_strain_rate * reference_strain_rate
                                            :
                                            0.5 * strain_rate*strain_rate);

      // In later timesteps, we still need to care about cases of very small
      // strain rates. We expect the viscosity to approach the maximum_viscosity
      // in these cases. This check prevents a division-by-zero.
      if (std::sqrt(strain_rate_dev_inv2) <= std::numeric_limits<double>::min())
        return maximum_viscosity;

      // To avoid negative yield strengths and eventually viscosities,
      // we make sure the pressure is not negative
      const double strength = ( (dim==3)
                                ?
                                ( 6.0 * cohesion * std::cos(phi) + 2.0 * std::max(pressure,0.0) * std::sin(phi) )
                                / ( std::sqrt(3.0) * ( 3.0 + std::sin(phi) ) )
                                :
                                cohesion * std::cos(phi) + std::max(pressure,0.0) * std::sin(phi) );

      // Rescale the viscosity back onto the yield surface
      const double viscosity = strength / ( 2.0 * std::sqrt(strain_rate_dev_inv2) );

      // Cut off the viscosity between a minimum and maximum value to avoid
      // a numerically unfavourable large viscosity range.
      const double effective_viscosity = std::min(std::max(viscosity,minimum_viscosity),maximum_viscosity);//1.0 / ( ( 1.0 / ( viscosity + minimum_viscosity ) ) + ( 1.0 / maximum_viscosity ) );

      return effective_viscosity;

    }


    template <int dim>
    double
    DruckerPrager<dim>::
    reference_viscosity () const
    {
      return reference_eta;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    reference_density () const
    {
      return reference_rho;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    reference_thermal_expansion_coefficient () const
    {
      return thermal_alpha;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    specific_heat (const double,
                   const double,
                   const std::vector<double> &, /*composition*/
                   const Point<dim> &) const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    reference_cp () const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    thermal_conductivity (const double,
                          const double,
                          const std::vector<double> &, /*composition*/
                          const Point<dim> &) const
    {
      return thermal_k;
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    reference_thermal_diffusivity () const
    {
      return thermal_k/(reference_rho*reference_specific_heat);
    }

    template <int dim>
    double
    DruckerPrager<dim>::
    density (const double temperature,
             const double,
             const std::vector<double> &, /*composition*/
             const Point<dim> &) const
    {
      return reference_rho * (1 - thermal_alpha * (temperature - reference_T));
    }


    template <int dim>
    double
    DruckerPrager<dim>::
    thermal_expansion_coefficient (const double,
                                   const double,
                                   const std::vector<double> &, /*composition*/
                                   const Point<dim> &) const
    {
      return thermal_alpha;
    }


    template <int dim>
    double
    DruckerPrager<dim>::
    compressibility (const double,
                     const double,
                     const std::vector<double> &, /*composition*/
                     const Point<dim> &) const
    {
      return 0.0;
    }



    template <int dim>
    bool
    DruckerPrager<dim>::
    is_compressible () const
    {
      return false;
    }



    template <int dim>
    void
    DruckerPrager<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Drucker Prager");
        {
          prm.declare_entry ("Reference density", "3300",
                             Patterns::Double (0),
                             "The reference density $\\rho_0$. Units: $kg/m^3$.");
          prm.declare_entry ("Reference temperature", "293",
                             Patterns::Double (0),
                             "The reference temperature $T_0$. The reference temperature is used "
                             "in the density calculation. Units: $K$.");
          prm.declare_entry ("Reference viscosity", "1e22",
                             Patterns::Double (0),
                             "The value of the reference viscosity $\\eta_0$. Units: $kg/m/s$.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0),
                             "The value of the thermal conductivity $k$. "
                             "Units: $W/m/K$.");
          prm.declare_entry ("Reference specific heat", "1250",
                             Patterns::Double (0),
                             "The value of the specific heat $cp$. "
                             "Units: $J/kg/K$.");
          prm.declare_entry ("Thermal expansion coefficient", "2e-5",
                             Patterns::Double (0),
                             "The value of the thermal expansion coefficient $\\beta$. "
                             "Units: $1/K$.");
          prm.enter_subsection ("Viscosity");
          {
            prm.declare_entry ("Minimum viscosity", "1e19",
                               Patterns::Double (0),
                               "The value of the minimum viscosity cutoff $\\eta_min$. Units: $Pa\\;s$.");
            prm.declare_entry ("Maximum viscosity", "1e24",
                               Patterns::Double (0),
                               "The value of the maximum viscosity cutoff $\\eta_max$. Units: $Pa\\;s$.");
            prm.declare_entry ("Reference strain rate", "1e-15",
                               Patterns::Double (0),
                               "The value of the initial strain rate prescribed during the "
                               "first nonlinear iteration $\\dot{\\epsilon}_ref$. Units: $1/s$.");
            prm.declare_entry ("Angle of internal friction", "0",
                               Patterns::Double (0),
                               "The value of the angle of internal friction $\\phi$. "
                               "For a value of zero, in 2D the von Mises "
                               "criterion is retrieved. Angles higher than 30 degrees are "
                               "harder to solve numerically. Units: degrees.");
            prm.declare_entry ("Cohesion", "2e7",
                               Patterns::Double (0),
                               "The value of the cohesion $C$. Units: $Pa$.");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    DruckerPrager<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Drucker Prager");
        {
          reference_rho              = prm.get_double ("Reference density");
          reference_T                = prm.get_double ("Reference temperature");
          reference_eta              = prm.get_double ("Reference viscosity");
          thermal_k                  = prm.get_double ("Thermal conductivity");
          reference_specific_heat    = prm.get_double ("Reference specific heat");
          thermal_alpha              = prm.get_double ("Thermal expansion coefficient");
          prm.enter_subsection ("Viscosity");
          {
            minimum_viscosity        = prm.get_double ("Minimum viscosity");
            maximum_viscosity        = prm.get_double ("Maximum viscosity");
            reference_strain_rate    = prm.get_double ("Reference strain rate");
            // Convert degrees to radians
            phi                      = prm.get_double ("Angle of internal friction") * numbers::PI/180.0;
            cohesion                 = prm.get_double ("Cohesion");
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // Declare dependencies on solution variables
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::none;
      this->model_dependence.viscosity = NonlinearDependence::strain_rate;
      this->model_dependence.density = NonlinearDependence::none;

      if (phi==0.0)
        this->model_dependence.viscosity |= NonlinearDependence::pressure;

      if (thermal_alpha != 0)
        this->model_dependence.density = NonlinearDependence::temperature;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(DruckerPrager,
                                   "drucker prager",
                                   "A material model that has constant values "
                                   "for all coefficients but the density and viscosity. The defaults for all "
                                   "coefficients are chosen to be similar to what is believed to be correct "
                                   "for Earth's mantle. All of the values that define this model are read "
                                   "from a section ``Material model/Drucker Prager'' in the input file, see "
                                   "Section~\\ref{parameters:Material_model/Drucker Prager}."
                                   "Note that the model does not take into account any dependencies of "
                                   "material properties on compositional fields. "
                                   "\n\n"
                                   "The viscosity is computed according to the Drucker Prager frictional "
                                   "plasticity criterion (non-associative) based on a user-defined "
                                   "internal friction angle $\\phi$ and cohesion $C$. In 3D: "
                                   " $\\sigma_y = \\frac{6 C \\cos(\\phi)}{\\sqrt(3) (3+\\sin(\\phi))} + "
                                   "\\frac{2 P \\sin(\\phi)}{\\sqrt(3) (3+\\sin(\\phi))}$, "
                                   "where $P$ is the pressure. "
                                   "See for example Zienkiewicz, O. C., Humpheson, C. and Lewis, R. W. (1975), "
                                   "G\\'{e}otechnique 25, No. 4, 671-689. "
                                   "With this formulation we circumscribe instead of inscribe the Mohr Coulomb "
                                   "yield surface. "
                                   "In 2D the Drucker Prager yield surface is the same "
                                   "as the Mohr Coulomb surface: "
                                   " $\\sigma_y = P \\sin(\\phi) + C \\cos(\\phi)$. "
                                   "Note that in 2D for $\\phi=0$, these criteria "
                                   "revert to the von Mises criterion (no pressure dependence). "
                                   "See for example Thieulot, C. (2011), PEPI 188, 47-68. "
                                   "\n\n"
                                   "Note that we enforce the pressure to be positive to prevent negative "
                                   "yield strengths and viscosities. "
                                   "\n\n"
                                   "We then use the computed yield strength to scale back the viscosity on "
                                   "to the yield surface using the Viscosity Rescaling Method described in "
                                   "Kachanov, L. M. (2004), Fundamentals of the Theory of Plasticity, "
                                   "Dover Publications, Inc. (Not Radial Return.)"
                                   "A similar implementation can be found in GALE "
                                   "(https://geodynamics.org/cig/software/gale/gale-manual.pdf). "
                                   "\n\n"
                                   "To avoid numerically unfavourably large (or even negative) viscosity ranges, "
                                   "we cut off the viscosity with a user-defined minimum and maximum viscosity: "
                                   "$\\eta_eff = \\frac{1}{\\frac{1}{\\eta_min + \\eta}+ "
                                   "\\frac{1}{\\eta_max}}$. "
                                   "\n\n"
                                   "Note that this model uses the formulation that assumes an incompressible "
                                   "medium despite the fact that the density follows the law "
                                   "$\\rho(T)=\\rho_0(1-\\beta(T-T_{\\text{ref}}))$. ")
  }
}
