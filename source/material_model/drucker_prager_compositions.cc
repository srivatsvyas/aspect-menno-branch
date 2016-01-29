/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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

#include <aspect/material_model/drucker_prager_compositions.h>
#include <aspect/utilities.h>

using namespace dealii;

namespace aspect
{
  namespace
  {
    std::vector<double>
    get_vector_double (const std::string &parameter, const unsigned int n_fields, ParameterHandler &prm)
    {
      std::vector<double> parameter_list;
      parameter_list = Utilities::string_to_double(Utilities::split_string_list(prm.get (parameter)));
      if (parameter_list.size() == 1)
        parameter_list.resize(n_fields, parameter_list[0]);

      AssertThrow(parameter_list.size() == n_fields,
                  ExcMessage("Length of " + parameter + " list (size "+ std::to_string(parameter_list.size()) +") must be either one,"
                             " or n_compositional_fields+1 (= " + std::to_string(n_fields) + ")."));

      return parameter_list;
    }
  }

  namespace MaterialModel
  {
    template <int dim>
    double
    DruckerPragerCompositions<dim>::
    compute_second_invariant(const SymmetricTensor<2,dim> strain_rate, const double min_strain_rate) const
    {

      const double edot_ii_strict = std::sqrt(strain_rate*strain_rate); // V6
      //if(trace(strain_rate)!=0)
      //std::cout << "Strain rate = " << strain_rate << ", dev sr = " << deviator(strain_rate) << ", trace sr = " << trace(strain_rate) << ", trace dev sr = " << trace(deviator(strain_rate)) <<  std::endl;
      //const double edot_ii_strict = std::sqrt(deviator(strain_rate)*deviator(strain_rate)); // V7
      //const double edot_ii_strict = std::sqrt(std::fabs(second_invariant(deviator(strain_rate)))); // V8
      //const double edot_ii_strict = std::sqrt(std::fabs(second_invariant(strain_rate))); // V9//std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));//2 * std::sqrt(0.5*deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));
      const double edot_ii =  std::max(edot_ii_strict, min_strain_rate*min_strain_rate);
      return edot_ii;
    }

    template <int dim>
    double
    DruckerPragerCompositions<dim>::
    compute_viscosity(const double edot_ii,const double pressure,const int comp,const double prefactor,const double alpha, const double eref, const double min_visc, const double max_visc) const
    {
      double viscosity;
      if (comp == 0)
        {

          const double strength = ( (dim==3)
                                    ?
                                    ( 6.0 * cohesion[comp] * std::cos(phi[comp] * numbers::PI/180) + 6.0 * std::max(pressure,0.0) * std::sin(phi[comp] * numbers::PI/180) )
                                    / ( std::sqrt(3.0) * ( 3.0 + std::sin(phi[comp]) ) )
                                    :
                                    cohesion[comp] * std::cos(phi[comp] * numbers::PI/180) + std::max(pressure,0.0) * std::sin(phi[comp] * numbers::PI/180) );

          // Rescale the viscosity back onto the yield surface
          if (strength != 0 && edot_ii != 0)
            viscosity = strength / ( 2.0 * edot_ii );
          else
            viscosity = ref_visc;

          viscosity =ref_visc*viscosity/(ref_visc+viscosity);
        }
      else
        {
          viscosity = prefactor;
        }


      return std::max(std::min(viscosity,max_visc),min_visc);
      //return prefactor * std::pow(edot_ii,alpha);

      //return (prefactor / (2 * eref)) * (1/std::pow(eref,alpha)) * pow(edot_ii+eref*eref,alpha/2);

      /*double eref = 1e-4 * std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));
      if(eref == 0)
        eref = 1e-4 * min_strain_rate[c];
      composition_viscosities[c] = //prefactor[c] * std::pow(edot_ii,stress_exponent_inv);*/
      //(stress_exponent_inv * composition_viscosities[c] / edot_ii) * in.strain_rate[i];
      /**
       * EQ 1: This one seems to work
       */
      //std::pow(prefactor[c],-stress_exponent_inv) * std::pow(edot_ii,stress_exponent_inv-1);
      /**
       * EQ 2: Raids equation of powerlaw
       */
      //prefactor[c] * std::pow(std::pow(std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i])+eref*eref),2),alpha/2);
      //compute_viscosity(edot_ii,prefactor[c],alpha);//prefactor[c] * std::pow(edot_ii,alpha);
      //std::cout << composition_viscosities[c] << " = " << prefactor[c] << " * std::pow(" << edot_ii << "," << alpha << ")" << std::endl;
      /**
       * EQ FINAL
       */
      //std::max(std::min(std::pow(prefactor[c],-stress_exponent_inv) * std::pow(edot_ii,stress_exponent_inv-1), max_visc[c]), min_visc[c]);

    }

    template <int dim>
    void
    DruckerPragerCompositions<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      //set up additional output for the derivatives
      MaterialModelDerivatives<dim> *derivatives;
      derivatives = out.template get_additional_output<MaterialModelDerivatives<dim> >();

      for (unsigned int i=0; i < in.temperature.size(); ++i)
        {
          // const Point<dim> position = in.position[i];
          const double temperature = in.temperature[i];
          const double pressure = in.pressure[i];
          const Point<dim> position = in.position[i];

          // Averaging composition-field dependent properties
          // Compositions
          //This assert may be help when designing tests for this material model
          AssertThrow(in.composition[i].size()+1 == n_fields,
                      ExcMessage("Number of compositional fields + 1 not equal to number of fields given in input file."));

          const std::vector<double> volume_fractions = Utilities::compute_volume_fractions(in.composition[i]);
          double density = 0.0;
          for (unsigned int c=0; c < volume_fractions.size(); ++c)
            {
              //not strictly correct if thermal expansivities are different, since we are interpreting
              //these compositions as volume fractions, but the error introduced should not be too bad.
        	  double temperature_factor = 1;
        	  if (this->get_adiabatic_conditions().is_initialized())
        		  temperature_factor = (1 - thermal_expansivities[c] * (temperature - this->get_adiabatic_conditions().temperature(position)));
              //const double temperature_factor = (1.0 - thermal_expansivities[c] * (temperature - reference_T_list[c]));
              density += volume_fractions[c] * densities[c] * std::exp(reference_compressibility * pressure) * temperature_factor;
            }

          // thermal expansivities
          double thermal_expansivity = 0.0;
          for (unsigned int c=0; c < volume_fractions.size(); ++c)
            thermal_expansivity += volume_fractions[c] * thermal_expansivities[c];

          // Specific heat at the given positions.
          double specific_heat = 0.0;
          for (unsigned int c=0; c < volume_fractions.size(); ++c)
            specific_heat += volume_fractions[c] * heat_capacity[c];

          // Thermal conductivity at the given positions.
          double thermal_conductivities = 0.0;
          for (unsigned int c=0; c < volume_fractions.size(); ++c)
            thermal_conductivities += volume_fractions[c] * thermal_diffusivity[c] * heat_capacity[c] * densities[c];

          // calculate effective viscosity
          if (in.strain_rate.size())
            {
              // This function calculates viscosities assuming that all the compositional fields
              // experience the same strain rate (isostrain). Since there is only one process in
              // this material model (a general powerlaw) we do not need to worry about how to
              // distribute the strain-rate and stress over the processes.
              std::vector<double> composition_viscosities(volume_fractions.size());
              std::vector<SymmetricTensor<2,dim> > composition_viscosities_derivatives(volume_fractions.size());
              std::vector<double> composition_dviscosities_dpressure(volume_fractions.size());

              for (unsigned int c=0; c < volume_fractions.size(); ++c)
                {
                  // If strain rate is zero (like during the first time step) set it to some very small number
                  // to prevent a division-by-zero, and a floating point exception.
                  // Otherwise, calculate the square-root of the norm of the second invariant of the deviatoric-
                  // strain rate (often simplified as epsilondot_ii)
                  //const double edot_ii_strict = std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));//std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));//2 * std::sqrt(0.5*deviator(in.strain_rate[i])*deviator(in.strain_rate[i]));
            	  const SymmetricTensor<2,dim> strain_rate = in.strain_rate[i];
            	  const SymmetricTensor<2,dim> deviator_strain_rate = deviator(in.strain_rate[i]);
            	  const double edot_ii = compute_second_invariant(deviator_strain_rate, min_strain_rate[c]);// std::max(edot_ii_strict, min_strain_rate[c]*min_strain_rate[c]);

                  // Find effective viscosities for each of the individual phases
                  // Viscosities should have same number of entries as compositional fields

                  // Power law equation
                  // edot_ii_i = A_i * stress_ii_i^{n_i} * d^{-m} \exp\left(-\frac{E_i^* + PV_i^*}{n_iRT}\right)
                  // where ii indicates the square root of the second invariant and
                  // i corresponds to diffusion or dislocation creep

                  // The isostrain condition implies that the viscosity averaging should be arithmetic (see above).
                  // We have given the user freedom to apply alternative bounds, because in diffusion-dominated
                  // creep (where n_diff=1) viscosities are stress and strain-rate independent, so the calculation
                  // of compositional field viscosities is consistent with any averaging scheme.
                  // TODO: Mailed Bob and asked why the averaging should be arithmetic. Bob replyed that this has to
                  //       do with effective medium theory. Have to look into this a bit more.
                  const double stress_exponent_inv = (1./stress_exponent[c]);//stress_exponent[c])-1;
                  const double alpha = stress_exponent_inv - 1;
                  const double eref = 0; //1e-20;//std::max(1e-15 * edot_ii,1e-15);

                  composition_viscosities[c] = compute_viscosity(edot_ii,pressure,c,prefactor[c],alpha,eref,min_visc[c],max_visc[c]);

                  Assert(dealii::numbers::is_finite(composition_viscosities[c]),ExcMessage ("Error: Viscosity is not finite."));

                  if (this->get_parameters().newton_theta != 0 && derivatives != NULL)
                    {
                      if (use_analytical_derivative)
                        {
                          //analytic
                          if (edot_ii >= min_strain_rate[c] && composition_viscosities[c] < max_visc[c] && composition_viscosities[c] > min_visc[c])// && composition_viscosities[c] < max_visc[c] && composition_viscosities[c] > min_visc[c])
                            {
                              const double cohesion = 1e8;
                              const double phi = 30 * numbers::PI/180;
                              //strictly speaking the derivative is this: 0.5 * ((1/stress_exponent)-1) * std::pow(2,2) * out.viscosities[i] * (1/(edot_ii*edot_ii)) * deviator(in.strain_rate[i])
                              composition_viscosities_derivatives[c] = -(1/(edot_ii*edot_ii*edot_ii))*deviator_strain_rate*(cohesion*std::cos(phi)+in.pressure[i]*std::sin(phi));//alpha * 1 * composition_viscosities[c] * (1/(edot_ii * edot_ii + eref * eref))  * in.strain_rate[i];
                              /**
                               * EQ 2: Raids euqation of powerlaw
                               */
                              //alpha * composition_viscosities[c] * (deviator(in.strain_rate[i])*deviator(in.strain_rate[i])/ std::pow(std::max(std::sqrt(deviator(in.strain_rate[i])*deviator(in.strain_rate[i])),min_strain_rate[c]),2));
                              /**
                               * EQ FINAL
                               */
                              //composition_viscosities[c] * (stress_exponent_inv-1) * prefactor[c] *  std::pow(edot_ii,stress_exponent_inv-1) * (1/(edot_ii * edot_ii)) * in.strain_rate[i];
                              //(stress_exponent_inv * composition_viscosities[c] / edot_ii) * in.strain_rate[i];
                              /**
                               * EQ 1: This one seems to work
                               */
                              //2 * (stress_exponent_inv-1) * composition_viscosities[c] * (1/(edot_ii*edot_ii)) * deviator(in.strain_rate[i]);
                              //std::cout << composition_viscosities_derivatives[i] << ", dev strt:" << deviator(in.strain_rate[i]) << std::endl;
                              //std::cout << "SN1: " << 2 * (stress_exponent_inv-1) * composition_viscosities[c] * (1/(edot_ii*edot_ii)) << ";" << std::flush;
                              //std::cout << "eq " << c << ": " << composition_viscosities_derivatives[c] << "= 2 * (" << stress_exponent_inv << "-1) * " << composition_viscosities[c] << "* (1/(" << edot_ii << "*" << edot_ii << ") * " << deviator(in.strain_rate[i]) << std::endl;
                            }
                          else
                            {
                              composition_viscosities_derivatives[c] = 0;
                            }
                        }
                      else
                        {
                          // finite difference
                          const double finite_difference_accuracy = 1e-7;
                          SymmetricTensor<2,dim> zerozero = SymmetricTensor<2,dim>();
                          SymmetricTensor<2,dim> onezero = SymmetricTensor<2,dim>();
                          SymmetricTensor<2,dim> oneone = SymmetricTensor<2,dim>();

                          zerozero[0][0] = 1;
                          onezero[1][0]  = 0.5; // because symmetry doubles this entry
                          oneone[1][1]   = 1;

                          SymmetricTensor<2,dim> strain_rate_zero_zero = deviator_strain_rate + std::fabs(deviator_strain_rate[0][0]) * finite_difference_accuracy * zerozero;
                          SymmetricTensor<2,dim> strain_rate_one_zero = deviator_strain_rate + std::fabs(deviator_strain_rate[1][0]) * finite_difference_accuracy * onezero;
                          SymmetricTensor<2,dim> strain_rate_one_one = deviator_strain_rate + std::fabs(deviator_strain_rate[1][1]) * finite_difference_accuracy * oneone;

                          //SymmetricTensor<2,dim> strain_rate_zero_zero = strain_rate + std::fabs(strain_rate[0][0]) * finite_difference_accuracy * zerozero;
                          //SymmetricTensor<2,dim> strain_rate_one_zero = strain_rate + std::fabs(strain_rate[1][0]) * finite_difference_accuracy * onezero;
                          //SymmetricTensor<2,dim> strain_rate_one_one = strain_rate + std::fabs(strain_rate[1][1]) * finite_difference_accuracy * oneone;

                          double edot_ii_fd;

                          edot_ii_fd = compute_second_invariant(strain_rate_zero_zero,0);
                          double eta_zero_zero = compute_viscosity(edot_ii_fd,pressure,c,prefactor[c],alpha,eref,min_visc[c],max_visc[c]);
                          double deriv_zero_zero = eta_zero_zero - composition_viscosities[c];

                          if (deriv_zero_zero != 0)
                            {
                              if (strain_rate_zero_zero[0][0] != 0)
                                {
                                  deriv_zero_zero /= std::fabs(strain_rate_zero_zero[0][0]) * finite_difference_accuracy;
                                }
                              else
                                {
                                  deriv_zero_zero = 0;
                                }

                            }

                          edot_ii_fd = compute_second_invariant(strain_rate_one_zero,0);
                          double eta_one_zero = compute_viscosity(edot_ii_fd,pressure,c,prefactor[c],alpha,eref,min_visc[c],max_visc[c]);
                          double deriv_one_zero = eta_one_zero - composition_viscosities[c];

                          if (deriv_one_zero != 0)
                            {
                              if (strain_rate_one_zero[1][0] != 0)
                                {
                                  deriv_one_zero /= std::fabs(strain_rate_one_zero[1][0]) * finite_difference_accuracy;
                                }
                              else
                                {
                                  deriv_one_zero = 0;
                                }
                            }

                          edot_ii_fd = compute_second_invariant(strain_rate_one_one,0);
                          double eta_one_one = compute_viscosity(edot_ii_fd,pressure,c,prefactor[c],alpha,eref,min_visc[c],max_visc[c]);
                          double deriv_one_one = eta_one_one - composition_viscosities[c];

                          if (eta_one_one != 0)
                            {
                              if (strain_rate_one_one[1][1] != 0)
                                {
                                  deriv_one_one /= std::fabs(strain_rate_one_one[1][1]) * finite_difference_accuracy;
                                }
                              else
                                {
                                  deriv_one_one = 0;
                                }
                            }

                          composition_viscosities_derivatives[c][0][0] = deriv_zero_zero;
                          composition_viscosities_derivatives[c][1][0] = deriv_one_zero;
                          composition_viscosities_derivatives[c][1][1] = deriv_one_one;


                          /**
                           * Now compute the derivative of the viscoisty to the pressure
                           */
                          double pressure_difference = in.pressure[i] + (std::fabs(in.pressure[i]) * 1e-7);

                          double pressure_difference_eta = compute_viscosity(edot_ii, pressure_difference,c,prefactor[c],alpha,eref,min_visc[c],max_visc[c]);
                          double deriv_pressure = pressure_difference_eta - composition_viscosities[c];

                          if (pressure_difference_eta != 0)
                            {
                              if (in.pressure[i] != 0)
                                {
                                  deriv_pressure /= std::fabs(in.pressure[i]) * 1e-7;
                                }
                              else
                                {
                                  deriv_pressure = 0;
                                }
                            }
                          composition_dviscosities_dpressure[c] = deriv_pressure;
                        }
                    }
                }
              out.viscosities[i] = Utilities::weighted_p_norm_average(volume_fractions, composition_viscosities, viscosity_averaging_p);
              Assert(dealii::numbers::is_finite(out.viscosities[i]),ExcMessage ("Error: Averaged viscosity is not finite."));

              if (this->get_parameters().newton_theta != 0 && derivatives != NULL)
                {
                  derivatives->dviscosities_dstrain_rate[i] = Utilities::derivatives_weighed_p_norm_average(out.viscosities[i],volume_fractions, composition_viscosities, composition_viscosities_derivatives, viscosity_averaging_p);
                  derivatives->dviscosities_dpressure[i] = Utilities::derivatives_weighed_p_norm_average(out.viscosities[i],volume_fractions, composition_viscosities, composition_dviscosities_dpressure, viscosity_averaging_p);//p_norm_average(volume_fractions, composition_dviscosities_dpressure, viscosity_averaging_p);//;//p_norm_average(in.composition[i], composition_dviscosities_dpressure, viscosity_averaging_p);


#ifdef DEBUG
                  for (int x = 0; x < dim; x++)
                    for (int y = 0; y < dim; y++)
                      if (!dealii::numbers::is_finite(derivatives->dviscosities_dstrain_rate[i][x][y]))
                        std::cout << "Error: Averaged viscosity to strain-rate devrivative is not finite." << std::endl;

                  //derivatives->dviscosities_dpressure[i]    = 0;*/
                  if (!dealii::numbers::is_finite(derivatives->dviscosities_dpressure[i]))
                    {
                      std::cout << "Error: Averaged viscosity to pressure devrivative is not finite. " << std::endl;
                      for (unsigned int c=0; c < volume_fractions.size(); ++c)
                        std::cout << composition_dviscosities_dpressure[c] << ",";
                      std::cout << std::endl;
                    }
                  Assert(dealii::numbers::is_finite(derivatives->dviscosities_dpressure[i]),ExcMessage ("Error: Averaged dviscosities_dpressure is not finite."));
                  for (int x = 0; x < dim; x++)
                    for (int y = 0; y < dim; y++)
                      Assert(dealii::numbers::is_finite(derivatives->dviscosities_dstrain_rate[i][x][y]),ExcMessage ("Error: Averaged dviscosities_dstrain_rate is not finite."));
                  //Assert(dealii::numbers::is_nan(out.viscosities[i]),ExcMessage ("Error: Averaged viscosity is not finite."));
#endif
                }
            }
          out.densities[i] = density;
          out.thermal_expansion_coefficients[i] = thermal_expansivity;
          // Specific heat at the given positions.
          out.specific_heat[i] = specific_heat;
          // Thermal conductivity at the given positions.
          out.thermal_conductivities[i] = thermal_conductivities;
          // Compressibility at the given positions.
          // The compressibility is given as
          // $\frac 1\rho \frac{\partial\rho}{\partial p}$.
          out.compressibilities[i] = reference_compressibility;
          // Pressure derivative of entropy at the given positions.
          out.entropy_derivative_pressure[i] = 0.0;
          // Temperature derivative of entropy at the given positions.
          out.entropy_derivative_temperature[i] = 0.0;
          // Change in composition due to chemical reactions at the
          // given positions. The term reaction_terms[i][c] is the
          // change in compositional field c at point i.
          for (unsigned int c=0; c < in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0.0;
        }
    }
    template <int dim>
    double
    DruckerPragerCompositions<dim>::
    reference_viscosity () const
    {
      return ref_visc;
    }

    template <int dim>
    double
    DruckerPragerCompositions<dim>::
    reference_density () const
    {
      return densities[0];
    }

    template <int dim>
    bool
    DruckerPragerCompositions<dim>::
    is_compressible () const
    {
      return (reference_compressibility != 0);
    }

    template <int dim>
    void
    DruckerPragerCompositions<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Compositional fields");
      {
        prm.declare_entry ("Number of fields", "0",
                           Patterns::Integer (0),
                           "The number of fields that will be advected along with the flow field, excluding "
                           "velocity, pressure and temperature.");
        prm.declare_entry ("List of conductivities of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of thermal conductivities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of capacities of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of heat capacities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of reftemps of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of reference temperatures equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of refdens of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of reference densities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("Thermal expansivities", "3.5e-5",
                           Patterns::List(Patterns::Double(0)),
                           "List of thermal expansivities for background mantle and compositional fields, "
                           "for a total of N+1 values, where N is the number of compositional fields. "
                           "If only one values is given, then all use the same value.  Units: $1 / K$");
        prm.declare_entry ("List of stress exponents of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of stress exponents equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of cohesion of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of initial viscosities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of angle of internal friction of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of initial viscosities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of initviscs of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of initial viscosities equal to the number of "
                           "compositional fields.");
        prm.declare_entry ("List of prefactors of fields", "",
                           Patterns::List (Patterns::Double(0)),
                           "A list of viscous viscosities equal to the number of "
                           "compositional fields.");

      }
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection ("Drucker prager compositions");
        {
          // Reference and minimum/maximum values
          prm.declare_entry ("Reference temperature", "293", Patterns::Double(0),
                             "For calculating density by thermal expansivity. Units: $K$");
          prm.declare_entry ("Minimum strain rate", "1.4e-20", Patterns::List(Patterns::Double(0)),
                             "Stabilizes strain dependent viscosity. Units: $1 / s$");
          prm.declare_entry ("Minimum viscosity", "1e10", Patterns::List(Patterns::Double(0)),
                             "Lower cutoff for effective viscosity. Units: $Pa s$");
          prm.declare_entry ("Maximum viscosity", "1e28", Patterns::List(Patterns::Double(0)),
                             "Upper cutoff for effective viscosity. Units: $Pa s$");
          prm.declare_entry ("Effective viscosity coefficient", "1.0", Patterns::List(Patterns::Double(0)),
                             "Scaling coefficient for effective viscosity.");
          prm.declare_entry ("Reference viscosity", "1e22", Patterns::List(Patterns::Double(0)),
                             "Reference viscosity for nondimensionalization. Units $Pa s$");
          prm.declare_entry ("Reference compressibility", "4e-12", Patterns::Double (0),
                             "The value of the reference compressibility. Units: $1/Pa$.");

          // averaging parameters
          prm.declare_entry ("Viscosity averaging p", "-1",
                             Patterns::Double(),
                             "This is the p value in the generalized weighed average eqation: "
                             " mean = \\frac{1}{k}(\\sum_{i=1}^k \\big(c_i \\eta_{\\text{eff}_i}^p)\\big)^{\\frac{1}{p}}. "
                             " Units: $Pa s$");

          // finite difference versus analytical
          prm.declare_entry ("Use analytical derivative", "false",
                             Patterns::Bool(),
                             "A bool indicating wether to use finite differences to compute the derivative or to use "
                             "the analytical derivative.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    DruckerPragerCompositions<dim>::parse_parameters (ParameterHandler &prm)
    {
      // can't use this->n_compositional_fields(), because some
      // tests never initiate the simulator, but uses the material
      // model directly.
      prm.enter_subsection ("Compositional fields");
      {
        n_fields = prm.get_integer ("Number of fields")+1;
        //AssertThrow(n_fields > 0, ExcMessage("This material model needs at least one compositional field."))
        reference_T_list = get_vector_double("List of reftemps of fields",n_fields,prm);//
        ref_visc_list = get_vector_double ("List of initviscs of fields",n_fields,prm);//
        thermal_diffusivity = get_vector_double("List of conductivities of fields",n_fields,prm);
        heat_capacity = get_vector_double("List of capacities of fields",n_fields,prm);

        // ---- Compositional parameters
        densities = get_vector_double("List of refdens of fields",n_fields,prm);
        thermal_expansivities = get_vector_double("Thermal expansivities",n_fields,prm);

        // Rheological parameters
        cohesion = get_vector_double("List of cohesion of fields",n_fields,prm);
        phi = get_vector_double("List of angle of internal friction of fields",n_fields,prm);
        prefactor = get_vector_double("List of prefactors of fields",n_fields,prm);
        stress_exponent = get_vector_double("List of stress exponents of fields",n_fields,prm);
      }
      prm.leave_subsection();

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection ("Drucker prager compositions");
        {

          // Reference and minimum/maximum values
          reference_T = prm.get_double("Reference temperature");
          ref_visc = prm.get_double ("Reference viscosity");
          min_strain_rate = get_vector_double("Minimum strain rate",n_fields,prm);
          min_visc = get_vector_double ("Minimum viscosity",n_fields,prm);
          max_visc = get_vector_double ("Maximum viscosity",n_fields,prm);
          veff_coefficient = get_vector_double ("Effective viscosity coefficient",n_fields,prm);
          reference_compressibility  = prm.get_double ("Reference compressibility");


          // averaging parameters
          viscosity_averaging_p = prm.get_double("Viscosity averaging p");

          use_analytical_derivative = prm.get_bool("Use analytical derivative");

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();


      // Declare dependencies on solution variables
      this->model_dependence.viscosity = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::strain_rate | NonlinearDependence::compositional_fields;
      this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
      this->model_dependence.compressibility = NonlinearDependence::none;
      this->model_dependence.specific_heat = NonlinearDependence::none;
      this->model_dependence.thermal_conductivity = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(DruckerPragerCompositions,
                                   "drucker prager compositions",
                                   " An implementation of a viscous rheology including diffusion"
                                   " and dislocation creep."
                                   " Compositional fields can each be assigned individual"
                                   " activation energies, reference densities, thermal expansivities,"
                                   " and stress exponents. The effective viscosity is defined as"
                                   " \n\n"
                                   " \\[v_\\text{eff} = \\left(\\frac{1}{v_\\text{eff}^\\text{diff}}+"
                                   " \\frac{1}{v_\\text{eff}^\\text{dis}}\\right)^{-1}\\]"
                                   " where"
                                   " \\[v_\\text{i} = 0.5 * A^{-\\frac{1}{n_i}} d^\\frac{m_i}{n_i}"
                                   " \\dot{\\varepsilon_i}^{\\frac{1-n_i}{n_i}}"
                                   " \\exp\\left(\\frac{E_i^* + PV_i^*}{n_iRT}\\right)\\]"
                                   " \n\n"
                                   " where $d$ is grain size, $i$ corresponds to diffusion or dislocation creep,"
                                   " $\\dot{\\varepsilon}$ is the square root of the second invariant of the"
                                   " strain rate tensor, $R$ is the gas constant, $T$ is temperature, "
                                   " and $P$ is pressure."
                                   " $A_i$ are prefactors, $n_i$ and $m_i$ are stress and grain size exponents"
                                   " $E_i$ are the activation energies and $V_i$ are the activation volumes."
                                   " \n\n"
                                   " The ratio of diffusion to dislocation strain rate is found by Newton's"
                                   " method, iterating to find the stress which satisfies the above equations."
                                   " The value for the components of this formula and additional"
                                   " parameters are read from the parameter file in subsection"
                                   " 'Material model/SimpleNonlinear'.")
  }
}
