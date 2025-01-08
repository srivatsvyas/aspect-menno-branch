/*
  Copyright (C) 2022 - 2023 by the authors of the ASPECT code.

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
 along with ASPECT; see the file LICENSE.  If not see
 <http://www.gnu.org/licenses/>.
 */

#include <aspect/particle/property/crystal_preferred_orientation.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/gravity_model/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <world_builder/grains.h>
#include <world_builder/world.h>

#include <aspect/citation_info.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      template <int dim>
      CrystalPreferredOrientation<dim>::CrystalPreferredOrientation ()
      {}



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize ()
      {
        CitationInfo::add("CPO");
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        this->random_number_generator.seed(random_number_seed+my_rank);

        // Don't assert when called by the unit tester.
        if (this->simulator_is_past_initialization())
          {
            AssertThrow(this->introspection().compositional_name_exists("water"),
                        ExcMessage("Particle property CPO only works if"
                                   "there is a compositional field called water."));
            water_index = this->introspection().compositional_index_for_name("water");
          }
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::compute_random_rotation_matrix(Tensor<2,3> &rotation_matrix) const
      {
        
        // This function is based on an article in Graphic Gems III, written by James Arvo, Cornell University (p 116-120).
        // The original code can be found on  http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        // and is licenced according to this website with the following licence:
        //
        // "The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your own and
        // resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial. Giving credit
        // is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems with any Gems
        // code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held responsible. Basically,
        // don't be a jerk, and remember that anything free comes with no guarantee.""
        //
        // The book states in the preface the following: "As in the first two volumes, all of the C and C++ code in this book is in
        // the public domain, and is yours to study, modify, and use."

        // first generate three random numbers between 0 and 1 and multiply them with 2 PI or 2 for z. Note that these are not the same as phi_1, theta and phi_2.

        boost::random::uniform_real_distribution<double> uniform_distribution(0,1);
        double one   = uniform_distribution(this->random_number_generator);
        double two   = uniform_distribution(this->random_number_generator);
        double three = uniform_distribution(this->random_number_generator);

        boost::random::uniform_real_distribution<double> uniform_distribution1(0,numbers::PI/24);
        double rand_def = uniform_distribution1(this->random_number_generator);
        /*double theta = 2.0 * M_PI * one; // Rotation about the pole (Z)
        double phi   =  2.0 * M_PI * two; // For direction of pole deflection.
        double z     =   2.0 * three; //For magnitude of pole deflection.*/
        double theta;
        if(this-> get_time()!= 0)
          theta = 2.0 *  rand_def;
        else
          theta = 2.0 * M_PI * one; // Rotation about the pole (Z)
        double phi   = 2.0 * M_PI * two; // For direction of pole deflection.
        double z     = 2.0 * three; //For magnitude of pole deflection
        // SV : have to come up with a solution to create a misorientation of 10 degrees or more for olivine new grains.
        // SV : maybe write a separate block of code in the recrystalize_grains function to initialize the new grains with a misorientation wrt to the parent grain.

        // Compute a vector V used for distributing points over the sphere
        // via the reflection I - V Transpose(V).  This formulation of V
        // will guarantee that if x[1] and x[2] are uniformly distributed,
        // the reflected points will be uniform on the sphere.  Note that V
        // has length sqrt(2) to eliminate the 2 in the Householder matrix.

        double r  = std::sqrt( z );
        double Vx = std::sin( phi ) * r;
        double Vy = std::cos( phi ) * r;
        double Vz = std::sqrt( 2.f - z );

        // Compute the row vector S = Transpose(V) * R, where R is a simple
        // rotation by theta about the z-axis.  No need to compute Sz since
        // it's just Vz.

        double st = std::sin( theta );
        double ct = std::cos( theta );
        double Sx = Vx * ct - Vy * st;
        double Sy = Vx * st + Vy * ct;

        // Construct the rotation matrix  ( V Transpose(V) - I ) R, which
        // is equivalent to V S - R.

        rotation_matrix[0][0] = Vx * Sx - ct;
        rotation_matrix[0][1] = Vx * Sy - st;
        rotation_matrix[0][2] = Vx * Vz;
        rotation_matrix[1][0] = Vy * Sx + st;
        rotation_matrix[1][1] = Vy * Sy - ct;
        rotation_matrix[1][2] = Vy * Vz;
        rotation_matrix[2][0] = Vz * Sx;
        rotation_matrix[2][1] = Vz * Sy;
        rotation_matrix[2][2] = 1.0 - z;   // This equals Vz * Vz - 1.0
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::initialize_one_particle_property(const Point<dim> &,
                                                                         std::vector<double> &data) const
      {
        std::cout<<"total slots available = "<<n_grains<<"\t buffer slots = "<<n_grains_buffer<<std::endl;
        // the layout of the data vector per perticle is the following:
        // 1. M mineral times
        //    1.1  olivine deformation type   -> 1 double, at location
        //                                      => data_position + 0 + mineral_i * (n_grains * 20 + 2)
        //    2.1. Mineral volume fraction    -> 1 double, at location
        //                                      => data_position + 1 + mineral_i *(n_grains * 20 + 2)
        //    2.2. N grains times:
        //         2.1. volume fraction grain -> 1 double, at location:
        //                                      => data_position + 2 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 2 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.2. rotation matrix grain -> 9 (Tensor<2,dim>::n_independent_components) doubles, starts at:
        //                                      => data_position + 3 + i_grain * 20 + mineral_i * (n_grains * 20 + 2), or
        //                                      => data_position + 3 + i_grain * (2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.3. volume derivative grain -> 1 double at location:
        //                                      => data_position + 12 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 12 + i_graib * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.4. dislocation density grain -> 4 doubles, at location
        //                                      => data_position + 13 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 13 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.5. Max schid factor -> 1 double, at location
        //                                      => data_position + 17 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 17 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.6. RRSS for max scmid factor -> 1 double, at location
        //                                      => data_position + 18 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 18 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.7. Deformation mechanism factor -> 1 doubles, at location
        //                                      => data_position + 19 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                      => data_position + 19 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.8. Rate of recrystalization  -> 1 doubles, at location
        //                                       => data_position + 20 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                       => data_position + 20 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        //         2.9. Parent grain              -> 1 doubles, at location
        //                                       => data_position + 21 + i_grain * 20 + mineral_i *(n_grains * 20 + 2), or
        //                                       => data_position + 21 + i_grain * 2 * Tensor<2,3>::n_independent_components+ 2) + mineral_i * (n_grains * 10 + 2)
        // Note that we store exactly the same number of grains of all minerals (e.g. olivine and enstatite
        // grains), although their volume fractions may not be the same. We need a minimum amount
        // of grains per tracer to perform reliable statistics on it. This minimum is the same for all phases.
        // and enstatite.
        //
        // Furthermore, for this plugin the following dims are always 3. When using 2d an infinitely thin 3d domain is assumed.
        //
        // The rotation matrix is a direction cosine matrix, representing the orientation of the grain in the domain.
        // The fabric is determined later in the computations, so initialize it to -1.
        std::vector<double> deformation_type(n_minerals, -1.0);
        std::vector<std::vector<double >>volume_fractions_grains(n_minerals);
        std::vector<std::vector<Tensor<2,3>>> rotation_matrices_grains(n_minerals);
        std::vector<std::vector<double>>grain_boundary_velocity(n_minerals);
        std::vector<std::vector<std::array<double,4>>>dislocation_density_grains(n_minerals);
        std::vector<std::vector<double >>dominant_slip_system(n_minerals);
        std::vector<std::vector<double >>strain_energy(n_minerals);
        std::vector<std::vector<double >>surface_energy(n_minerals);
        std::vector<std::vector<double>> slip_rate(n_minerals);
        std::vector<std::vector<int>> grain_status(n_minerals);
        std::vector<std::vector<double>> strain_accumulated(n_minerals);
        std::vector<std::vector<double>> rx_fractions(n_minerals);
        std::vector<std::vector<double>> strain_difference(n_minerals);
        std::vector<std::vector<double>> dis_inv(n_minerals);
        std::vector<std::vector<double>> dif_inv(n_minerals);
        std::vector<std::vector<int>> lifetime(n_minerals);

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            volume_fractions_grains[mineral_i].resize(n_grains);
            rotation_matrices_grains[mineral_i].resize(n_grains);
            dislocation_density_grains[mineral_i].resize(n_grains);
            dominant_slip_system[mineral_i].resize(n_grains);
            grain_boundary_velocity[mineral_i].resize(n_grains);
            strain_accumulated[mineral_i].resize(n_grains);
            rx_fractions[mineral_i].resize(n_grains);
            strain_energy[mineral_i].resize(n_grains);
            surface_energy[mineral_i].resize(n_grains);
            slip_rate[mineral_i].resize(n_grains);
            strain_difference[mineral_i].resize(n_grains);
            dis_inv[mineral_i].resize(n_grains);
            dif_inv[mineral_i].resize(n_grains);
            lifetime[mineral_i].resize(n_grains);
            grain_status[mineral_i].resize(n_grains);
            // This will be set by the initial grain subsection.
            bool use_world_builder = false;
            if (use_world_builder)
              {
#ifdef ASPECT_WITH_WORLD_BUILDER
                AssertThrow(false,
                            ExcMessage("Not implemented."));
#else
                AssertThrow(false,
                            ExcMessage("The world builder was requested but not provided. Make sure that aspect is "
                                       "compiled with the World Builder and that you provide a world builder file in the input."));
#endif
              }
            else
              {

                switch (cpo_derivative_algorithm)
                  {
                    case CPODerivativeAlgorithm::drexpp:
                    {
                     
                      for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
                        {
                          //set volume fraction
                          if (grain_i < n_grains_init)
                            {
                              if (initialize_grains_with_normal_distribution)
                                {
                                   boost::random::normal_distribution<double> lognormal_distribution(initialize_grains_with_normal_distribution_mean,
                                                                                           initialize_grains_with_normal_distribution_standard_deviation);
                                  double grain_size = lognormal_distribution(this->random_number_generator);
                                  volume_fractions_grains[mineral_i][grain_i] = grain_size;
                                }
                              else
                              {
                                volume_fractions_grains[mineral_i][grain_i] = initial_grain_size;
                              }
                              grain_status[mineral_i][grain_i] = 0;
                            }
                          else
                           if(grain_i > n_grains - n_grains_buffer)
                            {
                              volume_fractions_grains[mineral_i][grain_i] = 0.;
                              grain_status[mineral_i][grain_i] = -2;
                            }
                          else
                           {
                              volume_fractions_grains[mineral_i][grain_i] = 0.;
                              grain_status[mineral_i][grain_i] = -1;
                           }
                          for (unsigned int i = 0; i<4; ++i)
                            {
                              dislocation_density_grains[mineral_i][grain_i][i] = 0.0;
                            }
                          this->compute_random_rotation_matrix(rotation_matrices_grains[mineral_i][grain_i]);
                          dominant_slip_system[mineral_i][grain_i] = 0.;
                          grain_boundary_velocity[mineral_i][grain_i] = 0.;
                          strain_energy[mineral_i][grain_i] = 0;
                          surface_energy[mineral_i][grain_i] = 0.;
                          slip_rate[mineral_i][grain_i] = 0;
                          strain_accumulated[mineral_i][grain_i] = 0.0;
                          strain_difference[mineral_i][grain_i] = 0.0;
                          rx_fractions[mineral_i][grain_i] = 0.;
                          dis_inv[mineral_i][grain_i] = 0;
                          dif_inv[mineral_i][grain_i] = 0;
                          lifetime[mineral_i][grain_i] = 0;
                        }
                      break;
                    }
                    case CPODerivativeAlgorithm::drex_2004:
                    {
                      for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
                        {
                          const double initial_volume_fraction = 1.0/n_grains;
                          volume_fractions_grains[mineral_i][grain_i] = initial_volume_fraction;
                          this->compute_random_rotation_matrix(rotation_matrices_grains[mineral_i][grain_i]);
                          for (unsigned int i = 0; i<4; ++i)
                            {
                              dislocation_density_grains[mineral_i][grain_i][i] = 0.0;
                            }
                          dominant_slip_system[mineral_i][grain_i] = 0.;
                          grain_boundary_velocity[mineral_i][grain_i] = 0.;
                          strain_energy[mineral_i][grain_i] = 0;
                          surface_energy[mineral_i][grain_i] = 0.;
                          slip_rate[mineral_i][grain_i] = 0;
                          grain_status[mineral_i][grain_i] = 0;
                          strain_accumulated[mineral_i][grain_i] = 0.0;
                          strain_difference[mineral_i][grain_i] = 0.0;
                          rx_fractions[mineral_i][grain_i] = 0.;
                          dis_inv[mineral_i][grain_i] = 0;
                          dif_inv[mineral_i][grain_i] = 0;
                          lifetime[mineral_i][grain_i] = 0;
                        }

                      break;
                    }
                    default:
                      break;
                  }
              }
          }


        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            data.emplace_back(deformation_type[mineral_i]);
            data.emplace_back(volume_fractions_minerals[mineral_i]);
            for (unsigned int grain_i = 0; grain_i < n_grains ; ++grain_i)
              {
                data.emplace_back(volume_fractions_grains[mineral_i][grain_i]);
                //std::cout<<volume_fractions_grains[mineral_i][grain_i]<<"\n";
                for (unsigned int i = 0; i < Tensor<2,3>::n_independent_components ; ++i)
                  {
                    const dealii::TableIndices<2> index = Tensor<2,3>::unrolled_to_component_indices(i);
                    data.emplace_back(rotation_matrices_grains[mineral_i][grain_i][index]);
                  }
                data.emplace_back(grain_boundary_velocity[mineral_i][grain_i]);
                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    data.emplace_back(dislocation_density_grains[mineral_i][grain_i][slip_system_i]);
                  }
                      data.emplace_back(grain_status[mineral_i][grain_i]);
                data.emplace_back(dominant_slip_system[mineral_i][grain_i]);
                data.emplace_back(strain_energy[mineral_i][grain_i]);
                data.emplace_back(surface_energy[mineral_i][grain_i]);
                data.emplace_back(rx_fractions[mineral_i][grain_i]);
                data.emplace_back(slip_rate[mineral_i][grain_i]);
                data.emplace_back(strain_accumulated[mineral_i][grain_i]);
                data.emplace_back(dis_inv[mineral_i][grain_i]);
                data.emplace_back(dif_inv[mineral_i][grain_i]);
                data.emplace_back(strain_difference[mineral_i][grain_i]);
                data.emplace_back(lifetime[mineral_i][grain_i]);
              }
          }
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::update_one_particle_property(const unsigned int data_position,
                                                                     const Point<dim> &position,
                                                                     const Vector<double> &solution,
                                                                     const std::vector<Tensor<1,dim>> &gradients,
                                                                     const ArrayView<double> &data) const
      {
        // STEP 1: Load data and preprocess it.

        // need access to the pressure, viscosity,
        // get velocity
        Tensor<1,dim> velocity;
        for (unsigned int i = 0; i < dim; ++i)
          velocity[i] = solution[this->introspection().component_indices.velocities[i]];

        // get velocity gradient tensor.
        Tensor<2,dim> velocity_gradient;
        for (unsigned int d=0; d<dim; ++d)
          velocity_gradient[d] = gradients[d];

        // Calculate strain rate from velocity gradients
        const SymmetricTensor<2,dim> strain_rate = symmetrize (velocity_gradient);
        const SymmetricTensor<2,dim> deviatoric_strain_rate
          = (this->get_material_model().is_compressible()
             ?
             strain_rate - 1./3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
             :
             strain_rate);

        const double pressure = solution[this->introspection().component_indices.pressure];
        const double temperature = solution[this->introspection().component_indices.temperature];
        const double water_content = solution[this->introspection().component_indices.compositional_fields[water_index]];

        // get the composition of the particle
        std::vector<double> compositions;
        for (unsigned int i = 0; i < this->n_compositional_fields(); ++i)
          {
            const unsigned int solution_component = this->introspection().component_indices.compositional_fields[i];
            compositions.push_back(solution[solution_component]);
          }

        const double dt = this->get_timestep();

        // even in 2d we need 3d strain-rates and velocity gradient tensors. So we make them 3d by
        // adding an extra dimension which is zero.
        SymmetricTensor<2,3> strain_rate_3d;
        strain_rate_3d[0][0] = strain_rate[0][0];
        strain_rate_3d[0][1] = strain_rate[0][1];
        //sym: strain_rate_3d[1][0] = strain_rate[1][0];
        strain_rate_3d[1][1] = strain_rate[1][1];

        if (dim == 3)
          {
            strain_rate_3d[0][2] = strain_rate[0][2];
            strain_rate_3d[1][2] = strain_rate[1][2];
            //sym: strain_rate_3d[2][0] = strain_rate[0][2];
            //sym: strain_rate_3d[2][1] = strain_rate[1][2];
            strain_rate_3d[2][2] = strain_rate[2][2];
          }
        Tensor<2,3> velocity_gradient_3d;
        velocity_gradient_3d[0][0] = velocity_gradient[0][0];
        velocity_gradient_3d[0][1] = velocity_gradient[0][1];
        velocity_gradient_3d[1][0] = velocity_gradient[1][0];
        velocity_gradient_3d[1][1] = velocity_gradient[1][1];
        if (dim == 3)
          {
            velocity_gradient_3d[0][2] = velocity_gradient[0][2];
            velocity_gradient_3d[1][2] = velocity_gradient[1][2];
            velocity_gradient_3d[2][0] = velocity_gradient[2][0];
            velocity_gradient_3d[2][1] = velocity_gradient[2][1];
            velocity_gradient_3d[2][2] = velocity_gradient[2][2];
          }

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {

            /**
            * Now we have loaded all the data and can do the actual computation.
            * The computation consists of two parts. The first part is computing
            * the derivatives for the directions and grain sizes. Then those
            * derivatives are used to advect the particle properties.
            */
            double sum_volume_mineral = 0;
            std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
            derivatives_grains = this->compute_derivatives(data_position,
                                                           data,
                                                           mineral_i,
                                                           strain_rate_3d,
                                                           velocity_gradient_3d,
                                                           position,
                                                           temperature,
                                                           pressure,
                                                           velocity,
                                                           compositions,
                                                           strain_rate,
                                                           deviatoric_strain_rate,
                                                           water_content);

            switch (advection_method)
              {
                case AdvectionMethod::forward_euler:

                  sum_volume_mineral = this->advect_forward_euler(data_position,
                                                                  data,
                                                                  mineral_i,
                                                                  dt,
                                                                  derivatives_grains);

                  break;

                case AdvectionMethod::backward_euler:
                  sum_volume_mineral = this->advect_backward_euler(data_position,
                                                                   data,
                                                                   mineral_i,
                                                                   dt,
                                                                   derivatives_grains);

                  break;
              }

            // normalize the volume fractions back to a total of 1 for each mineral
            double inv_sum_volume_mineral;

            if (cpo_derivative_algorithm == CPODerivativeAlgorithm::drex_2004)
              {
                inv_sum_volume_mineral = 1/sum_volume_mineral;

                Assert(std::isfinite(inv_sum_volume_mineral),
                       ExcMessage("inv_sum_volume_mineral is not finite. sum_volume_enstatite = "
                                  + std::to_string(sum_volume_mineral)));
              }
            else
              {
                inv_sum_volume_mineral = 1;
              }


            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                const double volume_fraction_grains = get_volume_fractions_grains(data_position,data,mineral_i,grain_i)*inv_sum_volume_mineral;
                set_volume_fractions_grains(data_position,data,mineral_i,grain_i,volume_fraction_grains);
                Assert(isfinite(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)),
                       ExcMessage("volume_fractions_grains[mineral_i]" + std::to_string(grain_i) + "] is not finite: "
                                  + std::to_string(get_volume_fractions_grains(data_position,data,mineral_i,grain_i)) + ", inv_sum_volume_mineral = "
                                  + std::to_string(inv_sum_volume_mineral) + "."));

                /**
                * Correct direction rotation matrices numerical error (orthnormality) after integration
                * Follows same method as in matlab version from Thissen (see https://github.com/cthissen/Drex-MATLAB/)
                * of finding the nearest orthonormal matrix using the SVD
                */
                Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(data_position,data,mineral_i,grain_i);
                for (size_t i = 0; i < 3; ++i)
                  {
                    for (size_t j = 0; j < 3; ++j)
                      {
                        Assert(!std::isnan(rotation_matrix[i][j]), ExcMessage("rotation_matrix is nan before orthogonalization."));
                      }
                  }
                
                rotation_matrix = dealii::project_onto_orthogonal_tensors(rotation_matrix);
                for (size_t i = 0; i < 3; ++i)
                  for (size_t j = 0; j < 3; ++j)
                    {
                      // I don't think this should happen with the projection, but D-Rex
                      // does not do the orthogonal projection, but just clamps the values
                      // to 1 and -1.
                      Assert(std::fabs(rotation_matrix[i][j]) <= 1.0,
                             ExcMessage("The rotation_matrix has a entry larger than 1."));

                      Assert(!std::isnan(rotation_matrix[i][j]),
                             ExcMessage("rotation_matrix is nan after orthoganalization: "
                                        + std::to_string(rotation_matrix[i][j])));

                      Assert(abs(rotation_matrix[i][j]) <= 1.0,
                             ExcMessage("3. rotation_matrix[" + std::to_string(i) + "][" + std::to_string(j) +
                                        "] is larger than one: "
                                        + std::to_string(rotation_matrix[i][j]) + " (" + std::to_string(rotation_matrix[i][j]-1.0) + "). rotation_matrix = \n"
                                        + std::to_string(rotation_matrix[0][0]) + " " + std::to_string(rotation_matrix[0][1]) + " " + std::to_string(rotation_matrix[0][2]) + "\n"
                                        + std::to_string(rotation_matrix[1][0]) + " " + std::to_string(rotation_matrix[1][1]) + " " + std::to_string(rotation_matrix[1][2]) + "\n"
                                        + std::to_string(rotation_matrix[2][0]) + " " + std::to_string(rotation_matrix[2][1]) + " " + std::to_string(rotation_matrix[2][2])));
                    }
              }
          }
      }



      template <int dim>
      UpdateTimeFlags
      CrystalPreferredOrientation<dim>::need_update() const
      {
        return update_time_step;
      }



      template <int dim>
      InitializationModeForLateParticles
      CrystalPreferredOrientation<dim>::late_initialization_mode () const
      {
        return InitializationModeForLateParticles::interpolate;
      }



      template <int dim>
      UpdateFlags
      CrystalPreferredOrientation<dim>::get_needed_update_flags () const
      {
        return update_values | update_gradients;
      }



      template <int dim>
      std::vector<std::pair<std::string, unsigned int>>
      CrystalPreferredOrientation<dim>::get_property_information() const
      {
        std::vector<std::pair<std::string,unsigned int>> property_information;

        for (unsigned int mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
          {
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " type",1));
            property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " volume fraction",1));

            for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
              {
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " volume fraction",1));

                for (unsigned int index = 0; index < Tensor<2,3>::n_independent_components; ++index)
                  {
                    property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " rotation_matrix " + std::to_string(index),1));
                  }
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " grain boundary velocity",1));

                for (unsigned int index = 0; index < 4; ++index)
                  {
                    property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " dislocation density " + std::to_string(index),1));
                  }
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Slip system",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Slip rate ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Strain energy ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Surface energy ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Grain status ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Accumulated Strain ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Rx fractions ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Diffusion strain rate ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Dislocation strain rate ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Strain difference ",1));
                property_information.push_back(std::make_pair("cpo mineral " + std::to_string(mineral_i) + " grain " + std::to_string(grain_i) + " Lifetime ",1));
              }
          }

        return property_information;
      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_forward_euler(const unsigned int cpo_index,
                                                             const ArrayView<double> &data,
                                                             const unsigned int mineral_i,
                                                             const double dt,
                                                             const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {

        double sum_volume_fractions = 0; 
        Tensor<2,3> rotation_matrix;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Do the volume fraction of the grain
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite before it is set."));
            double volume_fraction_grains = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            volume_fraction_grains =  volume_fraction_grains + dt * volume_fraction_grains * derivatives.first[grain_i];
            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i, volume_fraction_grains);
            Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                   + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                   + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

            sum_volume_fractions += get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);

            // Do the rotation matrix for this grain
            rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            rotation_matrix += dt * rotation_matrix * derivatives.second[grain_i];
            set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,rotation_matrix);
          }

        Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
        return sum_volume_fractions;

      }



      template <int dim>
      double
      CrystalPreferredOrientation<dim>::advect_backward_euler(const unsigned int cpo_index,
                                                              const ArrayView<double> &data,
                                                              const unsigned int mineral_i,
                                                              const double dt,
                                                              const std::pair<std::vector<double>, std::vector<Tensor<2,3>>> &derivatives) const
      {
        switch (cpo_derivative_algorithm)
          {
            case CPODerivativeAlgorithm::drex_2004:
            {
              double sum_volume_fractions = 0;

              Tensor<2,3> cosine_ref;
              for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                {
                  // Do the volume fraction of the grain
                  double vf_old = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  double vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  //std::cout<<"vf_new = "<<vf_new<<std::endl;
                  Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set."));
                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set. grain_i = "
                                                              + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                              + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                      vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + dt * (vf_new) * derivatives.first[grain_i];
                      set_grain_boundary_velocity(cpo_index,data,mineral_i,grain_i, derivatives.first[grain_i]);
                      Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                             + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                             + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));
                      
                      if (std::fabs(vf_new-vf_old) < property_advection_tolerance)
                        {
                          break;
                        }
                      vf_old = vf_new;
                    }
                  Assert(vf_new > 0, ExcMessage("The grain volume for grain" + std::to_string(grain_i) + " is negative."))
                  set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,vf_new);
                  sum_volume_fractions += vf_new;

                  // Do the rotation matrix for this grain
                  cosine_ref = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
                  Tensor<2,3> cosine_old = cosine_ref;
                  Tensor<2,3> cosine_new = cosine_ref;

                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      cosine_new = cosine_ref + dt * cosine_new * derivatives.second[grain_i];
                      if ((cosine_new-cosine_old).norm() < property_advection_tolerance)
                        {
                          break;
                        }
                      cosine_old = cosine_new;
                    }

                  set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,cosine_new);

                }
              Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
              return sum_volume_fractions;
              break;
            }

            case CPODerivativeAlgorithm::drexpp:
            {
              double sum_of_volumes = 0;
              double sum_volume_fractions = 0;
              Tensor<2,3> cosine_ref;

              for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                {
                  sum_of_volumes += (4./3.) * numbers::PI * std::pow(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)* 0.5,3);
                }

              for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
                {
                  // Do the volume fraction of the grain
                  double vf_old = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  double vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
                  Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set."));
                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      Assert(std::isfinite(vf_new),ExcMessage("vf_new is not finite before it is set. grain_i = "
                                                              + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                                                              + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                      if (this ->get_time() !=0)
                        vf_new = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) + dt * (((4./3.) * numbers::PI * std::pow(vf_new* 0.5,3))/sum_of_volumes) * derivatives.first[grain_i];
                      else
                        vf_new = vf_new;

                      set_strain_difference(cpo_index,data,mineral_i,grain_i,dt * (((4./3.) * numbers::PI * std::pow(vf_new* 0.5,3))/sum_of_volumes) * derivatives.first[grain_i]);
                      Assert(std::isfinite(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)),ExcMessage("volume_fractions[grain_i] is not finite. grain_i = "
                             + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                             + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));

                      Assert(vf_new >= 0,ExcMessage("volume_fractions[grain_i] is less than zero. grain_i = "
                             + std::to_string(grain_i) + ", volume_fractions[grain_i] = " + std::to_string(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i))
                             + ", derivatives.first[grain_i] = " + std::to_string(derivatives.first[grain_i])));
                      if (std::fabs(vf_new-vf_old) < property_advection_tolerance)
                        {
                          break;
                        }
                      vf_old = vf_new;
                     
                    }

                  set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,vf_new);
                  sum_volume_fractions += vf_new;

                  // Do the rotation matrix for this grain
                  cosine_ref = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
                  Tensor<2,3> cosine_old = cosine_ref;
                  Tensor<2,3> cosine_new = cosine_ref;

                  for (size_t iteration = 0; iteration < property_advection_max_iterations; ++iteration)
                    {
                      cosine_new = cosine_ref + dt * cosine_new * derivatives.second[grain_i];
                      if ((cosine_new-cosine_old).norm() < property_advection_tolerance)
                        {
                          break;
                        }
                      cosine_old = cosine_new;
                    }

                  set_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i,cosine_new);

                }
              Assert(sum_volume_fractions != 0, ExcMessage("The sum of all grain volume fractions of a mineral is equal to zero. This should not happen."));
              return sum_volume_fractions;
              break;
            }
            default:
              break;
          }
      }



      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives(const unsigned int cpo_index,
                                                            const ArrayView<double> &data,
                                                            const unsigned int mineral_i,
                                                            const SymmetricTensor<2,3> &strain_rate_3d,
                                                            const Tensor<2,3> &velocity_gradient_tensor,
                                                            const Point<dim> &position,
                                                            const double temperature,
                                                            const double pressure,
                                                            const Tensor<1,dim> &velocity,
                                                            const std::vector<double> &compositions,
                                                            const SymmetricTensor<2,dim> &strain_rate,
                                                            const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                                            const double water_content) const
      {
        std::pair<std::vector<double>, std::vector<Tensor<2,3>>> derivatives;
        switch (cpo_derivative_algorithm)
          {
            case CPODerivativeAlgorithm::spin_tensor:
            {
              return compute_derivatives_spin_tensor(velocity_gradient_tensor);
              break;
            }
            case CPODerivativeAlgorithm::drex_2004:
            {

              const DeformationType deformation_type = determine_deformation_type(deformation_type_selector[mineral_i],
                                                                                  position,
                                                                                  temperature,
                                                                                  pressure,
                                                                                  velocity,
                                                                                  compositions,
                                                                                  strain_rate,
                                                                                  deviatoric_strain_rate,
                                                                                  water_content);

              set_deformation_type(cpo_index,data,mineral_i,static_cast<unsigned int>(deformation_type));

              const std::array<double,4> ref_resolved_shear_stress = reference_resolved_shear_stress_from_deformation_type(deformation_type);

              return compute_derivatives_drex_2004(cpo_index,
                                                   data,
                                                   mineral_i,
                                                   strain_rate_3d,
                                                   velocity_gradient_tensor,
                                                   ref_resolved_shear_stress);
              break;
            }
            case CPODerivativeAlgorithm::drexpp:
            {
              const double pressure = 3e+8;
              const double t =this-> get_time();
              const DeformationType deformation_type = determine_deformation_type(deformation_type_selector[mineral_i],
                                                                                  position,
                                                                                  temperature,
                                                                                  pressure,
                                                                                  velocity,
                                                                                  compositions,
                                                                                  strain_rate,
                                                                                  deviatoric_strain_rate,
                                                                                  water_content);

              set_deformation_type(cpo_index,data,mineral_i,static_cast<unsigned int>(deformation_type));


              const std::array<double,4> ref_resolved_shear_stress = reference_resolved_shear_stress_from_deformation_type(deformation_type);

              //Compute the diffusion strain with the diffusion viscosity

              const double gravity_norm = this -> get_gravity_model().gravity_vector(position).norm();
              const double reference_density = (this->get_adiabatic_conditions().is_initialized())
                                               ?
                                               this->get_adiabatic_conditions().density(position)
                                               :
                                               3300.;

              // The phase index is set to invalid_unsigned_int, because it is only used internally
              // in phase_average_equation_of_state_outputs to loop over all existing phases
              MaterialModel::MaterialUtilities::PhaseFunctionInputs<dim> phase_inputs(temperature,
                                                                                      pressure,
                                                                                      this->get_geometry_model().depth(position),
                                                                                      gravity_norm * reference_density,
                                                                                      numbers::invalid_unsigned_int);

              std::vector<double> phase_function_values(phase_function.n_phase_transitions(), 0.0);

              // Compute value of phase fucntions

              for (unsigned int j = 0; j < phase_function.n_phase_transitions(); j++)
                {
                  phase_inputs.phase_index = j;
                  phase_function_values[j] = phase_function.compute_value(phase_inputs);
                }

              const std::vector<double> volume_fractions = MaterialModel::MaterialUtilities::compute_composition_fractions(compositions);
              // the diffusion_pre_viscosities is the diffusion viscosity without the grainsize

              std::vector<double> diffusion_pre_strain_rates(volume_fractions.size(), std::numeric_limits<double>::quiet_NaN());
              std::vector<double> diffusion_grain_size_exponent(volume_fractions.size(), std::numeric_limits<double>::quiet_NaN());
              std::vector<double> dislocation_strain_rates(volume_fractions.size(), std::numeric_limits<double>::quiet_NaN());
              const double strain_rate_inv = std::max(std::sqrt(std::max(-second_invariant(deviatoric_strain_rate), 0.)), min_strain_rate);

              // now compute the normal viscosity to be able to compute the stress
              // Create the material model inputs and outputs to
              // retrieve the current viscosity.

              MaterialModel::MaterialModelInputs<dim> in = MaterialModel::MaterialModelInputs<dim>(1, compositions.size());
              in.pressure[0] = pressure;
              in.temperature[0] = temperature;
              in.position[0] = position;
              in.strain_rate[0] = strain_rate;
              in.composition[0] = compositions;

              in.requested_properties = MaterialModel::MaterialProperties::viscosity;

              MaterialModel::MaterialModelOutputs<dim> out(1.,
                                                           this -> n_compositional_fields());

              this -> get_material_model().evaluate(in,out);

              // Compressive stress is positive in geosciences applications.
              SymmetricTensor<2, dim> stress = pressure * unit_symmetric_tensor<dim>();

              // Add elastic stresses if existent.
              AssertThrow(this->get_parameters().enable_elasticity == false, ExcMessage("Elasticity not supported when computing the CPO stress"));

              const double eta = out.viscosities[0];

              stress += -2. * eta * deviatoric_strain_rate;

              // Compute the deviatoric stress tensor after elastic stresses were added.
              const SymmetricTensor<2, dim> deviatoric_stress = deviator(stress);

              // Compute the second moment invariant of the deviatoric stress
              // in the same way as the second moment invariant of the deviatoric
              // strain rate is computed in the viscoplastic material model.
              // TODO to check if this is valid for the compressible case.
               SymmetricTensor<2,3> stress_3d;
               stress_3d[0][0] = deviatoric_stress[0][0];
               stress_3d[0][1] = deviatoric_stress[0][1];
               stress_3d[1][1] = deviatoric_stress[1][1];

              if (dim == 3)
              {
                stress_3d[0][2] = deviatoric_stress[0][2];
                stress_3d[1][2] = deviatoric_stress[1][2];
                stress_3d[2][2] = deviatoric_stress[2][2];
              }
              const std::array<double, dim> eigenvalues = dealii::eigenvalues(deviatoric_stress);
              const double differential_stress = eigenvalues[0] - eigenvalues[dim -1];
              
              for (unsigned int composition = 0; composition < volume_fractions.size(); composition++)
                {
                  const MaterialModel::Rheology::DiffusionCreepParameters p_dif = rheology_diff->compute_creep_parameters(composition,
                                                                                  phase_function_values,
                                                                                  phase_function.n_phase_transitions_for_each_composition());

                  /*
                    Power law creep equation
                    viscosity = 0.5 * A^(-1) * d^(m) * exp(( E + P * V ) / ( R * T ))
                    A : prefactor
                    d : grain size
                    m : grain size exponent
                    E : activation energy
                    P : pressure
                    V : activation volume
                    R : gas constant
                    T : temperature
                  */
                  diffusion_pre_strain_rates[composition] = std::exp((-1*(p_dif.activation_energy +
                                                                          pressure*p_dif.activation_volume)/
                                                                      (constants::gas_constant*temperature)));

                  diffusion_grain_size_exponent[composition] = p_dif.grain_size_exponent;

                  const MaterialModel::Rheology::DislocationCreepParameters p_dis = rheology_disl->compute_creep_parameters(composition,
                                                                                    phase_function_values,
                                                                                    phase_function.n_phase_transitions_for_each_composition());
                  dislocation_strain_rates[composition] =  p_dis.prefactor*
                                                           std::exp(-1*((p_dis.activation_energy + pressure * p_dis.activation_volume)/
                                                                        (constants::gas_constant * temperature )));
                }

              return compute_derivatives_drexpp(cpo_index,
                                                data,
                                                mineral_i,
                                                strain_rate_3d,
                                                velocity_gradient_tensor,
                                                ref_resolved_shear_stress,
                                                stress_3d,
                                                deviatoric_strain_rate,
                                                volume_fractions,
                                                diffusion_pre_strain_rates,
                                                diffusion_grain_size_exponent,
                                                dislocation_strain_rates);
              break;
            }
            default:
              AssertThrow(false, ExcMessage("Internal error."));
              break;
          }
        AssertThrow(false, ExcMessage("Internal error."));
        return derivatives;
      }



      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_spin_tensor(const Tensor<2,3> &velocity_gradient_tensor) const
      {
        // dA/dt = W * A, where W is the spin tensor and A is the rotation matrix
        // The spin tensor is defined as W = 0.5 * ( L - L^T ), where L is the velocity gradient tensor.
        const Tensor<2,3> spin_tensor = -0.5 *(velocity_gradient_tensor - dealii::transpose(velocity_gradient_tensor));

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(std::vector<double>(n_grains,0.0), std::vector<Tensor<2,3>>(n_grains, spin_tensor));
      }


      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_drex_2004(const unsigned int cpo_index,
                                                                      const ArrayView<double> &data,
                                                                      const unsigned int mineral_i,
                                                                      const SymmetricTensor<2,3> &strain_rate_3d,
                                                                      const Tensor<2,3> &velocity_gradient_tensor,
                                                                      const std::array<double,4> ref_resolved_shear_stress,
                                                                      const bool prevent_nondimensionalization) const
      {
        // This if statement is only there for the unit test. In normal situations it should always be set to false,
        // because the nondimensionalization should always be done (in this exact way), unless you really know what
        // you are doing.
        double nondimensionalization_value = 1.0;
        if (!prevent_nondimensionalization)
          {
            const std::array< double, 3 > eigenvalues = dealii::eigenvalues(strain_rate_3d);
            nondimensionalization_value = std::max(std::abs(eigenvalues[0]),std::abs(eigenvalues[2]));

            Assert(!std::isnan(nondimensionalization_value), ExcMessage("The second invariant of the strain rate is not a number."));
          }


        // Make the strain-rate and velocity gradient tensor non-dimensional
        // by dividing it through the second invariant
        const Tensor<2,3> strain_rate_nondimensional = nondimensionalization_value != 0 ? strain_rate_3d/nondimensionalization_value : strain_rate_3d;
        const Tensor<2,3> velocity_gradient_tensor_nondimensional = nondimensionalization_value != 0 ? velocity_gradient_tensor/nondimensionalization_value : velocity_gradient_tensor;

        // create output variables
        std::vector<double> deriv_volume_fractions(n_grains);
        std::vector<Tensor<2,3>> deriv_a_cosine_matrices(n_grains);
        std::vector<double> volume_derivative(n_grains);
        std::array<double,4>dislocation_density;
        std::vector<double> def_mech_factor(n_grains);
        std::vector<double> schmid_factor_max(n_grains);
        // create shortcuts
        const std::array<double, 4> &tau = ref_resolved_shear_stress;

        std::vector<double> strain_energy(n_grains);
        double mean_strain_energy = 0;

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Compute the Schmidt tensor for this grain (nu), s is the slip system.
            // We first compute beta_s,nu (equation 5, Kaminski & Ribe, 2001)
            // Then we use the beta to calculate the Schmidt tensor G_{ij} (Eq. 5, Kaminski & Ribe, 2001)
            Tensor<2,3> G;
            Tensor<1,3> w;
            Tensor<1,4> beta({1.0, 1.0, 1.0, 1.0});
            std::array<Tensor<1,3>,4> slip_normal_reference {{Tensor<1,3>({0,1,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),Tensor<1,3>({1,0,0})}};
            std::array<Tensor<1,3>,4> slip_direction_reference {{Tensor<1,3>({1,0,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,0,1})}};

            // these are variables we only need for olivine, but we need them for both
            // within this if block and the next ones
            // Ordered vector where the first entry is the max/weakest and the last entry is the inactive slip system.
            std::array<unsigned int,4> indices {};

            // compute G and beta
            Tensor<1,4> bigI;
            const Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            const Tensor<2,3> rotation_matrix_transposed = transpose(rotation_matrix);
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                const Tensor<2,3> slip_cross_product = outer_product(slip_direction_global,slip_normal_global);
                bigI[slip_system_i] = scalar_product(slip_cross_product,strain_rate_nondimensional);
              }

            if (bigI.norm() < 1e-10)
              {
                // In this case there is no shear, only (possibly) a rotation. So \gamma_y and/or G should be zero.
                // Which is the default value, so do nothing.
              }
            else
              {
                // compute the element wise absolute value of the element wise
                // division of BigI by tau (tau = ref_resolved_shear_stress).
                std::array<double,4> q_abs;
                std::array<double,4> schmid_factor;
                for (unsigned int i = 0; i < 4; ++i)
                  {
                    q_abs[i] = std::abs(bigI[i] / tau[i]);
                    schmid_factor[i] = q_abs[i]* (1/1.833);
                  }

                schmid_factor_max[grain_i] = *std::max_element(schmid_factor.begin(), schmid_factor.end());
                

                // here we find the indices starting at the largest value and ending at the smallest value
                // and assign them to special variables. Because all the variables are absolute values,
                // we can set them to a negative value to ignore them. This should be faster then deleting
                // the element, which would require allocation. (not tested)
                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    indices[slip_system_i] = std::distance(q_abs.begin(),std::max_element(q_abs.begin(), q_abs.end()));
                    q_abs[indices[slip_system_i]] = -1;
                  }

                
                // compute the ordered beta vector, which is the relative slip rates of the active slip systems.
                // Test whether the max element is not equal to zero.
                Assert(bigI[indices[0]] != 0.0, ExcMessage("Internal error: bigI is zero."));
                beta[indices[0]] = 1.0; // max q_abs, weak system (most deformation) "s=1"

                const double ratio = tau[indices[0]]/bigI[indices[0]];
                for (unsigned int slip_system_i = 1; slip_system_i < 4-1; ++slip_system_i)
                  {
                    beta[indices[slip_system_i]] = std::pow(std::abs(ratio * (bigI[indices[slip_system_i]]/tau[indices[slip_system_i]])), stress_exponent);
                  }
                beta[indices.back()] = 0.0;

                // Now compute the crystal rate of deformation tensor.
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    for (unsigned int j = 0; j < 3; ++j)
                      {
                        G[i][j] = 2.0 * (beta[0] * rotation_matrix[0][i] * rotation_matrix[1][j]
                                         + beta[1] * rotation_matrix[0][i] * rotation_matrix[2][j]
                                         + beta[2] * rotation_matrix[2][i] * rotation_matrix[1][j]
                                         + beta[3] * rotation_matrix[2][i] * rotation_matrix[0][j]);
                      }
                  }
              }

            // Now calculate the analytic solution to the deformation minimization problem
            // compute gamma (equation 7, Kaminiski & Ribe, 2001)

            // Top is the numerator and bottom is the denominator in equation 7.
            double top = 0;
            double bottom = 0;
            for (unsigned int i = 0; i < 3; ++i)
              {
                // Following the actual Drex implementation we use i+2, which differs
                // from the EPSL paper, which says gamma_nu depends on i+1
                const unsigned int i_offset = (i==0) ? (i+2) : (i-1);

                top = top - (velocity_gradient_tensor_nondimensional[i][i_offset]-velocity_gradient_tensor_nondimensional[i_offset][i])*(G[i][i_offset]-G[i_offset][i]);
                bottom = bottom - (G[i][i_offset]-G[i_offset][i])*(G[i][i_offset]-G[i_offset][i]);

                for (unsigned int j = 0; j < 3; ++j)
                  {
                    top = top + 2.0 * G[i][j]*velocity_gradient_tensor_nondimensional[i][j];
                    bottom = bottom + 2.0* G[i][j] * G[i][j];
                  }
              }
            // see comment on if all BigI are zero. In that case gamma should be zero.
            const double gamma = (bottom != 0.0) ? top/bottom : 0.0;
            // compute w (equation 8, Kaminiski & Ribe, 2001)
            // w is the Rotation rate vector of the crystallographic axes of grain
            w[0] = 0.5*(velocity_gradient_tensor_nondimensional[2][1]-velocity_gradient_tensor_nondimensional[1][2]) - 0.5*(G[2][1]-G[1][2])*gamma;
            w[1] = 0.5*(velocity_gradient_tensor_nondimensional[0][2]-velocity_gradient_tensor_nondimensional[2][0]) - 0.5*(G[0][2]-G[2][0])*gamma;
            w[2] = 0.5*(velocity_gradient_tensor_nondimensional[1][0]-velocity_gradient_tensor_nondimensional[0][1]) - 0.5*(G[1][0]-G[0][1])*gamma;

            // Compute strain energy for this grain (abbreviated Estr)
            // For olivine: DREX only sums over 1-3. But Christopher Thissen's matlab
            // code (https://github.com/cthissen/Drex-MATLAB) corrected
            // this and writes each term using the indices created when calculating bigI.
            // Note tau = RRSS = (tau_m^s/tau_o), this why we get tau^(p-n)

            dislocation_density = get_dislocation_density(cpo_index,data,mineral_i,grain_i);
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const double rhos = std::pow(tau[indices[slip_system_i]],exponent_p-stress_exponent) *
                                    std::pow(std::abs(gamma*beta[indices[slip_system_i]]),exponent_p/stress_exponent);
                dislocation_density[indices[slip_system_i]] = rhos;
                strain_energy[grain_i] += rhos * std::exp(-nucleation_efficiency * rhos * rhos);

                Assert(isfinite(strain_energy[grain_i]), ExcMessage("strain_energy[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(strain_energy[grain_i])
                                                                    + ", rhos (" + std::to_string(slip_system_i) + ") = " + std::to_string(rhos)
                                                                    + ", nucleation_efficiency = " + std::to_string(nucleation_efficiency) + "."));
              }
            set_dislocation_density(cpo_index,data,mineral_i,grain_i,dislocation_density);

            // compute the derivative of the rotation matrix: \frac{\partial a_{ij}}{\partial t}
            // (Eq. 9, Kaminski & Ribe 2001)
            deriv_a_cosine_matrices[grain_i] = 0;
            const double volume_fraction_grain = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            if (volume_fraction_grain >= threshold_GBS/n_grains)
              {
                deriv_a_cosine_matrices[grain_i] = Utilities::Tensors::levi_civita<3>() * w * nondimensionalization_value;
                // volume averaged strain energy
                mean_strain_energy += volume_fraction_grain * strain_energy[grain_i];

                Assert(isfinite(mean_strain_energy), ExcMessage("mean_strain_energy when adding grain " + std::to_string(grain_i) + " is not finite: " + std::to_string(mean_strain_energy)
                                                                + ", volume_fraction_grain = " + std::to_string(volume_fraction_grain) + "."));
              }
            else
              {
                strain_energy[grain_i] = 0;
              }
          }

        // Change of volume fraction of grains by grain boundary migration
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Different than D-Rex. Here we actually only compute the derivative and do not multiply it with the volume_fractions. We do that when we advect.
            deriv_volume_fractions[grain_i] = get_volume_fraction_mineral(cpo_index,data,mineral_i) * mobility * (mean_strain_energy - strain_energy[grain_i]) * nondimensionalization_value;
            set_grain_boundary_velocity(cpo_index,data,mineral_i,grain_i,deriv_volume_fractions[grain_i]);
            Assert(isfinite(deriv_volume_fractions[grain_i]),
                   ExcMessage("deriv_volume_fractions[" + std::to_string(grain_i) + "] is not finite: "
                              + std::to_string(deriv_volume_fractions[grain_i])));
          }

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(deriv_volume_fractions, deriv_a_cosine_matrices);
      }

      template <int dim>
      void
      CrystalPreferredOrientation<dim>::recrystalize_grains(const unsigned int cpo_index,
                                                            const ArrayView<double> &data,
                                                            const unsigned int mineral_i,
                                                            const std::vector<double> &recrystalized_grain_size,
                                                            const std::vector<double> &recrystalization_fractions,
                                                            const double bulk_piezometer,
                                                            std::vector<double> &strain_energy,
                                                            std::vector<bool> &rx_now) const
      {
        
        std::vector<std::size_t> permutation_vector;
        std::vector<std::size_t> buffer_permutation_vector;
        int buffer_vector_counter = 0;
        double n_rx_cumulative = 0;
        // Creating a vector of indices to track which slots are empty
        for(unsigned int i = 0;  i < n_grains ; ++i)
         {
          if((get_grain_status(cpo_index,data,mineral_i,i) == -1)&&(i < n_grains - n_grains_buffer))
          {
            permutation_vector.push_back(i);
          }
          if(i >= n_grains - n_grains_buffer)
          { 
            buffer_permutation_vector.push_back(i);
          }
         }
        
        std::sort(buffer_permutation_vector.begin(), buffer_permutation_vector.end(),
                  [&](std::size_t grain_i, std::size_t grain_j)
        {
          return get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) < get_volume_fractions_grains(cpo_index,data,mineral_i,grain_j);
        });
        
         for(unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
        {
          const double grain_size = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
          const double volume = (4.0/3.) * numbers::PI * std::pow(grain_size * 0.5 , 3.0);
          const double rx_volume = (4.0/3.) * numbers::PI * std::pow(recrystalized_grain_size[grain_i] * 0.5 , 3.0);
        
          Tensor<2,3> rotation_matrix ;
         int n_recrystalized_grains; 
         double replaced_grain_volume;

         if(volume >= 2. * rx_volume)
         {
            n_recrystalized_grains = (std::floor(recrystalization_fractions[grain_i] * (volume/rx_volume)));
            //std::cout<<" nrx should be = "<<std::floor(recrystalization_fractions[grain_i] * (volume/rx_volume))<<"but is assigned as "<<n_recrystalized_grains<<std::endl;
         }
         else
            n_recrystalized_grains =0;

         set_slip_rate(cpo_index,data,mineral_i,grain_i,n_recrystalized_grains);
         
         n_rx_cumulative += n_recrystalized_grains;
         
         double unrx_portion;
          if(volume != 0.)
            unrx_portion = (volume -(n_recrystalized_grains * rx_volume)/volume)*recrystalization_fractions[grain_i];
          
          set_rx_fraction(cpo_index,data,mineral_i,grain_i,unrx_portion);
          double temp_grains = n_recrystalized_grains;
          set_dif_invariant(cpo_index,data,mineral_i,grain_i,grain_size);
          set_slip_rate(cpo_index,data,mineral_i,grain_i,n_recrystalized_grains);  

          double left_overs = volume - (n_recrystalized_grains * rx_volume);
          double left_over_grain_size = 2.0 * std::pow((left_overs* (3.0/4.0) * (1.0/numbers::PI)),(1.0/3.0));

          if(n_recrystalized_grains >= 1.)
          {

          if(left_over_grain_size < recrystalized_grain_size[grain_i])
            {
              n_recrystalized_grains += -1;
              left_overs = volume - (n_recrystalized_grains * rx_volume);
              left_over_grain_size = 2.0 * std::pow((left_overs* (3.0/4.0) * (1.0/numbers::PI)),(1.0/3.0));
            }  
                        
            set_volume_fractions_grains(cpo_index,data,mineral_i,grain_i,left_over_grain_size);
            set_dis_invariant(cpo_index,data,mineral_i,grain_i,left_over_grain_size);
            if(permutation_vector.size() >= n_recrystalized_grains)
              {
                for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < n_recrystalized_grains  ; ++recrystalize_grain_i)
                   {
                      int random_var = std::rand() % permutation_vector.size();
                      if(get_volume_fractions_grains(cpo_index,data,mineral_i,permutation_vector[random_var]) != 0.)
                        std::cout<<"I am rxxing existing grains outside the buffer"<<std::endl;
                      
                      set_volume_fractions_grains(cpo_index,data,mineral_i,permutation_vector[random_var],recrystalized_grain_size[grain_i]);
                      this->compute_random_rotation_matrix(rotation_matrix);
                      set_rotation_matrix_grains(cpo_index,data,mineral_i,permutation_vector[random_var],rotation_matrix);            
                      set_volume_fractions_grains(cpo_index,data,mineral_i,permutation_vector[random_var],recrystalized_grain_size[grain_i]);
                      set_grain_status(cpo_index,data,mineral_i,permutation_vector[random_var],1);
                      set_strain_accumulated(cpo_index,data,mineral_i,permutation_vector[random_var],0.0);
                      set_lifetime(cpo_index,data,mineral_i,permutation_vector[random_var],0.0);
                      rx_now[permutation_vector[random_var]] = true;
                      set_strain_energy(cpo_index,data,mineral_i,permutation_vector[random_var], 0.0);
                      set_surface_energy(cpo_index,data,mineral_i,permutation_vector[random_var], 0.0);
                      permutation_vector.erase(permutation_vector.begin() + random_var);            
                  }
              }
            else
              {
                 for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < temp_grains  ; ++recrystalize_grain_i)
                   {
                    if(get_volume_fractions_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter + recrystalize_grain_i]) !=0.)
                      {
                        replaced_grain_volume +=(4.0/3.0) * numbers::PI * std::pow(0.5 * get_volume_fractions_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter+recrystalize_grain_i]),3.0);
                        set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter+recrystalize_grain_i],0.);
                      }
                    }

                 if(replaced_grain_volume > 0.0)
                   {
                      set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],2.0 *std::pow((3.0/4.0)*(replaced_grain_volume/numbers::PI),1./3.));
                      rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter]);
                      set_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],rotation_matrix);            
                      set_grain_status(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],2);
                      set_strain_accumulated(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],0.0);
                      set_lifetime(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],0.0);
                      rx_now[buffer_permutation_vector[buffer_vector_counter]] = true;
                      set_strain_energy(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter], 0.0);
                      set_surface_energy(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter], 0.0);
                      buffer_vector_counter++;
                   }
                  for (unsigned int recrystalize_grain_i = 0; recrystalize_grain_i < temp_grains  ; ++recrystalize_grain_i)
                   {
                      set_volume_fractions_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],recrystalized_grain_size[grain_i]);
                      rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter]);
                      set_rotation_matrix_grains(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],rotation_matrix);            
                      set_grain_status(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],3);
                      set_strain_accumulated(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],0.0);
                      set_lifetime(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter],0.0);
                      rx_now[buffer_permutation_vector[buffer_vector_counter]] = true;
                      set_strain_energy(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter], 0.0);
                      set_surface_energy(cpo_index,data,mineral_i,buffer_permutation_vector[buffer_vector_counter], 0.0);
                      buffer_vector_counter++;
                   }
              }  
          }
       }
       std::cout<<"total grains rxxed during this timestep = "<<n_rx_cumulative<<std::endl;
      }

      template <int dim>
      std::pair<std::vector<double>, std::vector<Tensor<2,3>>>
      CrystalPreferredOrientation<dim>::compute_derivatives_drexpp(const unsigned int cpo_index,
                                                                   const ArrayView<double> &data,
                                                                   const unsigned int mineral_i,
                                                                   const SymmetricTensor<2,3> &strain_rate,
                                                                   const Tensor<2,3> &velocity_gradient_tensor,
                                                                   const std::array<double,4> ref_resolved_shear_stress,
                                                                   const SymmetricTensor<2,3> &deviatoric_stress,
                                                                   const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                                                   const std::vector<double> &volume_fractions,
                                                                   const std::vector<double> &diffusion_pre_strain_rates,
                                                                   const std::vector<double> &diffusion_grain_size_exponent,
                                                                   const std::vector<double> &dislocation_strain_rates) const
      {
        // create output variables
        std::vector<double> deriv_volume_fractions(n_grains);
        std::vector<Tensor<2,3>> deriv_a_cosine_matrices(n_grains);
        std::vector<double> volume_derivative(n_grains);
        std::array<double, 4>dislocation_density;

        // create shorcuts
        const std::array<double, 4> &tau = ref_resolved_shear_stress;

        // create local variables
        std::vector<double> strain_energy(n_grains);
        std::vector<double> recrystalized_fractions(n_grains);
        std::vector<double> grain_boundary_sliding_fractions(n_grains);
        std::vector<double> def_mech_factor(n_grains);
        std::vector<Tensor<1,3>> spin_vectors(n_grains);
        std::vector<double> schmid_factor_max(n_grains);
        double recrystalization_increment;
        std::vector<double> recrystalized_grain_volume(n_grains);
        std::vector<std::array<Tensor<2,3>,4>> global_slip_system(n_grains);
        std::vector<double> accumulated_strain(n_grains);
        std::vector<bool> rx_now(n_grains);
        const double t =this-> get_time();
        std::vector<double> strain_accumulated(n_grains);
        std::vector<double> strain_increment(n_grains);
        std::vector<int> lifetime(n_grains);
        std::vector<std::array<double,4>> schmid_factor(n_grains);
        std::vector<double> rho_scale(n_grains);
        std::vector<SymmetricTensor<2,3>> diffusion_strain_rate(n_grains);
        std::vector<SymmetricTensor<2,3>> dislocation_strain_rate(n_grains);
        std::vector<Tensor<2,3>> diffusion_velocity_gradient(n_grains);
        std::vector<Tensor<2,3>> dislocation_velocity_gradient(n_grains);
       
        // Calculation of the differential stress
        const std::array< double, 3 > eigenvalues = dealii::eigenvalues(strain_rate);
        const double nondimensionalization_value = std::max(std::abs(eigenvalues[0]),std::abs(eigenvalues[2]));
        const std::array<double, 3> eigstress = dealii::eigenvalues(deviatoric_stress);
        const double differential_stress = eigstress[0] - eigstress[dim -1];
        
        // Constants -> the values below are for olivine alone (SV: Do I add the citations?)
        const double shear_modulus = 8.0 * std::pow(10.0,10.0);
        const double burgers_vector = 5.0 * std::pow(10.0,-10.0);

        // Rheology
         // Divvying up the strain rate tensor and velocity gradient tensor into diffusion and dislocation creep
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
        {
         if((get_volume_fractions_grains(cpo_index,data, mineral_i, grain_i) > 0.0) && (this->get_time() != 0))
         { 
          /*
           const double diffusion_prefactor = MaterialModel::MaterialUtilities::average_value(volume_fractions, diffusion_pre_strain_rates, MaterialModel::MaterialUtilities::harmonic);
           const double diffusion_exponent = MaterialModel::MaterialUtilities::average_value(volume_fractions,  diffusion_grain_size_exponent, MaterialModel::MaterialUtilities::harmonic);
           const double dislocation_prefactor = MaterialModel::MaterialUtilities::average_value(volume_fractions,  dislocation_strain_rates, MaterialModel::MaterialUtilities::harmonic);
           
           // Calculating diffusion creep strain rate
           diffusion_strain_rate[grain_i] = diffusion_prefactor * deviatoric_stress * std::pow(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i), diffusion_exponent);
           dislocation_strain_rate[grain_i] = strain_rate - diffusion_strain_rate[grain_i];
          
           set_dis_invariant(cpo_index,data,mineral_i,grain_i, std::sqrt(std::max(-second_invariant(dislocation_strain_rate[grain_i]), 0.)));
           set_dif_invariant(cpo_index,data,mineral_i,grain_i, std::sqrt(std::max(-second_invariant(diffusion_strain_rate[grain_i]), 0.)));
           
           // Calculating velocity gradient tensor for different mechansims
           diffusion_velocity_gradient[grain_i] = diffusion_strain_rate[grain_i];
           dislocation_velocity_gradient[grain_i] = velocity_gradient_tensor - diffusion_velocity_gradient[grain_i];
          */
                     
           dislocation_strain_rate[grain_i] = strain_rate;
           dislocation_velocity_gradient[grain_i] = velocity_gradient_tensor;
         }
         else 
         {
           diffusion_strain_rate[grain_i] = 0.;
           dislocation_strain_rate[grain_i] = 0.;
          
           rho_scale[grain_i] = 0.;
           //set_dis_invariant(cpo_index,data,mineral_i,grain_i, 0.);
           //set_dif_invariant(cpo_index,data,mineral_i,grain_i, 0.);
           
           // Calculating velocity gradient tensors 
           diffusion_velocity_gradient[grain_i] = 0.;
           dislocation_velocity_gradient[grain_i] = 0.;
         }
        }
        
        // first compute the amount of slip, G, strain accumulated and dislocation density for n_grains, as long as grain is initialized, i.e grain size is not equal to 0
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // Compute the Schmidt tensor for this grain (nu), s is the slip system.
            // We first compute beta_s,nu (equation 5, Kaminski & Ribe, 2001)
            // Then we use the beta to calculate the Schmidt tensor G_{ij} (Eq. 5, Kaminski & Ribe, 2001)
            //Tensor<2,3> G;
            Tensor<1,4> beta({1.0, 1.0, 1.0, 1.0});
            std::array<Tensor<1,3>,4> slip_normal_reference {{Tensor<1,3>({0,1,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,1,0}),Tensor<1,3>({1,0,0})}};
            std::array<Tensor<1,3>,4> slip_direction_reference {{Tensor<1,3>({1,0,0}),Tensor<1,3>({1,0,0}),Tensor<1,3>({0,0,1}),Tensor<1,3>({0,0,1})}};

            // these are variables we only need for olivine, but we need them for both
            // within this if block and the next ones
            // Ordered vector where the first entry is the max/weakest and the last entry is the inactive slip system.
            std::array<unsigned int,4> indices {};

            // compute G and beta
            Tensor<1,4> bigI;
            const Tensor<2,3> rotation_matrix = get_rotation_matrix_grains(cpo_index,data,mineral_i,grain_i);
            const Tensor<2,3> rotation_matrix_transposed = transpose(rotation_matrix);
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                const Tensor<2,3> slip_cross_product =(outer_product(slip_direction_global,slip_normal_global));
                global_slip_system[grain_i][slip_system_i] = slip_cross_product;
                bigI[slip_system_i] = scalar_product(slip_cross_product,dislocation_strain_rate[grain_i]);
              }

            // The value in this if statement is arbitrary. Has to be further checked out to for a more "realistic value".
            if (bigI.norm() < 1e-30)
              {
                // In this case there is no shear, only (possibly) a rotation. So \gamma_y and/or G should be zero.
                // Which is the default value, so do nothing.
              }
            else
              {

                if(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.0)
                {
                  lifetime[grain_i] = get_lifetime(cpo_index,data,mineral_i,grain_i) + this->get_timestep();
                  set_lifetime(cpo_index,data,mineral_i,grain_i,lifetime[grain_i]);
                }
                else
                {
                  set_lifetime(cpo_index,data,mineral_i,grain_i, 0);
                }

                // compute the element wise abso>lute value of the element wise
                // division of BigI by tau (tau = ref_resolved_shear_stress).
                std::array<double,4> q_abs;

                for (unsigned int i = 0; i < 4; ++i)
                  {
                    q_abs[i] = std::abs(bigI[i] / tau[i]);
                  }

                // here we find the indices starting at the largest value and ending at the smallest value
                // and assign them to special variables. Because all the variables are absolute values,
                // we can set them to a negative value to ignore them. This should be faster then deleting
                // the element, which would require allocation. (not tested)
                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    indices[slip_system_i] = std::distance(q_abs.begin(),std::max_element(q_abs.begin(), q_abs.end()));
                    q_abs[indices[slip_system_i]] = -1;
                  }

                set_dom_slip_system(cpo_index,data,mineral_i,grain_i,tau[indices[0]]);
                
                // compute the ordered beta vector, which is the relative slip rates of the active slip systems.
                // Test whether the max element is not equal to zero.
                Assert(bigI[indices[0]] != 0.0, ExcMessage("Internal error: bigI is zero."));
                beta[indices[0]] = 1.0; // max q_abs, weak system (most deformation) "s=1"

                const double ratio = tau[indices[0]]/bigI[indices[0]];
                for (unsigned int slip_system_i = 1; slip_system_i < 4-1; ++slip_system_i)
                  {
                    beta[indices[slip_system_i]] = std::pow(std::abs(ratio * (bigI[indices[slip_system_i]]/tau[indices[slip_system_i]])), drexpp_stress_exponent[mineral_i]);
                  }
                beta[indices.back()] = 0.0;

                
                // Now compute the crystal rate of deformation tensor. equation 4 of Kaminski&Ribe 2001
                // rotation_matrix_transposed = inverse of rotation matrix
                // (see Engler et al., 2024 book: Intro to Texture analysis chp 2.3.2 The Rotation Matrix)
                // this transform the crystal reference frame to specimen reference frame
                Tensor<2,3> schmidt_tensor;
                for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
                  {
                    const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                    const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                    const Tensor<2,3> slip_cross_product = outer_product(slip_direction_global,slip_normal_global);
                    schmidt_tensor += 2.0 * beta[slip_system_i] * slip_cross_product;
                  }

            // Now calculate the analytic solution to the deformation minimization problem
            // compute gamma (equation 7, Kaminiski & Ribe, 2001)

            // Top is the numerator and bottom is the denominator in equation 7.
            double top = 0;
            double bottom = 0;
            for (unsigned int i = 0; i < 3; ++i)
              {
                // Following the actual Drex implementation we use i+2, which differs
                // from the EPSL paper, which says gamma_nu depends on i+1
                const unsigned int i_offset = (i==0) ? (i+2) : (i-1);

                top = top - (dislocation_velocity_gradient[grain_i][i][i_offset]-dislocation_velocity_gradient[grain_i][i_offset][i])*(schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i]);
                bottom = bottom - (schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i])*(schmidt_tensor[i][i_offset]-schmidt_tensor[i_offset][i]);

                for (unsigned int j = 0; j < 3; ++j)
                  {
                    top = top + 2.0 * schmidt_tensor[i][j]*dislocation_velocity_gradient[grain_i][i][j];
                    bottom = bottom + 2.0* schmidt_tensor[i][j] * schmidt_tensor[i][j];
                  }
              }
            // see comment on if all BigI are zero. In that case gamma should be zero.
            const double gamma = (bottom != 0.0) ? top/bottom : 0.0;
            set_slip_rate(cpo_index,data,mineral_i,grain_i,gamma);
            // compute w (equation 8, Kaminiski & Ribe, 2001)
            // w is the Rotation rate vector of the crystallographic axes of grain

            spin_vectors[grain_i] = Tensor<1,3>
                                    (
            {
              0.5*(dislocation_velocity_gradient[grain_i][2][1]-dislocation_velocity_gradient[grain_i][1][2]) - 0.5*(schmidt_tensor[2][1]-schmidt_tensor[1][2]) *gamma,
              0.5*(dislocation_velocity_gradient[grain_i][0][2]-dislocation_velocity_gradient[grain_i][2][0]) - 0.5*(schmidt_tensor[0][2]-schmidt_tensor[2][0]) *gamma,
              0.5*(dislocation_velocity_gradient[grain_i][1][0]-dislocation_velocity_gradient[grain_i][0][1]) - 0.5*(schmidt_tensor[1][0]-schmidt_tensor[0][1]) *gamma
            });
            
            // Calculate the amount of strain accumulated by dislocation glide. 
            Tensor<2,3> local_strain_rate;
            local_strain_rate = (schmidt_tensor * gamma) - (Utilities::Tensors::levi_civita<3>()*spin_vectors[grain_i]);
            SymmetricTensor<2,3>strrate = symmetrize (local_strain_rate);
            if((lifetime[grain_i] > 0) && this->get_time() !=0)
            {
              strain_increment[grain_i] = (this->get_timestep() * std::sqrt(std::max(-second_invariant(strrate), 0.)));
              strain_accumulated[grain_i] =  get_strain_accumulated(cpo_index,data,mineral_i,grain_i) + strain_increment[grain_i];
            }
            else
            {
              strain_accumulated[grain_i]= 0.;
            }
             set_strain_accumulated(cpo_index,data,mineral_i,grain_i,strain_accumulated[grain_i]);           
            
            // Compute sstrain energy for this grain (abbreviated Estr)
            // For olivine: DREX only sums over 1-3. But Christopher Thissen's matlab
            // code (https://github.com/cthissen/Drex-MATLAB) corrected
            // this and writes each term using the indices created when calculating bigI.
            // Note tau = RRSS = (tau_m^s/tau_o), this why we get tau^(p-n)

          if(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.)
          {        
            for (unsigned int slip_system_i = 0; slip_system_i < 4; ++slip_system_i)
              {
                const Tensor<1,3> slip_normal_global = rotation_matrix_transposed*slip_normal_reference[slip_system_i];
                const Tensor<1,3> slip_direction_global = rotation_matrix_transposed*slip_direction_reference[slip_system_i];
                const Tensor<2,3> slip_cross_product = outer_product(slip_direction_global,slip_normal_global);
                
                const double non_dimensionalization = std::sqrt(std::max(-second_invariant(dislocation_strain_rate[grain_i]), 0.));
                const double e_s = scalar_product(slip_cross_product,strrate);
               
                std::vector<double> ref(volume_fractions.size(), std::numeric_limits<double>::quiet_NaN());
              
                double rho_ref;
                // Calculation of rho_ref
                for (unsigned int composition = 0; composition < volume_fractions.size(); composition++)
                  {
                    ref[composition] = std::pow(non_dimensionalization/dislocation_strain_rates[composition],1./3.5);
                  }
                const double stress_ref = MaterialModel::MaterialUtilities::average_value(volume_fractions, ref, MaterialModel::MaterialUtilities::harmonic);
                rho_ref = std::pow(stress_ref/(0.5 * shear_modulus * burgers_vector),exponent_p);
                
                const double rhos = rho_ref * std::pow(tau[indices[slip_system_i]],exponent_p-stress_exponent) *
                                    std::pow(std::abs(e_s/non_dimensionalization),exponent_p/stress_exponent);

                dislocation_density[slip_system_i] = rhos;
                
                Assert(isfinite(strain_energy[grain_i]), ExcMessage("strain_energy[" + std::to_string(grain_i) + "] is not finite: " + std::to_string(strain_energy[grain_i])
                                                                    + ", rhos (" + std::to_string(slip_system_i) + ") = " + std::to_string(rhos)
                                                                    + ", nucleation_efficiency = " + std::to_string(nucleation_efficiency) + "."));
              }
            set_dislocation_density(cpo_index,data,mineral_i,grain_i,dislocation_density);
          }
          }
            
        }

        double total_volume = 0.;
        double temp_vals = 0.;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
        {
          if(lifetime[grain_i] > 0) 
          {
            const double volume = (4./3.) * numbers::PI * std::pow(0.5 * get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i),3.);
            temp_vals = temp_vals + volume * strain_increment[grain_i];
            total_volume += volume;
          }
        }
      
      
        /*
             Calculation of the recrystallized grain size is done using the piezometer proposed by Van der Waal (1993).
             d_{recrystallized} = A * \sigma^{m}
             where
             A - prefactor
             m - stress exponent

             The values for olivine is taken from Van der Waal (1993)
             The values for pyroxene is taken from Speciale et al (2021)
           */

        double bulk_piezometer;
        std::array<double, 2> A = {{0.015,std::pow(10,3.8)}};
        std::array<double, 2> m = {{-1.33, -1.28}};
        if(t!= 0)
        {
          bulk_piezometer = A[mineral_i] * std::pow(differential_stress/1e6,m[mineral_i]);
        }
        else 
        {
          bulk_piezometer = 0.5;
        }
        for (unsigned int grain_i = 0; grain_i < n_grains; grain_i++)
          {
            recrystalized_grain_volume[grain_i] = bulk_piezometer;
          }
        
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            if((get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.) && (get_strain_accumulated(cpo_index,data,mineral_i,grain_i) >= 0.25))
              {
                recrystalized_fractions[grain_i] = get_rx_fraction(cpo_index,data,mineral_i,grain_i);
                recrystalized_fractions[grain_i] += (avrami_slope_input * (get_strain_accumulated(cpo_index,data,mineral_i,grain_i) - 0.25));
                if (recrystalized_fractions[grain_i] > 1.0)
                  recrystalized_fractions[grain_i] = 1.0;
              }
                else
                {
                  recrystalized_fractions[grain_i] = 0.0;
                }
            set_rx_fraction(cpo_index,data,mineral_i,grain_i,recrystalized_fractions[grain_i]);
          }
        
        this->recrystalize_grains(cpo_index,
                                  data,
                                  mineral_i,
                                  recrystalized_grain_volume,
                                  recrystalized_fractions,
                                  bulk_piezometer,
                                  strain_energy,
                                  rx_now);
        
        double sum_grain;
        double nnz;
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
        {
         sum_grain += get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
        
         if(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.)
          {
            nnz += 1.0;
          }
        }
        const double mean_grain_size = sum_grain/nnz;

        for(unsigned int grain_i = 0; grain_i <n_grains; ++grain_i)
        {
          const std::array<double,4> dd = get_dislocation_density(cpo_index,data,mineral_i,grain_i);
          double ssd;
          for(unsigned int slip_system_i = 0; slip_system_i <4; ++slip_system_i)
          {
            ssd += dd[slip_system_i];
          }
          
          if(get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) > 0.)
          {
            strain_energy[grain_i] = 0.5 *  ssd * burgers_vector* burgers_vector * shear_modulus;
          }
          set_strain_energy(cpo_index,data,mineral_i,grain_i,strain_energy[grain_i]);
        }


        double mean_strain_energy = 0.0;
        double sum_volume = 0. ;

        for(unsigned int grain_i = 0; grain_i<n_grains; ++grain_i)
        {
          if(this-> get_time() != 0)
          {
            const double grain_size = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            const double volume = (4./3.) * numbers::PI * std::pow(0.5 * grain_size,3.);
            sum_volume += volume;
            mean_strain_energy += (volume * strain_energy[grain_i]);
          }
        }

        if(total_volume !=0. )
          mean_strain_energy = mean_strain_energy/total_volume;
        Assert(isfinite(mean_strain_energy), ExcMessage("mean_strain_energy is not finite: " + std::to_string(mean_strain_energy) + "."));
        
        // Change of volume fraction of grains by grain boundary migration
        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            // compute the derivative of the rotation matrix: \frac{\partial a_{ij}}{\partial t}
            // (Eq. 9, Kaminski & Ribe 2001)
            deriv_a_cosine_matrices[grain_i] =  Utilities::Tensors::levi_civita<3>() * spin_vectors[grain_i];
          }

        for (unsigned int grain_i = 0; grain_i < n_grains; ++grain_i)
          {
            double volume_fraction_grain = get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i);
            if ((get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i) != 0.0))
              {
                // Different than D-Rex. Here we actually only compute the derivative and do not multiply it with the volume_fractions. We do that when we advect.
                const double driving_force = (mean_strain_energy - strain_energy[grain_i]) - (interfacial_energy*((2./get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)) - (2./mean_grain_size)));
                set_surface_energy(cpo_index,data,mineral_i,grain_i,(interfacial_energy*((2./get_volume_fractions_grains(cpo_index,data,mineral_i,grain_i)) - (2./mean_grain_size))));
                deriv_volume_fractions[grain_i] = get_volume_fraction_mineral(cpo_index,data,mineral_i) *  drexpp_mobility[mineral_i] * driving_force;
                set_grain_boundary_velocity(cpo_index,data,mineral_i,grain_i,deriv_volume_fractions[grain_i]);
                }
            else
              {
                deriv_volume_fractions[grain_i] = 0.;
                set_grain_boundary_velocity(cpo_index,data,mineral_i,grain_i,deriv_volume_fractions[grain_i]);
              }
          }

        return std::pair<std::vector<double>, std::vector<Tensor<2,3>>>(deriv_volume_fractions, deriv_a_cosine_matrices);
      }

      template<int dim>
      DeformationType
      CrystalPreferredOrientation<dim>::determine_deformation_type(const DeformationTypeSelector deformation_type_selector,
                                                                   const Point<dim> &position,
                                                                   const double temperature,
                                                                   const double pressure,
                                                                   const Tensor<1,dim> &velocity,
                                                                   const std::vector<double> &compositions,
                                                                   const SymmetricTensor<2,dim> &strain_rate,
                                                                   const SymmetricTensor<2,dim> &deviatoric_strain_rate,
                                                                   const double water_content) const
      {
        // Now compute what type of deformation takes place.
        switch (deformation_type_selector)
          {
            case DeformationTypeSelector::passive:
              return DeformationType::passive;
            case DeformationTypeSelector::olivine_a_fabric:
              return DeformationType::olivine_a_fabric;
            case DeformationTypeSelector::olivine_b_fabric:
              return DeformationType::olivine_b_fabric;
            case DeformationTypeSelector::olivine_c_fabric:
              return DeformationType::olivine_c_fabric;
            case DeformationTypeSelector::olivine_d_fabric:
              return DeformationType::olivine_d_fabric;
            case DeformationTypeSelector::olivine_e_fabric:
              return DeformationType::olivine_e_fabric;
            case DeformationTypeSelector::enstatite:
              return DeformationType::enstatite;
            case DeformationTypeSelector::olivine_karato_2008:
              // construct the material model inputs and outputs
              // Since this function is only evaluating one particle,
              // we use 1 for the amount of quadrature points.
              MaterialModel::MaterialModelInputs<dim> material_model_inputs(1,this->n_compositional_fields());
              material_model_inputs.position[0] = position;
              material_model_inputs.temperature[0] = temperature;
              material_model_inputs.pressure[0] = pressure;
              material_model_inputs.velocity[0] = velocity;
              material_model_inputs.composition[0] = compositions;
              material_model_inputs.strain_rate[0] = strain_rate;

              MaterialModel::MaterialModelOutputs<dim> material_model_outputs(1,this->n_compositional_fields());
              this->get_material_model().evaluate(material_model_inputs, material_model_outputs);
              double eta = material_model_outputs.viscosities[0];

              const SymmetricTensor<2,dim> stress = 2*eta*deviatoric_strain_rate +
                                                    pressure * unit_symmetric_tensor<dim>();
              const std::array< double, dim > eigenvalues = dealii::eigenvalues(stress);
              double differential_stress = eigenvalues[0]-eigenvalues[dim-1];
              return determine_deformation_type_karato_2008(differential_stress, water_content);

          }

        AssertThrow(false, ExcMessage("Internal error. Deformation type not implemented."));
        return DeformationType::passive;
      }


      template<int dim>
      DeformationType
      CrystalPreferredOrientation<dim>::determine_deformation_type_karato_2008(const double stress, const double water_content) const
      {
        constexpr double MPa = 1e6;
        constexpr double ec_line_slope = -500./1050.;
        if (stress > (380. - 0.05 * water_content)*MPa)
          {
            if (stress > (625. - 2.5 * water_content)*MPa)
              {
                return DeformationType::olivine_b_fabric;
              }
            else
              {
                return DeformationType::olivine_d_fabric;
              }
          }
        else
          {
            if (stress < (625.0 -2.5 * water_content)*MPa)
              {
                return DeformationType::olivine_a_fabric;
              }
            else
              {
                if (stress < (500.0 + ec_line_slope*-100. + ec_line_slope * water_content)*MPa)
                  {
                    return DeformationType::olivine_e_fabric;
                  }
                else
                  {
                    return DeformationType::olivine_c_fabric;
                  }
              }
          }
      }


      template<int dim>
      std::array<double,4>
      CrystalPreferredOrientation<dim>::reference_resolved_shear_stress_from_deformation_type(DeformationType deformation_type,
          double max_value) const
      {
        std::array<double,4> ref_resolved_shear_stress;
        switch (deformation_type)
          {
            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_a_fabric :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 3;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_b_fabric :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = 2;
              ref_resolved_shear_stress[2] = 1;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // from Kaminski and Ribe, GJI 2004 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_c_fabric :
              ref_resolved_shear_stress[0] = 3;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = 2;
              ref_resolved_shear_stress[3] = 1;
              break;

            // from Kaminski and Ribe, GRL 2002 and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_d_fabric :
              ref_resolved_shear_stress[0] = 1;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = 3;
              ref_resolved_shear_stress[3] = max_value;
              break;

            // Kaminski, Ribe and Browaeys, GJI, 2004 (same as in the matlab code) and
            // Becker et al., 2007 (http://www-udc.ig.utexas.edu/external/becker/preprints/bke07.pdf)
            case DeformationType::olivine_e_fabric :
              ref_resolved_shear_stress[0] = 2;
              ref_resolved_shear_stress[1] = 1;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 3;
              break;

            // from Kaminski and Ribe, GJI 2004.
            // Todo: this one is not used in practice, since there is an optimization in
            // the code. So maybe remove it in the future.
            case DeformationType::enstatite :
              ref_resolved_shear_stress[0] = max_value;
              ref_resolved_shear_stress[1] = max_value;
              ref_resolved_shear_stress[2] = max_value;
              ref_resolved_shear_stress[3] = 1;
              break;

            default:
              AssertThrow(false,
                          ExcMessage("Deformation type enum with number " + std::to_string(static_cast<unsigned int>(deformation_type))
                                     + " was not found."));
              break;
          }
        return ref_resolved_shear_stress;
      }

      template<int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_grains() const
      {
        return n_grains;
      }



      template<int dim>
      unsigned int
      CrystalPreferredOrientation<dim>::get_number_of_minerals() const
      {
        return n_minerals;
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("Crystal Preferred Orientation");
            {
              prm.declare_entry ("Random number seed", "1",
                                 Patterns::Integer (0),
                                 "The seed used to generate random numbers. This will make sure that "
                                 "results are reproducible as long as the problem is run with the "
                                 "same number of MPI processes. It is implemented as final seed = "
                                 "user seed + MPI Rank. ");

              prm.declare_entry ("Number of grains per particle", "50",
                                 Patterns::Integer (1),
                                 "The number of grains of each different mineral "
                                 "each particle contains.");

              prm.declare_entry ("Property advection method", "Backward Euler",
                                 Patterns::Anything(),
                                 "Options: Forward Euler, Backward Euler");

              prm.declare_entry ("Property advection tolerance", "1e-10",
                                 Patterns::Double(0),
                                 "The Backward Euler property advection method involve internal iterations. "
                                 "This option allows for setting a tolerance. When the norm of tensor new - tensor old is "
                                 "smaller than this tolerance, the iteration is stopped.");

              prm.declare_entry ("Property advection max iterations", "100",
                                 Patterns::Integer(0),
                                 "The Backward Euler property advection method involve internal iterations. "
                                 "This option allows for setting the maximum number of iterations. Note that when the iteration "
                                 "is ended by the max iteration amount an assert is thrown.");

              prm.declare_entry ("CPO derivatives algorithm", "Spin tensor",
                                 Patterns::List(Patterns::Anything()),
                                 "Options: Spin tensor, D-Rex 2004, D-Rex++");

              prm.enter_subsection("Initial grains");
              {
                prm.declare_entry("Model name","Uniform grains and random uniform rotations",
                                  Patterns::Anything(),
                                  "The model used to initialize the CPO for all particles. "
                                  "Currently 'Uniform grains and random uniform rotations' is the only valid option.");

                prm.declare_entry ("Minerals", "Olivine: Karato 2008, Enstatite",
                                   Patterns::List(Patterns::Anything()),
                                   "This determines what minerals and fabrics or fabric selectors are used used for the LPO/CPO calculation. "
                                   "The options are Olivine: Passive, A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric, "
                                   "Olivine: E-fabric, Olivine: Karato 2008 or Enstatite. Passive sets all RRSS entries to the maximum. The "
                                   "Karato 2008 selector selects a fabric based on stress and water content as defined in "
                                   "figure 4 of the Karato 2008 review paper (doi: 10.1146/annurev.earth.36.031207.124120).");

                prm.declare_entry ("Volume fractions minerals", "0.7, 0.3",
                                   Patterns::List(Patterns::Double(0)),
                                   "The volume fractions for the different minerals. "
                                   "There need to be the same number of values as there are minerals."
                                   "Note that the currently implemented scheme is incompressible and "
                                   "does not allow chemical interaction or the formation of new phases");
              }
              prm.leave_subsection ();

              prm.enter_subsection("D-Rex 2004");
              {
                prm.declare_entry ("Mobility", "125",
                                   Patterns::Double(0),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Volume fractions minerals", "0.5, 0.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "The volume fraction for the different minerals. "
                                   "There need to be the same amount of values as there are minerals");

                prm.declare_entry ("Stress exponents", "3.5",
                                   Patterns::Double(0),
                                   "This is the power law exponent that characterizes the rheology of the "
                                   "slip systems. It is used in equation 11 of Kaminski et al., 2004.");

                prm.declare_entry ("Exponents p", "1.5",
                                   Patterns::Double(0),
                                   "This is exponent p as defined in equation 11 of Kaminski et al., 2004. ");

                prm.declare_entry ("Nucleation efficiency", "5",
                                   Patterns::Double(0),
                                   "This is the dimensionless nucleation rate as defined in equation 8 of "
                                   "Kaminski et al., 2004. ");

                prm.declare_entry ("Threshold GBS", "0.3",
                                   Patterns::Double(0),
                                   "The Dimensionless Grain Boundary Sliding (GBS) threshold. "
                                   "This is a grain size threshold below which grain deform by GBS and "
                                   "become strain-free grains.");
              }
              prm.leave_subsection();

              prm.enter_subsection("D-Rex++");
              {
                prm.declare_entry ("Number of initial grains","500",
                                   Patterns::List(Patterns::Double(0)),
                                   "Initial no. of grains we want to start the model with." );
                
                prm.declare_entry ("Number of buffer grains","500",
                                   Patterns::List(Patterns::Double(0)),
                                   "Initial no. of grains we want to start the model with." );
                
                prm.declare_entry ("Mobility", "125",
                                   Patterns::List(Patterns::Double(0)),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");
                
                prm.declare_entry ("Interfacial Energy", "0.1",
                                   Patterns::List(Patterns::Double(0)),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Avrami Slope Input", "0.15",
                                   Patterns::Double(0),
                                   "The dimensionless intrinsic grain boundary mobility for both olivine and enstatite.");

                prm.declare_entry ("Volume fractions minerals", "0.5, 0.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "The volume fraction for the different minerals. "
                                   "There need to be the same amount of values as there are minerals");

                prm.declare_entry ("Stress exponents", "3.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is the power law exponent that characterizes the rheology of the "
                                   "slip systems. It is used in equation 11 of Kaminski et al., 2004.");

                prm.declare_entry ("Exponents p", "1.5",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is exponent p as defined in equation 11 of Kaminski et al., 2004. ");

                prm.declare_entry ("Initial grain size", "1e-6",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is intial grain size we choose to prescribe to Drex ++ ");

                prm.declare_entry ("Initialize grain with normal distribution", "false",
                                   Patterns::Bool(),
                                   "Initialize grains grains with a normal distribution instead of a uniform distribution. "
                                   "The value provided by the distribution is cut of between 0 and 1 and all values are normalized "
                                   "between 0 and 1.");

                prm.declare_entry ("Initial grains mean", "0.5",
                                   Patterns::Double(0),
                                   "The mean of the initial grain size.");

                prm.declare_entry ("Initial grains standard deviation", "0.1",
                                   Patterns::Double(0),
                                   "The standard deviation of the initial grain size. "
                                   "Note that values will be cut of and normalized between 0 and 1.");

                prm.declare_entry ("Nucleation efficiency", "5",
                                   Patterns::List(Patterns::Double(0)),
                                   "This is the dimensionless nucleation rate as defined in equation 8 of "
                                   "Kaminski et al., 2004. ");

                prm.declare_entry ("Threshold GBS", "0.3",
                                   Patterns::Double(0),
                                   "The Dimensionless Grain Boundary Sliding (GBS) threshold. "
                                   "This is a grain size threshold below which grain deform by GBS and "
                                   "become strain-free grains.");
              }
              prm.leave_subsection();
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();
      }



      template <int dim>
      void
      CrystalPreferredOrientation<dim>::parse_parameters (ParameterHandler &prm)
      {
        AssertThrow(dim == 3, ExcMessage("CPO computations are currently only supported for 3d models. "
                                         "2d computations will work when this assert is removed, but you will need to make sure that the "
                                         "correct 3d strain-rate and velocity gradient tensors are provided to the algorithm."));

        prm.enter_subsection("Postprocess");
        {
          prm.enter_subsection("Particles");
          {
            prm.enter_subsection("Crystal Preferred Orientation");
            {
              random_number_seed = prm.get_integer ("Random number seed");
              n_grains = prm.get_integer("Number of grains per particle");

              property_advection_tolerance = prm.get_double("Property advection tolerance");
              property_advection_max_iterations = prm.get_integer ("Property advection max iterations");

              const std::string temp_cpo_derivative_algorithm = prm.get("CPO derivatives algorithm");

              if (temp_cpo_derivative_algorithm == "Spin tensor")
                {
                  cpo_derivative_algorithm = CPODerivativeAlgorithm::spin_tensor;
                }
              else if (temp_cpo_derivative_algorithm ==  "D-Rex 2004")
                {
                  cpo_derivative_algorithm = CPODerivativeAlgorithm::drex_2004;
                }
              else if (temp_cpo_derivative_algorithm ==  "D-Rex++")
                {
                  cpo_derivative_algorithm = CPODerivativeAlgorithm::drexpp;
                }
              else if (temp_cpo_derivative_algorithm ==  "D-Rex++")
                {
                  cpo_derivative_algorithm = CPODerivativeAlgorithm::drexpp;
                }
              else
                {
                  AssertThrow(false,
                              ExcMessage("The CPO derivatives algorithm needs to be one of the following: "
                                         "Spin tensor, D-Rex 2004."));
                }

              const std::string temp_advection_method = prm.get("Property advection method");
              if (temp_advection_method == "Forward Euler")
                {
                  advection_method = AdvectionMethod::forward_euler;
                }
              else if (temp_advection_method == "Backward Euler")
                {
                  advection_method = AdvectionMethod::backward_euler;
                }
              else
                {
                  AssertThrow(false, ExcMessage("particle property advection method not found: \"" + temp_advection_method + "\""));
                }

              prm.enter_subsection("Initial grains");
              {
                const std::string model_name = prm.get("Model name");
                AssertThrow(model_name == "Uniform grains and random uniform rotations",
                            ExcMessage("No model named " + model_name + "for CPO particle property initialization. "
                                       + "Only the model \"Uniform grains and random uniform rotations\" is available."));
                const std::vector<std::string> temp_deformation_type_selector = dealii::Utilities::split_string_list(prm.get("Minerals"));
                //prm.enter_subsection("Uniform grains and random uniform rotations");
                //{
                n_minerals = temp_deformation_type_selector.size();
                deformation_type_selector.resize(n_minerals);

                for (size_t mineral_i = 0; mineral_i < n_minerals; ++mineral_i)
                  {
                    if (temp_deformation_type_selector[mineral_i] == "Passive")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::passive;
                      }
                    else if (temp_deformation_type_selector[mineral_i] == "Olivine: Karato 2008")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_karato_2008;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: A-fabric")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_a_fabric;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: B-fabric")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_b_fabric;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: C-fabric")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_c_fabric;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: D-fabric")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_d_fabric;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Olivine: E-fabric")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::olivine_e_fabric;
                      }
                    else if (temp_deformation_type_selector[mineral_i] ==  "Enstatite")
                      {
                        deformation_type_selector[mineral_i] = DeformationTypeSelector::enstatite;
                      }
                    else
                      {
                        AssertThrow(false,
                                    ExcMessage("The fabric needs to be assigned one of the following comma-delimited values: Olivine: Karato 2008, "
                                               "Olivine: A-fabric, Olivine: B-fabric, Olivine: C-fabric, Olivine: D-fabric,"
                                               "Olivine: E-fabric, Enstatite, Passive."));
                      }
                  }

                volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
                double volume_fractions_minerals_sum = 0;
                for (auto fraction : volume_fractions_minerals)
                  {
                    volume_fractions_minerals_sum += fraction;
                  }

                AssertThrow(abs(volume_fractions_minerals_sum-1.0) < 2.0 * std::numeric_limits<double>::epsilon(),
                            ExcMessage("The sum of the CPO volume fractions should be one."));
                //}
                //prm.leave_subsection();
              }
              prm.leave_subsection();

              prm.enter_subsection("D-Rex 2004");
              {
                mobility = prm.get_double("Mobility");
                volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
                stress_exponent = prm.get_double("Stress exponents");
                exponent_p = prm.get_double("Exponents p");
                nucleation_efficiency = prm.get_double("Nucleation efficiency");
                threshold_GBS = prm.get_double("Threshold GBS");
              }
              prm.leave_subsection();

              prm.enter_subsection("D-Rex++");
              {
                n_grains_init = prm.get_double("Number of initial grains");
                n_grains_buffer = prm.get_double("Number of buffer grains");
                drexpp_mobility = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Mobility")));
                volume_fractions_minerals = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Volume fractions minerals")));
                drexpp_stress_exponent = Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Stress exponents")));
                drexpp_exponent_p =  Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Exponents p")));
                avrami_slope_input = prm.get_double("Avrami Slope Input");
                interfacial_energy = prm.get_double("Interfacial Energy");
                initialize_grains_with_normal_distribution = prm.get_bool("Initialize grain with normal distribution");
                initialize_grains_with_normal_distribution_mean= prm.get_double("Initial grains mean");
                initialize_grains_with_normal_distribution_standard_deviation = prm.get_double("Initial grains standard deviation");
                drexpp_nucleation_efficiency =  Utilities::string_to_double(dealii::Utilities::split_string_list(prm.get("Nucleation efficiency")));
                threshold_GBS = prm.get_double("Threshold GBS");
                initial_grain_size = prm.get_double("Initial grain size");
              }
              prm.leave_subsection();
            }
            prm.leave_subsection ();
          }
          prm.leave_subsection ();
        }
        prm.leave_subsection ();


        prm.enter_subsection("Material model");
        {
          prm.enter_subsection ("Visco Plastic");
          {
            // Phase transition parameters
            phase_function.initialize_simulator (this->get_simulator());
            phase_function.parse_parameters (prm);

            // Retrieve the list of composition names
            const std::vector<std::string> list_of_composition_names = this->introspection().get_composition_names();

            // Establish that a background field is required here
            const bool has_background_field = true;

            thermal_diffusivities = Utilities::parse_map_to_double_array (prm.get("Thermal diffusivities"),
                                                                          list_of_composition_names,
                                                                          has_background_field,
                                                                          "Thermal diffusivities");

            define_conductivities = prm.get_bool ("Define thermal conductivities");

            thermal_conductivities = Utilities::parse_map_to_double_array (prm.get("Thermal conductivities"),
                                                                           list_of_composition_names,
                                                                           has_background_field,
                                                                           "Thermal conductivities");

            rheology_diff = std::make_unique<MaterialModel::Rheology::DiffusionCreep<dim>>();
            rheology_diff->initialize_simulator (this->get_simulator());
            rheology_diff->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function.n_phases_for_each_composition()));

            rheology_disl = std::make_unique<MaterialModel::Rheology::DislocationCreep<dim>>();
            rheology_disl->initialize_simulator (this->get_simulator());
            rheology_disl->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function.n_phases_for_each_composition()));

            rheology_vipl = std::make_unique<MaterialModel::Rheology::ViscoPlastic<dim>>();
            rheology_vipl->initialize_simulator (this->get_simulator());
            rheology_vipl->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(phase_function.n_phases_for_each_composition()));
            min_strain_rate = rheology_vipl->min_strain_rate;
          }
          prm.leave_subsection();
        }
        prm.leave_subsection();

      }
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY(CrystalPreferredOrientation,
                                        "crystal preferred orientation",
                                        "WARNING: all the CPO plugins are a work in progress and not ready for production use yet. "
                                        "See https://github.com/geodynamics/aspect/issues/3885 for current status and alternatives. "
                                        "The plugin manages and computes the evolution of Lattice/Crystal Preferred Orientations (LPO/CPO) "
                                        "on particles. Each ASPECT particle can be assigned many grains. Each grain is assigned a size and a orientation "
                                        "matrix. This allows for CPO evolution tracking with polycrystalline kinematic CrystalPreferredOrientation evolution models such "
                                        "as D-Rex (Kaminski and Ribe, 2001; Kaminski et al., 2004).")
    }
  }
}
