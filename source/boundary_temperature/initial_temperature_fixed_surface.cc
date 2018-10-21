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


#include <aspect/boundary_temperature/initial_temperature_fixed_surface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace BoundaryTemperature
  {
// ------------------------------ InitialTemperatureFixedSurface -------------------

    template <int dim>
    double
    InitialTemperatureFixedSurface<dim>::
    boundary_temperature (const types::boundary_id             boundary_id,
                          const Point<dim>                    &location) const
    {
      if (boundary_id == surface_boundary_id)
        return surface_boundary_temperature;
      else
        return this->get_initial_temperature().initial_temperature(location);
    }


    template <int dim>
    double
    InitialTemperatureFixedSurface<dim>::
    minimal_temperature (const std::set<types::boundary_id> &) const
    {
      return min_temperature;
    }



    template <int dim>
    double
    InitialTemperatureFixedSurface<dim>::
    maximal_temperature (const std::set<types::boundary_id> &) const
    {
      return max_temperature;
    }



    template <int dim>
    void
    InitialTemperatureFixedSurface<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary temperature model");
      {
        prm.enter_subsection("Initial temperature fixed surface");
        {
          prm.declare_entry ("Surface boundary indicator", "5",
                             Patterns::Anything(),
                             "Surface boundary indicator.");
          prm.declare_entry ("Surface boundary temperature", "273.15",
                             Patterns::Double (),
                             "Minimal temperature. Units: K.");
          prm.declare_entry ("Minimal temperature", "0",
                             Patterns::Double (),
                             "Minimal temperature. Units: K.");
          prm.declare_entry ("Maximal temperature", "3773",
                             Patterns::Double (),
                             "Maximal temperature. Units: K.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    InitialTemperatureFixedSurface<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary temperature model");
      {
        prm.enter_subsection("Initial temperature fixed surface");
        {
          //std::cout << "Flag pp 1" << std::endl;
          const GeometryModel::Interface<dim> &geometry_model = this->get_geometry_model();
          // std::cout << "Flag pp 2" << std::endl;
          surface_boundary_id = geometry_model.translate_symbolic_boundary_name_to_id(prm.get("Surface boundary indicator"));
          //std::cout << "Flag pp 3" << std::endl;
          surface_boundary_temperature = prm.get_double ("Surface boundary temperature");
          min_temperature = prm.get_double ("Minimal temperature");
          max_temperature = prm.get_double ("Maximal temperature");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTemperature
  {
    ASPECT_REGISTER_BOUNDARY_TEMPERATURE_MODEL(InitialTemperatureFixedSurface,
                                               "initial temperature fixed surface",
                                               "A model in which the temperature at the boundary "
                                               "is chosen to be the same as given in the initial "
                                               "conditions."
                                               "\n\n"
                                               "Because this class simply takes what the initial "
                                               "temperature had described, this class can not "
                                               "know certain pieces of information such as the "
                                               "minimal and maximal temperature on the boundary. "
                                               "For operations that require this, for example in "
                                               "postprocessing, this boundary temperature model "
                                               "must therefore be told what the minimal and "
                                               "maximal values on the boundary are. This is done "
                                               "using parameters set in section ``Boundary temperature model/Initial temperature''.")
  }
}
