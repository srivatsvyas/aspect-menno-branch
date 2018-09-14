/*
  Copyright (C) 2011 - 2017 by the authors of the ASPECT code.

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

#ifndef _aspect_material_model_cpp_h
#define _aspect_material_model_cpp_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

//#if ASPECT_USE_SHARED_LIBS==1
#  include <dlfcn.h>
#  ifdef ASPECT_HAVE_LINK_H
#    include <link.h>
#  endif
#endif

namespace aspect
{
  namespace MaterialModel
  {
    using namespace dealii;

    /**
     * A material model that compiles user-defined C++
     * code at runtime.
     *
     * @ingroup MaterialModels
     */
    template <int dim>
    class CPP : public MaterialModel::Interface<dim>, public ::aspect::SimulatorAccess<dim>
    {
      public:

        virtual bool is_compressible () const;

        virtual double reference_viscosity () const;

        virtual void evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                              MaterialModel::MaterialModelOutputs<dim> &out) const;


        /**
         * @name Functions used in dealing with run-time parameters
         * @{
         */
        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);
        /**
         * @}
         */

      private:
        double ref_visc;
        bool compressible;
        typedef void (*eval_t)(const MaterialModel::MaterialModelInputs<dim> &in,
                               MaterialModel::MaterialModelOutputs<dim> &out,
                               ::aspect::SimulatorAccess<dim> simulator);
        eval_t material_eval;
    };

  }
}

#endif
