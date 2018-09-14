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


#include <aspect/material_model/cpp.h>


namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    bool
    CPP<dim>::
    is_compressible () const
    {
      return compressible;
    }

    template <int dim>
    double
    CPP<dim>::
    reference_viscosity () const
    {
      return ref_visc;
    }

    template <int dim>
    void
    CPP<dim>::
    evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
             MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      material_eval(in, out, this->get_simulator());
    }


    NonlinearDependence::Dependence
    str2dep (const std::string depstr)
    {

      if (depstr == "none")
        {
          return NonlinearDependence::none;
        }
      else if (depstr == "compositional_fields")
        {
          return NonlinearDependence::compositional_fields;
        }
      else if (depstr == "pressure")
        {
          return NonlinearDependence::pressure;
        }
      else if (depstr == "strain_rate")
        {
          return NonlinearDependence::strain_rate;
        }
      else if (depstr == "temperature")
        {
          return NonlinearDependence::temperature;
        }
      else
        {
          AssertThrow(false, ExcMessage("Nonlinear dependencies must be one of "
                                        "non|compositional_fields|pressure|strain_rate|temperature."));
        }
    }


    void
    assert_valid_deps (std::vector<std::string> list, std::string param_name)
    {

      AssertThrow(list.size() == 1
                  || (std::find(list.begin(),list.end(),"none") == list.end()),
                  ExcMessage("The property 'none' in the parameter '"+param_name+"' "
                             "cannot be combined with any other options."));
    }

    int
    execute (std::string cmd, bool assert_success)
    {
      AssertThrow(system((char *)0) != 0,
                  ExcMessage("The 'CPP' material model requires a command-processor, "
                             "which appears to be unavailable on this system."));

      const int error = system(cmd.c_str());
      if (assert_success && error != 0)
        {
          std::string err_str = (error<0 ? "-" : "") +
                                Utilities::int_to_string(std::abs(error));

          AssertThrow(false, ExcMessage("Command <" +
                                        cmd +
                                        "> failed with error code: " +
                                        err_str +
                                        ". Terminating process."));
        }
      return error;
    }

    int
    execute (std::string cmd)
    {
      execute(cmd, true);
      return 0;
    }

    void
    generate_cmakelists (const std::string cdir)
    {
      std::ofstream cmfile;
      cmfile.open((cdir + "CMakeLists.txt").c_str(), std::ios_base::out);
      cmfile << "CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)\n"
             << "\n"
             << "FIND_PACKAGE(Aspect QUIET HINTS "<<ASPECT_SOURCE_DIR<<" ${Aspect_DIR} $ENV{ASPECT_DIR})\n"
             << "\n"
             << "IF (NOT Aspect_FOUND)\n"
             << "  MESSAGE(FATAL_ERROR \"\\n\"\n"
             << "    \"Could not find a valid ASPECT build/installation directory. \"\n"
             << "    \"Please specify the directory where you are building ASPECT by passing\\n\"\n"
             << "    \"   -D Aspect_DIR=<path to ASPECT>\\n\"\n"
             << "    \"to cmake or by setting the environment variable ASPECT_DIR in your shell \"\n"
             << "    \"before calling cmake. See the section 'How to write a plugin' in the \"\n"
             << "    \"manual for more information.\")\n"
             << "ENDIF ()\n"
             << "\n"
             << "DEAL_II_INITIALIZE_CACHED_VARIABLES()\n"
             << "\n"
             << "SET(TARGETmaterial \"material\")\n"
             << "PROJECT(${TARGETmaterial})\n"
             << "\n"
             << "ADD_LIBRARY(${TARGETmaterial} SHARED material.cc)\n"
             << "ASPECT_SETUP_PLUGIN(${TARGETmaterial})\n";
      cmfile.close();
    }

    void
    generate_src (const unsigned int dim,
                  const std::vector<std::string> user_includes,
                  const std::string user_variable_defs,
                  const std::string user_update_code,
                  const std::string user_viscosity_function,
                  const std::string user_density_function,
                  const std::string user_thermal_k_function,
                  const std::string user_thermal_exp_function,
                  const std::string user_specific_heat_function,
                  const std::string user_compressibility_function,
                  const std::string user_entropy_derivative_p_function,
                  const std::string user_entropy_derivative_t_function,
                  const std::string user_reaction_function,
                  const std::string fname)
    {
      const std::string dimstr  = Utilities::int_to_string(dim);
      const std::string argdefs = "const Point<"+dimstr+"> &position,\n"
                                  "const double &temperature,\n"
                                  "const double &pressure,\n"
                                  "const Tensor<1,"+dimstr+"> &pressure_gradient,\n"
                                  "const std::vector<double> &composition,\n"
                                  "const SymmetricTensor<2,"+dimstr+"> &strain_rate,\n"
                                  "::aspect::SimulatorAccess<"+dimstr+"> &simulator";
      const std::string args    = "in.position[_i],\nin.temperature[_i],\nin.pressure[_i],\nin.pressure_gradient[_i],\n"
                                  "in.composition[_i],\nin.strain_rate[_i],\nsimulator";

      std::ofstream srcfile;
      srcfile.open((fname).c_str(), std::ios_base::out);
      for (unsigned int i = 0; i<user_includes.size(); ++i)
        srcfile << "#include <" << user_includes[i] << ">\n";
      srcfile   << "#include <aspect/material_model/interface.h>\n"
                << "#include <aspect/simulator_access.h>\n"
                << "namespace aspect {\n"
                << "  namespace MaterialModel {\n"
                << "    using namespace dealii;\n"
                << "    class Local_\n"
                << "      {\n"
                << "        public:\n"
                << "          const unsigned int dim = "<<dim<<";\n"
                << "\n"    << user_variable_defs << ";\n"
                << "          void update(" << argdefs << ") {\n" << user_update_code << ";\n}\n"
                << "          double viscosity (" << argdefs << ") {\n" << user_viscosity_function << ";\n}\n"
                << "          double density (" << argdefs << ") {\n" << user_density_function << ";\n}\n"
                << "          double thermal_k (" << argdefs << ") {\n" << user_thermal_k_function << ";\n}\n"
                << "          double thermal_exp (" << argdefs << ") {\n" << user_thermal_exp_function << ";\n}\n"
                << "          double specific_heat (" << argdefs << ") {\n" << user_specific_heat_function << ";\n}\n"
                << "          double compressibility (" << argdefs << ") {\n" << user_compressibility_function << ";\n}\n"
                << "          double entropy_derivative_p (" << argdefs << ") {\n" << user_entropy_derivative_p_function << ";\n}\n"
                << "          double entropy_derivative_t (" << argdefs << ") {\n" << user_entropy_derivative_t_function << ";\n}\n"
                << "          double reaction (const int c,\n" << argdefs << ") {\n" << user_reaction_function << ";\n}\n"
                << "      } local;\n"
                << "\n"
                << "    extern \"C\" void\n"
                << "    material_eval(const MaterialModel::MaterialModelInputs<"<<dim<<"> &in,\n"
                << "                  MaterialModel::MaterialModelOutputs<"<<dim<<"> &out,\n"
                << "                  ::aspect::SimulatorAccess<"<<dim<<"> &simulator)\n"
                << "    {\n"
                << "\n"
                << "        for (unsigned int _i=0; _i<in.position.size(); ++_i)\n"
                << "          {\n"
                << "            local.update(" << args << ");\n";
      if (user_viscosity_function != "")
        srcfile << "            out.viscosities[_i] = local.viscosity (" << args << ");\n";
      if (user_density_function != "")
        srcfile << "            out.densities[_i] = local.density (" << args << ");\n";
      if (user_thermal_k_function != "")
        srcfile << "            out.thermal_conductivities[_i] = local.thermal_k (" << args << ");\n";
      if (user_thermal_exp_function != "")
        srcfile << "            out.thermal_expansion_coefficients[_i] = local.thermal_exp (" << args << ");\n";
      if (user_specific_heat_function != "")
        srcfile << "            out.specific_heat[_i] = local.specific_heat (" << args << ");\n";
      if (user_compressibility_function != "")
        srcfile << "            out.compressibilities[_i] = local.compressibility (" << args << ");\n";
      if (user_entropy_derivative_p_function != "")
        srcfile << "            out.entropy_derivative_pressure[_i] = local.entropy_derivative_p (" << args << ");\n";
      if (user_entropy_derivative_t_function != "")
        srcfile << "            out.entropy_derivative_temperature[_i] = local.entropy_derivative_t (" << args << ");\n";
      if (user_reaction_function != "")
        srcfile << "            for (unsigned int c=0; c<in.composition[_i].size(); ++c)\n"
                << "              out.reaction_terms[_i][c] = local.reaction (c, " << args << ");\n";
      srcfile   << "        }\n"
                << "    }\n"
                << "  }\n"
                << "}\n";
      srcfile.close();

      if (execute("which astyle >/dev/null", false) == 0)
        if (execute("[ -f '" ASPECT_SOURCE_DIR "/doc/astyle.rc' ]", false) == 0)
          execute("astyle --options=" ASPECT_SOURCE_DIR "/doc/astyle.rc '"+fname+"'", false);
        else
          execute("astyle '"+fname+"'", false);
      else if (execute("which astyle2.04 >/dev/null", false) == 0)
        if (execute("[ -f '" ASPECT_SOURCE_DIR "/doc/astyle.rc' ]", false) == 0)
          execute("astyle2.04 --options=" ASPECT_SOURCE_DIR "/doc/astyle.rc '"+fname+"'", false);
        else
          execute("astyle2.04 '"+fname+"'", false);
      else if (execute("which indent >/dev/null", false) == 0)
        execute("indent '"+fname+"'", false);
    }


    template <int dim>
    void
    CPP<dim>::declare_parameters (ParameterHandler &prm)
    {
      const std::string pattern_of_deps
        = "none|compositional_fields|pressure|strain_rate|temperature";
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("C plus plus");
        {
          prm.declare_entry ("Is compressible", "false",
                             Patterns::Bool (),
                             "Whether or not the model is compressible.");
          prm.declare_entry ("Reference viscosity", "5e24",
                             Patterns::Double (0),
                             "The value of the reference viscosity $\\eta$. Units: $kg/m/s$.");
          prm.declare_entry ("Libraries", "",
                             Patterns::List (Patterns::Anything()),
                             "List additional library headers to include in material model. "
                             "For example: `set Libraries =  stdio.h, math.h`");
          prm.declare_entry ("Global variable definitions", "",
                             Patterns::Anything (),
                             "Define variables accessible to all material functions.");
          prm.declare_entry ("Update function", "",
                             Patterns::Anything (),
                             "Contents of a function that can modify global variables "
                             "defined in 'Material model/C plus plus/Global variable "
                             "definitions'. It is invoked before material properties "
                             "are calculated in a new location."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Viscosity function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating viscosity. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Density function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating density. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Thermal conductivity function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating thermal conductivity. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Thermal expansivity function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating thermal expansivity. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Specific heat function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating specific heat. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Compressibility function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating compressibility. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Entropy derivative pressure function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating entropy derivative with respect to pressure. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Entropy derivative temperature function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating entropy derivative with respect to temperature. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("Reaction function", "",
                             Patterns::Anything (),
                             "Contents of a function for calculating reaction terms. It has access to "
                             "all variables defined in 'Material model/C plus plus/Global variable "
                             "definitions'. It receives an integer called `c' that denotes the particular "
                             "compositional field for which to calculate a reaction rate."
                             "\n\nThe state variables `position', `temperature', `pressure', "
                             "`pressure_gradient', `velocity', `composition', `strain_rate' are "
                             "available within this function, as well as a SimulatorAccess object "
                             "called `simulator'.");
          prm.declare_entry ("List of viscosity dependencies", "none",
                             Patterns::MultipleSelection(pattern_of_deps),
                             "A comma separated list of nonlinear dependencies for viscosity. "
                             "The following options are available:\n\n"
                             +
                             pattern_of_deps);
          prm.declare_entry ("List of density dependencies", "none",
                             Patterns::MultipleSelection(pattern_of_deps),
                             "A comma separated list of nonlinear dependencies for density. "
                             "The following options are available:\n\n"
                             +
                             pattern_of_deps);
          prm.declare_entry ("List of compressibility dependencies", "none",
                             Patterns::MultipleSelection(pattern_of_deps),
                             "A comma separated list of nonlinear dependencies for compressibility. "
                             "The following options are available:\n\n"
                             +
                             pattern_of_deps);
          prm.declare_entry ("List of specific heat dependencies", "none",
                             Patterns::MultipleSelection(pattern_of_deps),
                             "A comma separated list of nonlinear dependencies for specific heat. "
                             "The following options are available:\n\n"
                             +
                             pattern_of_deps);
          prm.declare_entry ("List of thermal conductivity dependencies", "none",
                             Patterns::MultipleSelection(pattern_of_deps),
                             "A comma separated list of nonlinear dependencies for thermal conductivity. "
                             "The following options are available:\n\n"
                             +
                             pattern_of_deps);

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    CPP<dim>::parse_parameters (ParameterHandler &prm)
    {
#if ASPECT_USE_SHARED_LIBS==1
      prm.enter_subsection ("Material model");
      {
        prm.enter_subsection ("C plus plus");
        {
          compressible               = prm.get_bool ("Is compressible");
          ref_visc                   = prm.get_double ("Reference viscosity");

          const std::vector<std::string> user_includes          = Utilities::split_string_list(prm.get ("Libraries"));
          const std::string user_variable_defs                  = prm.get ("Global variable definitions");
          const std::string user_update_code                    = prm.get ("Update function");
          const std::string user_visc_function                  = prm.get ("Viscosity function");
          const std::string user_density_function               = prm.get ("Density function");
          const std::string user_thermal_k_function             = prm.get ("Thermal conductivity function");
          const std::string user_thermal_exp_function           = prm.get ("Thermal expansivity function");
          const std::string user_specific_heat_function         = prm.get ("Specific heat function");
          const std::string user_compressibility_function       = prm.get ("Compressibility function");
          const std::string user_entropy_derivative_p_function  = prm.get ("Entropy derivative pressure function");
          const std::string user_entropy_derivative_t_function  = prm.get ("Entropy derivative temperature function");
          const std::string user_reaction_function              = prm.get ("Reaction function");

          const std::vector<std::string> visc_deps =
            Utilities::split_string_list (prm.get ("List of viscosity dependencies"));
          const std::vector<std::string> density_deps =
            Utilities::split_string_list (prm.get ("List of density dependencies"));
          const std::vector<std::string> compressibility_deps =
            Utilities::split_string_list (prm.get ("List of compressibility dependencies"));
          const std::vector<std::string> specific_heat_deps =
            Utilities::split_string_list (prm.get ("List of specific heat dependencies"));
          const std::vector<std::string> thermal_conductivity_deps =
            Utilities::split_string_list (prm.get ("List of thermal conductivity dependencies"));

          assert_valid_deps (visc_deps,
                             "Material model/C plus plus/List of viscosity dependencies");
          assert_valid_deps (density_deps,
                             "Material model/C plus plus/List of density dependencies");
          assert_valid_deps (compressibility_deps,
                             "Material model/C plus plus/List of compressibility dependencies");
          assert_valid_deps (specific_heat_deps,
                             "Material model/C plus plus/List of specific heat dependencies");
          assert_valid_deps (thermal_conductivity_deps,
                             "Material model/C plus plus/List of thermal conductivity dependencies");

          // Declare dependencies on solution variables
          this->model_dependence.viscosity              = NonlinearDependence::uninitialized;
          this->model_dependence.density                = NonlinearDependence::uninitialized;
          this->model_dependence.compressibility        = NonlinearDependence::uninitialized;
          this->model_dependence.specific_heat          = NonlinearDependence::uninitialized;
          this->model_dependence.thermal_conductivity   = NonlinearDependence::uninitialized;

          for (unsigned int i=0; i<visc_deps.size(); ++i)
            this->model_dependence.viscosity |= str2dep(visc_deps[i]);
          for (unsigned int i=0; i<density_deps.size(); ++i)
            this->model_dependence.density |= str2dep(density_deps[i]);
          for (unsigned int i=0; i<compressibility_deps.size(); ++i)
            this->model_dependence.compressibility |= str2dep(compressibility_deps[i]);
          for (unsigned int i=0; i<specific_heat_deps.size(); ++i)
            this->model_dependence.specific_heat |= str2dep(specific_heat_deps[i]);
          for (unsigned int i=0; i<thermal_conductivity_deps.size(); ++i)
            this->model_dependence.thermal_conductivity |= str2dep(thermal_conductivity_deps[i]);

          std::string cdir = this->get_output_directory() + "cpp_material/";
          std::string sfname = cdir + "material.cc";
          std::string ofname = cdir + "libmaterial.so";

          if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            {
              execute ("mkdir -p " + cdir);

              generate_src (dim,
                            user_includes,
                            user_variable_defs,
                            user_update_code,
                            user_visc_function,
                            user_density_function,
                            user_thermal_k_function,
                            user_thermal_exp_function,
                            user_specific_heat_function,
                            user_compressibility_function,
                            user_entropy_derivative_p_function,
                            user_entropy_derivative_t_function,
                            user_reaction_function,
                            sfname);

              generate_cmakelists (cdir);

              std::cout << "Compiling material model..." << std::endl;
              execute ("cd '" + cdir + "' && cmake . > build.log 2>&1 && make >> build.log 2>&1");

              std::cout << "Loading cpp material model..." << std::endl;
            }

          MPI_Barrier(MPI_COMM_WORLD);

          void *handle = dlopen (ofname.c_str(), RTLD_LAZY);
          AssertThrow (handle != nullptr,
                       ExcMessage (std::string("Could not successfully load shared library <")
                                   + ofname + ">. The operating system reports "
                                   + "that the error is this: <"
                                   + dlerror() + ">."));
          material_eval = (eval_t) dlsym(handle, "material_eval");

          if (dealii::Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            std::cout << std::endl;
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
#else
      AssertThrow(false, ExcMessage("The 'cpp' material model requires "
                                    "that Aspect can load shared libraries. "
                                    "Please check your build configuration."));
#endif
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(CPP,
                                   "cpp",
                                   "A material model whose evaluate function is dynamically "
                                   "compiled from code defined in section 'Material model/C "
                                   "plus plus' of the parameter file.")
  }
}
