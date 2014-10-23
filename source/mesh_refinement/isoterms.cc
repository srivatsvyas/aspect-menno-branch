/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


//#include <aspect/simulator.h>
#include <aspect/mesh_refinement/isoterms.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/derivative_approximation.h>
//#include <math.h>
//#include <aspect/simulator_access.h>
//#include <aspect/utilities.h>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    void
    Isoterms<dim>::tag_additional_cells () const
    { 
      if(this->get_dof_handler().n_dofs() != 0){
	
      /// Gather all information on the state of the system ///
      LinearAlgebra::BlockVector vec_distributed (this->introspection().index_sets.system_partitioning,
                                                  this->get_mpi_communicator());

      const Quadrature<dim> quadrature(this->get_fe().base_element(this->introspection().base_elements.temperature).get_unit_support_points());
      std::vector<types::global_dof_index> local_dof_indices (this->get_fe().dofs_per_cell);
      FEValues<dim> fe_values (this->get_mapping(),
                               this->get_fe(),
                               quadrature,
                               update_quadrature_points | update_values);
      std::vector<double> pressure_values(quadrature.size());
      std::vector<double> temperature_values(quadrature.size());

      /// the values of the compositional fields are stored as blockvectors for each field
      /// we have to extract them in this structure
      std::vector<std::vector<double> > prelim_composition_values (this->n_compositional_fields(),
                                                                   std::vector<double> (quadrature.size()));
      std::vector<std::vector<double> > composition_values (quadrature.size(),
                                                            std::vector<double> (this->n_compositional_fields()));

      typename MaterialModel::Interface<dim>::MaterialModelInputs in(quadrature.size(), this->n_compositional_fields());
      typename MaterialModel::Interface<dim>::MaterialModelOutputs out(quadrature.size(), this->n_compositional_fields());

      typename DoFHandler<dim>::active_cell_iterator
      cell = this->get_dof_handler().begin_active(),
      endc = this->get_dof_handler().end();
      for (; cell!=endc; ++cell)
        if (cell->is_locally_owned())
          {
              bool coarsen = false;
              bool clear_refine = false;
	      bool refine = false;
	      bool clear_coarsen = false;
	       fe_values.reinit(cell);
            fe_values[this->introspection().extractors.pressure].get_function_values (this->get_solution(),
                                                                                      pressure_values);
            fe_values[this->introspection().extractors.temperature].get_function_values (this->get_solution(),
                                                                                         temperature_values);
            for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
              fe_values[this->introspection().extractors.compositional_fields[c]].get_function_values (this->get_solution(),
                  prelim_composition_values[c]);

            cell->get_dof_indices (local_dof_indices);
            in.position = fe_values.get_quadrature_points();
            in.strain_rate.resize(0);// we are not reading the viscosity
	   for (unsigned int i=0; i<quadrature.size(); ++i)
              {
                in.temperature[i] = temperature_values[i];
                in.pressure[i] = pressure_values[i];
		for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                  in.composition[i][c] = prelim_composition_values[c][i];
	      }
	      this->get_material_model().evaluate(in, out);
	   
	   /// all data has been gethered, now do something with it ///
	      
	   std::vector<std::vector<int> > isoterms_i(n_isoterms_outer.size(),std::vector<int>(2));
	   /// isoterms_i[0] is the minimum, and isoterms_i[1] is the maximum cell level.
	   
	   /// convert max,max-1,max-2 etc and min, min-1,min-2 to numbers in integer
	   for ( unsigned int it1 = 0; it1 != isoterms_i.size(); ++it1 )
		{
		if(isoterms[it1][0].compare(0,3,"max")==0){ 
	      if(isoterms[it1][0].compare(0,4,"max-")==0){
		std::vector<std::string> tmpNumber = Utilities::split_string_list(isoterms[it1][0],'-');
		isoterms_i[it1][0]=maximum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
	      }else{
		isoterms_i[it1][0]=maximum_refinement_level;
	      }
	    }else if(isoterms[it1][0].compare(0,3,"min")==0){
	       if(isoterms[it1][0].compare(0,4,"min+")==0){
		std::vector<std::string> tmpNumber = Utilities::split_string_list(isoterms[it1][0],'+');
		isoterms_i[it1][0]=minimum_refinement_level+Utilities::string_to_int(tmpNumber[1]);
	      }else{
		isoterms_i[it1][0]=minimum_refinement_level;
	      }
	    }else if(!isoterms[it1][0].empty() && isoterms[it1][0].find_first_not_of("0123456789") == std::string::npos){
	      isoterms_i[it1][0]=Utilities::string_to_int(isoterms[it1][0]);
	    }else{
	      
	      AssertThrow (true,
                         ExcMessage ("Not able to read input at inputline "
				     +
				     Utilities::int_to_string(it1)
				     +
				     ": "  
				     +
				     isoterms[it1][0] 
				     )
			  );
	      
	     //std::cout << "can't convert max or min at point 0: " << isoterms[it1][0] << std::endl; 
	    }
	    
	    if(isoterms[it1][1].compare(0,3,"max")==0){ 
	      if(isoterms[it1][1].compare(0,4,"max-")==0){
		std::vector<std::string> tmpNumber = Utilities::split_string_list(isoterms[it1][1],'-');
		isoterms_i[it1][1]=maximum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
	      }else{
		isoterms_i[it1][1]=maximum_refinement_level;
	      }
	    }else if(isoterms[it1][1].compare(0,3,"min")==0){
	       if(isoterms[it1][1].compare(0,4,"min+")==0){
		std::vector<std::string> tmpNumber = Utilities::split_string_list(isoterms[it1][1],'+');
		isoterms_i[it1][1]=minimum_refinement_level-Utilities::string_to_int(tmpNumber[1]);
	      }else{
		isoterms_i[it1][1]=minimum_refinement_level;
	      }
	    }else if(!isoterms[it1][1].empty() && isoterms[it1][1].find_first_not_of("0123456789") == std::string::npos){
	      isoterms_i[it1][1]=Utilities::string_to_int(isoterms[it1][1]);
	    }else{
	       AssertThrow (true,
                         ExcMessage ("Not able to read input at inputline "
				     +
				     Utilities::int_to_string(it1)
				     +
				     ": "  
				     +
				     isoterms[it1][0] 
				     )
			  );
	     //std::cout << "can't convert max or min at point 1: " << isoterms[it1][1] << std::endl; 
	    }
	  }
	    /// end convert max,max-1,max-2 etc and min, min-1,min-2 to numbers in integer
	   
            for (unsigned int i=0; i<quadrature.size(); ++i)
              {
		for ( unsigned int it1 = 0; it1 != n_isoterms_outer.size(); ++it1 )
		{
	   
		/// deterine if cell should be refined or coarsened		
		//for(unsigned int j=0; j<isoterms[3].size();++j){
		  //std::cout << "isoterms 3j: " << isoterms[it1][3] << "," << isoterms[it1][2] << std::endl;//<< "," << n_isoterms_inner[1]<< "," << n_isoterms_inner[0] << ",temp: " << in.temperature[i] << std::endl;
		  //std::cout << "level " << it1 << "," << cell->level() <<  ": " << isoterms_i[it1][0] << ", " << isoterms_i[it1][1] << std::endl;
		  if(i_excludeComposition>0 && i_excludeComposition<=this->n_compositional_fields()){
                if(rint(in.temperature[i]) <= Utilities::string_to_int(isoterms[it1][3]) && rint(in.temperature[i]) >= Utilities::string_to_int(isoterms[it1][2]) && in.composition[it1][i_excludeComposition]<0.5){
		   
                    ///min
	      if (cell->level() <= isoterms_i[it1][0]){
                    clear_coarsen = true;
	      }
              if (cell->level() <  isoterms_i[it1][0])
                {
                  refine = true;
		  //std::cout << "refine " << it1 << "," << cell->level() <<  ": " << isoterms_i[it1][0] << ", " << isoterms_i[it1][1] << std::endl;
                  break;
                }
                    
		  ///max
                  if (cell->level() >= isoterms_i[it1][1]){
                    clear_refine = true;
		  }
                  if (cell->level() >  isoterms_i[it1][1])
                    {
                      coarsen = true;
		      //std::cout << "coarsen " << it1 << "," << cell->level() <<  ": " << isoterms_i[it1][1] << std::endl;
                      break;
                    }
                   
                    }
		  }else{
		    if(rint(in.temperature[i]) <= Utilities::string_to_int(isoterms[it1][3]) && rint(in.temperature[i]) >= Utilities::string_to_int(isoterms[it1][2])){
		   
                    ///min
	      if (cell->level() <= isoterms_i[it1][0]){
                    clear_coarsen = true;
	      }
              if (cell->level() <  isoterms_i[it1][0])
                {
                  refine = true;
		  //std::cout << "refine " << it1 << "," << cell->level() <<  ": " << isoterms_i[it1][0] << ", " << isoterms_i[it1][1] << std::endl;
                  break;
                }
                    
		  ///max
                  if (cell->level() >= isoterms_i[it1][1]){
                    clear_refine = true;
		  }
                  if (cell->level() >  isoterms_i[it1][1])
                    {
                      coarsen = true;
		      //std::cout << "coarsen " << it1 << "," << cell->level() <<  ": " << isoterms_i[it1][1] << std::endl;
                      break;
                    }
                   
                    }
		  }
		//}
		}
	      }
              
              /// coarsen or refine cell
              if (clear_refine)
                cell->clear_refine_flag ();
              if (coarsen)
                cell->set_coarsen_flag ();
	       if (clear_coarsen)
                cell->clear_coarsen_flag ();
              if (refine)
                cell->set_refine_flag ();
            }
        }
    }

    template <int dim>
    void
    Isoterms<dim>::
    declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {

        prm.enter_subsection("Isoterms");
        {
          /**
           * Choose the coordinates to evaluate the maximum refinement level
           * function. The function can be declared in dependence of depth,
           * cartesian coordinates or spherical coordinates. Note that the order
           * of spherical coordinates is r,phi,theta and not r,theta,phi, since
           * this allows for dimension independent expressions.
           */
          prm.declare_entry ("Exclude composition", "-1",
                             Patterns::Double (),
                             "Number of mininum and maximum areas defined by isoterms are going to be defined."
                             );
		  prm.declare_entry ("List of isoterms", "",
                             Patterns::List(Patterns::List(Patterns::Anything(),0,100000000,","),0,100000000,";"),
                             "Number of mininum and maximum areas defined by isoterms are going to be defined."
                             );
							 
          /**
           * Let the function that describes the maximal level of refinement
           * as a function of position declare its parameters.
           * This defines the maximum refinement level each cell should have,
           * and that can not be exceeded by coarsening.
           */
          //Functions::ParsedFunction<dim>::declare_parameters (prm, 1);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    Isoterms<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh refinement");
      {
	minimum_refinement_level =Utilities::string_to_int(prm.get ("Minimum refinement level"));
	maximum_refinement_level = Utilities::string_to_int(prm.get("Initial global refinement")) + Utilities::string_to_int(prm.get("Initial adaptive refinement"));
        prm.enter_subsection("Isoterms");
        {
		i_excludeComposition = Utilities::string_to_int(prm.get("Exclude composition"));
		///////////////////////////////////////////////////////////////////////////////////////////
		n_isoterms_outer = Utilities::split_string_list(prm.get ("List of isoterms"),';');
	  int size_element = n_isoterms_outer.size();
	  isoterms.resize(size_element,std::vector<std::string>(4,"0"));
	  for ( unsigned int it1 = 0; it1 != n_isoterms_outer.size(); ++it1 )
	  {
	    n_isoterms_inner = Utilities::split_string_list(n_isoterms_outer[it1]);
	    //std::cout << "isoout: " << n_isoterms_inner[0] << "," << n_isoterms_inner[1] << "," << n_isoterms_inner[2] << "," << n_isoterms_inner[3] << std::endl;
	    
	    isoterms[it1][0]=n_isoterms_inner[0];
	    isoterms[it1][1]=n_isoterms_inner[1];
	    /// The temperatures for the min and max isoterms
	    isoterms[it1][2]=n_isoterms_inner[2];//Utilities::string_to_double(n_isoterms_inner[2]);
	    isoterms[it1][3]=n_isoterms_inner[3];//Utilities::string_to_double(n_isoterms_inner[3]);
	    //std::cout << "end: " << isoterms[0][0] << "," << isoterms[0][1] << "; " << isoterms[1][0] << "," << isoterms[1][1] << "; " << isoterms[2][0] << "," << isoterms[2][1] << std::endl;

	  }
          /*AssertThrow (isoterms.size() == wkz_n_zones,
                       ExcMessage("Invalid input parameter file: Wrong number of entries in List of beginpoints"));*/
					   
		//////////////////////////////////////////////////////////////////////////////////////////////////////
		
          /*for(int iii=0,prm.get ("Numer of isoterms"),iii++){
		  
		  }*/
           /* coordinate_system = depth;
          else if (prm.get ("Coordinate system") == "cartesian")
            coordinate_system = cartesian;
          else if (prm.get ("Coordinate system") == "spherical")
            coordinate_system = spherical;
          else
            AssertThrow (false, ExcNotImplemented());*/

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MeshRefinement
  {
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(Isoterms,
                                              "isoterms",
                                              "A mesh refinement criterion that ensures a "
                                              "maximum refinement level described by an "
                                              "explicit formula with the depth or position "
                                              "as argument. Which coordinate representation "
                                              "is used is determined by an input parameter. "
                                              "Whatever the coordinate system chosen, the "
                                              "function you provide in the input file will "
                                              "by default depend on variables 'x', 'y' and "
                                              "'z' (if in 3d). However, the meaning of these "
                                              "symbols depends on the coordinate system. In "
                                              "the Cartesian coordinate system, they simply "
                                              "refer to their natural meaning. If you have "
                                              "selected 'depth' for the coordinate system, "
                                              "then 'x' refers to the depth variable and 'y' "
                                              "and 'z' will simply always be zero. If you "
                                              "have selected a spherical coordinate system, "
                                              "then 'x' will refer to the radial distance of "
                                              "the point to the origin, 'y' to the azimuth "
                                              "angle and 'z' to the polar angle measured "
                                              "positive from the north pole. Note that the "
                                              "order of spherical coordinates is r,phi,theta "
                                              "and not r,theta,phi, since this allows for "
                                              "dimension independent expressions. "
                                              "After evaluating the function, its values are "
                                              "rounded to the nearest integer."
					      "\n\n"
					      "The format of these "
					      "functions follows the syntax understood by the "
					      "muparser library, see Section~\\ref{sec:muparser-format}.")
  }
}
