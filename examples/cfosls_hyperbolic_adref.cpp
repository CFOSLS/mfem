///                           MFEM(with 4D elements) CFOSLS for 3D/4D transport equation
///                                      with adaptive mesh refinement,
///                                     solved by a preconditioner MINRES.
///
/// The problem considered in this example is
///                             du/dt + b * u = f (either 3D or 4D in space-time)
/// casted in the CFOSLS formulation
/// 1) either in Hdiv-L2 case:
///                             (K sigma, sigma) -> min
/// where sigma is from H(div), u is recovered (as an element of L^2) from sigma = b * u,
/// and K = (I - bbT / || b ||);
/// 2) or in Hdiv-H1-L2 case
///                             || sigma - b * u || ^2 -> min
/// where sigma is from H(div) and u is from H^1;
/// minimizing in all cases under the constraint
///                             div sigma = f.
///
/// The problem is discretized using RT, linear Lagrange and discontinuous constants in 3D/4D.
/// The current 3D tests are either in cube (preferred) or in a cylinder, with a rotational velocity field b.
///
/// The problem is then solved with adaptive mesh refinement (AMR).
/// Unlike other AMR examples for transport equation, here the simplest approach is used.
/// At each iteration a preconditioned MINRES is used to solve the problem and update the local error
/// which trigger the next mesh refinement.
/// For more complicated setups, look at cfosls_hyperbolic_adref_Hcurl_new.cpp or cfosls_hyperbolic_adref_Hcurl.cpp.
///
/// This example demonstrates usage of such classes from mfem/cfosls/ as
/// FOSLSProblem and FOSLSEstimator.
///
/// (**) This code was tested in serial and in parallel.
/// (***) The example was tested for memory leaks with valgrind, in 3D.
///
/// Typical run of this example: ./cfosls_hyperbolic_adref_Hcurl_new --whichD 3 --spaceS L2 -no-vis
/// If you want to use the Hdiv-H1-L2 formulation, you will need not only change --spaceS option but also
/// change the source code, around 4.
///
/// Another examples on adaptive mesh refinement are cfosls_laplace_adref_Hcurl_new.cpp,
/// cfosls_laplace_adref_Hcurl.cpp and cfosls_hyperbolic_adref_Hcurl.cpp.


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

using namespace std;
using namespace mfem;
using std::shared_ptr;
using std::make_shared;

// Defines required estimator components, such as integrators, grid function structure
// for a given problem and given version of the functional
void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version,
                               std::vector<std::pair<int,int> >& grfuns_descriptor,
                               Array<ParGridFunction*>& extra_grfuns,
                               Array2D<BilinearFormIntegrator *> & integs, bool verbose);

int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = 8;

    int ser_ref_levels  = 0;
    int par_ref_levels  = 0;

    // These must be consistent with what formulation is used below.
    // Search for "using FormulType" below
    const char *space_for_S = "H1";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"

    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    bool visualization = 0;

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";
    //const char *mesh_file = "../data/cube4d_low.MFEM";
    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";
    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    //mesh_file = "../data/pmesh_check.mesh";
    //mesh_file = "../data/cube_3d_moderate.mesh";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS transport equation in a simple AMR setting\n";

    // 2. Parse command-line options.

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&feorder, "-o", "--feorder",
                   "Finite element order (polynomial degree).");
    args.AddOption(&ser_ref_levels, "-sref", "--sref",
                   "Number of serial refinements 4d mesh.");
    args.AddOption(&par_ref_levels, "-pref", "--pref",
                   "Number of parallel refinements 4d mesh.");
    args.AddOption(&nDimensions, "-dim", "--whichD",
                   "Dimension of the space-time problem.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&prec_option, "-precopt", "--prec-option",
                   "Preconditioner choice (0, 1 or 2 for now).");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S (H1 or L2).");
    args.AddOption(&space_for_sigma, "-spacesigma", "--spacesigma",
                   "Space for sigma (Hdiv or H1).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");

    args.Parse();
    if (!args.Good())
    {
       if (verbose)
       {
          args.PrintUsage(cout);
       }
       MPI_Finalize();
       return 1;
    }
    if (verbose)
    {
       args.PrintOptions(cout);
    }

    if (verbose)
    {
        if (strcmp(space_for_sigma,"Hdiv") == 0)
            std::cout << "Space for sigma: Hdiv \n";
        else
            std::cout << "Space for sigma: H1 \n";

        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
        {
            std::cout << "S: is ";
            std::cout << "eliminated from the system \n";
        }
    }

    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0,
                "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0,
                "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0),
                "Sigma from H1vec must be coupled with S from H1!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    // 3. Reading the mesh and performing a prescribed number of serial and parallel refinements
    Mesh *mesh = NULL;

    shared_ptr<ParMesh> pmesh;

    if (nDimensions == 3 || nDimensions == 4)
    {
        if (verbose)
            cout << "Reading a " << nDimensions << "d mesh from the file " << mesh_file << endl;
        ifstream imesh(mesh_file);
        if (!imesh)
        {
             std::cerr << "\nCan not open mesh file: " << mesh_file << '\n' << std::endl;
             MPI_Finalize();
             return -2;
        }
        else
        {
            mesh = new Mesh(imesh, 1, 1);
            imesh.close();
        }
    }
    else //if nDimensions is no 3 or 4
    {
        if (verbose)
            cerr << "Case nDimensions = " << nDimensions << " is not supported \n"
                 << flush;
        MPI_Finalize();
        return -1;

    }

    if (mesh) // if only serial mesh was generated previously, parallel mesh is initialized here
    {
        for (int l = 0; l < ser_ref_levels; l++)
            mesh->UniformRefinement();

        if ( verbose )
            cout << "Creating parmesh(" << nDimensions <<
                    "d) from the serial mesh (" << nDimensions << "d)" << endl << flush;
        pmesh = make_shared<ParMesh>(comm, *mesh);
        delete mesh;
    }

    for (int l = 0; l < par_ref_levels; l++)
       pmesh->UniformRefinement();

    //if(dim==3) pmesh->ReorientTetMesh();

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 4. Define the problem to be solved (CFOSLS Hdiv-L2 or Hdiv-H1 formulation, e.g.)
    int dim = nDimensions;

   // Hdiv-H1 case
   MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Hdiv-H1-L2 formulation must have space_for_S = `H1` \n");
   using FormulType = CFOSLSFormulation_HdivH1Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
   using ProblemType = FOSLSProblem_HdivH1L2hyp;

   /*
   // Hdiv-L2 case
   MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Hdiv-L2-L2 formulation must have space_for_S = `L2` \n");
   using FormulType = CFOSLSFormulation_HdivL2Hyper;
   using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
   using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
   using ProblemType = FOSLSProblem_HdivL2hyp;
   */

   // 5. Creating an instance of the FOSLSProblem to be solved.

   FormulType * formulat = new FormulType (dim, numsol, verbose);
   FEFormulType * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrCondsType * bdr_conds = new BdrCondsType(*pmesh);

   ProblemType * problem = new ProblemType
           (*pmesh, *bdr_conds, *fe_formulat, prec_option, verbose);

   // 6. Creating the error estimator

   int fosls_func_version = 1;
   if (verbose)
    std::cout << "fosls_func_version = " << fosls_func_version << "\n";

   /// The descriptor describes the grid functions used in the error estimator
   /// each pair (which corresponds to a grid function used in the estimator)
   /// has the form <a,b>, where:
   /// 1) a pair of the form <1,b> means that the corresponding grid function
   /// is one of the grid functions inside the FOSLSProblem, and b
   /// equals its index in grfuns array
   /// 2) a pair of the for <-1,b> means that the grid function is in the extra
   /// grid functions (additional argument in the estimator construction)
   /// and b is its index inside the extra grfuns array.
   /// (*) The user should take care of updating the extra grfuns, if they
   /// are not a part of the problem (e.g., defined on a different pfespace)
   std::vector<std::pair<int,int> > grfuns_descriptor;
   Array<ParGridFunction*> extra_grfuns;
   Array2D<BilinearFormIntegrator *> integs;

   // Create components required for the FOSLSEstimator: bilinear form integrators and grid functions
   DefineEstimatorComponents(problem, fosls_func_version, grfuns_descriptor, extra_grfuns, integs, verbose);

   FOSLSEstimator * estimator;
   estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
   problem->AddEstimator(*estimator);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.5);

   // 7. The main AMR loop. At each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 200000;

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();

   bool compute_error = true;

   if (verbose)
       std::cout << "Running AMR ... \n";

   // upper limit on the number of AMR iterations
   int max_iter_amr = 3; // 21;

   for (int it = 0; it < max_iter_amr ; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n";
       }

       // 7.1 Solving the problem
       problem->Solve(verbose, compute_error);

       // to make sure that problem has grfuns which correspond to the computed problem solution
       // we explicitly distribute the solution here
       // though for now the coordination already happens in ComputeError()
       problem->DistributeToGrfuns(problem->GetSol());

       // 7.2. Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);
           ParGridFunction * S;

           if (problem->GetFEformulation().Nunknowns() >= 2)
               S = problem->GetGrFun(1);
           else // only sigma = Hdiv-L2 formulation with eliminated S
               S = (dynamic_cast<ProblemType*>(problem))->RecoverS(problem->GetSol().GetBlock(0));

           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream sigma_sock(vishost, visport);
           sigma_sock << "parallel " << num_procs << " " << myid << "\n";
           sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                  << it <<"'" << flush;

           socketstream s_sock(vishost, visport);
           s_sock << "parallel " << num_procs << " " << myid << "\n";
           s_sock << "solution\n" << *pmesh << *S << "window_title 'S, AMR iter No."
                  << it <<"'" << flush;
       }

       // 7.3. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.
       refiner.Apply(*problem->GetParMesh());
       if (refiner.Stop())
       {
          if (verbose)
          {
             cout << "Stopping criterion satisfied. Stop." << endl;
          }
          break;
       }

       // 7.4 Updating the problem
       problem->Update();
       problem->BuildSystem(verbose);

       global_dofs = problem->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

       /*
        * Just an example of how Mesh and parGridFunction slicing can be done
       if (it == 0 || it == 1)
       {
           double t0 = 0.1;
           double Nmoments = 4;
           double deltat = 0.2;

           ComputeSlices(*problem->GetParMesh(), t0, Nmoments, deltat, myid);
       }
       */

       /*
       Vector& solution = problem->GetSol();
       BlockVector sol_viewer(solution.GetData(), problem->GetTrueOffsets());
       ParGridFunction * sol_sigma_h = new ParGridFunction(problem->GetPfes(0));
       sol_sigma_h->SetFromTrueDofs(sol_viewer.GetBlock(0));
       sol_sigma_h->ComputeSlices (t0, Nmoments, deltat, myid, false);
       delete sol_sigma_h;
       */
   }

   // 8. Free the used memory.
   for (int i = 0; i < extra_grfuns.Size(); ++i)
       if (extra_grfuns[i])
           delete extra_grfuns[i];
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           if (integs(i,j))
               delete integs(i,j);

   delete problem;
   delete estimator;

   delete bdr_conds;
   delete formulat;
   delete fe_formulat;

   MPI_Finalize();
   return 0;
}


// See it's declaration
void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version, std::vector<std::pair<int,int> >& grfuns_descriptor,
                          Array<ParGridFunction*>& extra_grfuns, Array2D<BilinearFormIntegrator *> & integs, bool verbose)
{
    int numfoslsfuns = -1;

    if (verbose)
        std::cout << "fosls_func_version = " << fosls_func_version << "\n";

    if (fosls_func_version == 1)
        numfoslsfuns = 2;
    else if (fosls_func_version == 2)
        numfoslsfuns = 3;

    // extra_grfuns.SetSize(0); // must come by default
    if (fosls_func_version == 2)
        extra_grfuns.SetSize(1);

    grfuns_descriptor.resize(numfoslsfuns);

    integs.SetSize(numfoslsfuns, numfoslsfuns);
    for (int i = 0; i < integs.NumRows(); ++i)
        for (int j = 0; j < integs.NumCols(); ++j)
            integs(i,j) = NULL;

    Hyper_test* Mytest = dynamic_cast<Hyper_test*>
            (problem->GetFEformulation().GetFormulation()->GetTest());
    MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

    const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
            GetFunctSpacesDescriptor();

    // version 1, only || sigma - b S ||^2, or || K sigma ||^2
    if (fosls_func_version == 1)
    {
        // this works
        grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
        grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(0,0) = new VectorFEMassIntegrator;
        else // sigma is from H1vec
            integs(0,0) = new ImproperVectorMassIntegrator;

        integs(1,1) = new MassIntegrator(*Mytest->GetBtB());

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());
        else // sigma is from H1
            integs(1,0) = new MixedVectorScalarIntegrator(*Mytest->GetMinB());
    }
    else if (fosls_func_version == 2)
    {
        // version 2, only || sigma - b S ||^2 + || div bS - f ||^2
        MFEM_ASSERT(problem->GetFEformulation().Nunknowns() == 2 && (*space_names_funct)[1] == SpaceName::H1,
                "Version 2 works only if S is from H1 \n");

        // this works
        grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
        grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
        grfuns_descriptor[2] = std::make_pair<int,int>(-1, 0);

        int numblocks = problem->GetFEformulation().Nblocks();

        extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
        extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs());

        if ( (*space_names_funct)[0] == SpaceName::HDIV) // sigma is from Hdiv
            integs(0,0) = new VectorFEMassIntegrator;
        else // sigma is from H1vec
            integs(0,0) = new ImproperVectorMassIntegrator;

        integs(1,1) = new H1NormIntegrator(*Mytest->GetBBt(), *Mytest->GetBtB());

        integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());

        // integrators related to f (rhs side)
        integs(2,2) = new MassIntegrator;
        integs(1,2) = new MixedDirectionalDerivativeIntegrator(*Mytest->GetMinB());
    }
    else
    {
        MFEM_ABORT("Unsupported version of fosls functional \n");
    }
}

