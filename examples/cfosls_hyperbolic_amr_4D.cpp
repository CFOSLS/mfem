#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#include "ls_temp.cpp"

// if active, no serial AMR in 4D is done
//#define ONLY_PAR_UR

// A test with a rotating Gaussian hill in the cubic domain.
// Originally, the test was for the cylinder, hence "cylinder" test,
// but later, to avoid errors from circle boundary approximation
// the same test was considered in the cubic domain
#define CYLINDER_CUBE_TEST

// Defines whether boundary conditions for CYLINDER_CUBE_TEST are overconstraining (see below)
// The actual inflow for a rotation when the space domain is [-1,1]^2 is actually two corners
// and one has to split bdr attributes for the faces, which is quite a pain
// Instead, we prescribe homogeneous bdr conditions at the entire boundary except for the top,
// since the solution is 0 at the boundary anyway. This is overconstraining but works ok.
#define OVERCONSTRAINED

// only for one-time test to compare MARS vs. MFEM refinement in terms of AMR performance
//#define SPECIAL_3DCASE

using namespace std;
using namespace mfem;

// Defines required estimator components, such as integrators, grid function structure
// for a given problem and given version of the functional
void DefineEstimatorComponents(FOSLSProblem * problem, int fosls_func_version,
                               std::vector<std::pair<int,int> >& grfuns_descriptor,
                               Array<ParGridFunction*>& extra_grfuns,
                               Array2D<BilinearFormIntegrator *> & integs, bool verbose);

// Rearranges boundary attributes for a rotational test in a cube so that essential boundary is
// exactly the inflow boundary, not all the bottom and lateral boundary (which would be overconstraining
// but work since the solution is localized  strictly inside the domain)
// Used when CYLINDER_CUBE_TEST is defined, but OVERCONSTRAINED is not
void ReArrangeBdrAttributes(Mesh* mesh4cube);


int main(int argc, char *argv[])
{
    // 1. Initialize MPI
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);
    bool verbose = (myid == 0);

    MPI_Comm comm_myid;
    MPI_Comm_split(comm, myid, 0, &comm_myid );

    StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    
    // 2. Parse command-line options.

    const char *space_for_S = "L2";     // "H1" or "L2"

    //const char *mesh_file = "../data/cube4d_96.MFEM";
    const char *mesh_file = "../data/cube4d_24.MFEM";
    int order = 0;
    bool visualization = 1;
    int numofrefinement = 1;
#ifndef ONLY_PAR_UR
    //int maxdofs = 900000;
    double error_frac = .95;
    double betavalue = 1000000;//0.1;
    int strat = 1;
#endif
    int numsol = -4;
    int prec_option = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&space_for_S, "-spaceS", "--spaceS",
                   "Space for S (H1 or L2).");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
#ifndef ONLY_PAR_UR
    //args.AddOption(&maxdofs, "-r", "-refine","-r");
    args.AddOption(&strat, "-rs", "--refinementstrategy", "Which refinement strategy to implement for the LS Refiner");
    args.AddOption(&error_frac, "-ef","--errorfraction", "Weight in Dorfler Marking Strategy");
    args.AddOption(&betavalue, "-b","--beta", "Beta in the Difference Term of Estimator");
#endif

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    if (verbose)
    {
        std::cout << "error_frac: " << error_frac << "\n";
        std::cout << "betavalue: " << betavalue << "\n";
        std::cout << "strat: " << strat << "\n";
    }

    if (verbose)
    {
        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";
    }

    // 1.5 Define the formulation to use for transport equation.
    /*
    // Hdiv-H1 case
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Hdiv-H1-L2 formulation must have space_for_S = `H1` \n");
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
    */

    // Hdiv-L2 case
    MFEM_ASSERT(strcmp(space_for_S,"L2") == 0, "Hdiv-L2-L2 formulation must have space_for_S = `L2` \n");
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2hyp;

    // 2. Read the mesh from the given mesh file.
#ifdef CYLINDER_CUBE_TEST
#ifndef OVERCONSTRAINED
    {
        MFEM_ABORT("In 4D due to the boundary attributes for a space-time cylinder, "
                  "we cannot get rid of the overconstraining so easily as in 3D. This "
                  "case has not been implemented");
    }
#endif
    numsol = 88;
    mesh_file = "../data/cube_4d_96_-11x02.mesh";

#ifdef SPECIAL_3DCASE
    numsol = 8;
    mesh_file = "../data/cube_3d_-11x02_notoverconstrained.mesh";
#endif

    if (verbose)
        std::cout << "numsol = " << numsol << "\n";

    /*
    if (verbose)
        std::cout << "WARNING: CYLINDER_CUBE_TEST works only when the domain is a cube [0,1]^(d+1)! \n";

    // Posprocessing the mesh in case of a "cylinder" test running in the cube
    // For n = 3, if the domain cube was [0,1]^2 x [0,T] before, here we want to stretch
    // it in (x,y) plane to cover [-1,1]^2 x [0,1] instead.
    // Similarly in the case n = 4. For n = 4, the z-direction of the cube after post-processing is [0,2],
    // as well as the time direction.
    Vector vert_coos;
    mesh->GetVertices(vert_coos);
    int nv = mesh->GetNV();
    for (int vind = 0; vind < nv; ++vind)
    {
        for (int j = 0; j < dim; ++j)
        {
            if (j < dim - 1) // shift only in (x,y)
            {
                // translation by -0.5 in space variables
                vert_coos(j*nv + vind) -= 0.5;
                // dilation so that the resulting mesh covers [-1,1] ^d in space
                vert_coos(j*nv + vind) *= 2.0;
            }
            else // dilation in time (if n = 3) and also in z-direction (if n = 4) so that
                 // final time interval is [0,2] for these directions
            //if (j == dim - 1)
                vert_coos(j*nv + vind) *= 2.0;
        }
    }
    mesh->SetVertices(vert_coos);
    */

    /*
    std::string filename_mesh;
    filename_mesh = "blablabla_4d.mesh";
    std::ofstream ofid(filename_mesh);
    ofid.precision(8);
    mesh->Print(ofid);

    MPI_Finalize();
    return 0;
    */

    // if we don't want to overconstrain the solution (see comments before
    // #define OVERCONSTRAINED), we need to manually rearrange boundary attributes
#ifndef OVERCONSTRAINED
    // rearranging boundary attributes which are now assigned to cube faces,
    // not to bot + square corners + top as we need
    MFEM_ASSERT(dim == 3, "Current implementation of ReArrangeBdrAttributes works only for n = 3 \n");
    ReArrangeBdrAttributes(mesh);

    /*
    std::string filename_mesh;
    filename_mesh = "checkmesh_cube4transport.mesh";
    std::ofstream ofid(filename_mesh);
    ofid.precision(8);
    mesh->Print(ofid);

    MPI_Finalize();
    return 0;
    */
#endif

#endif // for #ifdef CYLINDER_CUBE_TEST

    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    for (int l = 0; l < numofrefinement; l++)
        mesh->UniformRefinement();

    // 3. Define weak f.e. formulation for the problem at hand
    // and create FOSLSProblem on top of them
    FormulType * formulat = new FormulType (dim, numsol, verbose);
    FEFormulType * fe_formulat = new FEFormulType(*formulat, order);

#ifndef ONLY_PAR_UR
    // Perform adaptive refinement in serial

    if (myid == 0)
    {
        int global_dofs;
        int max_dofs = 45000000;
        int max_amr_iter = 11;
        int it_viz_step = 2;
#ifdef SPECIAL_3DCASE
        int it_print_step = 2;
        bool output_solution = 1;
        bool glvis_visualize = false;
        max_amr_iter = 21;
#endif

        ParMesh * pmesh = new ParMesh(comm_myid, *mesh);
        pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

        BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

        ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                                 prec_option, verbose);

        // 4. Creating the estimator
        std::vector<std::pair<int,int> > grfuns_descriptor;
        Array<ParGridFunction*> extra_grfuns;
        Array2D<BilinearFormIntegrator *> integs;

        int fosls_func_version = 1;

        // Create components required for the FOSLSEstimator: bilinear form integrators and grid functions
        DefineEstimatorComponents(problem, fosls_func_version, grfuns_descriptor, extra_grfuns, integs, verbose);

        FOSLSEstimator * estimator;
        estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
        problem->AddEstimator(*estimator);

        // cannot work due to the absence of LcoalRefinement in 4D for MFEM meshes
        //ThresholdRefiner refiner(*estimator);
        //refiner.SetTotalErrorFraction(0.5);

        NDLSRefiner * refiner = new NDLSRefiner(*estimator);
        refiner->SetTotalErrorFraction(error_frac);
        refiner->SetTotalErrorNormP(2.0);
        refiner->SetRefinementStrategy(strat);
        refiner->SetBetaCalc(0);
        refiner->SetBetaConstants(betavalue);
        refiner->version_difference = false;

        std::cout << "#dofs for the first solve: " << problem->GlobalTrueProblemSize() << "\n";
        problem->Solve(verbose, true);

        if (visualization)
        {
            problem->DistributeToGrfuns(problem->GetSol());
            ParGridFunction * sigma = problem->GetGrFun(0);

#ifdef SPECIAL_3DCASE
            if (glvis_visualize)
            {
                char vishost[] = "localhost";
                int  visport   = 19916;

                socketstream sigma_sock(vishost, visport);
                sigma_sock << "parallel " << num_procs << " " << myid << "\n";
                sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                       << 0 <<"'" << flush;

                ParGridFunction * sigma_ex = new ParGridFunction(problem->GetPfes(0));
                BlockVector * exactsol_proj = problem->GetExactSolProj();
                sigma_ex->SetFromTrueDofs(exactsol_proj->GetBlock(0));

                socketstream sigmaex_sock(vishost, visport);
                sigmaex_sock << "parallel " << num_procs << " " << myid << "\n";
                sigmaex_sock << "solution\n" << *pmesh << *sigma_ex << "window_title 'sigma exact, AMR iter No."
                       << 0 <<"'" << flush;

                delete sigma_ex;
                delete exactsol_proj;
            }
            if (output_solution)
            {
                // don't know what exactly ref is used for
                int ref = 1;

                //std::ofstream fp_sigma("sigma_test_it0.vtk");
                std::string filename_sig;
                filename_sig = "sigma_mars_it_";
                filename_sig.append(std::to_string(0));
                if (num_procs > 1)
                {
                    filename_sig.append("_proc_");
                    filename_sig.append(std::to_string(myid));
                }
                filename_sig.append(".vtk");
                std::ofstream fp_sigma(filename_sig);

                pmesh->PrintVTK(fp_sigma, ref, true);
                //pmesh->PrintVTK(fp_sigma);

                std::string field_name_sigma("sigma_h");
                sigma->SaveVTK(fp_sigma, field_name_sigma, ref);
            }
#else
            //if ( (it + 1) % it_viz_step == 0 || it + 1 == max_amr_iter - 1)
            {
                // creating mesh slices (and printing them in VTK format in a file for paraview)
                std::stringstream mesh_fname;
                mesh_fname << "slicedmesh_it_" << 0 << "_";
                ComputeSlices (*pmesh, 0.1, 4, 0.399, myid, num_procs, mesh_fname.str().c_str());

                // sigma
                std::stringstream sigma_fname;
                sigma_fname << "sigma_it_" << 0 << "_slices_";
                ComputeSlices (*sigma, 0.1, 4, 0.399, myid, num_procs, false, sigma_fname.str().c_str());
            }
#endif
        }

        for (int it = 0; it < max_amr_iter; ++it)
        {
            if (verbose)
            {
               cout << "\nAMR iteration " << it << "\n";
               cout << "Refining the mesh ... \n";
            }

            refiner->Apply(*mesh);
            if(refiner->Stop())
            {
                cout<< "Maximum number of dofs has been reached \n";
                break;
            }

            // cannot use Update() because CoarseToFine transformations are not generated
            // through MARS wrapper right now
            //problem->Update();
            //problem->BuildSystem(verbose);

            delete estimator;
            delete problem;

            mesh->AllocateSwappedElements();
            Mesh * mesh_copy = new Mesh(*mesh);
            delete mesh;
            mesh = mesh_copy;

            ParMesh * pmesh_copy = new ParMesh(comm_myid, *mesh);
            delete pmesh;
            pmesh = pmesh_copy;

            pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

            problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                                     prec_option, verbose);

            estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
            problem->AddEstimator(*estimator);

            delete refiner;
            refiner = new NDLSRefiner(*estimator);
            refiner->SetTotalErrorFraction(error_frac);
            refiner->SetTotalErrorNormP(2.0);
            refiner->SetRefinementStrategy(strat);
            refiner->SetBetaCalc(0);
            refiner->SetBetaConstants(betavalue);
            refiner->version_difference = false;

            global_dofs = problem->GlobalTrueProblemSize();
            if (global_dofs > max_dofs)
            {
               if (verbose)
                  cout << "Reached the maximum number of dofs, current problem "
                          "#dofs = " << global_dofs << ". Stop. \n";
               break;
            }

            if (verbose)
            {
               cout << "\nAMR iteration " << it << "\n";
               cout << "Number of unknowns: " << global_dofs << "\n";
            }

            problem->Solve(verbose, true);

            if (visualization)
            {
                problem->DistributeToGrfuns(problem->GetSol());
                ParGridFunction * sigma = problem->GetGrFun(0);

#ifdef SPECIAL_3DCASE
                if (glvis_visualize && ( ( (it + 1) % it_viz_step == 0 || it + 1 == max_amr_iter - 1)) )
                {
                    char vishost[] = "localhost";
                    int  visport   = 19916;

                    socketstream sigma_sock(vishost, visport);
                    sigma_sock << "parallel " << num_procs << " " << myid << "\n";
                    sigma_sock << "solution\n" << *pmesh << *sigma << "window_title 'sigma, AMR iter No."
                           << it + 1 <<"'" << flush;

                    ParGridFunction * sigma_ex = new ParGridFunction(problem->GetPfes(0));
                    BlockVector * exactsol_proj = problem->GetExactSolProj();
                    sigma_ex->SetFromTrueDofs(exactsol_proj->GetBlock(0));

                    socketstream sigmaex_sock(vishost, visport);
                    sigmaex_sock << "parallel " << num_procs << " " << myid << "\n";
                    sigmaex_sock << "solution\n" << *pmesh << *sigma_ex << "window_title 'sigma exact, AMR iter No."
                           << it + 1 << "'" << flush;

                    delete sigma_ex;
                    delete exactsol_proj;
                }

                if (output_solution && (it + 1) % it_print_step == 0)
                {
                    // don't know what exactly ref is used for
                    int ref = 1;

                    //std::ofstream fp_sigma("sigma_test_it0.vtk");
                    std::string filename_sig;
                    filename_sig = "sigma_mars_it_";
                    filename_sig.append(std::to_string(it + 1));
                    if (num_procs > 1)
                    {
                        filename_sig.append("_proc_");
                        filename_sig.append(std::to_string(myid));
                    }
                    filename_sig.append(".vtk");
                    std::ofstream fp_sigma(filename_sig);

                    pmesh->PrintVTK(fp_sigma, ref, true);
                    //pmesh->PrintVTK(fp_sigma);

                    std::string field_name_sigma("sigma_h");
                    sigma->SaveVTK(fp_sigma, field_name_sigma, ref);
                }
#else
                if ( (it + 1) % it_viz_step == 0 || it + 1 == max_amr_iter - 1)
                {
                    // creating mesh slices (and printing them in VTK format in a file for paraview)
                    std::stringstream mesh_fname;
                    mesh_fname << "slicedmesh_it_" << it + 1 << "_";
                    ComputeSlices (*pmesh, 0.1, 4, 0.399, myid, num_procs, mesh_fname.str().c_str());

                    // sigma
                    std::stringstream sigma_fname;
                    sigma_fname << "sigma_it_" << it + 1 << "_slices_";
                    ComputeSlices (*sigma, 0.1, 4, 0.399, myid, num_procs, false, sigma_fname.str().c_str());
                }
#endif
            }

        }

        delete problem;
        delete refiner;
        delete estimator;
        for (int i = 0; i < extra_grfuns.Size(); ++i)
            if (extra_grfuns[i])
                delete extra_grfuns[i];
        for (int i = 0; i < integs.NumRows(); ++i)
            for (int j = 0; j < integs.NumCols(); ++j)
                if (integs(i,j))
                    delete integs(i,j);

        delete bdr_conds;
        delete pmesh;
    }

    // Have to print and (later) read the mesh from file, because it's broekn as Mesh after 4D serial AMR through MARS
    std::stringstream fname;
    //fname << "mesh_file_temp.MFEM";
    fname << "final_amr_mesh_" << dim << ".mesh";

    if (myid == 0)
    {
        std::ofstream ofid(fname.str().c_str());
        ofid.precision(8);
        mesh->Print(ofid);
        delete mesh;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;

    //mesh = new Mesh("../data/cube4d_96.MFEM", 1, 1);
    std::ifstream ifid(fname.str().c_str());
    mesh = new Mesh(ifid, 1, 1);
#endif

    // Perform uniform refinement in parallel
    if (verbose)
        std::cout << "Parallel uniform refinement stage \n";
    int nprefs = 2;

    //std::cout << "Got here 0 \n" << std::flush;
    //MPI_Barrier(MPI_COMM_WORLD);
    ParMesh * pmesh = new ParMesh(comm, *mesh);
    //std::cout << "Got here 1 \n" << std::flush;
    //MPI_Barrier(MPI_COMM_WORLD);
    delete mesh;
    BdrConditions * bdr_conds = new BdrCondsType(*pmesh);
    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    ProblemType * problem = new ProblemType (*pmesh, *bdr_conds, *fe_formulat,
                                             prec_option, verbose);
    int global_dofs;
    int max_dofs_prefs = 1600000;

    for (int pref = 0; pref < nprefs; ++pref)
    {
        global_dofs = problem->GlobalTrueProblemSize();
        if (verbose)
        {
           cout << "\nUR iteration " << pref << "\n";
           cout << "Number of unknowns: " << global_dofs << "\n";
        }
        if (global_dofs > max_dofs_prefs)
        {
           if (verbose)
              cout << "Global #dofs: " << global_dofs << ". Reached the maximum number of dofs. Stop. \n";
           break;
        }

        problem->Solve(verbose, true);

        if (pref < nprefs - 1)
        {
            pmesh->UniformRefinement();
            problem->Update();
            problem->BuildSystem(verbose);
        }

    }

    delete fe_formulat;
    delete formulat;

    delete problem;
    delete bdr_conds;
    delete pmesh;

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

// See it's declaration and comments inside
void ReArrangeBdrAttributes(Mesh* mesh4cube)
{
    Mesh * mesh = mesh4cube;

    int dim = mesh->Dimension();

    int nbe = mesh->GetNBE();

    // Assume we have [-1,1] x [-1,1] x [0,2]
    // cube faces:
    // 0: bottom (z = 0)
    // 1: x = -1
    // 2: y = -1
    // 3: x = 1
    // 4: y = 1
    // 5: z = 2
    // final boundary parts (=attributes):
    // 1: bottom cube face
    // lateral boundary parts at the square corners:
    // 2: (face == 1 && y < 0) || (face == 2 && x < 0), here face = cube face
    // 3: (face == 2 && x >=0) || (face == 3 && y < 0)
    // 4: (face == 3 && y >=0) || (face == 4 && x >=0)
    // 5: (face == 4 && x < 0) || (face == 1 && y >=0)
    // and
    // 6: top cube face
    for (int beind = 0; beind < nbe; ++beind)
    {
        //std::cout << "beind = " << beind << "\n";
        // determine which cube face the be belongs to, via vertex coordinate dispersion
        int cubeface = -1;
        Element * bel = mesh->GetBdrElement(beind);
        int be_nv = bel->GetNVertices();
        Array<int> vinds(be_nv);
        bel->GetVertices(vinds);
        //vinds.Print();

        //std::cout << "mesh nv =  " << pmesh->GetNV() << "\n";

        Array<double> av_coords(dim);
        av_coords = 0.0;
        for (int vno = 0; vno < be_nv; ++vno)
        {
            //std::cout << "vinds[vno] = " << vinds[vno] << "\n";
            double * vcoos = mesh->GetVertex(vinds[vno]);
            for (int coo = 0; coo < dim; ++coo)
            {
                //std::cout << vcoos[coo] << " ";
                av_coords[coo] += vcoos[coo];
            }
            //std::cout << "\n";
        }
        //av_coords.Print();

        for (int coo = 0; coo < dim; ++coo)
        {
            av_coords[coo] /= 1.0 * be_nv;
        }

        //std::cout << "average be coordinates: \n";
        //av_coords.Print();

        int face_coo = -1;
        for (int coo = 0; coo < dim; ++coo)
        {
            bool coo_fixed = true;
            //std::cout << "coo = " << coo << "\n";
            for (int vno = 0; vno < be_nv; ++vno)
            {
                double * vcoos = mesh->GetVertex(vinds[vno]);
                //std::cout << "vcoos[coo] = " << vcoos[coo] << "\n";
                if (fabs(vcoos[coo] - av_coords[coo]) > 1.0e-13)
                    coo_fixed = false;
            }
            if (coo_fixed)
            {
                if (face_coo > -1)
                {
                    MFEM_ABORT("Found a second coordinate which is fixed \n");
                }
                else
                {
                    face_coo = coo;
                }
            }
        }

        MFEM_ASSERT(face_coo != -1,"Didn't find a fixed coordinate \n");

        double value = av_coords[face_coo];
        if (face_coo == 0 && fabs(value - (-1.0)) < 1.0e-13)
            cubeface = 1;
        if (face_coo== 1 && fabs(value - (-1.0)) < 1.0e-13)
            cubeface = 2;
        if (face_coo == 2 && fabs(value - (0.0)) < 1.0e-13)
            cubeface = 0;
        if (face_coo == 0 && fabs(value - (1.0)) < 1.0e-13)
            cubeface = 3;
        if (face_coo == 1 && fabs(value - (1.0)) < 1.0e-13)
            cubeface = 4;
        if (face_coo == 2 && fabs(value - (2.0)) < 1.0e-13)
            cubeface = 5;

        //std::cout << "cubeface = " << cubeface << "\n";

        // determine to which vertical stripe of the cube face the be belongs to
        Array<int> signs(dim);
        int x,y;
        if (cubeface != 0 && cubeface != 5)
        {
            for (int coo = 0; coo < dim; ++coo)
            {
                Array<int> coo_signs(be_nv);
                for (int vno = 0; vno < be_nv; ++vno)
                {
                    double * vcoos = mesh->GetVertex(vinds[vno]);
                    if (vcoos[coo] - 0.0 > 1.0e-13)
                        coo_signs[vno] = 1;
                    else if (vcoos[coo] - 0.0 < -1.0e-13)
                        coo_signs[vno] = -1;
                    else
                        coo_signs[vno] = 0;
                }

                int first_sign = -2; // anything which is not -1, 0 or 1
                for (int i = 0; i < be_nv; ++i)
                {
                    if (abs(coo_signs[i]) > 0)
                    {
                        first_sign = coo_signs[i];
                        break;
                    }
                }

                MFEM_ASSERT(first_sign != -2, "All signs were 0 for this coordinate, i.e. triangle"
                                              " is on the line");

                for (int i = 0; i < be_nv; ++i)
                {
                    if (coo_signs[i] != 0 && coo_signs[i] != first_sign)
                    {
                        MFEM_ABORT("Boundary element intersects the face middle line");
                    }
                }

                signs[coo] = first_sign;
            }
            x = signs[0];
            y = signs[1];
        }

        switch(cubeface)
        {
        case 0:
            bel->SetAttribute(1);
            break;
        case 1:
        {
            if (y < 0)
                bel->SetAttribute(2);
            else
                bel->SetAttribute(5);
        }
            break;
        case 2:
            if (x < 0)
                bel->SetAttribute(2);
            else
                bel->SetAttribute(3);
            break;
        case 3:
            if (y < 0)
                bel->SetAttribute(3);
            else
                bel->SetAttribute(4);
            break;
        case 4:
            if (x < 0)
                bel->SetAttribute(5);
            else
                bel->SetAttribute(4);
            break;
        case 5:
            bel->SetAttribute(6);
            break;
        default:
        {
            MFEM_ABORT("Could not find a cube face for the boundary element \n");
        }
            break;
        }
    } // end of loop ove be elements

}

