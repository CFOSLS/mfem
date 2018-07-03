//                                MFEM(with 4D elements) CFOSLS for 3D/4D hyperbolic equation
//                                  with adaptive refinement involving a div-free formulation
//
// Compile with: make
//
// Description:  This example code solves a simple 3D/4D hyperbolic problem over [0,1]^3(4)
//               corresponding to the saddle point system
//                                  sigma_1 = u * b
//							 		sigma_2 - u        = 0
//                                  div_(x,t) sigma    = f
//                       with b = vector function (~velocity),
//						 NO boundary conditions (which work only in case when b * n = 0 pointwise at the domain space boundary)
//						 and initial condition:
//                                  u(x,0)            = 0
//               Here, we use a given exact solution
//                                  u(xt) = uFun_ex(xt)
//               and compute the corresponding r.h.s.
//               We discretize with Raviart-Thomas finite elements (sigma), continuous H1 elements (u) and
//					  discontinuous polynomials (mu) for the lagrange multiplier.
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <iomanip>
#include <list>

// if passive, the mesh is simply uniformly refined at each iteration
#define AMR

// activates the setup when the solution is sought for as a sum of a particular solution
// and a divergence-free correction
#define DIVFREE_SETUP

//#define NEWINTERFACE
//#define MG_DIVFREEPREC

#define RECOARSENING_AMR

// activates using the solution at the previous mesh as a starting guess for the next problem
#define CLEVER_STARTING_GUESS

// activates using a (simpler & cheaper) preconditioner for the problems, simple Gauss-Seidel
//#define USE_GS_PREC

// changes the particular solution problem to a pure div sigma = f problem
//#define PUREDIVCONSTRAINT

#define MULTILEVEL_PARTSOL

// activates using the particular solution at the previous mesh as a starting guess
// when finding the next particular solution (i.e., particular solution on the next mesh)
//#define CLEVER_STARTING_PARTSOL

//#define DEBUGGING_CASE

// is wrong, because the inflow for a rotation when the space domain is [-1,1]^2 is actually two corners
// and one has to split bdr attributes for the faces, which is quite a pain
#define CYLINDER_CUBE_TEST

using namespace std;
using namespace mfem;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

int main(int argc, char *argv[])
{
    int num_procs, myid;
    bool visualization = 1;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    bool verbose = (myid == 0);

    int nDimensions     = 3;
    int numsol          = -33;

#ifdef CYLINDER_CUBE_TEST
    numsol = 8;
#endif

    int ser_ref_levels  = 2;
    int par_ref_levels  = 0;

    const char *formulation = "cfosls"; // "cfosls" or "fosls"
    const char *space_for_S = "L2";     // "H1" or "L2"
    const char *space_for_sigma = "Hdiv"; // "Hdiv" or "H1"

    /*
    // Hdiv-H1 case
    using FormulType = CFOSLSFormulation_HdivH1Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivH1Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivH1_Hyper;
    using ProblemType = FOSLSProblem_HdivH1L2hyp;
 #ifdef DIVFREE_SETUP
    using DivfreeFormulType = CFOSLSFormulation_HdivH1DivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivH1DivfreeHyper;
 #endif
    */

    // Hdiv-L2 case
    using FormulType = CFOSLSFormulation_HdivL2Hyper;
    using FEFormulType = CFOSLSFEFormulation_HdivL2Hyper;
    using BdrCondsType = BdrConditions_CFOSLS_HdivL2_Hyper;
    using ProblemType = FOSLSProblem_HdivL2L2hyp;
#ifdef DIVFREE_SETUP
    using DivfreeFormulType = CFOSLSFormulation_HdivL2DivfreeHyp;
    using DivfreeFEFormulType = CFOSLSFEFormulation_HdivL2DivfreeHyper;
#endif

    // solver options
    int prec_option = 1; //defines whether to use preconditioner or not, and which one

    const char *mesh_file = "../data/cube_3d_moderate.mesh";
    //const char *mesh_file = "../data/square_2d_moderate.mesh";

    //const char *mesh_file = "../data/cube4d_low.MFEM";

    //const char *mesh_file = "../data/cube4d.MFEM";
    //const char *mesh_file = "dsadsad";
    //const char *mesh_file = "../data/orthotope3D_moderate.mesh";
    //const char *mesh_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * mesh_file = "../data/orthotope3D_fine.mesh";

    //const char * meshbase_file = "../data/sphere3D_0.1to0.2.mesh";
    //const char * meshbase_file = "../data/sphere3D_0.05to0.1.mesh";
    //const char * meshbase_file = "../data/sphere3D_veryfine.mesh";
    //const char * meshbase_file = "../data/beam-tet.mesh";
    //const char * meshbase_file = "../data/escher-p3.mesh";
    //const char * meshbase_file = "../data/orthotope3D_moderate.mesh";
    //const char * meshbase_file = "../data/orthotope3D_fine.mesh";
    //const char * meshbase_file = "../data/square_2d_moderate.mesh";
    //const char * meshbase_file = "../data/square_2d_fine.mesh";
    //const char * meshbase_file = "../data/square-disc.mesh";
    //const char *meshbase_file = "dsadsad";
    //const char * meshbase_file = "../data/circle_fine_0.1.mfem";
    //const char * meshbase_file = "../data/circle_moderate_0.2.mfem";

    int feorder         = 0;

    if (verbose)
        cout << "Solving (С)FOSLS Transport equation with MFEM & hypre \n";

    OptionsParser args(argc, argv);
    //args.AddOption(&mesh_file, "-m", "--mesh",
    //               "Mesh file to use.");
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
    args.AddOption(&formulation, "-form", "--formul",
                   "Formulation to use (cfosls or fosls).");
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
        if (strcmp(formulation,"cfosls") == 0)
            std::cout << "formulation: CFOSLS \n";
        else
            std::cout << "formulation: FOSLS \n";

        if (strcmp(space_for_sigma,"Hdiv") == 0)
            std::cout << "Space for sigma: Hdiv \n";
        else
            std::cout << "Space for sigma: H1 \n";

        if (strcmp(space_for_S,"H1") == 0)
            std::cout << "Space for S: H1 \n";
        else
            std::cout << "Space for S: L2 \n";

        if (strcmp(space_for_S,"L2") == 0)
            std::cout << "S: is eliminated from the system \n";
    }

    if (verbose)
        std::cout << "Running tests for the paper: \n";

    //mesh_file = "../data/netgen_cylinder_mesh_0.1to0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_moderate_0.2.mesh";
    //mesh_file = "../data/pmesh_cylinder_fine_0.1.mesh";

    //mesh_file = "../data/pmesh_check.mesh";
    mesh_file = "../data/cube_3d_moderate.mesh";

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "WARNING: CYLINDER_CUBE_TEST works only when the domain is a cube [0,1]! \n";
#endif


    if (verbose)
        std::cout << "For the records: numsol = " << numsol
                  << ", mesh_file = " << mesh_file << "\n";

#ifdef AMR
    if (verbose)
        std::cout << "AMR active \n";
#else
    if (verbose)
        std::cout << "AMR passive \n";
#endif

#ifdef DIVFREE_SETUP
    if (verbose)
        std::cout << "DIVFREE_SETUP active \n";
#else
    if (verbose)
        std::cout << "DIVFREE_SETUP passive \n";
#endif

#ifdef CLEVER_STARTING_GUESS
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_GUESS passive \n";
#endif

#if defined(CLEVER_STARTING_PARTSOL) && !defined(MULTILEVEL_PARTSOL)
    MFEM_ABORT("CLEVER_STARTING_PARTSOL cannot be active if MULTILEVEL_PARTSOL is not \n");
#endif

#ifdef CLEVER_STARTING_PARTSOL
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL active \n";
#else
    if (verbose)
        std::cout << "CLEVER_STARTING_PARTSOL passive \n";
#endif

#ifdef USE_GS_PREC
    if (verbose)
        std::cout << "USE_GS_PREC active (overwrites the prec_option) \n";
#else
    if (verbose)
        std::cout << "USE_GS_PREC passive \n";
#endif

#ifdef MULTILEVEL_PARTSOL
    if (verbose)
        std::cout << "MULTILEVEL_PARTSOL active \n";
#else
    if (verbose)
        std::cout << "MULTILEVEL_PARTSOL passive \n";
#endif

#ifdef PUREDIVCONSTRAINT
    if (verbose)
        std::cout << "PUREDIVCONSTRAINT active \n";
#else
    if (verbose)
        std::cout << "PUREDIVCONSTRAINT passive \n";
#endif

#ifdef CYLINDER_CUBE_TEST
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST active \n";
#else
    if (verbose)
        std::cout << "CYLINDER_CUBE_TEST passive \n";
#endif

#ifdef NEWINTERFACE
    if (verbose)
        std::cout << "NEWINTERFACE active \n";
#else
    if (verbose)
        std::cout << "NEWINTERFACE passive \n";
#endif

#ifdef MG_DIVFREEPREC
    if (verbose)
        std::cout << "MG_DIVFREEPREC active \n";
#else
    if (verbose)
        std::cout << "MG_DIVFREEPREC passive \n";
#endif

#ifdef RECOARSENING_AMR
    if (verbose)
        std::cout << "RECOARSENING_AMR active \n";
#else
    if (verbose)
        std::cout << "RECOARSENING_AMR passive \n";
#endif


    MFEM_ASSERT(strcmp(formulation,"cfosls") == 0 || strcmp(formulation,"fosls") == 0, "Formulation must be cfosls or fosls!\n");
    MFEM_ASSERT(strcmp(space_for_S,"H1") == 0 || strcmp(space_for_S,"L2") == 0, "Space for S must be H1 or L2!\n");
    MFEM_ASSERT(strcmp(space_for_sigma,"Hdiv") == 0 || strcmp(space_for_sigma,"H1") == 0, "Space for sigma must be Hdiv or H1!\n");

    MFEM_ASSERT(!strcmp(space_for_sigma,"H1") == 0 || (strcmp(space_for_sigma,"H1") == 0 && strcmp(space_for_S,"H1") == 0), "Sigma from H1vec must be coupled with S from H1!\n");

    if (verbose)
        std::cout << "Number of mpi processes: " << num_procs << "\n";

    StopWatch chrono;

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
    //mesh = new Mesh(2, 2, 2, Element::HEXAHEDRON, 1);

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
    {
       pmesh->UniformRefinement();
    }

    int dim = nDimensions;

#ifdef CYLINDER_CUBE_TEST
    Vector vert_coos;
    pmesh->GetVertices(vert_coos);
    int nv = pmesh->GetNV();
    for (int vind = 0; vind < nv; ++vind)
    {
        for (int j = 0; j < dim; ++j)
        {
            if (j < dim - 1) // shift only in space
            {
                // translation by -0.5 in space variables
                vert_coos(j*nv + vind) -= 0.5;
                // dilation so that the resulting mesh covers [-1,1] ^d in space
                vert_coos(j*nv + vind) *= 2.0;
            }
            // dilation in time so that final time interval is [0,2]
            if (j == dim - 1)
                vert_coos(j*nv + vind) *= 2.0;
        }
    }
    pmesh->SetVertices(vert_coos);

    /*
    std::stringstream fname;
    fname << "checkmesh.mesh";
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh->Print(ofid);

    MPI_Finalize();
    return 0;
    */

#endif

#ifdef DIVFREE_SETUP
    if (dim == 3 && feorder > 0)
    {
        if (verbose)
            std::cout << "WARNING: For Nedelec f.e. we need to call ReorientTetMesh for the inital mesh "
                         "when feorder > 1. However, current MFEM implementation then does not allow "
                         "to call UniformRefinement(), the code fails inside STable3D::operator()";
        if (verbose)
            std::cout << "Calling ReorientTetMesh in 3D since feorder is more than 0 \n";
        pmesh->ReorientTetMesh();
    }
#endif

    pmesh->PrintInfo(std::cout); if(verbose) cout << endl;

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.

    int numblocks = 1;

    if (strcmp(space_for_S,"H1") == 0)
        numblocks++;
    if (strcmp(formulation,"cfosls") == 0)
        numblocks++;

    if (verbose)
        std::cout << "Number of blocks in the formulation: " << numblocks << "\n";

   if (verbose)
       std::cout << "Running AMR ... \n";

   FOSLSFormulation * formulat = new FormulType (dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat = new FEFormulType(*formulat, feorder);
   BdrConditions * bdr_conds = new BdrCondsType(*pmesh);

#ifdef CYLINDER_CUBE_TEST
   delete bdr_conds;
   MFEM_ASSERT(pmesh->bdr_attributes.Max() == 6, "For CYLINDER_CUBE_TEST there must be"
                                                 " a bdr aittrbute for each face");

   std::vector<Array<int>* > bdr_attribs_data(formulat->Nblocks());
   for (int i = 0; i < formulat->Nblocks(); ++i)
       bdr_attribs_data[i] = new Array<int>(pmesh->bdr_attributes.Max());

   if (strcmp(space_for_S,"L2") == 0)
   {
       *bdr_attribs_data[0] = 1;
       (*bdr_attribs_data[0])[5] = 0;
   }
   else // S from H^1
   {
       *bdr_attribs_data[0] = 0;
       *bdr_attribs_data[1] = 1;
       (*bdr_attribs_data[1])[5] = 0;
   }
   *bdr_attribs_data[formulat->Nblocks() - 1] = 0;

   bdr_conds = new BdrConditions(*pmesh, formulat->Nblocks());
   bdr_conds->Set(bdr_attribs_data);
#endif

#ifdef DIVFREE_SETUP
   DivfreeFormulType * formulat_divfree = new DivfreeFormulType (dim, numsol, verbose);
   DivfreeFEFormulType * fe_formulat_divfree = new DivfreeFEFormulType(*formulat_divfree, feorder);
#endif

   /*
   // Hdiv-L2 case
   int numfoslsfuns = 1;

   std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);
   // this works
   grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   integs(0,0) = new VectorFEMassIntegrator(*Mytest.Ktilda);

   FOSLSEstimator * estimator;

   estimator = new FOSLSEstimator(*problem, grfuns_descriptor, NULL, integs, verbose);
   */

#if 0
   std::cout << "Special test! \n";

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose);
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructDivfreeDops();

   // with uniform refinement it works
   //hierarchy->GetFinestParMesh()->UniformRefinement();

   // with non-uniform refinement it doesn't work
   int nmarked = 5;
   Array<int> els_to_refine(nmarked);
   for (int i = 0; i < nmarked; ++i)
       els_to_refine[i] = hierarchy->GetFinestParMesh()->GetNE()/2 + i;
   hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

   hierarchy->Update();

   bool assemble_system = true;
   ProblemType * problem = new ProblemType(*hierarchy, 0, *bdr_conds,
                                           *fe_formulat, verbose, assemble_system);

   // checking that B * P * coarse_sigma = constant within each coarse element
   // outside DivConstraintClass

   // this works
   /*
   ParDiscreteLinearOperator div(hierarchy->GetSpace(SpaceName::HDIV,0), hierarchy->GetSpace(SpaceName::L2,0));
   div.AddDomainInterpolator(new DivergenceInterpolator);
   div.Assemble();
   div.Finalize();
   HypreParMatrix * Divergence = div.ParallelAssemble();
   */

   // works incorrectly, i.e., the check fails
   ParMixedBilinearForm div(hierarchy->GetSpace(SpaceName::HDIV,0), hierarchy->GetSpace(SpaceName::L2,0));
   div.AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   div.Assemble();
   div.Finalize();
   HypreParMatrix * Divergence = div.ParallelAssemble();


   // works incorrectly, i.e., the check fails
   //HypreParMatrix* Divergence = (HypreParMatrix*)(&problem->GetOp_nobnd()->GetBlock(numblocks - 1, 0));

   ParGridFunction * sigma_coarse_pgfun = new ParGridFunction(hierarchy->GetSpace(SpaceName::HDIV,1));
   //VectorFunctionCoefficient testvec_coeff(dim, testHdivfun);
   //sigma_coarse_pgfun->ProjectCoefficient(testvec_coeff);
   sigma_coarse_pgfun->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetSigma());
   Vector sigma_c(hierarchy->GetSpace(SpaceName::HDIV,1)->TrueVSize());
   sigma_coarse_pgfun->ParallelProject(sigma_c);

   Vector Psigma_c(Divergence->Width());
   hierarchy->GetTruePspace(SpaceName::HDIV,0)->Mult(sigma_c, Psigma_c);
   Vector BPsigma_c(Divergence->Height());
   Divergence->Mult(Psigma_c, BPsigma_c);

   SparseMatrix * AE_e = Transpose(*hierarchy->GetPspace(SpaceName::L2, 0));

   for (int AE = 0; AE < hierarchy->GetPmesh(1)->GetNE(); ++AE)
   {
       double row_av = 0.0;
       double rowsum = 0.0;
       int row_length = AE_e->RowSize(AE);
       int * elinds = AE_e->GetRowColumns(AE);
       for (int j = 0; j < row_length; ++j)
           rowsum += BPsigma_c[elinds[j]];
       row_av = rowsum / row_length;

       if (rowsum > 1.0e-14)
       {
           bool bad_row = false;
           for (int j = 0; j < row_length; ++j)
               if (fabs(BPsigma_c[elinds[j]] - row_av) > 1.0e-13)
               {
                   bad_row = true;
                   break;
               }

           if (bad_row)
           {
               std::cout << "AE " << AE << ": \n";
               std::cout << "rowsum = " << rowsum << "\n";
               std::cout << "row_av = " << row_av << "\n";
               for (int j = 0; j < row_length; ++j)
               {
                   std::cout << BPsigma_c[elinds[j]] << " ";
               }
               std::cout << "\n";

           }
       }

   }

   MPI_Finalize();
   return 0;
#endif

#if 0
   std::cout << "Special test 2! \n";

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose);
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructDivfreeDops();

   ParBilinearForm mass_coarse(hierarchy->GetSpace(SpaceName::L2, 0));
   mass_coarse.AddDomainIntegrator(new MassIntegrator);
   mass_coarse.Assemble();
   mass_coarse.Finalize();
   SparseMatrix * mass_coarse_spmat = mass_coarse.LoseMat();

   // with uniform refinement it works
   //hierarchy->GetFinestParMesh()->UniformRefinement();

   // with non-uniform refinement it doesn't work
   int nmarked = 5;
   Array<int> els_to_refine(nmarked);
   for (int i = 0; i < nmarked; ++i)
       els_to_refine[i] = hierarchy->GetFinestParMesh()->GetNE()/2 + i;
   hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

   hierarchy->Update();


   ParBilinearForm mass_fine(hierarchy->GetSpace(SpaceName::L2, 0));
   mass_fine.AddDomainIntegrator(new MassIntegrator);
   mass_fine.Assemble();
   mass_fine.Finalize();
   SparseMatrix * mass_fine_spmat = mass_fine.LoseMat();

   SparseMatrix * P_L2 = hierarchy->GetPspace(SpaceName::L2, 0);

   SparseMatrix * coarsened_mass_spmat = mfem::RAP(*P_L2, *mass_fine_spmat, *P_L2);

   SparseMatrix diff(*coarsened_mass_spmat);
   diff.Add(-1.0, *mass_coarse_spmat);

   std::cout << "|| W_H - P^T W_h P || = " << diff.MaxNorm() << "\n";

   MPI_Finalize();
   return 0;
#endif

//#if 0
   bool with_hcurl = false;
#ifdef DIVFREE_SETUP
   with_hcurl = true;
#endif

   GeneralHierarchy * hierarchy = new GeneralHierarchy(1, *pmesh, feorder, verbose, with_hcurl);
#ifdef DIVFREE_SETUP
   hierarchy->ConstructDofTrueDofs();
   hierarchy->ConstructDivfreeDops();
#endif
   FOSLSProblHierarchy<ProblemType, GeneralHierarchy> * prob_hierarchy = new
           FOSLSProblHierarchy<ProblemType, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat, prec_option, verbose);

   ProblemType * problem = prob_hierarchy->GetProblem(0);

#ifdef MULTILEVEL_PARTSOL
   const Array<SpaceName>* space_names_funct = problem->GetFEformulation().GetFormulation()->
           GetFunctSpacesDescriptor();
#endif

#ifdef DIVFREE_SETUP


#ifdef NEWINTERFACE
   FOSLSDivfreeProblem* problem_divfree = hierarchy->BuildDynamicProblem<FOSLSDivfreeProblem>
           (*bdr_conds, *fe_formulat_divfree, prec_option, verbose);

   problem_divfree->ConstructDivfreeHpMats();
   problem_divfree->CreateOffsetsRhsSol();
   BlockOperator * problem_divfree_op = ConstructDivfreeProblemOp(*problem_divfree, *problem);
   problem_divfree->ResetOp(*problem_divfree_op);

   hierarchy->AttachProblem(problem_divfree);
   ComponentsDescriptor * divfree_descriptor;
   {
       bool with_Schwarz = false;
       bool optimized_Schwarz = false;
       bool with_Hcurl = false;
       bool with_coarsest_partfinder = false;
       bool with_coarsest_hcurl = false;
       bool with_monolithic_GS = true;
       divfree_descriptor = new ComponentsDescriptor(with_Schwarz, optimized_Schwarz,
                                                     with_Hcurl, with_coarsest_partfinder,
                                                     with_coarsest_hcurl, with_monolithic_GS);
   }

   MultigridToolsHierarchy * mgtools_divfree_hierarchy =
           new MultigridToolsHierarchy(*hierarchy, problem_divfree->GetAttachedIndex(), *divfree_descriptor);
   GeneralMultigrid * GeneralMGprec;

   problem_divfree->InitSolver(verbose);
#ifdef MG_DIVFREEPREC
   int num_levels = hierarchy->Nlevels();

   int coarsest_level = num_levels - 1;
   Operator* CoarseSolver_mg = new CGSolver(comm);
   ((CGSolver*)CoarseSolver_mg)->SetAbsTol(sqrt(1e-32));
   ((CGSolver*)CoarseSolver_mg)->SetRelTol(sqrt(1e-12));
   ((CGSolver*)CoarseSolver_mg)->SetMaxIter(100);
   ((CGSolver*)CoarseSolver_mg)->SetPrintLevel(0);
   ((CGSolver*)CoarseSolver_mg)->SetOperator(*mgtools_divfree_hierarchy->GetOps()[coarsest_level]);
   ((CGSolver*)CoarseSolver_mg)->iterative_mode = false;

   BlockDiagonalPreconditioner * CoarsePrec_mg =
           new BlockDiagonalPreconditioner(mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->ColOffsets());

   HypreParMatrix &blk00 = (HypreParMatrix&)(mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->GetBlock(0,0));
   HypreSmoother * precU = new HypreSmoother(blk00, HypreSmoother::Type::l1GS, 1);
   ((BlockDiagonalPreconditioner*)CoarsePrec_mg)->SetDiagonalBlock(0, precU);

   Array<Operator*> casted_monolitGSSmoothers(num_levels - 1);
   for (int l = 0; l < casted_monolitGSSmoothers.Size(); ++l)
       casted_monolitGSSmoothers[l] = mgtools_divfree_hierarchy->GetMonolitGSSmoothers()[l];

   GeneralMGprec =
           new GeneralMultigrid(num_levels,
                                //P_mg,
                                mgtools_divfree_hierarchy->GetPs_bnd(),
                                //Ops_mg,
                                mgtools_divfree_hierarchy->GetOps(),
                                *CoarseSolver_mg,
                                //*mgtools_divfree_hierarchy->GetCoarsestSolver_Hcurl(),
                                //Smoo_mg);
                                casted_monolitGSSmoothers);
   problem_divfree->ChangeSolver();
   problem_divfree->SetPrec(*GeneralMGprec);
   problem_divfree->UpdateSolverPrec();
#else
   // creating a preconditioner for the divfree problem
   problem_divfree->CreatePrec(*problem_divfree->GetOp(), prec_option, verbose);
   problem_divfree->ChangeSolver();
   problem_divfree->UpdateSolverPrec();
#endif

#else

   FOSLSProblHierarchy<FOSLSDivfreeProblem, GeneralHierarchy> * divfreeprob_hierarchy =
           new FOSLSProblHierarchy<FOSLSDivfreeProblem, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds, *fe_formulat_divfree, prec_option, verbose);

   FOSLSDivfreeProblem * problem_divfree = divfreeprob_hierarchy->GetProblem(0);
#endif

#ifdef MULTILEVEL_PARTSOL
   bool optimized_localsolvers = true;
   bool with_hcurl_smoothers = true;
   DivConstraintSolver * partsol_finder;
#ifdef PUREDIVCONSTRAINT
   FOSLSFormulation * formulat_lapl = new CFOSLSFormulation_MixedLaplace(dim, numsol, verbose);
   FOSLSFEFormulation * fe_formulat_lapl = new
           CFOSLSFEFormulation_MixedLaplace(*formulat_lapl, feorder);
   std::vector<const Array<int>*> bdr_conditions_laplace(2);
   bdr_conditions_laplace[0] = &bdr_conds->GetBdrAttribs(0);
   bdr_conditions_laplace[1] = &bdr_conds->GetBdrAttribs(1);
   BdrConditions * bdr_conds_lapl = new BdrConditions(bdr_conditions_laplace);

   FOSLSProblHierarchy<FOSLSProblem, GeneralHierarchy> * specialprob_hierarchy = new
           FOSLSProblHierarchy<FOSLSProblem, GeneralHierarchy>
           (*hierarchy, 1, *bdr_conds_lapl, *fe_formulat_lapl, 0, verbose);

   FOSLSProblem * special_problem = specialprob_hierarchy->GetProblem(0);
   ReplaceBlockByIdentityHpmat(*special_problem->GetOp(), 0);

   partsol_finder = new DivConstraintSolver
           (*special_problem, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);

#else // if using the original FOSLS problem when finding the particular solution
   partsol_finder = new DivConstraintSolver
           (*problem, *hierarchy, optimized_localsolvers, with_hcurl_smoothers, verbose);
#endif

#endif// endif for MULTILEVEL_PARTSOL

#endif

   // testing DivConstraintSolver
#if 0
   {
       Vector * partsol_guess = new Vector(partsol_finder->Size());
       *partsol_guess = 0.0;
       BlockVector partsol_vec(problem->GetTrueOffsetsFunc());
       Vector& div_rhs = problem->GetRhs().GetBlock(numblocks - 1);
       partsol_finder->FindParticularSolution(*partsol_guess, partsol_vec, div_rhs, verbose);

       {
           HypreParMatrix & Constr = (HypreParMatrix&)(problem->GetOp()->GetBlock(numblocks - 1, 0));
           Vector tempc(Constr.Height());
           Constr.Mult(partsol_vec.GetBlock(0), tempc);
           tempc -= problem->GetRhs().GetBlock(numblocks - 1);
           double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
           MFEM_ASSERT (res_constr_norm < 1.0e-12, "first place");
       }

       // with uniform refinement it works
       //hierarchy->GetFinestParMesh()->UniformRefinement();

       // with non-uniform refinement it doesn't work
       int nmarked = 5;
       Array<int> els_to_refine(nmarked);
       for (int i = 0; i < nmarked; ++i)
           els_to_refine[i] = hierarchy->GetFinestParMesh()->GetNE()/2 + i;
       hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);
       partsol_finder->UpdateProblem(*problem);
       partsol_finder->Update();

       // one more time
       //hierarchy->GetFinestParMesh()->UniformRefinement();

       // with non-uniform refinement it doesn't work
       nmarked = 6;
       els_to_refine.SetSize(nmarked);
       for (int i = 0; i < nmarked; ++i)
           els_to_refine[i] = hierarchy->GetFinestParMesh()->GetNE()/3 + i;
       hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);

       partsol_finder->UpdateProblem(*problem);
       partsol_finder->Update();


       Vector * partsol_guess2 = new Vector(partsol_finder->Size());
       *partsol_guess2 = 0.0;
       BlockVector partsol_vec2(problem->GetTrueOffsetsFunc());
       Vector& div_rhs2 = problem->GetRhs().GetBlock(numblocks - 1);
       partsol_finder->FindParticularSolution(*partsol_guess2, partsol_vec2, div_rhs2, verbose);

       {
           HypreParMatrix & Constr = (HypreParMatrix&)(problem->GetOp()->GetBlock(numblocks - 1, 0));
           Vector tempc(Constr.Height());
           Constr.Mult(partsol_vec2.GetBlock(0), tempc);
           tempc -= problem->GetRhs().GetBlock(numblocks - 1);
           double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
           MFEM_ASSERT (res_constr_norm < 1.0e-12, "second place");
       }

       MPI_Finalize();
       return 0;
   }
#endif

   int numfoslsfuns = -1;

   int fosls_func_version = 1;
   if (verbose)
    std::cout << "fosls_func_version = " << fosls_func_version << "\n";

   if (fosls_func_version == 1)
       numfoslsfuns = 2;
   else if (fosls_func_version == 2)
       numfoslsfuns = 3;

   int numblocks_funct = 1;
   if (strcmp(space_for_S,"H1") == 0)
       ++numblocks_funct;

   Array<ParGridFunction*> extra_grfuns(0);
   if (fosls_func_version == 2)
       extra_grfuns.SetSize(1);

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

   std::vector<std::pair<int,int> > grfuns_descriptor(numfoslsfuns);

   Array2D<BilinearFormIntegrator *> integs(numfoslsfuns, numfoslsfuns);
   for (int i = 0; i < integs.NumRows(); ++i)
       for (int j = 0; j < integs.NumCols(); ++j)
           integs(i,j) = NULL;

   Hyper_test* Mytest = dynamic_cast<Hyper_test*>
           (problem->GetFEformulation().GetFormulation()->GetTest());
   MFEM_ASSERT(Mytest, "Unsuccessful cast into Hyper_test* \n");

   // version 1, only || sigma - b S ||^2, or || K sigma ||^2
   if (fosls_func_version == 1)
   {
       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           integs(0,0) = new VectorFEMassIntegrator;
       else // sigma is from H1vec
           integs(0,0) = new ImproperVectorMassIntegrator;

       integs(1,1) = new MassIntegrator(*Mytest->GetBtB());

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
           integs(1,0) = new VectorFEMassIntegrator(*Mytest->GetMinB());
       else // sigma is from H1
           integs(1,0) = new MixedVectorScalarIntegrator(*Mytest->GetMinB());
   }
   else if (fosls_func_version == 2)
   {
       // version 2, only || sigma - b S ||^2 + || div bS - f ||^2
       MFEM_ASSERT(strcmp(space_for_S,"H1") == 0, "Version 2 works only if S is from H1 \n");

       // this works
       grfuns_descriptor[0] = std::make_pair<int,int>(1, 0);
       grfuns_descriptor[1] = std::make_pair<int,int>(1, 1);
       grfuns_descriptor[2] = std::make_pair<int,int>(-1, 0);

       extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
       extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().GetFormulation()->GetTest()->GetRhs());

       if (strcmp(space_for_sigma,"Hdiv") == 0) // sigma is from Hdiv
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

   FOSLSEstimator * estimator;

   if (fosls_func_version == 2)
   {
       estimator = new FOSLSEstimatorOnHier<ProblemType, GeneralHierarchy>
               (*prob_hierarchy, 0, grfuns_descriptor, &extra_grfuns, integs, verbose);
   }
   else
       estimator = new FOSLSEstimatorOnHier<ProblemType, GeneralHierarchy>
               (*prob_hierarchy, 0, grfuns_descriptor, NULL, integs, verbose);

   problem->AddEstimator(*estimator);

   ThresholdRefiner refiner(*estimator);
   refiner.SetTotalErrorFraction(0.9); // 0.5

#ifdef CLEVER_STARTING_GUESS
   BlockVector * coarse_guess;
#endif

#ifdef DEBUGGING_CASE
   Vector * checkdiff;
#endif

   Array<Vector*> div_rhs_lvls(0);
   Array<BlockVector*> partsol_lvls(0);
   Array<BlockVector*> partsol_funct_lvls(0);

   // 12. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.
   const int max_dofs = 300000;//1600000; 400000;

   HYPRE_Int global_dofs = problem->GlobalTrueProblemSize();
   std::cout << "starting n_el = " << hierarchy->GetFinestParMesh()->GetNE() << "\n";

   double fixed_rtol = 1.0e-12; // 1.0e-10
   double fixed_atol = 1.0e-15;

   double initial_res_norm = -1.0;

   bool compute_error = true;

   // Main loop (with AMR or uniform refinement depending on the predefined macro AMR)
   for (int it = 0; ; it++)
   {
       if (verbose)
       {
          cout << "\nAMR iteration " << it << "\n";
          cout << "Number of unknowns: " << global_dofs << "\n\n";
       }

#ifdef DIVFREE_SETUP
       // finding a particular solution
       partsol_lvls.Prepend(new BlockVector(problem->GetTrueOffsets()));
       *partsol_lvls[0] = 0.0;

#ifdef MULTILEVEL_PARTSOL
       partsol_funct_lvls.Prepend(new BlockVector(problem->GetTrueOffsetsFunc()));
#endif

       div_rhs_lvls.Prepend(new Vector(problem->GetRhs().GetBlock(numblocks - 1).Size()));
       *div_rhs_lvls[0] = problem->GetRhs().GetBlock(numblocks - 1);
       if (verbose && it == 0)
           std::cout << "div_rhs norm = " << div_rhs_lvls[0]->Norml2() / sqrt (div_rhs_lvls[0]->Size()) << "\n";

#ifdef RECOARSENING_AMR
       if (verbose)
           std::cout << "Starting re-coarsening and re-solving part \n";

       // recoarsening constraint rhsides from finest to coarsest level
       for (int l = 1; l < div_rhs_lvls.Size(); ++l)
           hierarchy->GetTruePspace(SpaceName::L2,l - 1)->MultTranspose(*div_rhs_lvls[l-1], *div_rhs_lvls[l]);

       // re-solving all the problems with coarsened rhs, from coarsest to finest
       // and using the previous soluition as a starting guess
       int coarsest_lvl = prob_hierarchy->Nlevels() - 1;
       for (int l = coarsest_lvl; l > 0; --l) // l = 0 could be included actually after testing
       {
           ProblemType * problem_l = prob_hierarchy->GetProblem(l);
           FOSLSDivfreeProblem * problem_l_divfree = divfreeprob_hierarchy->GetProblem(l);

           /*
           problem_l_divfree->ConstructDivfreeHpMats();
           problem_l_divfree->CreateOffsetsRhsSol();
           BlockOperator * problem_l_divfree_op = ConstructDivfreeProblemOp(*problem_l_divfree, *problem_l);
           problem_l_divfree->ResetOp(*problem_l_divfree_op);
           divfreeprob_hierarchy->ConstructCoarsenedOps();

           problem_l_divfree->InitSolver(verbose);
           // creating a preconditioner for the divfree problem
           problem_l_divfree->CreatePrec(*problem_l_divfree->GetOp(), prec_option, verbose);
           problem_l_divfree->ChangeSolver();
           problem_l_divfree->UpdateSolverPrec();
           */

           // finding a new particular solution for the new rhs
#ifdef MULTILEVEL_PARTSOL
           Vector partsol_guess(partsol_funct_lvls[l]->Size());//partsol_finder->Size());
           partsol_guess = 0.0;

#ifdef      CLEVER_STARTING_PARTSOL
           if (l < coarsest_lvl)
           {
               BlockVector partsol_guess_viewer(partsol_guess.GetData(), problem_l->GetTrueOffsetsFunc());
               for (int blk = 0; blk < numblocks_funct; ++blk)
                   hierarchy->GetTruePspace((*space_names_funct)[blk], l - 1)->Mult
                           (partsol_funct_lvls[l + 1]->GetBlock(blk), partsol_guess_viewer.GetBlock(blk));
           }
#endif
           partsol_finder->FindParticularSolution(l, (HypreParMatrix&)problem_l->GetOp_nobnd()->GetBlock(numblocks_funct,0),
                                                  partsol_guess, *partsol_funct_lvls[l], *div_rhs_lvls[l], verbose);

           for (int blk = 0; blk < numblocks_funct; ++blk)
               partsol_lvls[l]->GetBlock(blk) = partsol_funct_lvls[l]->GetBlock(blk);
#else
           HypreParMatrix * B_hpmat = dynamic_cast<HypreParMatrix*>(&problem_l->GetOp()->GetBlock(numblocks - 1,0));
           ParGridFunction * partsigma = FindParticularSolution(problem_l->GetPfes(0), *B_hpmat, *div_rhs_lvls[l], verbose);
           partsigma->ParallelProject(partsol_lvls[l]->GetBlock(0));
#endif // for #else for #ifdef MULTILEVEL_PARTSOL

           // a check that the particular solution does satisfy the divergence constraint after all
           HypreParMatrix & Constr = (HypreParMatrix&)(problem_l->GetOp()->GetBlock(numblocks - 1, 0));
           Vector tempc(Constr.Height());
           Constr.Mult(partsol_lvls[l]->GetBlock(0), tempc);
           tempc -= *div_rhs_lvls[l];
           double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
           MFEM_ASSERT (res_constr_norm < 1.0e-10, "");

#ifdef NEW_INTERFACE
           MFEM_ABORT("Not ready yet \n");
#endif // end of #ifdef NEW_INTERFACE

           //  creating the solution and right hand side for the divfree problem
           BlockVector rhs(problem_l_divfree->GetTrueOffsets());

           BlockVector temp(problem_l->GetTrueOffsets());
           problem_l->GetOp()->Mult(*partsol_lvls[l], temp);
           temp *= -1;
           temp += problem_l->GetRhs();

           const HypreParMatrix * divfree_hpmat = &problem_l_divfree->GetDivfreeHpMat();
           divfree_hpmat->MultTranspose(temp.GetBlock(0), rhs.GetBlock(0));
           if (strcmp(space_for_S,"H1") == 0)
               rhs.GetBlock(1) = temp.GetBlock(1);

           // solving the div-free problem
#ifdef CLEVER_STARTING_GUESS
           // if it's not the coarsest level, we reuse the previous solution as a starting guess
           if (l < coarsest_lvl)
               divfreeprob_hierarchy->GetTrueP(l)->Mult
                       (divfreeprob_hierarchy->GetProblem(l + 1)->GetSol(), problem_l_divfree->GetSol());

           // checking the residual
           BlockVector res(problem_l_divfree->GetTrueOffsets());
           problem_l_divfree->GetOp()->Mult(problem_l_divfree->GetSol(), res);
           res -= rhs;

           double res_norm = ComputeMPIVecNorm(comm, res, "", false);
           if (it == 0)
               initial_res_norm = res_norm;

           if (verbose)
               std::cout << "Initial res norm for div-free problem at level # "
                         << l << " = " << res_norm << "\n";

           double adjusted_rtol = fixed_rtol * initial_res_norm / res_norm;
           if (verbose)
               std::cout << "adjusted rtol = " << adjusted_rtol << "\n";

           problem_l_divfree->SetRelTol(adjusted_rtol);
           problem_l_divfree->SetAbsTol(fixed_atol);
#ifdef USE_GS_PREC
           if (it > 0)
           {
               prec_option = 100;
               problem_l_divfree->ResetPrec(prec_option);
           }
#endif

           problem_l_divfree->SolveProblem(rhs, problem_l_divfree->GetSol(), verbose, false);
#else
           problem_l_divfree->SolveProblem(rhs, verbose, false);
#endif // for #else for #ifdef CLEVER_STARTING_GUESS

       }

       if (verbose)
           std::cout << "Re-coarsening and re-solving part has been finished\n";

#endif // end of #ifdef RECOARSENING_AMR

#ifdef MULTILEVEL_PARTSOL

#ifdef DEBUGGING_CASE
       if (it == 0)
       {
           checkdiff = new Vector(div_rhs_lvls[0]->Size());
           *checkdiff = *div_rhs_lvls[0];
       }

       if (it == 1)
       {
           Vector temp(hierarchy->GetTruePspace(SpaceName::L2,0)->Width());
           hierarchy->GetTruePspace(SpaceName::L2,0)->MultTranspose(*div_rhs_lvls[0], temp);
           *checkdiff -= temp;

           checkdiff->Print();

           if (verbose)
               std::cout << "|| f_H - P^T f_h || " <<
                            checkdiff->Norml2() / sqrt (checkdiff->Size()) << "\n";

           /*
           *checkdiff += temp;
           Vector temp2;
           Vector finer_buff;
           // temp2 = Qh_1 f_h, living on the fine level
           partsol_finder->NewProjectFinerL2ToCoarser(0, div_rhs, temp2, finer_buff);

           // temp = P^T  * temp2
           hierarchy->GetTruePspace(SpaceName::L2,0)->MultTranspose(temp2, temp);
           *checkdiff -= temp;

           if (verbose)
               std::cout << "|| f_H - P^T Qh_1 f_h || " <<
                            checkdiff->Norml2() / sqrt (checkdiff->Size()) << "\n";
           */
       }

       if (it == 2)
       {
           MPI_Finalize();
           return 0;
       }
#endif

#ifdef PUREDIVCONSTRAINT
       BlockVector partsol_vec(special_problem->GetTrueOffsetsFunc());
       MFEM_ASSERT(partsol_vec.Size() == partsol_finder->Size(), "Something went wrong");
       partsol_finder->FindParticularSolution(*partsol_guess, partsol_vec, *div_rhs_lvls[0], verbose);
       true_partsol.GetBlock(0) = partsol_vec.GetBlock(0);
#else

       // define a starting guess for the particular solution finder
       Vector partsol_guess(partsol_finder->Size());

#ifdef CLEVER_STARTING_PARTSOL
       if (it > 0)
       {
           BlockVector partsol_guess_viewer(partsol_guess.GetData(), problem->GetTrueOffsetsFunc());
           for (int blk = 0; blk < numblocks_funct; ++blk)
               hierarchy->GetTruePspace((*space_names_funct)[blk], 0)
                       ->Mult(partsol_lvls[1]->GetBlock(blk), partsol_guess_viewer.GetBlock(blk));
       }
       else
#endif
       {
           partsol_guess = 0.0;
       }

       partsol_finder->FindParticularSolution(partsol_guess, *partsol_funct_lvls[0], *div_rhs_lvls[0], verbose);

       for (int blk = 0; blk < numblocks_funct; ++blk)
           partsol_lvls[0]->GetBlock(blk) = partsol_funct_lvls[0]->GetBlock(blk);
#endif

#else // not a multilevel particular solution finder
       HypreParMatrix * B_hpmat = dynamic_cast<HypreParMatrix*>(&problem->GetOp()->GetBlock(numblocks - 1,0));
       ParGridFunction * partsigma = FindParticularSolution(problem->GetPfes(0), *B_hpmat, *div_rhs_lvls[0], verbose);
       partsigma->ParallelProject(partsol_lvls[0]->GetBlock(0));
#endif
       // a check that the particular solution does satisfy the divergence constraint after all
       HypreParMatrix & Constr = (HypreParMatrix&)(problem->GetOp()->GetBlock(numblocks - 1, 0));
       Vector tempc(Constr.Height());
       Constr.Mult(partsol_lvls[0]->GetBlock(0), tempc);
       tempc -= *div_rhs_lvls[0];//problem->GetRhs().GetBlock(numblocks_funct);
       double res_constr_norm = ComputeMPIVecNorm(comm, tempc, "", false);
       MFEM_ASSERT (res_constr_norm < 1.0e-10, "");

       // creating the operator for the div-free problem
#ifdef NEWINTERFACE
       if (it > 0)
       {
           problem_divfree->InitSolver(verbose);

#ifdef MG_DIVFREEPREC

           int num_levels = hierarchy->Nlevels();

           int coarsest_level = num_levels - 1;
           Operator* CoarseSolver_mg = new CGSolver(comm);
           ((CGSolver*)CoarseSolver_mg)->SetAbsTol(sqrt(1e-32));
           ((CGSolver*)CoarseSolver_mg)->SetRelTol(sqrt(1e-12));
           ((CGSolver*)CoarseSolver_mg)->SetMaxIter(100);
           ((CGSolver*)CoarseSolver_mg)->SetPrintLevel(0);
           ((CGSolver*)CoarseSolver_mg)->SetOperator(*mgtools_divfree_hierarchy->GetOps()[coarsest_level]);
           ((CGSolver*)CoarseSolver_mg)->iterative_mode = false;

           BlockDiagonalPreconditioner * CoarsePrec_mg =
                   new BlockDiagonalPreconditioner(mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->ColOffsets());

           HypreParMatrix &blk00 = (HypreParMatrix&)(mgtools_divfree_hierarchy->GetBlockOps()[coarsest_level]->GetBlock(0,0));
           HypreSmoother * precU = new HypreSmoother(blk00, HypreSmoother::Type::l1GS, 1);
           ((BlockDiagonalPreconditioner*)CoarsePrec_mg)->SetDiagonalBlock(0, precU);

           casted_monolitGSSmoothers.SetSize(num_levels - 1);
           for (int l = 0; l < casted_monolitGSSmoothers.Size(); ++l)
               casted_monolitGSSmoothers[l] = mgtools_divfree_hierarchy->GetMonolitGSSmoothers()[l];

           GeneralMGprec =
                   new GeneralMultigrid(num_levels,
                                        //P_mg,
                                        mgtools_divfree_hierarchy->GetPs_bnd(),
                                        //Ops_mg,
                                        mgtools_divfree_hierarchy->GetOps(),
                                        *CoarseSolver_mg,
                                        //*mgtools_divfree_hierarchy->GetCoarsestSolver_Hcurl(),
                                        //Smoo_mg);
                                        casted_monolitGSSmoothers);
           problem_divfree->ChangeSolver();
           problem_divfree->SetPrec(*GeneralMGprec);
           problem_divfree->UpdateSolverPrec();
#else
           // creating a preconditioner for the divfree problem
           problem_divfree->CreatePrec(*problem_divfree->GetOp(), prec_option, verbose);
           problem_divfree->ChangeSolver();
           problem_divfree->UpdateSolverPrec();
#endif
       }
#else
       problem_divfree->ConstructDivfreeHpMats();
       problem_divfree->CreateOffsetsRhsSol();
       BlockOperator * problem_divfree_op = ConstructDivfreeProblemOp(*problem_divfree, *problem);
       problem_divfree->ResetOp(*problem_divfree_op);
       divfreeprob_hierarchy->ConstructCoarsenedOps();

       problem_divfree->InitSolver(verbose);
       // creating a preconditioner for the divfree problem
       problem_divfree->CreatePrec(*problem_divfree->GetOp(), prec_option, verbose);
       problem_divfree->ChangeSolver();
       problem_divfree->UpdateSolverPrec();
#endif

       //  creating the solution and right hand side for the divfree problem
       BlockVector rhs(problem_divfree->GetTrueOffsets());

       BlockVector temp(problem->GetTrueOffsets());
       problem->GetOp()->Mult(*partsol_lvls[0], temp);
       temp *= -1;
       temp += problem->GetRhs();

       const HypreParMatrix * divfree_hpmat = &problem_divfree->GetDivfreeHpMat();
       divfree_hpmat->MultTranspose(temp.GetBlock(0), rhs.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0)
           rhs.GetBlock(1) = temp.GetBlock(1);

       // solving the div-free problem
#ifdef CLEVER_STARTING_GUESS
       // if it's not the first iteration we reuse the previous solution as a starting guess
       if (it > 0)
           divfreeprob_hierarchy->GetTrueP(0)->Mult(*coarse_guess, problem_divfree->GetSol());

       // checking the residual
       BlockVector res(problem_divfree->GetTrueOffsets());
       problem_divfree->GetOp()->Mult(problem_divfree->GetSol(), res);
       res -= rhs;

       double res_norm = ComputeMPIVecNorm(comm, res, "", false);
       if (it == 0)
           initial_res_norm = res_norm;

       if (verbose)
           std::cout << "Initial res norm for div-free problem at iteration # "
                     << it << " = " << res_norm << "\n";

       double adjusted_rtol = fixed_rtol * initial_res_norm / res_norm;
       if (verbose)
           std::cout << "adjusted rtol = " << adjusted_rtol << "\n";

       problem_divfree->SetRelTol(adjusted_rtol);
       problem_divfree->SetAbsTol(fixed_atol);
#ifdef USE_GS_PREC
       if (it > 0)
       {
           prec_option = 100;
           problem_divfree->ResetPrec(prec_option);
       }
#endif

       //std::cout << "checking rhs norm for the first solve: " <<
                    //rhs.Norml2() /  sqrt (rhs.Size()) << "\n";

       problem_divfree->SolveProblem(rhs, problem_divfree->GetSol(), verbose, false);
#else
       problem_divfree->SolveProblem(rhs, verbose, false);
#endif

       // checking the residual afterwards
       {
           BlockVector res(problem_divfree->GetTrueOffsets());
           problem_divfree->GetOp()->Mult(problem_divfree->GetSol(), res);
           res -= rhs;

           double res_norm = ComputeMPIVecNorm(comm, res, "", false);
           if (verbose)
               std::cout << "Res norm after solving the div-free problem at iteration # "
                         << it << " = " << res_norm << "\n\n";
       }

       // special testing cheaper preconditioners!
       /*
       if (verbose)
           std::cout << "Performing a special check for the preconditioners iteration counts! \n";

       BlockVector special_guess(problem_divfree->GetTrueOffsets());
       special_guess = problem_divfree->GetSol();

       int el_index = problem_divfree->GetParMesh()->GetNE() / 2;
       for (int blk = 0; blk < problem_divfree->GetFEformulation().Nblocks(); ++blk)
       {
           ParFiniteElementSpace * pfes = problem_divfree->GetPfes(blk);

           Array<int> dofs;
           MFEM_ASSERT(num_procs == 1, "This works only in serial");
           pfes->GetElementDofs(el_index, dofs);

           for (int i = 0; i < dofs.Size(); ++i)
               //special_guess.GetBlock(blk)[dofs[i]] = 0.0;
               special_guess.GetBlock(blk)[dofs[i]] = problem_divfree->GetSol().GetBlock(blk)[dofs[i]] * 0.5;
       }

       BlockVector check_diff(problem_divfree->GetTrueOffsets());
       check_diff = special_guess;
       check_diff -= problem_divfree->GetSol();
       double check_diff_norm = ComputeMPIVecNorm(comm, check_diff, "", false);

       if (verbose)
           std::cout << "|| sol - special_guess || = " << check_diff_norm << "\n";

       int nnz_count = 0;
       for (int i = 0; i < check_diff.Size(); ++i)
           if (fabs(check_diff[i]) > 1.0e-8)
               ++nnz_count;

       if (verbose)
           std::cout << "nnz_count in the diff = " << nnz_count << "\n";

       std::cout << "checking rhs norm for the second solve: " <<
                    rhs.Norml2() /  sqrt (rhs.Size()) << "\n";
       problem_divfree->SolveProblem(rhs, special_guess, verbose, false);

       MPI_Finalize();
       return 0;
       */

       /// converting the solution back into sigma from Hdiv inside the problem
       /// (adding a particular solution as a part of the process)
       /// and checking the accuracy of the resulting solution

       BlockVector& problem_sol = problem->GetSol();
       problem_sol = 0.0;

       BlockVector& problem_divfree_sol = problem_divfree->GetSol();

#ifdef CLEVER_STARTING_GUESS
       if (it > 0)
           delete coarse_guess;
       coarse_guess = new BlockVector(problem_divfree->GetTrueOffsets());
       *coarse_guess = problem_divfree_sol;
#endif

       problem_divfree->GetDivfreeHpMat().Mult(1.0, problem_divfree_sol.GetBlock(0), 1.0, problem_sol.GetBlock(0));
       if (strcmp(space_for_S,"H1") == 0)
           problem_sol.GetBlock(1) = problem_divfree_sol.GetBlock(1);

       problem_sol += *partsol_lvls[0];

       if (compute_error)
           problem->ComputeError(problem_sol, verbose, false);

       // to make sure that problem has grfuns in correspondence with the problem_sol we compute here
       // though for now its coordination already happens in ComputeError()
       problem->DistributeToGrfuns(problem_sol);
#else // the case when the original problem is solved, i.e., no div-free formulation is used

#ifdef CLEVER_STARTING_GUESS
       // if it's not the first iteration we reuse the previous solution as a starting guess
       if (it > 0)
           prob_hierarchy->GetTrueP(0)->Mult(*coarse_guess, problem->GetSol());

       // checking the residual
       BlockVector res(problem->GetTrueOffsets());
       problem->GetOp()->Mult(problem->GetSol(), res);
       res -= problem->GetRhs();

       double res_norm = ComputeMPIVecNorm(comm, res, "", false);
       if (it == 0)
           initial_res_norm = res_norm;

       if (verbose)
           std::cout << "Initial res norm at iteration # " << it << " = " << res_norm << "\n";

       double adjusted_rtol = fixed_rtol * initial_res_norm / res_norm;
       if (verbose)
           std::cout << "adjusted rtol = " << adjusted_rtol << "\n";

       problem->SetRelTol(adjusted_rtol);
       problem->SetAbsTol(fixed_atol);
#ifdef USE_GS_PREC
       if (it > 0)
       {
           prec_option = 100;
           std::cout << "Resetting prec with the Gauss-Seidel preconditioners \n";
           problem->ResetPrec(prec_option);
       }
#endif

       //std::cout << "checking rhs norm for the first solve: " <<
                    //problem->GetRhs().Norml2() /  sqrt (problem->GetRhs().Size()) << "\n";

       problem->SolveProblem(problem->GetRhs(), problem->GetSol(), verbose, false);

       // checking the residual afterwards
       {
           BlockVector res(problem->GetTrueOffsets());
           problem->GetOp()->Mult(problem->GetSol(), res);
           res -= problem->GetRhs();

           double res_norm = ComputeMPIVecNorm(comm, res, "", false);
           if (verbose)
               std::cout << "Res norm after solving the problem at iteration # "
                         << it << " = " << res_norm << "\n";
       }

#else
       problem->Solve(verbose, false);
#endif

      BlockVector& problem_sol = problem->GetSol();
      if (compute_error)
      {
          problem->ComputeError(problem_sol, verbose, true);
          problem->ComputeBndError(problem_sol);
      }

      // special testing cheaper preconditioners!
      /*
      if (verbose)
          std::cout << "Performing a special check for the preconditioners iteration counts! \n";

      prec_option = 100;
      problem->ResetPrec(prec_option);

      BlockVector special_guess(problem->GetTrueOffsets());
      special_guess = problem->GetSol();

      int special_num = 1;
      Array<int> el_indices(special_num);
      for (int i = 0; i < special_num; ++i)
          el_indices[i] = problem->GetParMesh()->GetNE() / 2 + i;

      std::cout << "Number of elements where the sol was changed: " <<
                   special_num << "(" <<  special_num * 100.0 /
                   problem->GetParMesh()->GetNE() << "%) \n";

      for (int blk = 0; blk < problem->GetFEformulation().Nblocks(); ++blk)
      {
          ParFiniteElementSpace * pfes = problem->GetPfes(blk);

          Array<int> dofs;
          MFEM_ASSERT(num_procs == 1, "This works only in serial");

          for (int elind = 0; elind < el_indices.Size(); ++elind)
          {
              pfes->GetElementDofs(el_indices[elind], dofs);

              for (int i = 0; i < dofs.Size(); ++i)
                  //special_guess.GetBlock(blk)[dofs[i]] = 0.0;
                  special_guess.GetBlock(blk)[dofs[i]] =
                    problem->GetSol().GetBlock(blk)[dofs[i]] * 0.9;
          }
      }

      BlockVector check_diff(problem->GetTrueOffsets());
      check_diff = special_guess;
      check_diff -= problem->GetSol();
      double check_diff_norm = ComputeMPIVecNorm(comm, check_diff, "", false);

      if (verbose)
          std::cout << "|| sol - special_guess || = " << check_diff_norm << "\n";

      int nnz_count = 0;
      for (int i = 0; i < check_diff.Size(); ++i)
          if (fabs(check_diff[i]) > 1.0e-8)
              ++nnz_count;

      if (verbose)
          std::cout << "nnz_count in the diff = " << nnz_count << "\n";

      {
          // checking the residual
          BlockVector res(problem->GetTrueOffsets());
          problem->GetOp()->Mult(special_guess, res);
          res -= problem->GetRhs();

          double res_norm = ComputeMPIVecNorm(comm, res, "", false);

          if (verbose)
              std::cout << "Initial res norm for the second solve = " << res_norm << "\n";

          double adjusted_rtol = fixed_rtol * initial_res_norm / res_norm;
          if (verbose)
              std::cout << "adjusted rtol = " << adjusted_rtol << "\n";

          problem->SetRelTol(adjusted_rtol);
          problem->SetAbsTol(fixed_atol);
      }


      std::cout << "checking rhs norm for the second solve: " <<
                   problem->GetRhs().Norml2() /  sqrt (problem->GetRhs().Size()) << "\n";
      problem->SolveProblem(problem->GetRhs(), special_guess, verbose, false);

      //problem_sol = problem->GetSol();
      //if (compute_error)
          //problem->ComputeError(problem_sol, verbose, true);

      MPI_Finalize();
      return 0;
      */

#ifdef CLEVER_STARTING_GUESS
       if (it > 0)
           delete coarse_guess;
       coarse_guess = new BlockVector(problem->GetTrueOffsets());
       *coarse_guess = problem_sol;
#endif

#endif

       // 17. Send the solution by socket to a GLVis server.
       if (visualization)
       {
           ParGridFunction * sigma = problem->GetGrFun(0);
           ParGridFunction * S;
           if (strcmp(space_for_S,"H1") == 0)
               S = problem->GetGrFun(1);
           else
               S = (dynamic_cast<FOSLSProblem_HdivL2L2hyp*>(problem))->RecoverS();

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

       // 18. Call the refiner to modify the mesh. The refiner calls the error
       //     estimator to obtain element errors, then it selects elements to be
       //     refined and finally it modifies the mesh. The Stop() method can be
       //     used to determine if a stopping criterion was met.

#ifdef DIVFREE_SETUP
       if (feorder > 0)
       {
           MFEM_ABORT("feorder > 0 is not supported with divfree setup since Nedelec f.e. with"
                      " feorder > 0 requires ReorientTetMesh() which prevents from using any kind "
                      "of refinement afterwards \n");
       }
#endif

#ifdef AMR
       int nel_before = hierarchy->GetFinestParMesh()->GetNE();

       // testing with only 1 element marked for refinement
       //Array<int> els_to_refine(1);
       //els_to_refine = hierarchy->GetFinestParMesh()->GetNE() / 2;
       //hierarchy->GetFinestParMesh()->GeneralRefinement(els_to_refine);

       // true AMR
       refiner.Apply(*hierarchy->GetFinestParMesh());
       int nmarked_el = refiner.GetNumMarkedElements();
       if (verbose)
       {
           std::cout << "Marked elements percentage = " << 100 * nmarked_el * 1.0 / nel_before << " % \n";
           std::cout << "nmarked_el = " << nmarked_el << ", nel_before = " << nel_before << "\n";
           int nel_after = hierarchy->GetFinestParMesh()->GetNE();
           std::cout << "nel_after = " << nel_after << "\n";
           std::cout << "number of elements introduced = " << nel_after - nel_before << "\n";
           std::cout << "percentage (w.r.t to # before) of elements introduced = " <<
                        100.0 * (nel_after - nel_before) * 1.0 / nel_before << "% \n\n";
       }

       if (visualization)
       {
           const Vector& local_errors = estimator->GetLocalErrors();
           if (feorder == 0)
               MFEM_ASSERT(local_errors.Size() == problem->GetPfes(numblocks_funct)->TrueVSize(), "");

           FiniteElementCollection * l2_coll;
           if (feorder > 0)
               l2_coll = new L2_FECollection(0, dim);

           ParFiniteElementSpace * L2_space;
           if (feorder == 0)
               L2_space = problem->GetPfes(numblocks_funct);
           else
               L2_space = new ParFiniteElementSpace(problem->GetParMesh(), l2_coll);
           ParGridFunction * local_errors_pgfun = new ParGridFunction(L2_space);
           local_errors_pgfun->SetFromTrueDofs(local_errors);
           char vishost[] = "localhost";
           int  visport   = 19916;

           socketstream amr_sock(vishost, visport);
           amr_sock << "parallel " << num_procs << " " << myid << "\n";
           amr_sock << "solution\n" << *pmesh << *local_errors_pgfun <<
                         "window_title 'local errors, AMR iter No." << it <<"'" << flush;

           if (feorder > 0)
           {
               delete l2_coll;
               delete L2_space;
           }
       }

#else
       hierarchy->GetFinestParMesh()->UniformRefinement();
#endif

       if (refiner.Stop())
       {
          if (verbose)
             cout << "Stopping criterion satisfied. Stop. \n";
          break;
       }

       bool recoarsen = true;
       prob_hierarchy->Update(recoarsen);
       problem = prob_hierarchy->GetProblem(0);

#ifdef PUREDIVCONSTRAINT
       specialprob_hierarchy->Update(recoarsen);
       special_problem = specialprob_hierarchy->GetProblem(0);
       ReplaceBlockByIdentityHpmat(*special_problem->GetOp(), 0);
#endif

#ifdef DIVFREE_SETUP

#ifdef      NEWINTERFACE
       hierarchy->Update();
#ifdef      MG_DIVFREEPREC
       mgtools_divfree_hierarchy->GetProblem()->Update();
       problem_divfree->ConstructDivfreeHpMats();
       problem_divfree->CreateOffsetsRhsSol();
       BlockOperator * problem_divfree_op = ConstructDivfreeProblemOp(*problem_divfree, *problem);
       problem_divfree->ResetOp(*problem_divfree_op);

       mgtools_divfree_hierarchy->Update(recoarsen);
#endif
#else
       divfreeprob_hierarchy->Update(false);
       problem_divfree = divfreeprob_hierarchy->GetProblem(0);
#endif // for ifdef NEWINTERFACE

#ifdef      MULTILEVEL_PARTSOL

#ifdef          PUREDIVCONSTRAINT
       partsol_finder->UpdateProblem(*special_problem);
#else
       partsol_finder->UpdateProblem(*problem);
#endif
       partsol_finder->Update();
#else
       delete partsigma;
#endif // endif for MULTILEVEL_PARTSOL

#endif // endif for DIVFREE_SETUP

       if (fosls_func_version == 2)
       {
           // first option is just to delete and recreate the extra grid function
           // this is slightly different from the old approach when the pgfun was
           // updated (~ interpolated)
           /*
           delete extra_grfuns[0];
           extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
           extra_grfuns[0]->ProjectCoefficient(*problem->GetFEformulation().
                                               GetFormulation()->GetTest()->GetRhs());
           */

           // second option is to project it (which is quiv. to Update() in the
           // old variant w/o hierarchies
           Vector true_temp1(prob_hierarchy->GetProblem(1)->GetPfes(numblocks - 1)->TrueVSize());
           extra_grfuns[0]->ParallelProject(true_temp1);

           Vector true_temp2(prob_hierarchy->GetProblem(0)->GetPfes(numblocks - 1)->TrueVSize());
           prob_hierarchy->GetHierarchy().GetTruePspace(SpaceName::L2, 0)->Mult(true_temp1, true_temp2);
           delete extra_grfuns[0];
           extra_grfuns[0] = new ParGridFunction(problem->GetPfes(numblocks - 1));
           extra_grfuns[0]->SetFromTrueDofs(true_temp2);
       }

       // checking #dofs after the refinement
       global_dofs = problem->GlobalTrueProblemSize();

       if (global_dofs > max_dofs)
       {
          if (verbose)
             cout << "Reached the maximum number of dofs. Stop. \n";
          break;
       }

   }

   MPI_Finalize();
   return 0;
//#endif
}



