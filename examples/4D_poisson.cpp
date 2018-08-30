//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Description:  This example code solves a the Poisson equation
//                                  -laplacian(p) = g
//               by means of the least squares finite element method. Introducing the flux
//               unknown u = -grad(p) we have the system of equations:
//                                  div(u) = g
//                                  u+grad(p) = 0,
//               with dirichlet boundary conditions for p. This system of equations is then put into
//               variational form by the least squares method. Here, we use a given exact solution
//               and the corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (u) and piecewise discontinuous
//               polynomials (p).
//
//
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

#include "ls_temp.cpp"

using namespace std;
using namespace mfem;

//Define needed global constants
const double pi =3.141592653589793238462643383279502884;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
double gFun(const Vector & x);
double pboundary_condition(const Vector & x);

int main(int argc, char *argv[])
{
    StopWatch chrono;
    chrono.Clear();
    chrono.Start();
    
    cout.precision(16);
    // 1. Parse command-line options.
    const char *mesh_file = "../data/cube4d_24.MFEM";
    int order = 0;
    bool visualization = 1;
    int numofrefinement = 0;
    int maxdofs = 900000;
    double error_frac = .50;
    double betavalue = 0.1;
    int strat = 1;
    bool verbose = true;
    
    ofstream output;
    output.open("Output.txt");
    
    ofstream contraction;
    contraction.open("contraction.txt");
    contraction << "(DOF) (Face Estimated Error)  (Estimated Error) (DOF New Mesh) (Face Estimated Error New Mesh) (Estimated Error New Mesh)" <<endl;
    
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&maxdofs, "-r", "-refine","-r");
    args.AddOption(&strat, "-rs", "--refinementstrategy", "Which refinement strategy to implement for the LS Refiner");
    args.AddOption(&error_frac, "-ef","--errorfraction", "Weight in Dorfler Marking Strategy");
    args.AddOption(&betavalue, "-b","--beta", "Beta in the Difference Term of Estimator");

    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);
    
    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
    //    the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();
    
    // 3. Refine the mesh to increase the resolution. In this example we do
    //    numofrefinement of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than numofrefinement
    //    elements.
    {

        for (int l = 0; l < numofrefinement; l++)
        {
            mesh->UniformRefinement();
        }
    }
    
    // 4. Define a finite element space on the mesh. Here we use the
    //    Raviart-Thomas finite elements of the specified order.
    FiniteElementCollection *hdiv_coll;
    if ( dim == 4 )
    {
        hdiv_coll = new RT0_4DFECollection;
        if(verbose)
            cout << "RT: order 0 for 4D" << endl;
    }
    else
    {
        hdiv_coll = new RT_FECollection(order, dim);
        if(verbose)
            cout << "RT: order " << order << " for 3D" << endl;
    }

    if (dim == 4)
        MFEM_ASSERT(order==0, "Only lowest order elements are support in 4D!");
    FiniteElementCollection *h1_coll;
    if (dim == 4)
    {
        h1_coll = new LinearFECollection;
        if (verbose)
            cout << "H1 in 4D: linear elements are used" << endl;
    }
    else
    {
        h1_coll = new H1_FECollection(order+1, dim);
        if(verbose)
            cout << "H1: order " << order + 1 << " for 3D" << endl;
    }
    
    FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
    FiniteElementSpace *W_space = new FiniteElementSpace(mesh, h1_coll);
    
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream u_sock;
    socketstream p_sock;

    if (visualization) {
        u_sock.open(vishost, visport);
        p_sock.open(vishost,visport);
    }
    
    
    
    // 6. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient k(1.0);
    ConstantCoefficient zero(0.0);
    ConstantCoefficient negativeone(-1.0);
    FunctionCoefficient gcoeff(gFun);
    
    FunctionCoefficient pcoeff(pFun_ex);
    
    //boundary condition
    FunctionCoefficient pboundary(pboundary_condition);
    
    
    // 7. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet)
    
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    //Define the gridfunctions that hold the solutions to the differential equation
    GridFunction u,p;
    
    //Define an estimator associated with the least squares formulation. Generates an estimate of the error of each element, which is used to designate which elements to refine.
    
    LeastSquaresEstimator estimator(&u,&p,gcoeff);
    
    // Define a refinement scheme, this finds all elements such that their estimated error is greater than some fraction of the normalized total error, and refines those elements and any other elements required to keep the mesh conforming.
    
    NDLSRefiner refiner(estimator);
    refiner.SetTotalErrorFraction(error_frac);
    refiner.SetTotalErrorNormP(2.0);
    refiner.SetRefinementStrategy(strat);
    refiner.SetBetaCalc(0);
    refiner.SetBetaConstants(betavalue);
    refiner.version_difference =false;
    
    Array<int> block_offsets(3);
    block_offsets[0] = 0;
    
    //The adaptive mesh Refinement loop. This continues until the number of degress of freedom of the system exceeds maxdofs
    for (int iter = 0; ;++iter){
        
        std::cout << " AMR loop: " <<iter <<endl;
        
        
        //(re)define the required components of the block_offsets
        block_offsets[1] = R_space->GetVSize();
        block_offsets[2] = W_space->GetVSize();
        block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";
    
    // 8. Define the blockvectors for the rhs and solution of the linear system.
       
        
        BlockVector rhs(block_offsets),x(block_offsets);
        rhs = 0.0;
        
        

            x = 0.0;

  
        //assemble the right hand side
    LinearForm *fform(new LinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    //fform->AddDomainIntegrator(new VectorFEDomainDivFIntegrator(gcoeff));
    fform->AddDomainIntegrator(new VectordivDomainLFIntegrator(gcoeff));
    fform->Assemble();
    
    // 9. Assemble the finite element matrices for the operator
    //
    //                            D = [ M  B ]
    //                                [ B^T   X  ]
    
    
    //set up the boundary condition for p
    GridFunction pbcondition(W_space);
    pbcondition.ProjectCoefficient(pboundary);
    
        //Assemble the stiffness matrix
    BilinearForm *mVarf(new BilinearForm(R_space));
    MixedBilinearForm *bVarf(new MixedBilinearForm(W_space,R_space));
    BilinearForm *xVarf( new BilinearForm(W_space));
    
    mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
    mVarf->AddDomainIntegrator(new DivDivIntegrator(k));
    mVarf->Assemble();
    mVarf->Finalize();
    SparseMatrix &M(mVarf->SpMat());
    
    xVarf->AddDomainIntegrator(new DiffusionIntegrator);
    xVarf->Assemble();
    // xVarf->EliminateEssentialBC(ess_bdr, pbcondition, rhs.GetBlock(1));
    xVarf->Finalize();
    SparseMatrix &X (xVarf->SpMat());
    
        
    bVarf->AddDomainIntegrator(new MixedVectorGradientIntegrator(negativeone));
    bVarf->Assemble();
    // bVarf->EliminateTrialDofs(ess_bdr, pbcondition, rhs.GetBlock(0));
    bVarf->Finalize();
        
    SparseMatrix & B(bVarf->SpMat());
    SparseMatrix *BT = Transpose(B);
        
    BlockMatrix darcyMatrix(block_offsets);
    darcyMatrix.SetBlock(0,0, &M);
    darcyMatrix.SetBlock(0,1, &B);
    darcyMatrix.SetBlock(1,0, BT);
    darcyMatrix.SetBlock(1,1, &X);
        
    //Symmetric Blockwise Gauss-Seidel preconditioning
       Solver *invM = new GSSmoother(M);
       Solver *invX = new GSSmoother(X);
        
        invM->iterative_mode =false;
        invX->iterative_mode =false;
        
        BlockDiagonalPreconditioner prec(block_offsets);
        prec.SetDiagonalBlock(0, invM);
        prec.SetDiagonalBlock(1, invX);
        
    //11. Solve using Conjuagate Gradient.
    PCG(darcyMatrix,prec,rhs,x, 3, 5000, 1e-6, 0.0);

    
    // 12. Associate with u and p the information from the blockvector x. Compute the L2 error norms.
    u.SetSpace(R_space);
    u.SetSize(x.BlockSize(0));
    u.SetVector(x.GetBlock(0), 0);
    p.SetSpace(W_space);
    p.SetSize(x.BlockSize(1));
    p.SetVector(x.GetBlock(1), 0);

    int order_quad = max(2, 2*order+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }
    
    double err_p  = p.ComputeL2Error(pcoeff, irs);
    double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);
    
        // 12. Save the refined mesh and the solution. This output can be viewed later
        //     using GLVis: "glvis -m refined.mesh -g sol.gf".
        ofstream mesh_ofs("refined.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);
 
    // 15. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        u_sock.precision(14);
        u_sock << "solution\n" << *mesh << u << "window_title 'Sigma'" << endl;
        p_sock.precision(14);
        p_sock << "solution\n" << *mesh << p << "window_title 'Solution'" <<flush;
    }
        std::cout << "Face Estimated Error = " << refiner.GetTotalErrorEstimate(*mesh) <<endl;
        contraction << block_offsets.Last() << " " << refiner.GetTotalErrorEstimate(*mesh) << " ";
        //Display Useful Information
        std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";
        estimator.GetLocalErrors(); 
        std::cout << "Estimated Error = " << estimator.GetTotalError()<<endl;
        std::cout << "Actual Number of Elements: " << mesh->GetNE() << endl;
        contraction << estimator.GetTotalError() << " ";
        
        refiner.Apply(*mesh);
        if(refiner.Stop()){
            cout<<"refiner satisfied interal condition"<<endl;
            break;
        }
      

        

        if(maxdofs<block_offsets.Last())
        {
            cout<<"max dofs reached"<<endl;
            break;
        }
       

       delete R_space;
       delete W_space;

       R_space = new FiniteElementSpace(mesh, hdiv_coll);
       W_space = new FiniteElementSpace(mesh, h1_coll);
        // output the estimate for the solutions projected onto the new mesh.
    
        
    // 16. Free the used memory.
    delete fform;
    delete BT;
    delete mVarf;
    delete bVarf;
    delete xVarf;
    delete invX;
    delete invM;
}
    
    chrono.Stop();
    cout<<"Time Elapsed: " <<chrono.RealTime()<<endl;
    delete W_space;
    delete R_space;
    delete h1_coll;
    delete hdiv_coll;
    delete mesh;
    output<<chrono.RealTime()<<"0"<<"0"<<"0"<<endl;
    output.close();
    contraction.close();
    u_sock.close();
    p_sock.close();
    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi =x[0];
    double yi = x[1];
    if(x.Size() ==2){

        //double r2 = sqrt(xi*xi+yi*yi);
       // double omega = atan2(yi,xi);
       // double zi(0.0);
        //u[0] = 2.0/3.0*(xi*(sin(2.0/3.0*omega+pi/3.0))-yi*cos(2.0/3.0*omega+pi/3.0))/pow(r2,4.0/3.0);
        //u[1] = 2.0/3.0*(yi*(sin(2.0/3.0*omega+pi/3.0))+xi*cos(2.0/3.0*omega+pi/3.0))/pow(r2,4.0/3.0);
        u[0] = cos(xi*pi)*sin(yi*pi)*pi;
        u[1] = sin(xi*pi)*cos(yi*pi)*pi;
    }
    if (x.Size() == 3)
    {
        double zi = x[2];
        double lambda = 2.0/3.0;
        double r = sqrt(xi*xi+yi*yi+zi*zi);
        u[0] =lambda*xi*pow(r,(lambda-2.0));
        u[1] =lambda*yi*pow(r,(lambda-2.0));
        u[2] =lambda*zi*pow(r,(lambda-2.0));
    }
}

// True Solution of p
double pFun_ex(const Vector & x)
{
    if(x.Size()==2){
    double xi =x[0];
    double yi = x[1];
    //double r = sqrt(xi*xi+yi*yi);
    //double omega = atan2(yi,xi);
    //return pow(r,2.0/3.0)*sin((2.0/3.0)*omega+pi/3.0);
    return sin(pi*xi)*sin(yi*pi);
    } else if(x.Size() ==3) {
        double xi =x[0];
        double yi = x[1];
        double zi = x[2];
        double lambda = 2.0/3.0;
        double r = sqrt(xi*xi+yi*yi+zi*zi);
        return pow(r,lambda);
    } else if (x.Size() ==4){
        return 0.0;
    }
    mfem_error("Input should be 2 or 3 dimensional");
    return 0.0;
}

//RHS of the original Laplace equation. Take the laplacian of the true solution.
double gFun(const Vector & x)
{
    if(x.Size()==2){
        //double xi =x[0];
        //double yi = x[1];
        //double r = sqrt(xi*xi+yi*yi);
       // double omega = atan2(yi,xi);
        return -1.0;//-2*pi*pi*sin(xi*pi)*sin(pi*yi);
    } else if(x.Size() ==3) {
        double xi =x[0];
        double yi = x[1];
        double zi = x[2];
        double lambda =2.0/3.0;
        double r = sqrt(xi*xi+yi*yi+zi*zi);
        return lambda*(lambda+1)*pow(r,lambda-2.0);
    }else if(x.Size() ==4) {
        return -1.0;
    }
    mfem_error("x should have dimension 2 or 3" );
    return 0.0;
}

double pboundary_condition(const Vector & x)
{
    if(x.Size()==2)
    {
        return 0;// pFun_ex(x);
    } else {
    return pFun_ex(x);
    }
}
