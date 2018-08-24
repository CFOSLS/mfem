#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>


using namespace std;
using namespace mfem;
using std::unique_ptr;


class DivPart
{

public:

// Returns the particular solution, sigma
void div_part( int ref_levels,
               SparseMatrix *M_fine,
               SparseMatrix *B_fine,
               Vector &G_fine,
               Vector &F_fine,
               Array< SparseMatrix*> &P_W,
               Array< SparseMatrix*> &P_R,
               Array< SparseMatrix*> &Element_Elementc,
               Array< SparseMatrix*> &Element_dofs_R,
               Array< SparseMatrix*> &Element_dofs_W,
               HypreParMatrix * d_td_coarse_R,
               HypreParMatrix * d_td_coarse_W,
               Vector &sigma
              )
              {


StopWatch chrono;

Vector sol_p_c2f;
Vector vec1;

Vector rhs_l;
Vector comp;
Vector F_coarse;


Vector total_sig(P_R[0]->Height());
total_sig = .0;

chrono.Clear();
chrono.Start();

for (int l=0; l < ref_levels; l++)
     {
           // 1. Obtaining the relation Dofs_Coarse_Element
       SparseMatrix *R_t = Transpose(*Element_dofs_R[l]);
       SparseMatrix *W_t = Transpose(*Element_dofs_W[l]);

       MFEM_ASSERT(R_t->Width() == Element_Elementc[l]->Height() ,
                   "Element_Elementc matrix and R_t does not match");

       SparseMatrix *W_AE = Mult(*W_t,*Element_Elementc[l]);
       SparseMatrix *R_AE = Mult(*R_t,*Element_Elementc[l]);

       // 2. For RT elements, we impose boundary condition equal zero,
       //   see the function: GetInternalDofs2AE to obtained them

       SparseMatrix intDofs_R_AE;
       GetInternalDofs2AE(*R_AE,intDofs_R_AE);

       //  AE elements x localDofs stored in AE_R & AE_W
       SparseMatrix *AE_R =  Transpose(intDofs_R_AE);
       SparseMatrix *AE_W = Transpose(*W_AE);


       // 3. Right hand size at each level is of the form:
       //
       //   rhs = F - (P_W[l])^T inv((P_W[l]^T)(P_W[l]))(P_W^T)F

       rhs_l.SetSize(P_W[l]->Height());

       if(l ==0)
            rhs_l = F_fine;

       if (l>0)
           rhs_l = comp;

       comp.SetSize(P_W[l]->Width());

       F_coarse.SetSize(P_W[l]->Height());

       P_W[l]->MultTranspose(rhs_l,comp);

       SparseMatrix * P_WT = Transpose(*P_W[l]);
       SparseMatrix * P_WTxP_W = Mult(*P_WT,*P_W[l]);
       Vector Diag(P_WTxP_W->Size());
       Vector invDiag(P_WTxP_W->Size());
       P_WTxP_W->GetDiag(Diag);

       for(int m=0; m < P_WTxP_W->Size(); m++)
          invDiag(m) = comp(m)/Diag(m);


       P_W[l]->Mult(invDiag,F_coarse);



       rhs_l -=F_coarse;

       MFEM_ASSERT(rhs_l.Sum()<= 9e-11,
                   "Average of rhs at each level is not zero: " << rhs_l.Sum());


       if (l> 0) {

          // 4. Creating matrices for the coarse problem:
           SparseMatrix *P_WT2 = Transpose(*P_W[l-1]);
           SparseMatrix *P_RT2 = Transpose(*P_R[l-1]);

           SparseMatrix *B_PR = Mult(*B_fine, *P_R[l-1]);
           B_fine = Mult(*P_WT2, *B_PR);

           SparseMatrix *M_PR = Mult(*M_fine, *P_R[l-1]);
           M_fine = Mult(*P_RT2, *M_PR);

         }

       //5. Setting for the coarse problem
       DenseMatrix sub_M;
       DenseMatrix sub_B;
       DenseMatrix sub_BT;
       DenseMatrix invBB;

       Vector sub_F;
       Vector sub_G;

       //Vector to Assamble the solution at level l
       Vector u_loc_vec(AE_W->Width());
       Vector p_loc_vec(AE_R->Width());

       u_loc_vec =0.0;
       p_loc_vec =0.0;


     for( int e = 0; e < AE_R->Height(); e++){

         Array<int> Rtmp_j(AE_R->GetRowColumns(e), AE_R->RowSize(e));
         Array<int> Wtmp_j(AE_W->GetRowColumns(e), AE_W->RowSize(e));

         // Setting size of Dense Matrices
         sub_M.SetSize(Rtmp_j.Size());
         sub_B.SetSize(Wtmp_j.Size(),Rtmp_j.Size());
         sub_BT.SetSize(Rtmp_j.Size(),Wtmp_j.Size());
         sub_G.SetSize(Rtmp_j.Size());
         sub_F.SetSize(Wtmp_j.Size());

         // Obtaining submatrices:
         M_fine->GetSubMatrix(Rtmp_j,Rtmp_j, sub_M);
         B_fine->GetSubMatrix(Wtmp_j,Rtmp_j, sub_B);
         sub_BT.Transpose(sub_B);

         sub_G  = .0;
         sub_F  = .0;

         rhs_l.GetSubVector(Wtmp_j, sub_F);


         Vector sig(Rtmp_j.Size());

        if (e ==1){
         MFEM_ASSERT(sub_F.Sum()<= 9e-11,
                   "checking global average at each level " << sub_F.Sum());
      }
         Vector sub_FF = sub_F;

        // Solving local problem:
         Local_problem(sub_M, sub_B, sub_G, sub_F,sig);

     if ( e ==1){

         // Checking if the local problems satisfy the condition
         Vector fcheck(Wtmp_j.Size());
         fcheck =.0;
         sub_B.Mult(sig, fcheck);
         fcheck-=sub_FF;
         MFEM_ASSERT(fcheck.Norml2()<= 9e-11,
                   "checking global average at each level " << fcheck.Norml2());


         }

     p_loc_vec.AddElementVector(Rtmp_j,sig);

     }

       Vector fcheck2(u_loc_vec.Size());
       fcheck2 = .0;
       B_fine->Mult(p_loc_vec, fcheck2);
       fcheck2-=rhs_l;
       MFEM_ASSERT(fcheck2.Norml2()<= 9e-11,
                   "checking global solution at each level " << fcheck2.Norml2());


        // Final Solution ==
        if (l>0){
          for (int k = l-1; k>=0; k--){

            vec1.SetSize(P_R[k]->Height());
            P_R[k]->Mult(p_loc_vec, vec1);
            p_loc_vec = vec1;

          }
        }

        total_sig +=p_loc_vec;

        MFEM_ASSERT(fcheck2.Norml2()<= 9e+9,
                   "checking global solution added" << total_sig.Norml2());

    }


 // The coarse problem::

     SparseMatrix *M_coarse;
     SparseMatrix *B_coarse;
     Vector FF_coarse(P_W[ref_levels-1]->Width());

     rhs_l +=F_coarse;
     P_W[ref_levels-1]->MultTranspose(rhs_l, FF_coarse );

     SparseMatrix *P_WT2 = Transpose(*P_W[ref_levels-1]);
         SparseMatrix *P_RT2 = Transpose(*P_R[ref_levels-1]);

         SparseMatrix *B_PR = Mult(*B_fine, *P_R[ref_levels-1]);
         B_coarse = Mult(*P_WT2, *B_PR);

         SparseMatrix *M_PR = Mult(*M_fine, *P_R[ref_levels-1]);

         M_coarse =  Mult(*P_RT2, *M_PR);

         Vector sig_c(B_coarse->Width());

         auto d_td_M = d_td_coarse_R->LeftDiagMult(*M_coarse);
         HypreParMatrix *d_td_T = d_td_coarse_R->Transpose();

         HypreParMatrix *M_Global = ParMult(d_td_T, d_td_M);

         auto B_Global = d_td_coarse_R->LeftDiagMult(*B_coarse,d_td_coarse_W->GetColStarts());
         HypreParMatrix *BT = B_Global->Transpose();

     Vector Truesig_c(B_Global->Width());

   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = M_Global->Width();
   block_offsets[2] = B_Global->Height();
   block_offsets.PartialSum();

   BlockOperator coarseMatrix(block_offsets);
   coarseMatrix.SetBlock(0,0, M_Global);
   coarseMatrix.SetBlock(0,1, BT);
   coarseMatrix.SetBlock(1,0, B_Global);


   BlockVector trueX(block_offsets), trueRhs(block_offsets);
   trueRhs =0;
   trueRhs.GetBlock(1)= FF_coarse;


   // 9. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement

   HypreParMatrix *MinvBt = B_Global->Transpose();
   HypreParVector *Md = new HypreParVector(MPI_COMM_WORLD, M_Global->GetGlobalNumRows(),
                                           M_Global->GetRowStarts());
   M_Global->GetDiag(*Md);

   MinvBt->InvScaleRows(*Md);
   HypreParMatrix *S = ParMult(B_Global, MinvBt);

   //HypreSolver *invM, *invS;
   auto invM = new HypreDiagScale(*M_Global);
   auto invS = new HypreBoomerAMG(*S);
   invS->SetPrintLevel(0);
      invM->iterative_mode = false;
   invS->iterative_mode = false;

   BlockDiagonalPreconditioner *darcyPr = new BlockDiagonalPreconditioner(
      block_offsets);
   darcyPr->SetDiagonalBlock(0, invM);
   darcyPr->SetDiagonalBlock(1, invS);

   // 12. Solve the linear system with MINRES.
   //     Check the norm of the unpreconditioned residual.

   int maxIter(50000);
   double rtol(1.e-12);
   double atol(1.e-12);


   MINRESSolver solver(MPI_COMM_WORLD);
   solver.SetAbsTol(atol);
   solver.SetRelTol(rtol);
   solver.SetMaxIter(maxIter);
   solver.SetOperator(coarseMatrix);
   solver.SetPreconditioner(*darcyPr);
   solver.SetPrintLevel(0);
   trueX = 0.0;
   solver.Mult(trueRhs, trueX);
   chrono.Stop();

   //cout << "CG converged in " << solver.GetNumIterations() << " iterations" <<endl;
   cout << "MINRES solver took " << chrono.RealTime() << "s. \n";

   Truesig_c = trueX.GetBlock(0);

   d_td_coarse_R->Mult(Truesig_c,sig_c);
  for (int k = ref_levels-1; k>=0; k--){

            vec1.SetSize(P_R[k]->Height());
            P_R[k]->Mult(sig_c, vec1);
            sig_c.SetSize(P_R[k]->Height());
            sig_c = vec1;

        }

   total_sig+=sig_c;
   sigma.SetSize(total_sig.Size());
   sigma = total_sig;




};

void Dofs_AE(SparseMatrix &Element_Dofs, const SparseMatrix &Element_Element_coarse, SparseMatrix &Dofs_Ae)
{
        // Returns a SparseMatrix with the relation dofs to Element coarse.
        SparseMatrix *R_T = Transpose(Element_Dofs);
        SparseMatrix *Dofs_AE = Mult(*R_T,Element_Element_coarse);
        SparseMatrix *AeDofs = Transpose(*Dofs_AE);
        Dofs_Ae = *AeDofs;
};


        void Elem2Dofs(const FiniteElementSpace &fes, SparseMatrix &Element_to_dofs)
{
  // Returns a SparseMatrix with the relation Element to Dofs
  int * I = new int[fes.GetNE()+1];
  Array<int> vdofs_R;
  Array<int> dofs_R;

  I[0] = 0;
  for (int i = 0; i < fes.GetNE(); i++)
    {
      fes.GetElementVDofs(i, vdofs_R);
      I[i+1] = I[i] + vdofs_R.Size();
    }
  int * J = new int[I[fes.GetNE()]];
  double * data = new double[I[fes.GetNE()]];

  for (int i = 0; i<fes.GetNE(); i++)
    {
      // Returns indexes of dofs in array for ith' elements'
      fes.GetElementVDofs(i,vdofs_R);
      fes.AdjustVDofs(vdofs_R);
      for (int j = I[i];j<I[i+1];j++)
        {
          J[j] = vdofs_R[j-I[i]];
          data[j] =1;
        }

    }
  SparseMatrix A(I,J,data,fes.GetNE(), fes.GetVSize());
  Element_to_dofs.Swap(A);
};

void GetInternalDofs2AE(const SparseMatrix &R_AE, SparseMatrix &B)
{
  /* Returns a SparseMatrix with the relation InteriorDofs to Coarse Element.
   * This is use for the Raviart-Thomas dofs, which vanish at the
   * boundary of the coarse elements.
   *
   * row.Size() ==2, means, it share by 2 AE
   *
   * For the lowest order case:
   * row.Size()==1, and data=1, means bdry
   * row.Size()==1, and data=2, means interior
   */

  int nnz=0;
  int * R_AE_i = R_AE.GetI();
  int * R_AE_j = R_AE.GetJ();
  double * R_AE_data = R_AE.GetData();

  int * out_i = new int [R_AE.Height()+1];

  // Find Hdivdofs_interior_AE
  for (int i=0; i<R_AE.Height(); i++)
    {
      out_i[i]= nnz;
      for (int j= R_AE_i[i]; j< R_AE_i[i+1]; j++)
        if (R_AE_data[j]==2)
           nnz++; // If the degree is share by two elements
    }
  out_i[R_AE.Height()] = nnz;

  int * out_j = new int[nnz];
  double * out_data = new double[nnz];
  nnz = 0;

  for (int i=0; i< R_AE.Height(); i++)
    for (int j=R_AE_i[i]; j<R_AE_i[i+1]; j++)
      if (R_AE_data[j] == 2)
        out_j[nnz++] = R_AE_j[j];

  // Forming the data array:
  std::fill_n(out_data, nnz, 1);

  SparseMatrix out(out_i, out_j, out_data, R_AE.Height(),
                   R_AE.Width());
  B.Swap(out);
};



void Local_problem(const DenseMatrix &sub_M,  DenseMatrix &sub_B, Vector &Sub_G, Vector &sub_F, Vector &sigma){
                  // Returns sigma local

                  DenseMatrixInverse invM_loc(sub_M);
                  DenseMatrix sub_BT(sub_B.Width(), sub_B.Height());
                  sub_BT.Transpose(sub_B);

                  DenseMatrix invM_BT(sub_B.Width());
          invM_loc.Mult(sub_BT,invM_BT);


          /* Solving the local problem:
                  *
              * Msig + B^tu = G
              * Bsig        = F
              *
              * sig =  M^{-1} B^t(-u) + M^{-1} G
              *
              * B M^{-1} B^t (-u) = F
              */

          DenseMatrix B_invM_BT;
          B_invM_BT = 0.0;
          B_invM_BT.SetSize(sub_B.Height());

          Mult(sub_B, invM_BT, B_invM_BT);

          Vector one(sub_B.Height());
          one = 0.0;
          one[0] =1;
          B_invM_BT.SetRow(0,0);
          B_invM_BT.SetCol(0,0);
          B_invM_BT.SetCol(0,one);
          B_invM_BT(0,0)=1;


          DenseMatrixInverse inv_BinvMBT(B_invM_BT);

          Vector invMG(sub_M.Size());
          invM_loc.Mult(Sub_G,invMG);

          sub_F[0] = 0;
          Vector uu(sub_B.Height());
          inv_BinvMBT.Mult(sub_F, uu);
          invM_BT.Mult(uu,sigma);
          sigma += invMG;
};




};




void pFun_ex(const Vector & x, Vector & u); // Exact Solution
double fFun(const Vector & x); // source f
double uexact(const Vector & x);



int main(int argc, char *argv[])
{
  StopWatch chrono;

  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  bool verbose = (myid == 0);

  // 2. Access to the mesh file.

  //const char *mesh_file = "../data/beam-quad.mesh"; // 2D
  const char *mesh_file = "../data/square1.mesh";

 // Mesh *mesh = new Mesh(mesh_file, 1, 0);
  Mesh *mesh = new Mesh(2,2,2,mfem::Element::HEXAHEDRON,1);

  // number of levels
  int ref_levels = 2;
  int order = 0; // lowest order
  bool visualization = 0;


  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&ref_levels, "-r", "--ref",
                 "Finite element refinement.");

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


  int dim = mesh->Dimension();
  mesh->UniformRefinement();
  //mesh->UniformRefinement();


   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   // Setting Finite elements
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   ParFiniteElementSpace *R_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
   ParFiniteElementSpace *W_space = new ParFiniteElementSpace(pmesh, l2_coll);

   ParFiniteElementSpace *coarseR_space = new ParFiniteElementSpace(pmesh, hdiv_coll);
   ParFiniteElementSpace *coarseW_space = new ParFiniteElementSpace(pmesh, l2_coll);

   // Input to the algorithm::

   Array< SparseMatrix*> P_W(ref_levels);
   Array< SparseMatrix*> P_R(ref_levels);
   Array< SparseMatrix*> Element_dofs_R(ref_levels);
   Array< SparseMatrix*> Element_dofs_W(ref_levels);
  // Array< int * > Sol_sig_level(ref_levels);

   const SparseMatrix* P_W_local;
   const SparseMatrix* P_R_local;

   // Dofs_TrueDofs at each space:

   auto d_td_coarse_R = coarseR_space->Dof_TrueDof_Matrix();
   auto d_td_coarse_W = coarseW_space->Dof_TrueDof_Matrix();

   DivPart divp;


   for (int l = 0; l < ref_levels+1; l++){
     if (l > 0){
       W_space->Update();
       R_space->Update();
       pmesh->UniformRefinement();

       P_W_local = ((const SparseMatrix *)W_space->GetUpdateOperator());
       P_R_local = ((const SparseMatrix *)R_space->GetUpdateOperator());

       SparseMatrix* R_Element_to_dofs1 = new SparseMatrix();
       SparseMatrix* W_Element_to_dofs1 = new SparseMatrix();

       divp.Elem2Dofs(*R_space, *R_Element_to_dofs1);
       divp.Elem2Dofs(*W_space, *W_Element_to_dofs1);

       P_W[ref_levels -l] = new SparseMatrix ( *P_W_local);
       P_R[ref_levels -l] = new SparseMatrix ( *P_R_local);

       Element_dofs_R[ref_levels - l] = R_Element_to_dofs1;
       Element_dofs_W[ref_levels - l] = W_Element_to_dofs1;

     }
   }



   // Setting boundary conditions if any:
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr=0;
   ess_bdr[0]=1;
   R_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);
   FunctionCoefficient fcoeff(fFun);


   ParBilinearForm *mVarf(new ParBilinearForm(R_space));
   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->Assemble();
   mVarf->Finalize();
   SparseMatrix &M_fine(mVarf->SpMat());

   ParMixedBilinearForm *bVarf(new ParMixedBilinearForm(R_space, W_space));
   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();
   SparseMatrix &B_fine = bVarf->SpMat();

   SparseMatrix *M_local = &M_fine;


   SparseMatrix *B_local = &B_fine;

   //Right hand size
   Vector F_fine(P_W[0]->Height());
   Vector G_fine(P_R[0]->Height());


   ParLinearForm gform(R_space);
   gform.Update();
   gform.Assemble();

   ParLinearForm fform(W_space);
   fform.Update();
   fform.AddDomainIntegrator(new DomainLFIntegrator(fcoeff));
   fform.Assemble();

   F_fine = fform;
   G_fine = .0;

   FunctionCoefficient uex(uexact);


   Vector sigma;

   divp.div_part(ref_levels,
            M_local, B_local,
            G_fine,
            F_fine,
            P_W, P_R, P_W,
            Element_dofs_R,
            Element_dofs_W,
            d_td_coarse_R,
            d_td_coarse_W,
            sigma);


   Vector sth(F_fine.Size());

   bVarf->SpMat().Mult(sigma, sth);
   sth -= F_fine;

   cout<< "final check " << sth.Norml2() <<endl;




   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete pmesh;
 //  delete W_coarse_space;

   MPI_Finalize();

   return 0;
}


void pFun_ex(const Vector & x, Vector & p)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   // u(0) = - exp(xi)*sin(yi)*cos(zi);
   //u(1) = - exp(xi)*cos(yi)*cos(zi);
   p(0) = (2*xi - 1)*(yi*yi - yi)*cos(zi);
   p(1) = (2*yi - 1)*(xi*xi - xi)*cos(zi);

}


double uexact(const Vector &x)
{
        double xi(x(0));
        double yi(x(1));

        return (xi*xi-xi)*(yi*yi-yi);
}

double fFun(const Vector & x)
{
  double xi(x(0));
  double yi(x(1));

  if (x.Size() ==3)
    {
   double  zi = x(2);

      return xi*xi-xi-2*yi*zi;
    }
  //return (2*xi-1 + 2*yi -1);
  return sin(2*M_PI*xi)*sin(2*M_PI*yi);
 // return (2*yi*yi-2*yi + 2*xi*xi-2*xi)-(-2/3);
  //return (3*yi*yi -1  + 3*xi*xi -1);
}




