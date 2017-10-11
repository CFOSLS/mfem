#include "mfem.hpp"

using namespace mfem;
using namespace std;
using std::unique_ptr;

#define NEW_STUFF

#ifdef NEW_STUFF

#if 0
// DenseBlockMatrix class
class BlockDenseMatrix : public DenseMatrix
{
public:
   //! Constructor for BlockDenseMatrices with the same block-structure for rows and
   //! columns.
   /**
    *  offsets: offsets that mark the start of each row/column block (size
    *  nRowBlocks+1).  Note: BlockDenseMatrix will not own/copy the data contained
    *  in offsets.
    */
   BlockDenseMatrix(const Array<int> & offsets);
   //! Constructor for general BlockDenseMatrices.
   /**
    *  row_offsets: offsets that mark the start of each row block (size
    *  nRowBlocks+1).  col_offsets: offsets that mark the start of each column
    *  block (size nColBlocks+1).  Note: BlockDenseMatrix will not own/copy the
    *  data contained in offsets.
    */
   BlockDenseMatrix(const Array<int> & row_offsets, const Array<int> & col_offsets);

   //! Add block op in the block-entry (iblock, iblock).
   /**
    * iblock: The block will be inserted in location (iblock, iblock).
    * op: the Operator to be inserted.
    * c: optional scalar multiple for this block.
    */
   void SetDiagonalBlock(int iblock, DenseMatrix *op, double c = 1.0);
   //! Add a block op in the block-entry (iblock, jblock).
   /**
    * irow, icol: The block will be inserted in location (irow, icol).
    * op: the Operator to be inserted.
    * c: optional scalar multiple for this block.
    */
   void SetBlock(int iRow, int iCol, Operator *op, double c = 1.0);

   //! Return the number of row blocks
   int NumRowBlocks() const { return nRowBlocks; }
   //! Return the number of column blocks
   int NumColBlocks() const { return nColBlocks; }

   //! Check if block (i,j) is a zero block
   int IsZeroBlock(int i, int j) const { return (op(i,j)==NULL) ? 1 : 0; }
   //! Return a reference to block i,j
   Operator & GetBlock(int i, int j)
   { MFEM_VERIFY(op(i,j), ""); return *op(i,j); }
   //! Return the coefficient for block i,j
   double GetBlockCoef(int i, int j) const
   { MFEM_VERIFY(op(i,j), ""); return coef(i,j); }
   //! Set the coefficient for block i,j
   void SetBlockCoef(int i, int j, double c)
   { MFEM_VERIFY(op(i,j), ""); coef(i,j) = c; }

   //! Return the row offsets for block starts
   Array<int> & RowOffsets() { return row_offsets; }
   //! Return the columns offsets for block starts
   Array<int> & ColOffsets() { return col_offsets; }

   /// Operator application
   virtual void Mult (const Vector & x, Vector & y) const;

   /// Action of the transpose operator
   virtual void MultTranspose (const Vector & x, Vector & y) const;

   ~BlockOperator();

   //! Controls the ownership of the blocks: if nonzero, BlockOperator will
   //! delete all blocks that are set (non-NULL); the default value is zero.
   int owns_blocks;

private:
   //! Number of block rows
   int nRowBlocks;
   //! Number of block columns
   int nColBlocks;
   //! Row offsets for the starting position of each block
   Array<int> row_offsets;
   //! Column offsets for the starting position of each block
   Array<int> col_offsets;
   //! 2D array that stores each block of the operator.
   Array2D<Operator *> op;
   //! 2D array that stores a coefficient for each block of the operator.
   Array2D<double> coef;

   //! Temporary Vectors used to efficiently apply the Mult and MultTranspose methods.
   mutable BlockVector xblock;
   mutable BlockVector yblock;
   mutable Vector tmp;
};
#endif


// Implements a multilevel solver for a minimization problem
// J(sigma,S) -> min under the constraint div sigma = f
// The solver is implemented algebraically, i.e.
// it takes the functional derivative as a 2x2 block operator
class GeneralMinConstrSolver : public IterativeSolver
{
private:
    const int strategy;    // reserved parameter for minimzation strategy
    int num_levels;
    int current_iterate;   // the behaviour is slightly different for the first iterate
    const Array< SparseMatrix*>& AE_e;
    const Array< SparseMatrix*>& el_to_dofs_R;
    const Array< SparseMatrix*>& el_to_dofs_H;
    const Array< SparseMatrix*>& el_to_dofs_W;
    const Array< SparseMatrix*>& P_R;
    const Array< SparseMatrix*>& P_H;
    const Array< SparseMatrix*>& P_W;
    const BlockMatrix& Funct;
    const SparseMatrix& Constr;

    bool higher_order;

    // temporary variables
    // FIXME: is it a good practice? should they be mutable?
    mutable SparseMatrix *AE_edofs_W;
    mutable SparseMatrix *AE_eintdofs_R;
    mutable SparseMatrix *AE_eintdofs_H;

protected:
    void SetUpRhsConstr(int level, Vector& Qlminus1_f, Vector& rhs_l);
    void SetUpLocal(int level);
    void SolveLocalProblems(int level, Vector &rhs_l);
    void SetUpCoarseProblem();
    void SolveCoarseProblem();
    SparseMatrix* GetElToIntDofs(int level, const char* Space);
    void SolveLocalProblem(DenseMatrix& sub_A, DenseMatrix& sub_B, Vector& sub_F, Vector& sub_G, Vector& sub_sig,
                           DenseMatrix& sub_C, DenseMatrix& sub_D, Vector& sub_s );
public:
    // constructor (empty for now)
    GeneralMinConstrSolver(int NumLevels,
                           const Array< SparseMatrix*> &AE_to_e,
                           const Array< SparseMatrix*> &El_to_dofs_R, const Array< SparseMatrix*> &El_to_dofs_H, const Array< SparseMatrix*> &El_to_dofs_W,
                           const Array< SparseMatrix*> &Proj_R, const Array< SparseMatrix*> &Proj_H, const Array< SparseMatrix*> &Proj_W,
                           const BlockMatrix& FunctBlockMat, const SparseMatrix& ConstrMat,
                           bool Higher_Order_Elements = false)
         : IterativeSolver(), strategy(0), num_levels(NumLevels), current_iterate(0),
           AE_e(AE_to_e),
           el_to_dofs_R(El_to_dofs_R), el_to_dofs_H(El_to_dofs_H), el_to_dofs_W(El_to_dofs_W),
           P_R(Proj_R), P_H(Proj_H), P_W(Proj_W),
           Funct(FunctBlockMat),
           Constr(ConstrMat),
           higher_order(Higher_Order_Elements)
    {}
    // main solving routine
    void Solve(const Vector& rhs, Vector& sigma, Vector& S);

};

void GeneralMinConstrSolver::Solve(const Vector& rhs, Vector& sigma, Vector& S)
{
    // 0. preliminaries

    // righthand side (from the divergence constraint) at level l
    Vector rhs_l(rhs.Size());
    rhs_l = rhs;
    // temporary storage for Q_{l-1} f
    Vector Qlminus1_f(rhs.Size());
    Qlminus1_f = rhs;

    // 1. loop over levels
    for (int l = 0; l < num_levels - 1; ++l)
    {
        // 1.1 set up the righthand side at level l
        SetUpRhsConstr(l, Qlminus1_f, rhs_l);

        SetUpLocal(l);

        // 1.2 solve the local problems at level l
        // FIXME: all factors of local matrices can be stored after the first solver iteration
        SolveLocalProblems(l, rhs_l);

    } // end of loop over levels

    // 2. set up coarse problem
    SetUpCoarseProblem();

    // 3. solve coarse problem
    SolveCoarseProblem();

    // 4. assemble the final solution

    ++current_iterate;

    return;
}

// Righthand side at level l is of the form:
//   rhs_l = (Q_l - Q_{l+1}) where Q_k is an orthogonal L2-projector: W -> W_k
// or, equivalently,
//   rhs_l = (I - Pi_{l-1,l}) rhs_{l-1},
// where Pi_{k,k+1} is an orthogonal L2-projector W_{k+1} -> W_k,
// and rhs_{l-1} = Q_{l-1} f (setting Q_0 = Id)
// Hence,
//   Pi_{l-1,l} = P_l * inv(P_l^T P_l) * P_l^T
// where P_l columns compose the basis of the coarser space.
void GeneralMinConstrSolver::SetUpRhsConstr(int level, Vector& Qlminus1_f, Vector& rhs_l)
{
    // 1.
    rhs_l = Qlminus1_f;

    // 2. Computing Qlminus1_f (new): = Q_l f = Pi_{l-1,l} * (Q_{l-1} f)
    // FIXME: memory efficiency can be increased by using pre-allocated work array here
    Vector temp(P_W[level]->Height());

    P_W[level]->MultTranspose(Qlminus1_f,temp);

    // FIXME: Can't this be done in a local way, without large mat-mat multiplication?
    SparseMatrix * P_WT = Transpose(*P_W[level]);
    SparseMatrix * P_WTxP_W = mfem::Mult(*P_WT,*P_W[level]);
    Vector Diag(P_WTxP_W->Size());
    Vector invDiag(P_WTxP_W->Size());
    P_WTxP_W->GetDiag(Diag);

    for ( int m = 0; m < P_WTxP_W->Size(); ++m)
        invDiag(m) = temp(m) / Diag(m);

    // FIXME: Unchecked alternative is below
    // something like temp (new) (AE) = temp(AE) / number of fine elements inside given AE
    // which is a corresponding row size in AE_E
    // and using temp intead if invDiag

    Vector F_coarse;
    P_W[level]->Mult(invDiag,F_coarse);

    // 3. Setting rhs_l = Q_{l-1} f - Pi_{l-1,l} * Q_{l-1} f
    rhs_l -= Qlminus1_f;

    return;
}

// Returns a pointer to a SparseMatrix which stores
// the relation between agglomareated elements (AEs)
// and fine-grid internal (w.r.t. to AEs) dofs.
// FIXME: for now works only for the lowest order case
// For higher order RT elements there will be two parts,
// one for boundary dofs at fine-grid element faces (implemented like here)
// and a different treatment for internal (w.r.t. to fine elements) dofs
// And don't forget about the boundary dofs!
SparseMatrix * GeneralMinConstrSolver::GetElToIntDofs(int level, const char* Space)
{
    SparseMatrix *el_to_dofs;
    if (strcmp(Space,"R") == 0)
        el_to_dofs = el_to_dofs_R[level];
    else if (strcmp(Space,"H") == 0)
        el_to_dofs = el_to_dofs_H[level];
    else
        MFEM_ABORT("Bad value of Space, must be W or H!");


    // creating dofs_to_AE relation table
    SparseMatrix * dofs_AE_R = Transpose(*mfem::Mult(*AE_e[level], *el_to_dofs));
    int ndofs = dofs_AE_R->Height();

    /*
     * out-dated
    // creating a vector which marks internal (w.r.t to fine-grid elements) dofs
    // over all fine-grid dofs
    Array<bool> dof_is_inside;
    if (higher_order)
    {
        // FIXME: actually, one needs only to know the values at the diagonal
        SparseMatrix * dofs_el_dofs = mfem::Mult(*el_to_dofs_R[level], *Transpose(*el_to_dofs_R[level]));
        double * dofs_el_dofs_data = dofs_el_dofs->GetData();
        int * dofs_el_dofs_i = dofs_el_dofs->GetI();

        dof_is_inside.SetSize(ndofs);
        for (int dof = 0; dof < ndofs; ++dof)
        {
            dof_is_inside[dof] = false;
            if (dofs_el_dofs_data[dofs_el_dofs_i[dof]] == 1)
                dof_is_inside[dof] = true;
        }

        delete dofs_el_dofs;
    }
    */

    // stores for each dof true if it is at the boundary of the entire domain
    // or false otherwise.
    Array<bool> boundary_dofs;
    if (higher_order)
    {
        for (int dof = 0; dof < ndofs; ++dof)
        {
            // FIXME: maybe it should be an input parameter for the class, boundary dofs for all levels, obtained from the f.e. spaces
            boundary_dofs[dof] = false;
        }
    }

    int * dofs_AE_R_i = dofs_AE_R->GetI();
    int * dofs_AE_R_j = dofs_AE_R->GetJ();
    double * dofs_AE_R_data = dofs_AE_R->GetData();

    int * innerdofs_AE_i = new int [ndofs + 1];

    // computing the number of internal degrees of freedom in all AEs
    int nnz = 0;
    for (int dof = 0; dof < ndofs; ++dof)
    {
        innerdofs_AE_i[dof]= nnz;
        for (int j = dofs_AE_R_i[dof]; j < dofs_AE_R_i[dof+1]; ++j)
            /*
            if (higher_order)
            {
                // if a dof is shared by two fine grid elements inside a single AE
                // OR (!) it is inside a fine-grid element,
                // then it is an internal dof for this AE
                if (dofs_AE_R_data[j] == 2 || dof_is_inside[dof])
                    nnz++;
            }
            else
            {
                // if a dof is shared by two fine grid elements inside a single AE,
                // then it is an internal dof for this AE
                if (dofs_AE_R_data[j] == 2)
                    nnz++;
            }
            */

            // if a dof is shared by two fine grid elements inside a single AE
            // OR [ if a dof belongs to only one AE and to only one fine-grid element
            // and is not at the domain boundary],
            // then it is an internal dof for this AE
            if (dofs_AE_R_data[j] == 2 || (higher_order && !boundary_dofs[dof] && dofs_AE_R_i[dof+1] - dofs_AE_R_i[dof] == 1 && dofs_AE_R_data[j] == 1))
                nnz++;

    }
    innerdofs_AE_i[ndofs] = nnz;

    // allocating j and data arrays for the created relation table
    int * innerdofs_AE_j = new int[nnz];
    double * innerdofs_AE_data = new double[nnz];

    int nnz_count = 0;
    for (int dof = 0; dof < ndofs; ++dof)
        for (int j = dofs_AE_R_i[dof]; j < dofs_AE_R_i[dof+1]; ++j)
            if (dofs_AE_R_data[j] == 2)
                innerdofs_AE_j[nnz_count++] = dofs_AE_R_j[j];

    std::fill_n(innerdofs_AE_data, nnz, 1);

    // creating a relation between internal fine-grid dofs (w.r.t to AE) and AEs,
    // keeeping zero rows for non-internal dofs
    SparseMatrix * innerdofs_AE = new SparseMatrix(innerdofs_AE_i, innerdofs_AE_j, innerdofs_AE_data,
                                                   dofs_AE_R->Height(), dofs_AE_R->Width());

    delete dofs_AE_R; // FIXME: or it can be saved and re-used if needed

    return Transpose(*innerdofs_AE);
}

// Computes prerequisites required for solving local problems at level l
// such as relation tables between AEs and internal fine-grid dofs
// and maybe smth else ... ?
void GeneralMinConstrSolver::SetUpLocal(int level)
{
    SparseMatrix* el_to_intdofs_R = GetElToIntDofs(level, "R");
    SparseMatrix* el_to_intdofs_H = GetElToIntDofs(level, "W");

    AE_edofs_W = mfem::Mult(*AE_e[level], *el_to_dofs_W[level]);
    AE_eintdofs_R = mfem::Mult(*AE_e[level], *el_to_intdofs_R);
    AE_eintdofs_H = mfem::Mult(*AE_e[level], *el_to_intdofs_H);

    return;
}


void GeneralMinConstrSolver::SolveLocalProblems(int level, Vector& rhs_l)
{
    // FIXME: factorization can be done only at the first solver iterate, stored and then re-used
    DenseMatrix sub_A;
    DenseMatrix sub_C;
    DenseMatrix sub_D;
    DenseMatrix sub_DT;
    DenseMatrix sub_B;
    DenseMatrix sub_BT;

    Vector sub_F;
    Vector sub_G;

    // vectors for assembled solution at level l
    Vector sig_loc_vec(AE_eintdofs_R->Width());
    sig_loc_vec = 0.0;

    Vector s_loc_vec(AE_eintdofs_H->Width());
    s_loc_vec = 0.0;

    SparseMatrix A_fine = Funct.GetBlock(0,0);
    SparseMatrix B_fine = Constr;
    SparseMatrix *C_fine, *D_fine;
    int nblocks = Funct.NumRowBlocks();
    if (nblocks > 1 && strategy == 0) // we have sigma and S in the functional and minimize over both
    {
        *C_fine = Funct.GetBlock(1,1);
        *D_fine = Funct.GetBlock(0,1);
    }

    // loop over all AE, solving a local problem in each AE
    int nAE = AE_edofs_W->Height();
    for( int AE = 0; AE < nAE; ++AE)
    {
        Array<int> Rtmp_j(AE_eintdofs_R->GetRowColumns(AE), AE_eintdofs_R->RowSize(AE));
        Array<int> * Htmp_j_pt;
        if (nblocks > 1 && strategy == 0) // we have sigma and S in the functional and minimize over both
            Htmp_j_pt = new Array<int> (AE_eintdofs_H->GetRowColumns(AE), AE_eintdofs_H->RowSize(AE));
        Array<int> Wtmp_j(AE_edofs_W->GetRowColumns(AE), AE_edofs_W->RowSize(AE));

        // Setting size of Dense Matrices
        sub_A.SetSize(Rtmp_j.Size());
        sub_B.SetSize(Wtmp_j.Size(),Rtmp_j.Size());
        //sub_BT.SetSize(Rtmp_j.Size(),Wtmp_j.Size());
        if (nblocks > 1 && strategy == 0) // we have sigma and S in the functional and minimize over both
        {
            sub_C.SetSize(Htmp_j_pt->Size());
            sub_D.SetSize(Htmp_j_pt->Size(),Rtmp_j.Size());
            //sub_DT.SetSize(Rtmp_j.Size(), Htmp_j_pt->Size());
        }

        //sub_G.SetSize(Rtmp_j.Size());
        //sub_F.SetSize(Wtmp_j.Size());

        // Obtaining submatrices:
        A_fine.GetSubMatrix(Rtmp_j,Rtmp_j, sub_A);
        B_fine.GetSubMatrix(Wtmp_j,Rtmp_j, sub_B);
        sub_BT.Transpose(sub_B);
        if (nblocks > 1 && strategy == 0) // we have sigma and S in the functional and minimize over both
        {
            C_fine->GetSubMatrix(*Htmp_j_pt,*Htmp_j_pt, sub_C);
            D_fine->GetSubMatrix(*Htmp_j_pt,Rtmp_j, sub_D);
            sub_DT.Transpose(sub_D);
        }

        sub_G  = .0;

        rhs_l.GetSubVector(Wtmp_j, sub_F);

        Vector sub_sig(Rtmp_j.Size());
        Vector sub_s(Htmp_j_pt->Size());

        MFEM_ASSERT(sub_F.Sum()<= 9e-11, "checking local average at each level " << sub_F.Sum());

        // Solving local problem at the agglomerate element AE:
        SolveLocalProblem(sub_A, sub_B, sub_F, sub_G, sub_sig, sub_C, sub_D, sub_s );

        sig_loc_vec.AddElementVector(Rtmp_j,sub_sig);
        s_loc_vec.AddElementVector(Rtmp_j,sub_s);

        // FIXME: is this a right way to delete Htmp_j?
        delete Htmp_j_pt;
    }

    std::cout << "SolveLocalProblems is not implemented yet! \n";
    return;
}

void GeneralMinConstrSolver::SolveLocalProblem (DenseMatrix& sub_A, DenseMatrix& sub_B, Vector& sub_F, Vector& sub_G, Vector& sub_sig,
                                                DenseMatrix& sub_C, DenseMatrix& sub_D, Vector& sub_s )
{
    return;
}

void GeneralMinConstrSolver::SetUpCoarseProblem()
{
    std::cout << "SetUpCoarseProblem is not implemented yet! \n";
    return;
}

void GeneralMinConstrSolver::SolveCoarseProblem()
{
    std::cout << "SolveCoarseProblem is not implemented yet! \n";
    return;
}

#endif

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
                   Vector &sigma,
                   Array<int>& ess_dof_coarsestlvl_list
                   )
    {
//        StopWatch chrono;

//        Vector sol_p_c2f;
        Vector vec1;

        Vector rhs_l;
        Vector comp;
        Vector F_coarse;

        Vector total_sig(P_R[0]->Height());
        total_sig = .0;

//        chrono.Clear();
//        chrono.Start();



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
            {
                //std::cout << "Diag(m) = " << Diag(m) << "\n";
                invDiag(m) = comp(m)/Diag(m);
            }

            //std::cout << "Diag(100) = " << Diag(100);
            //std::cout << "Diag(200) = " << Diag(200);
            //std::cout << "Diag(300) = " << Diag(300);


            P_W[l]->Mult(invDiag,F_coarse);



            rhs_l -=F_coarse;

            MFEM_ASSERT(rhs_l.Sum()<= 9e-11,
                        "Average of rhs at each level is not zero: " << rhs_l.Sum());


            if (l> 0) {

                // 4. Creating matrices for the coarse problem:
                SparseMatrix *P_WT2 = Transpose(*P_W[l-1]);
                SparseMatrix *P_RT2;
                if (M_fine)
                    P_RT2 = Transpose(*P_R[l-1]);

                SparseMatrix *B_PR = Mult(*B_fine, *P_R[l-1]);
                B_fine = Mult(*P_WT2, *B_PR);

                if (M_fine)
                {
                    SparseMatrix *M_PR = Mult(*M_fine, *P_R[l-1]);
                    M_fine = Mult(*P_RT2, *M_PR);
                }
            }

            //5. Setting for the coarse problem
            DenseMatrix sub_M;
            DenseMatrix sub_B;
            DenseMatrix sub_BT;
//            DenseMatrix invBB;

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
                if (M_fine)
                    sub_M.SetSize(Rtmp_j.Size());
                sub_B.SetSize(Wtmp_j.Size(),Rtmp_j.Size());
                sub_BT.SetSize(Rtmp_j.Size(),Wtmp_j.Size());
//                sub_G.SetSize(Rtmp_j.Size());
//                sub_F.SetSize(Wtmp_j.Size());

                // Obtaining submatrices:
                if (M_fine)
                    M_fine->GetSubMatrix(Rtmp_j,Rtmp_j, sub_M);
                B_fine->GetSubMatrix(Wtmp_j,Rtmp_j, sub_B);
                sub_BT.Transpose(sub_B);

//                sub_G  = .0;
//                sub_F  = .0;

                rhs_l.GetSubVector(Wtmp_j, sub_F);


                Vector sig(Rtmp_j.Size());

                MFEM_ASSERT(sub_F.Sum()<= 9e-11,
                            "checking local average at each level " << sub_F.Sum());

#ifdef MFEM_DEBUG
                Vector sub_FF = sub_F;
#endif

                // Solving local problem:
                Local_problem(sub_M, sub_B, sub_G, sub_F,sig);

#ifdef MFEM_DEBUG
                // Checking if the local problems satisfy the condition
                Vector fcheck(Wtmp_j.Size());
                fcheck =.0;
                sub_B.Mult(sig, fcheck);
                fcheck-=sub_FF;
                MFEM_ASSERT(fcheck.Norml2()<= 9e-11,
                            "checking local residual norm at each level " << fcheck.Norml2());
#endif

                p_loc_vec.AddElementVector(Rtmp_j,sig);

            } // end of loop over all elements at level l

#ifdef MFEM_DEBUG
            Vector fcheck2(u_loc_vec.Size());
            fcheck2 = .0;
            B_fine->Mult(p_loc_vec, fcheck2);
            fcheck2-=rhs_l;
            MFEM_ASSERT(fcheck2.Norml2()<= 9e-11,
                        "checking global solution at each level " << fcheck2.Norml2());
#endif

            // Final Solution ==
            if (l>0){
                for (int k = l-1; k>=0; k--){

                    vec1.SetSize(P_R[k]->Height());
                    P_R[k]->Mult(p_loc_vec, vec1);
                    p_loc_vec = vec1;

                }
            }

            total_sig +=p_loc_vec;

            MFEM_ASSERT(total_sig.Norml2()<= 9e+9,
                        "checking global solution added" << total_sig.Norml2());

        } // end of loop over levels

        // The coarse problem::

        SparseMatrix *M_coarse;
        SparseMatrix *B_coarse;
        Vector FF_coarse(P_W[ref_levels-1]->Width());

        rhs_l +=F_coarse;
        P_W[ref_levels-1]->MultTranspose(rhs_l, FF_coarse );

        SparseMatrix *P_WT2 = Transpose(*P_W[ref_levels-1]);
        SparseMatrix *P_RT2;
        if (M_fine)
            P_RT2 = Transpose(*P_R[ref_levels-1]);

        SparseMatrix *B_PR = Mult(*B_fine, *P_R[ref_levels-1]);
        B_coarse = Mult(*P_WT2, *B_PR);

        B_coarse->EliminateCols(ess_dof_coarsestlvl_list);

        if (M_fine)
        {
            SparseMatrix *M_PR = Mult(*M_fine, *P_R[ref_levels-1]);

            M_coarse =  Mult(*P_RT2, *M_PR);
            //std::cout << "M_coarse size = " << M_coarse->Height() << "\n";
            for ( int k = 0; k < ess_dof_coarsestlvl_list.Size(); ++k)
                if (ess_dof_coarsestlvl_list[k] !=0)
                    M_coarse->EliminateRowCol(k);
        }

        Vector sig_c(B_coarse->Width());

        auto B_Global = d_td_coarse_R->LeftDiagMult(*B_coarse,d_td_coarse_W->GetColStarts());
        Vector Truesig_c(B_Global->Width());

        if (M_fine)
        {
            auto d_td_M = d_td_coarse_R->LeftDiagMult(*M_coarse);
            HypreParMatrix *d_td_T = d_td_coarse_R->Transpose();

            HypreParMatrix *M_Global = ParMult(d_td_T, d_td_M);
            HypreParMatrix *BT = B_Global->Transpose();

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
            double rtol(1.e-16);
            double atol(1.e-16);

            MINRESSolver solver(MPI_COMM_WORLD);
            solver.SetAbsTol(atol);
            solver.SetRelTol(rtol);
            solver.SetMaxIter(maxIter);
            solver.SetOperator(coarseMatrix);
            solver.SetPreconditioner(*darcyPr);
            solver.SetPrintLevel(0);
            trueX = 0.0;
            solver.Mult(trueRhs, trueX);
//            chrono.Stop();

//            cout << "MINRES converged in " << solver.GetNumIterations() << " iterations" <<endl;
//            cout << "MINRES solver took " << chrono.RealTime() << "s. \n";
            Truesig_c = trueX.GetBlock(0);
        }
        else
        {
            int maxIter(50000);
            double rtol(1.e-16);
            double atol(1.e-16);

            HypreParMatrix *MinvBt = B_Global->Transpose();
            HypreParMatrix *S = ParMult(B_Global, MinvBt);

            auto invS = new HypreBoomerAMG(*S);
            invS->SetPrintLevel(0);
            invS->iterative_mode = false;

            Vector tmp_c(B_Global->Height());
            tmp_c = 0.0;

            CGSolver solver(MPI_COMM_WORLD);
            solver.SetAbsTol(atol);
            solver.SetRelTol(rtol);
            solver.SetMaxIter(maxIter);
            solver.SetOperator(*S);
            solver.SetPreconditioner(*invS);
            solver.SetPrintLevel(0);
            solver.Mult(FF_coarse, tmp_c);
//            chrono.Stop();

//            cout << "CG converged in " << solver.GetNumIterations() << " iterations" <<endl;
//            cout << "CG solver took " << chrono.RealTime() << "s. \n";
            MinvBt->Mult(tmp_c, Truesig_c);
        }

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
    }

    void Dofs_AE(SparseMatrix &Element_Dofs, const SparseMatrix &Element_Element_coarse, SparseMatrix &Dofs_Ae)
    {
        // Returns a SparseMatrix with the relation dofs to Element coarse.
        SparseMatrix *R_T = Transpose(Element_Dofs);
        SparseMatrix *Dofs_AE = Mult(*R_T,Element_Element_coarse);
        SparseMatrix *AeDofs = Transpose(*Dofs_AE);
        Dofs_Ae = *AeDofs;
    }


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
    }

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
    }

    void Local_problem(const DenseMatrix &sub_M,  DenseMatrix &sub_B, Vector &Sub_G, Vector &sub_F, Vector &sigma){
        // Returns sigma local


        DenseMatrix sub_BT(sub_B.Width(), sub_B.Height());
        sub_BT.Transpose(sub_B);

        DenseMatrix invM_BT;
        if (sub_M.Size() > 0)
        {
            DenseMatrixInverse invM_loc(sub_M);
            invM_loc.Mult(sub_BT,invM_BT);
        }

        /* Solving the local problem:
                  *
              * Msig + B^tu = G
              * Bsig        = F
              *
              * sig =  M^{-1} B^t(-u) + M^{-1} G
              *
              * B M^{-1} B^t (-u) = F
              */

        DenseMatrix B_invM_BT(sub_B.Height());

        if (sub_M.Size() > 0)
            Mult(sub_B, invM_BT, B_invM_BT);
        else
            Mult(sub_B, sub_BT, B_invM_BT);

//        Vector one(sub_B.Height());
//        one = 0.0;
//        one[0] =1;
        B_invM_BT.SetRow(0,0);
        B_invM_BT.SetCol(0,0);
//        B_invM_BT.SetCol(0,one);
        B_invM_BT(0,0)=1.;


        DenseMatrixInverse inv_BinvMBT(B_invM_BT);

//        Vector invMG(sub_M.Size());
//        invM_loc.Mult(Sub_G,invMG);

        sub_F[0] = 0;
        Vector uu(sub_B.Height());
        inv_BinvMBT.Mult(sub_F, uu);
        if (sub_M.Size() > 0)
            invM_BT.Mult(uu,sigma);
        else
            sub_BT.Mult(uu,sigma);
//        sigma += invMG;
    }

};

class MonolithicMultigrid : public Solver
{
private:
    class BlockSmoother : public BlockOperator
    {
    public:
        BlockSmoother(BlockOperator &Op)
            :
              BlockOperator(Op.RowOffsets()),
              A01((HypreParMatrix&)Op.GetBlock(0,1)),
              A10((HypreParMatrix&)Op.GetBlock(1,0)),
              offsets(Op.RowOffsets())
        {
            HypreParMatrix &A00 = (HypreParMatrix&)Op.GetBlock(0,0);
            HypreParMatrix &A11 = (HypreParMatrix&)Op.GetBlock(1,1);

            B00 = new HypreSmoother(A00);
            B11 = new HypreSmoother(A11);

            tmp01.SetSize(A00.Width());
            tmp02.SetSize(A00.Width());
            tmp1.SetSize(A11.Width());
        }

        virtual void Mult(const Vector & x, Vector & y) const
        {
            yblock.Update(y.GetData(), offsets);
            xblock.Update(x.GetData(), offsets);

            yblock.GetBlock(0) = 0.0;
            B00->Mult(xblock.GetBlock(0), yblock.GetBlock(0));

            tmp1 = xblock.GetBlock(1);
            A10.Mult(-1.0, yblock.GetBlock(0), 1.0, tmp1);
            B11->Mult(tmp1, yblock.GetBlock(1));
        }

        virtual void MultTranspose(const Vector & x, Vector & y) const
        {
            yblock.Update(y.GetData(), offsets);
            xblock.Update(x.GetData(), offsets);

            yblock.GetBlock(1) = 0.0;
            B11->Mult(xblock.GetBlock(1), yblock.GetBlock(1));

            tmp01 = xblock.GetBlock(0);
            A01.Mult(-1.0, yblock.GetBlock(1), 1.0, tmp01);
            B00->Mult(tmp01, yblock.GetBlock(0));
        }

        virtual void SetOperator(const Operator &op) { }

        ~BlockSmoother()
        {
            delete B00;
            delete B11;
            delete S;
        }

    private:
        HypreSmoother *B00;
        HypreSmoother *B11;
        HypreParMatrix &A01;
        HypreParMatrix &A10;
        HypreParMatrix *S;

        const Array<int> &offsets;
        mutable BlockVector xblock;
        mutable BlockVector yblock;
        mutable Vector tmp01;
        mutable Vector tmp02;
        mutable Vector tmp1;
    };

public:
    MonolithicMultigrid(BlockOperator &Operator,
                        const Array<BlockOperator*> &P,
                        Solver *CoarsePrec=NULL)
        :
          Solver(Operator.RowOffsets().Last()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        Operators_.Last() = &Operator;

        for (int l = Operators_.Size()-1; l >= 0; l--)
        {
            Array<int>& Offsets = Operators_[l]->RowOffsets();
            correction[l] = new Vector(Offsets.Last());
            residual[l] = new Vector(Offsets.Last());

            HypreParMatrix &A00 = (HypreParMatrix&)Operators_[l]->GetBlock(0,0);
            HypreParMatrix &A11 = (HypreParMatrix&)Operators_[l]->GetBlock(1,1);
            HypreParMatrix &A01 = (HypreParMatrix&)Operators_[l]->GetBlock(0,1);

            // Define smoothers
            Smoothers_[l] = new BlockSmoother(*Operators_[l]);

            // Define coarser level operators - two steps RAP (or P^T A P)
            if (l > 0)
            {
                HypreParMatrix& P0 = (HypreParMatrix&)P[l-1]->GetBlock(0,0);
                HypreParMatrix& P1 = (HypreParMatrix&)P[l-1]->GetBlock(1,1);

                unique_ptr<HypreParMatrix> P0T(P0.Transpose());
                unique_ptr<HypreParMatrix> P1T(P1.Transpose());

                unique_ptr<HypreParMatrix> A00P0( ParMult(&A00, &P0) );
                unique_ptr<HypreParMatrix> A11P1( ParMult(&A11, &P1) );
                unique_ptr<HypreParMatrix> A01P1( ParMult(&A01, &P1) );

                HypreParMatrix *A00_c(ParMult(P0T.get(), A00P0.get()));
                A00_c->CopyRowStarts();
                HypreParMatrix *A11_c(ParMult(P1T.get(), A11P1.get()));
                A11_c->CopyRowStarts();
                HypreParMatrix *A01_c(ParMult(P0T.get(), A01P1.get()));
                A01_c->CopyRowStarts();
                HypreParMatrix *A10_c(A01_c->Transpose());

                Operators_[l-1] = new BlockOperator(P[l-1]->ColOffsets());
                Operators_[l-1]->SetBlock(0, 0, A00_c);
                Operators_[l-1]->SetBlock(0, 1, A01_c);
                Operators_[l-1]->SetBlock(1, 0, A10_c);
                Operators_[l-1]->SetBlock(1, 1, A11_c);
                Operators_[l-1]->owns_blocks = 1;
            }
        }

        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(
                        ((HypreParMatrix&)Operator.GetBlock(0,0)).GetComm() );
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~MonolithicMultigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<BlockOperator*> &P_;

    Array<BlockOperator*> Operators_;
    Array<BlockSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void MonolithicMultigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void MonolithicMultigrid::MG_Cycle() const
{
    // PreSmoothing
    const BlockOperator& Operator_l = *Operators_[current_level];
    const BlockSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];
    Vector help(residual_l.Size());
    help = 0.0;

    Smoother_l.Mult(residual_l, correction_l);

    Operator_l.Mult(correction_l, help);
    residual_l -= help;

    // Coarse grid correction
    if (current_level > 0)
    {
        const BlockOperator& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(cor_cor, help);
        residual_l -= help;
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(cor_cor, help);
            residual_l -= help;
        }
    }

    // PostSmoothing
    Smoother_l.MultTranspose(residual_l, cor_cor);
    correction_l += cor_cor;
}

class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix &Operator,
              const Array<HypreParMatrix*> &P,
              Solver *CoarsePrec=NULL)
        :
          Solver(Operator.GetNumRows()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarseSolver(NULL),
          CoarsePrec_(CoarsePrec)
    {
        Operators_.Last() = &Operator;
        for (int l = Operators_.Size()-1; l > 0; l--)
        {
            // Two steps RAP
            unique_ptr<HypreParMatrix> PT( P[l-1]->Transpose() );
            unique_ptr<HypreParMatrix> AP( ParMult(Operators_[l], P[l-1]) );
            Operators_[l-1] = ParMult(PT.get(), AP.get());
            Operators_[l-1]->CopyRowStarts();
        }

        for (int l = 0; l < Operators_.Size(); l++)
        {
            Smoothers_[l] = new HypreSmoother(*Operators_[l]);
            correction[l] = new Vector(Operators_[l]->GetNumRows());
            residual[l] = new Vector(Operators_[l]->GetNumRows());
        }

        if (CoarsePrec)
        {
            CoarseSolver = new CGSolver(Operators_[0]->GetComm());
            CoarseSolver->SetRelTol(1e-8);
            CoarseSolver->SetMaxIter(50);
            CoarseSolver->SetPrintLevel(0);
            CoarseSolver->SetOperator(*Operators_[0]);
            CoarseSolver->SetPreconditioner(*CoarsePrec);
        }
    }

    virtual void Mult(const Vector & x, Vector & y) const;

    virtual void SetOperator(const Operator &op) { }

    ~Multigrid()
    {
        for (int l = 0; l < Operators_.Size(); l++)
        {
            delete Smoothers_[l];
            delete correction[l];
            delete residual[l];
        }
    }

private:
    void MG_Cycle() const;

    const Array<HypreParMatrix*> &P_;

    Array<HypreParMatrix*> Operators_;
    Array<HypreSmoother*> Smoothers_;

    mutable int current_level;

    mutable Array<Vector*> correction;
    mutable Array<Vector*> residual;

    mutable Vector res_aux;
    mutable Vector cor_cor;
    mutable Vector cor_aux;

    CGSolver *CoarseSolver;
    Solver *CoarsePrec_;
};

void Multigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;
    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    // PreSmoothing
    const HypreParMatrix& Operator_l = *Operators_[current_level];
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

    Smoother_l.Mult(residual_l, correction_l);
    Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);

    // Coarse grid correction
    if (current_level > 0)
    {
        const HypreParMatrix& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        current_level--;
        MG_Cycle();
        current_level++;

        cor_cor.SetSize(residual_l.Size());
        P_l.Mult(*correction[current_level-1], cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
    }
    else
    {
        cor_cor.SetSize(residual_l.Size());
        if (CoarseSolver)
        {
            CoarseSolver->Mult(residual_l, cor_cor);
            correction_l += cor_cor;
            Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        }
    }

    // PostSmoothing
    Smoother_l.Mult(residual_l, cor_cor);
    correction_l += cor_cor;
}

SparseMatrix * RemoveZeroEntries(const SparseMatrix& in)
{
    int * I = in.GetI();
    int * J = in.GetJ();
    double * Data = in.GetData();
    double * End = Data+in.NumNonZeroElems();

    int nnz = 0;
    for (double * data_ptr = Data; data_ptr != End; data_ptr++)
    {
        if (*data_ptr != 0)
            nnz++;
    }

    int * outI = new int[in.Height()+1];
    int * outJ = new int[nnz];
    double * outData = new double[nnz];
    nnz = 0;
    for (int i = 0; i < in.Height(); i++)
    {
        outI[i] = nnz;
        for (int j = I[i]; j < I[i+1]; j++)
        {
            if (Data[j] !=0)
            {
                outJ[nnz] = J[j];
                outData[nnz++] = Data[j];
            }
        }
    }
    outI[in.Height()] = nnz;

    return new SparseMatrix(outI, outJ, outData, in.Height(), in.Width());
}
