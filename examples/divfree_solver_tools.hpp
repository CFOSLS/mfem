#include "mfem.hpp"
#include "linalg/linalg.hpp"

#include <iterator>

using namespace mfem;
using namespace std;
using std::unique_ptr;


// PLAN:
// 1) Add righthand side everywhere in the functional residual computation and as an input parameter
// 2) Test that particular solution finder is working without smoother
// 2a) that with exact discrete solution as initial guess all the updates are 0 for the local problem solves
// 3) Test that particular solution finder is working with the smoother
// 4) Test that solver is working in the non-preconditioner mode
// 4) Figure out how it should be treated in the preconditioner mode
// 5) Test that the solver is working in the preconditioner mode

// QUESTION: Should I compute functional rhs at each level outside the solver using the bilinear form or
// just coarsen the functional righthand side given at the finest level?


#define MEMORY_OPTIMIZED

// activates a check for the correctness of local problem solve for the blocked case (with S)
#define CHECK_LOCALSOLVE

#define IMPROVEMENT

// activates some additional checks
//#define DEBUG_INFO

// FIXME: Is MG norm computed correctly?

// Checking routines used for debugging
// Vector dot product assembled over MPI
double ComputeMPIDotProduct(MPI_Comm comm, const Vector& vec1, const Vector& vec2)
{
    MFEM_ASSERT(vec1.Size() == vec2.Size(), "Sizes mismatch in ComputeMPIDotProduct()!");

    int local_size = vec1.Size();
    int global_size = 0;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

    double local_dotprod = vec1 * vec2;
    double global_norm = 0;
    MPI_Allreduce(&local_dotprod, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (global_norm < 0)
        std::cout << "MG norm is not a norm: dot product = " << global_norm << " less than zero! \n";
    global_norm = sqrt (global_norm / global_size);

    return global_norm;
}

// Vector norm assembled over MPI
double ComputeMPIVecNorm(MPI_Comm comm, const Vector& bvec, char const * string, bool print)
{
    int local_size = bvec.Size();
    int global_size = 0;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

    double local_normsq = bvec.Norml2() * bvec.Norml2();
    double global_norm = 0;
    MPI_Allreduce(&local_normsq, &global_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_norm = sqrt (global_norm / global_size);

    if (print)
        std::cout << "Norm " << string << global_norm << " ... \n";
    return global_norm;
}

// Computes and prints the norm of ( Funct * y, y )_2,h, assembled over all processes, input vectors on true dofs, matrix on dofs
// w/o proper assembly over shared dofs
double CheckFunctValue(MPI_Comm comm, const BlockMatrix& Funct, const std::vector<HypreParMatrix*> Dof_TrueDof, const BlockVector& truevec, char const * string, bool print)
{
    MFEM_ASSERT(Dof_TrueDof.size() - Funct.NumColBlocks() == 0,"CheckFunctValue: number of blocks mismatch \n");

    BlockVector vec(Funct.ColOffsets());
    for ( int blk = 0; blk < Funct.NumColBlocks(); ++blk)
    {
        Dof_TrueDof[blk]->Mult(truevec.GetBlock(blk), vec.GetBlock(blk));
    }

    BlockVector res(Funct.RowOffsets());
    Funct.Mult(vec, res);
    double local_func_norm = vec * res / sqrt (res.Size());
    double global_func_norm = 0;
    MPI_Allreduce(&local_func_norm, &global_func_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (print)
        std::cout << "Functional norm " << string << global_func_norm << " ... \n";
    return global_func_norm;
}

// Computes and prints the norm of ( Funct * y, y )_2,h, assembled over all processes, everything on truedofs
double CheckFunctValue(MPI_Comm comm, const BlockOperator& Funct, const BlockVector* truefunctrhs, const Array<int>& offsets, const BlockVector& truevec, char const * string, bool print)
{
    BlockVector trueres(offsets);
    Funct.Mult(truevec, trueres);
    double local_func_norm;
    if (truefunctrhs)
    {
        trueres -= *truefunctrhs;
        local_func_norm = truevec * trueres / sqrt (trueres.Size()) + (*truefunctrhs) * (*truefunctrhs) / sqrt(truefunctrhs->Size());
    }
    else // NULL case assumed to denote zero righthand side
        local_func_norm = truevec * trueres / sqrt (trueres.Size());
    double global_func_norm = 0;
    MPI_Allreduce(&local_func_norm, &global_func_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (print)
        std::cout << "Functional norm " << string << global_func_norm << " ... \n";
    return global_func_norm;
}


// Computes and prints the norm of || Constr * sigma - ConstrRhs ||_2,h
bool CheckConstrRes(Vector& sigma, const SparseMatrix& Constr, const Vector* ConstrRhs,
                                                char const* string)
{
    bool passed = true;
    Vector res_constr(Constr.Height());
    Constr.Mult(sigma, res_constr);
    //ofstream ofs("newsolver_out.txt");
    //res_constr.Print(ofs,1);
    if (ConstrRhs)
        res_constr -= *ConstrRhs;
    double constr_norm = res_constr.Norml2() / sqrt (res_constr.Size());
    if (fabs(constr_norm) > 1.0e-13)
    {
        std::cout << "Constraint residual norm " << string << ": "
                  << constr_norm << " ... \n";
        passed = false;
    }

    return passed;
}

// true for truedofs, false for dofs
bool CheckBdrError (const Vector& Candidate, const Vector& Given_bdrdata, const Array<int>& ess_bdr, bool dof_or_truedof)
{
    bool passed = true;
    double max_bdr_error = 0;
    if (dof_or_truedof) // for true dofs
    {
        for ( int i = 0; i < ess_bdr.Size(); ++i)
        {
            int tdof = ess_bdr[i];
            double bdr_error_dof = fabs(Given_bdrdata[tdof] - Candidate[tdof]);
            if ( bdr_error_dof > max_bdr_error )
                max_bdr_error = bdr_error_dof;
        }
    }
    else // for dofs
    {
        for ( int dof = 0; dof < Given_bdrdata.Size(); ++dof)
        {
            if (ess_bdr[dof] != 0.0)
            {
                double bdr_error_dof = fabs(Given_bdrdata[dof] - Candidate[dof]);
                if ( bdr_error_dof > max_bdr_error )
                    max_bdr_error = bdr_error_dof;
            }
        }
    }

    if (max_bdr_error > 1.0e-13)
    {
        std::cout << "CheckBdrError: Error, boundary values for the solution are wrong:"
                     " max_bdr_error = " << max_bdr_error << "\n";
        passed = false;
    }
    //else
        //std::cout << "CheckBdrError: boundary values are correct \n";

    return passed;
}


#if 0
class MultilevelSmoother : public Operator
{
protected:
    // number of levels where MultLevel is to be called
    const int num_levels;
    mutable Array<bool> finalized_lvls;
    mutable int print_level;
public:
    // constructor
    MultilevelSmoother (int Num_Levels) : num_levels(Num_Levels)
    {
        finalized_lvls.SetSize(num_levels);
        finalized_lvls = 0;
    }

    // general setup functions
    virtual void SetUpSmoother(int level, const BlockMatrix& SysMat_lvl) const = 0;

    // general functions for setting righthand side at the given level
    //virtual void ComputeRhsLevel(int level, const BlockVector& res_lvl);
    virtual void ComputeTrueRhsLevel(int level, const BlockVector& res_lvl);

    // main function which applies the smoother at the given level
    //virtual void MultLevel(int level, Vector& in, Vector& out) = 0;
    virtual void MultTrueLevel(int level, Vector& in, Vector& out) = 0;

    // legacy of the Operator class
    virtual void Mult (const Vector& x, Vector& y) const
    {
        MFEM_ABORT("Mult() should never be called from MultilevelSmoother and its descendants \n");
    }

    void SetPrintLevel(int PrintLevel)  {print_level = PrintLevel;}
    // getters
    int GetNumLevels() {return num_levels;}
    int GetPrintLevel() const { return print_level;}

    virtual void PrintAllOptions() const;

    virtual void Setup() const = 0;
};


void MultilevelSmoother::PrintAllOptions() const
{
    std::cout << "Multilevel smoother base options: \n";
    std::cout << "num_levels: " << num_levels << "\n";
    std::cout << "print_level: " << print_level << "\n";
    std::cout << "\n";
}

void MultilevelSmoother::SetUpSmoother(int level, const BlockMatrix& SysMat_lvl) const
{
    MFEM_ABORT("SetUpSmoother for a BlockMatrix argument is called from the abstract base"
                 " class but must have been redefined \n");
}

/*
void MultilevelSmoother::MultLevel(int level, Vector& in, Vector& out)
{
    MFEM_ABORT("MultLevel is called from the abstract base class but must have been redefined \n");
}

void MultilevelSmoother::ComputeRhsLevel(int level, const BlockVector& res_lvl)
{
    std::cout << "ComputeRhsLevel for a BlockVector argument is called from the abstract base"
                 " class but must have been redefined \n";
}
*/

void MultilevelSmoother::MultTrueLevel(int level, Vector& in, Vector& out)
{
    MFEM_ABORT("MultTrueLevel is called from the abstract base class but must have been redefined \n");
}

void MultilevelSmoother::ComputeTrueRhsLevel(int level, const BlockVector& res_lvl)
{
    std::cout << "ComputeTrueRhsLevel for a BlockVector argument is called from the abstract base"
                 " class but must have been redefined \n";
}
#endif

// TODO: Implement an abstract base for the coarsest problem solver. Maybe unnecessary now
// TODO: Implement an abstract base for the local problems solver. Maybe unnecessary now

class CoarsestProblemSolver : public Operator
{
private:
    const int numblocks;

    MPI_Comm comm;

    mutable bool finalized;

    // coarsest level local-to-process matrices
    mutable BlockMatrix* Op_blkspmat;
    mutable SparseMatrix* Constr_spmat;

    // Dof_TrueDof relation tables for each level for functional-related
    // variables and the L2 variable (constraint space).
    const std::vector<HypreParMatrix*> & dof_trueDof_blocks;
    const HypreParMatrix& dof_trueDof_L2;

    const std::vector<Array<int>* > & essbdrdofs_blocks;
    const std::vector<Array<int>* > & essbdrtruedofs_blocks;

    mutable Array<int> coarse_offsets;
    mutable BlockOperator* coarse_matrix;
    mutable BlockDiagonalPreconditioner * coarse_prec;
    mutable Array<int> coarse_rhsfunc_offsets;
    mutable BlockVector * coarse_rhsfunc;
    mutable BlockVector * coarsetrueX;
    mutable BlockVector * coarsetrueRhs;
    mutable IterativeSolver * coarseSolver;

    // all on true dofs
    mutable Array<int> block_offsets;
    mutable BlockVector* xblock;
    mutable BlockVector* yblock;

protected:
    void Setup() const;

public:
    CoarsestProblemSolver(BlockMatrix& Op_Blksmat, SparseMatrix& Constr_Spmat,
                          const std::vector<HypreParMatrix*>& D_tD_blks,
                          const HypreParMatrix& D_tD_L2,
                          const std::vector<Array<int>* >& EssBdrDofs_blks, const std::vector<Array<int> *> &EssBdrTrueDofs_blks);

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const { Mult(x,y, NULL); }

    void Mult(const Vector &x, Vector &y, Vector* rhs_constr) const;
};

CoarsestProblemSolver::CoarsestProblemSolver(BlockMatrix& Op_Blksmat,
                                             SparseMatrix& Constr_Spmat,
                                             const std::vector<HypreParMatrix*>& D_tD_blks,
                                             const HypreParMatrix& D_tD_L2,
                                             const std::vector<Array<int>* >& EssBdrDofs_blks,
                                             const std::vector<Array<int>* >& EssBdrTrueDofs_blks)
    : Operator(),
      numblocks(Op_Blksmat.NumRowBlocks()),
      comm(D_tD_L2.GetComm()),
      Op_blkspmat(&Op_Blksmat),
      Constr_spmat(&Constr_Spmat),
      dof_trueDof_blocks(D_tD_blks),
      dof_trueDof_L2(D_tD_L2),
      essbdrdofs_blocks(EssBdrDofs_blks),
      essbdrtruedofs_blocks(EssBdrTrueDofs_blks)
{
    finalized = false;

    block_offsets.SetSize(numblocks + 1);
    block_offsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        block_offsets[blk + 1] = dof_trueDof_blocks[blk]->Width();
    block_offsets.PartialSum();

    coarse_rhsfunc_offsets.SetSize(numblocks + 1);
    coarse_offsets.SetSize(numblocks + 2);

    Setup();
}

void CoarsestProblemSolver::Setup() const
{
    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    // 1. eliminating boundary conditions at coarse level
    const Array<int> * temp = essbdrdofs_blocks[0];

    Constr_spmat->EliminateCols(*temp);

    // old code which changes only diagonal blocks
    // incorrect for the block case
    /*
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> * temp = essbdrdofs_blocks[blk];
        for ( int dof = 0; dof < temp->Size(); ++dof)
            if ( (*temp)[dof] != 0)
            {
                // 1 var
                Op_blkspmat->GetBlock(blk,blk).EliminateRowCol(dof);
            }
    }
    */

    /*
     * doesn't work
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
    {
        const Array<int> * temp1 = essbdrdofs_blocks[blk1];
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            const Array<int> * temp2 = essbdrdofs_blocks[blk2];
            Op_blkspmat->GetBlock(blk1,blk2).EliminateCols(*temp2);

            for ( int dof1 = 0; dof1 < temp1->Size(); ++dof1)
                if ( (*temp1)[dof1] != 0)
                {
                    Op_blkspmat->GetBlock(blk1,blk2).EliminateRow(dof1, 1.0);
                }
        }
    }
    */

    /*
    // DOESN't WORK
    // new code, which takes care about the block structure
    // but requires ridiculuous temporary vectors
    // FIXME: Get rid of the temporary vectors
    BlockVector tempsol(Op_blkspmat->ColOffsets());
    tempsol = 0.0;
    BlockVector temprhs(Op_blkspmat->RowOffsets());
    temprhs = 0.0;

    int shift = 0;
    Array<int> & offsets = Op_blkspmat->RowOffsets();
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Array<int> * temp = essbdrdofs_blocks[blk];
        Array<int> temp_shifted(temp->Size());
        for ( int i = 0; i < temp->Size(); ++i)
            temp_shifted[i] = (*temp)[i] + shift;
        std::cout << "shift = " << shift << "\n";
        std::cout << "temp_shifted in coarsest solver setup \n";
        if (blk == 0)
            temp_shifted.Print();
        temp_shifted.Print();
        Op_blkspmat->EliminateRowCol(temp_shifted, tempsol, temprhs);
        shift += offsets[blk + 1];
    }
    */

    // latest version of the code
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
    {
        const Array<int> * temp1 = essbdrdofs_blocks[blk1];
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            const Array<int> * temp2 = essbdrdofs_blocks[blk2];
            Op_blkspmat->GetBlock(blk1,blk2).EliminateCols(*temp2);

            for ( int dof1 = 0; dof1 < temp1->Size(); ++dof1)
                if ( (*temp1)[dof1] != 0)
                {
                    if (blk1 == blk2)
                        Op_blkspmat->GetBlock(blk1,blk2).EliminateRow(dof1, 1.0);
                    else // doesn't set diagonal entry to 1
                        Op_blkspmat->GetBlock(blk1,blk2).EliminateRow(dof1);
                }
        }
    }

    /*
     * not needed if the original problem has non empty essential boundary
    // Getting rid of the one-dimensional kernel for lambda in the coarsest level problem
    (*Constr_lvls)[num_levels-1-1]->EliminateRow(0);
    (*Constr_lvls)[num_levels-1-1]->GetData()[0] = 1.0;
    (*rhs_constr)[0] = 0.0;
    */

    // 2. Creating the block matrix from the local parts using dof_truedof relation

    HypreParMatrix * Constr_d_td = dof_trueDof_blocks[0]->LeftDiagMult(
                *Constr_spmat, dof_trueDof_L2.GetColStarts());
    HypreParMatrix * d_td_L2_T = dof_trueDof_L2.Transpose();

    HypreParMatrix * Constr_global = ParMult(d_td_L2_T, Constr_d_td);
    Constr_global->CopyRowStarts();
    Constr_global->CopyColStarts();

    HypreParMatrix *ConstrT_global = Constr_global->Transpose();

    delete Constr_d_td;
    delete d_td_L2_T;

    //Array2D<HypreParMatrix*> Funct_d_td(numblocks, numblocks);
    //Array2D<HypreParMatrix*> d_td_T(numblocks, numblocks);

    Array2D<HypreParMatrix*> Funct_global(numblocks, numblocks);
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            auto Funct_d_td = dof_trueDof_blocks[blk2]->LeftDiagMult(Op_blkspmat->GetBlock(blk1,blk2),
                                                                     dof_trueDof_blocks[blk1]->GetRowStarts() );
            auto d_td_T = dof_trueDof_blocks[blk1]->Transpose();

            Funct_global(blk1, blk2) = ParMult(d_td_T, Funct_d_td);
            Funct_global(blk1, blk2)->CopyRowStarts();
            Funct_global(blk1, blk2)->CopyColStarts();

            //Funct_d_td[blk] = dof_trueDof_blocks[blk]->LeftDiagMult(Op_blkspmat->GetBlock(blk,blk));
            //d_td_T[blk] = dof_trueDof_blocks[blk]->Transpose();

            //Funct_global[blk] = ParMult(d_td_T[blk], Funct_d_td[blk]);
            //Funct_global[blk]->CopyRowStarts();
            //Funct_global[blk]->CopyColStarts();
        }

    //for ( int blk = 0; blk < numblocks; ++blk)
    //{
        //delete Funct_d_td[blk];
        //delete d_td_T[blk];
    //}


    coarse_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_offsets[blk + 1] = Funct_global(blk, blk)->Height();
    coarse_offsets[numblocks + 1] = Constr_global->Height();
    coarse_offsets.PartialSum();

    coarse_rhsfunc_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_rhsfunc_offsets[blk + 1] = coarse_offsets[blk + 1];
    coarse_rhsfunc_offsets.PartialSum();

    //std::cout << "coarse_rhsfunc offsets \n";
    //coarse_rhsfunc_offsets.Print();

    coarse_rhsfunc = new BlockVector(coarse_rhsfunc_offsets);

    coarse_matrix = new BlockOperator(coarse_offsets);
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
            coarse_matrix->SetBlock(blk1, blk2, Funct_global(blk1, blk2));
    coarse_matrix->SetBlock(0, numblocks, ConstrT_global);
    coarse_matrix->SetBlock(numblocks, 0, Constr_global);

    // coarse solution and righthand side vectors
    coarsetrueX = new BlockVector(coarse_offsets);
    coarsetrueRhs = new BlockVector(coarse_offsets);

    //coarse_offsets.Print();

    // preconditioner for the coarse problem

    std::vector<Operator*> Funct_prec(numblocks);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(numblocks <= 2, "Current implementation of coarsest level solver "
                                   "knows preconditioners only for sigma or (sigma,S) setups \n");
        if (blk == 0)
        {
            Funct_prec[blk] = new HypreDiagScale(*Funct_global(blk,blk));
            ((HypreDiagScale*)(Funct_prec[blk]))->iterative_mode = false;
        }
        else // blk == 1
        {
            Funct_prec[blk] = new HypreBoomerAMG(*Funct_global(blk, blk));
            ((HypreBoomerAMG*)(Funct_prec[blk]))->SetPrintLevel(0);
            ((HypreBoomerAMG*)(Funct_prec[blk]))->iterative_mode = false;
        }
    }

    HypreParMatrix *MinvBt = Constr_global->Transpose();
    HypreParVector *Md = new HypreParVector(comm, Funct_global(0,0)->GetGlobalNumRows(),
                                            Funct_global(0,0)->GetRowStarts());
    Funct_global(0,0)->GetDiag(*Md);
    MinvBt->InvScaleRows(*Md);
    HypreParMatrix *Schur = ParMult(Constr_global, MinvBt);
    Schur->CopyRowStarts();
    Schur->CopyColStarts();

    HypreBoomerAMG * invSchur = new HypreBoomerAMG(*Schur);
    invSchur->SetPrintLevel(0);
    invSchur->iterative_mode = false;

    coarse_prec = new BlockDiagonalPreconditioner(coarse_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_prec->SetDiagonalBlock(blk, Funct_prec[blk]);
    coarse_prec->SetDiagonalBlock(numblocks, invSchur);

    // coarse solver
    int maxIter(20000);
    double rtol(1.e-18);
    double atol(1.e-18);

    coarseSolver = new MINRESSolver(comm);
    coarseSolver->SetAbsTol(atol);
    coarseSolver->SetRelTol(rtol);
    coarseSolver->SetMaxIter(maxIter);
    coarseSolver->SetOperator(*coarse_matrix);
    if (coarse_prec)
        coarseSolver->SetPreconditioner(*coarse_prec);
    coarseSolver->SetPrintLevel(0);
    //Operator * coarse_id = new IdentityOperator(coarse_offsets[numblocks + 2] - coarse_offsets[0]);
    //coarseSolver->SetOperator(*coarse_id);

    finalized = true;
}

void CoarsestProblemSolver::Mult(const Vector &x, Vector &y, Vector* rhs_constr) const
{
    MFEM_ASSERT(finalized, "Mult() must not be called before the coarse solver was finalized \n");

    // x and y will be accessed through xblock and yblock as viewing structures
    xblock->Update(x.GetData(), block_offsets);
    yblock->Update(y.GetData(), block_offsets);

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Array<int> * temp = essbdrtruedofs_blocks[blk];

        //const Array<int> * temp2 = essbdrdofs_Func[blk][num_levels - 1];

        //temp->Print();

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            xblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            //std::cout << tdof << ": " << (*temp)[tdof] << " " << (*temp2)[tdof] << "\n";
        }

        //xblock->GetBlock(blk).Print();
        //std::cout << "Good? \n";
    }


    // 1. set up solution and righthand side vectors
    *coarsetrueX = 0.0;
    *coarsetrueRhs = 0.0;

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(coarsetrueRhs->GetBlock(blk).Size() == xblock->GetBlock(blk).Size(),
                    "Sizes mismatch when finalizing rhs at the coarsest level!\n");
        coarsetrueRhs->GetBlock(blk) = xblock->GetBlock(blk);
    }

    if (rhs_constr)
    {
        MFEM_ASSERT(coarsetrueRhs->GetBlock(numblocks).Size() == rhs_constr->Size(),
                    "Sizes mismatch when finalizing rhs at the coarsest level!\n");
        coarsetrueRhs->GetBlock(numblocks) = *rhs_constr;
    }

    std::cout << "coarsetruerhs \n";
    coarsetrueRhs->Print();
    ofstream ofs("coarserhs_inside solver.txt");
    ofs << coarsetrueRhs->Size() << "\n";
    coarsetrueRhs->Print(ofs,1);
    ofs.close();

    // 2. solve the linear system with preconditioned MINRES.
    coarseSolver->Mult(*coarsetrueRhs, *coarsetrueX);

    for ( int blk = 0; blk < numblocks; ++blk)
        yblock->GetBlock(blk) = coarsetrueX->GetBlock(blk);

    std::cout << "y \n";
    y.Print();

    return;
}

// ~ Non-overlapping Schwarz smoother based on agglomerated elements
// which provides zeros at the interfaces in the output
// TODO: Rename all the variables appropriately: now it's a mess of Op, Funct etc.
class LocalProblemSolver : public Operator
{
private:
    mutable bool finalized;

protected:
    int numblocks;

    // a parameter used in Get_AE_eintdofs to identify if one should additionally look
    // for fine-grid dofs which are internal to the fine-grid elements
    bool higher_order;

    const BlockMatrix& Op_blkspmat;
    const SparseMatrix& Constr_spmat;

    const std::vector<HypreParMatrix*>& d_td_blocks;

    // Relation tables which represent agglomerated elements-to-elements relation
    const SparseMatrix& AE_e;

    const BlockMatrix& el_to_dofs_Func;
    const SparseMatrix& el_to_dofs_L2;

    mutable SparseMatrix* AE_edofs_L2;
    mutable BlockMatrix* AE_eintdofs_blocks; // relation between AEs and internal (w.r.t to AEs) fine-grid dofs

    const std::vector<Array<int>* >& bdrdofs_blocks;
    const std::vector<Array<int>* >& essbdrdofs_blocks;

    // Store (except the coarsest) LU factors of the local
    // problems' matrices for each agglomerate (2 per agglomerate)
    mutable std::vector<std::vector<DenseMatrixInverse* > > LUfactors;

    // if true, local problems' matrices are computed in SolveLocalProblems()
    // but in the future one can compute them only once and reuse afterwards
    mutable Array2D<bool> compute_AEproblem_matrices;

    // all on true dofs
    mutable Array<int> block_offsets;
    mutable BlockVector* xblock;
    mutable BlockVector* yblock;

#ifdef COMPUTE_EXACTDISCRETESOL
    Vector * sigma, * S, * lambda;
#endif

protected:
    mutable bool optimized_localsolve;

#ifdef COMPUTE_EXACTDISCRETESOL
    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate, Vector* sigma_loc = NULL, Vector* S_loc = NULL, Vector* lambda_loc = NULL) const;
#else
    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate) const;
#endif
    // Optimized version of SolveLocalProblem where LU factors for the local
    // problem's matrices were computed during the setup via SaveLocalLUFactors()
    void SolveLocalProblemOpt(DenseMatrixInverse * inv_A, DenseMatrixInverse * inv_Schur, DenseMatrix& B, BlockVector &G,
                                               Vector& F, BlockVector &sol, bool is_degenerate) const;
    // an optional routine which can save LU factors for the local problems
    // solved at finer levels if needed. Should be redefined in the inheriting
    // classes in order to speed up iterations
    virtual void SaveLocalLUFactors() const;

    BlockMatrix* Get_AE_eintdofs(const BlockMatrix& el_to_dofs,
                                            const std::vector<Array<int>* > &dof_is_essbdr,
                                            const std::vector<Array<int>* > &dof_is_bdr) const;

    void Setup();

public:
    // main constructor
    LocalProblemSolver(const BlockMatrix& Op_Blksmat,
                       const SparseMatrix& Constr_Spmat,
                       const std::vector<HypreParMatrix*>& D_tD_blks,
                       const SparseMatrix& AE_el,
                       const BlockMatrix& El_to_Dofs_Func,
                       const SparseMatrix& El_to_Dofs_L2,
                       const std::vector<Array<int>* >& BdrDofs_blks,
                       const std::vector<Array<int>* >& EssBdrDofs_blks,
#ifdef COMPUTE_EXACTDISCRETESOL
                       Vector * sigma_input, Vector * S_input, Vector * lambda_input,
#endif
                       bool Optimized_LocalSolve,
                       bool Higher_Order)
        : Operator(),
          numblocks(Op_Blksmat.NumRowBlocks()),
          Op_blkspmat(Op_Blksmat), Constr_spmat(Constr_Spmat),
          d_td_blocks(D_tD_blks),
          AE_e(AE_el),
          el_to_dofs_Func(El_to_Dofs_Func),
          el_to_dofs_L2(El_to_Dofs_L2),
          bdrdofs_blocks(BdrDofs_blks),
          essbdrdofs_blocks(EssBdrDofs_blks)
    {
        finalized = 0;
        optimized_localsolve = Optimized_LocalSolve;
        higher_order = Higher_Order;
        compute_AEproblem_matrices.SetSize(numblocks + 1, numblocks + 1);
        compute_AEproblem_matrices = true;

        block_offsets.SetSize(numblocks + 1);
        block_offsets[0] = 0;
        for (int blk = 0; blk < numblocks; ++blk)
            block_offsets[blk + 1] = d_td_blocks[blk]->Width();
        block_offsets.PartialSum();

        if (optimized_localsolve)
        {
            compute_AEproblem_matrices = false;
            compute_AEproblem_matrices(numblocks, numblocks) = true;
        }

#ifdef COMPUTE_EXACTDISCRETESOL
        sigma = sigma_input;
        S = S_input;
        lambda = lambda_input;
#endif
        Setup();
    }

    void SetUp();

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const override { Mult(x,y, NULL); }

    // considers x as the righthand side
    // and returns y as a solution to all the local problems
    // (*) both x and y are vectors on true dofs
    virtual void Mult(const Vector &x, Vector &y, Vector* rhs_constr) const;

    // is public since one might want to use that to compute particular solution witn nonzero righthand side in the constraint
#ifdef COMPUTE_EXACTDISCRETESOL
    void SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol_update, Vector* localrhs_constr, Vector * sigma = NULL, Vector * S = NULL, Vector * lambda = NULL) const;
#else
    void SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol_update, Vector* localrhs_constr) const;
#endif
};

void LocalProblemSolver::Mult(const Vector &x, Vector &y, Vector * rhs_constr) const
{
    // x will be accessed through xblock as its view
    xblock->Update(x.GetData(), block_offsets);
    // y will be accessed through yblock as its view
    yblock->Update(y.GetData(), block_offsets);

#ifdef COMPUTE_EXACTDISCRETESOL
    SolveTrueLocalProblems( *xblock, *yblock, rhs_constr, sigma, S, lambda);
#else
    SolveTrueLocalProblems( *xblock, *yblock, rhs_constr);
#endif
}


void LocalProblemSolver::Setup()
{
    AE_edofs_L2 = mfem::Mult(AE_e, el_to_dofs_L2);
    AE_eintdofs_blocks = Get_AE_eintdofs(el_to_dofs_Func, essbdrdofs_blocks, bdrdofs_blocks);

    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    // (optionally) saves LU factors related to the local problems to be solved
    // for each agglomerate element
    if (optimized_localsolve)
        SaveLocalLUFactors();

    finalized = true;

}

#ifdef COMPUTE_EXACTDISCRETESOL
void LocalProblemSolver::SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol, Vector* localrhs_constr, Vector * sigma, Vector * S, Vector * lambda) const
#else
void LocalProblemSolver::SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol, Vector* localrhs_constr) const
#endif
{
    // FIXME: Get rid of temporary vectors;
    BlockVector lvlrhs_func(Op_blkspmat.ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
    {
        //std::cout << "truerhs_func block size = " << truerhs_func.GetBlock(blk).Size() << "\n";// << ", lvlrhsfunc blocksize = " << lvlrhs_func.GetBlock(blk).Size() << "\n";
        d_td_blocks[blk]->Mult(truerhs_func.GetBlock(blk), lvlrhs_func.GetBlock(blk));
    }
    BlockVector sol(Op_blkspmat.RowOffsets());
    sol = 0.0;

    DenseMatrix sub_Constr;
    Vector sub_rhsconstr;
    Array<int> sub_Func_offsets(numblocks + 1);

    Array2D<const SparseMatrix *> Op_blks(numblocks, numblocks);
    Array2D<DenseMatrix*> LocalAE_Matrices(numblocks, numblocks);
    std::vector<Array<int>*> Local_inds(numblocks);

#ifdef COMPUTE_EXACTDISCRETESOL
    /*
    // a preliminary check
    const SparseMatrix * Atemp = &(Op_blkspmat.GetBlock(0,0));

    Vector Asigma(Atemp->Height());
    Atemp->Mult(*sigma, Asigma);

    Vector BTlambda(Constr_spmat.Width());
    Constr_spmat.MultTranspose(*lambda, BTlambda);

    Vector vectemp1(Asigma.Size());
    vectemp1 = Asigma;
    vectemp1 += BTlambda;

    Vector Dsigma, DTS, CS;

    if (numblocks > 1)
    {
        const SparseMatrix * Dtemp = &(Op_blkspmat.GetBlock(1,0));
        const SparseMatrix * DTtemp = &(Op_blkspmat.GetBlock(0,1));
        const SparseMatrix * Ctemp = &(Op_blkspmat.GetBlock(1,1));

        Dsigma.SetSize(Dtemp->Height());
        Dtemp->Mult(*sigma, Dsigma);

        CS.SetSize(Ctemp->Height());
        Ctemp->Mult(*S, CS);

        Vector vectemp2(Dsigma.Size());
        vectemp2 = Dsigma;
        vectemp2 += CS;
        vectemp2 -= lvlrhs_func.GetBlock(1);

        ofstream ofs("Dsigma_inside local.txt");
        ofs << Dsigma.Size() << "\n";
        Dsigma.Print(ofs,1);
        ofs.close();

        ofstream ofs2("CS_inside local.txt");
        ofs2 << CS.Size() << "\n";
        CS.Print(ofs2,1);
        ofs2.close();

        ofstream ofs3("Gblock1_inside local.txt");
        ofs3 << lvlrhs_func.GetBlock(1).Size() << "\n";
        lvlrhs_func.GetBlock(1).Print(ofs3,1);
        ofs3.close();

        std::cout << "lvlrhsfunc, second block \n";
        lvlrhs_func.GetBlock(1).Print();

        vectemp2.Print();
        std::cout << "vectemp2 norm = " << vectemp2.Norml2() << "\n";

        DTS.SetSize(DTtemp->Height());
        DTtemp->Mult(*S, DTS);
        vectemp1 += DTS;
    }

    vectemp1.Print();
    std::cout << "vectemp1 norm = " << vectemp1.Norml2() << "\n";

    if (numblocks > 1)
    {
        std::cout << "lvlrhsfunc, second block \n";
        //for ( int i = 0; i < lvlrhs_func.GetBlock(1).Size(); ++i )
            //std::cout << lvlrhs_func.GetBlock(1)[i] << " " << (*bdrdofs_blocks[1])][i] << "\n";
        lvlrhs_func.GetBlock(1).Print();
    }
    */


#endif
    // loop over all AE, solving a local problem in each AE
    int nAE = AE_edofs_L2->Height();
    for( int AE = 0; AE < nAE; ++AE)
    {
#ifdef COMPUTE_EXACTDISCRETESOL
        Vector sigma_loc, S_loc, lambda_loc;
#endif

        //std::cout << "AE = " << AE << "\n";
        bool is_degenerate = true;
        sub_Func_offsets[0] = 0;
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            SparseMatrix AE_eintdofs_blk = AE_eintdofs_blocks->GetBlock(blk,blk);

            // FIXME: Is this necessary?
            // For each AE a new command?
            Array<int> tempview_inds(AE_eintdofs_blk.GetRowColumns(AE), AE_eintdofs_blk.RowSize(AE));
            Local_inds[blk] = new Array<int>;
            tempview_inds.Copy(*Local_inds[blk]);

            //bdrdofs_blocks[blk1]->Print();
            //essbdrdofs_blocks[blk]->Print();
            if (blk == 0) // degeneracy comes from Constraint matrix which involves only sigma = the first block
            {
                for (int i = 0; i < Local_inds[blk]->Size(); ++i)
                {
                    if ( (*bdrdofs_blocks[blk])[(*Local_inds[blk])[i]] != 0 &&
                         (*essbdrdofs_blocks[blk])[(*Local_inds[blk])[i]] == 0)
                    {
                        //std::cout << "i = " << i << "\n";
                        //std::cout << "bdrdofs[" << (*Local_inds[blk])[i] << "] = " << (*bdrdofs_blocks[blk])[(*Local_inds[blk])[i]] << "\n";
                        //std::cout << "essbdrdofs[" << (*Local_inds[blk])[i] << "] = " << (*essbdrdofs_blocks[blk])[(*Local_inds[blk])[i]] << "\n";
                        //std::cout << "then local problem is non-degenerate \n";
                        is_degenerate = false;
                        break;
                    }
                }
            } // end of if blk == 0
        } // end of loop over blocks

        for ( int blk1 = 0; blk1 < numblocks; ++blk1 )
        {
            for ( int blk2 = 0; blk2 < numblocks; ++blk2 )
            {
                if (blk1 == 0 && blk2 == 0) // handling L2 block (constraint)
                {
                    Array<int> Wtmp_j(AE_edofs_L2->GetRowColumns(AE), AE_edofs_L2->RowSize(AE));
                    if (compute_AEproblem_matrices(numblocks, numblocks))
                    {
                        sub_Constr.SetSize(Wtmp_j.Size(), Local_inds[blk1]->Size());
                        Constr_spmat.GetSubMatrix(Wtmp_j, *Local_inds[blk1], sub_Constr);
                    }

                    if (localrhs_constr)
                    {
                        localrhs_constr->GetSubVector(Wtmp_j, sub_rhsconstr);
                    }
                    else
                    {
                        sub_rhsconstr.SetSize(Wtmp_j.Size());
                        sub_rhsconstr = 0.0;
                    }

#ifdef COMPUTE_EXACTDISCRETESOL
                    lambda_loc.SetSize(Wtmp_j.Size());
                    lambda->GetSubVector(Wtmp_j, lambda_loc);
#endif

                } // end of special treatment of the first block involved into constraint

                sub_Func_offsets[blk1 + 1] = sub_Func_offsets[blk1] + Local_inds[blk1]->Size();

                if (compute_AEproblem_matrices(blk1,blk2))
                    Op_blks(blk1,blk2) = &(Op_blkspmat.GetBlock(blk1,blk2));

                if (compute_AEproblem_matrices(blk1,blk2))
                {
                    // FIXME: Memory leak here? New allocation for each AE but no free
                    // Setting size of Dense Matrices
                    LocalAE_Matrices(blk1,blk2) = new DenseMatrix(Local_inds[blk1]->Size(), Local_inds[blk2]->Size());

                    // Obtaining submatrices:
                    Op_blks(blk1,blk2)->GetSubMatrix(*Local_inds[blk1], *Local_inds[blk2], *LocalAE_Matrices(blk1,blk2));

                } // end of the block for non-optimized version


            }
        } // end of loop over all blocks in the functional

        BlockVector sub_Func(sub_Func_offsets);

        for ( int blk = 0; blk < numblocks; ++blk )
        {
            lvlrhs_func.GetBlock(blk).GetSubVector(*Local_inds[blk], sub_Func.GetBlock(blk));
        }

        /*
        std::cout << "sub_Func first block indices \n";
        Local_inds[0]->Print();
        std::cout << "sub_Func first block \n";
        sub_Func.GetBlock(0).Print();

        if (numblocks > 1)
        {
            std::cout << "sub_Func second block indices \n";
            std::cout << "size = " << Local_inds[1]->Size() << "\n";
            Local_inds[1]->Print();
            std::cout << "sub_Func second block \n";
            sub_Func.GetBlock(1).Print();
        }
        */

        BlockVector sol_loc(sub_Func_offsets);
        sol_loc = 0.0;

        //sub_rhsconstr.Print();
#ifdef COMPUTE_EXACTDISCRETESOL
        /*
        sigma_loc.SetSize(sol_loc.GetBlock(0).Size());
        sigma->GetSubVector(*Local_inds[0], sigma_loc);
        if (numblocks == 2)
        {
            S_loc.SetSize(sol_loc.GetBlock(1).Size());
            S->GetSubVector(*Local_inds[1], S_loc);
        }

        Vector vectemp1_loc(sol_loc.GetBlock(0).Size());
        vectemp1.GetSubVector(*Local_inds[0], vectemp1_loc);

        std::cout << "vectemp1_loc \n";
        vectemp1_loc.Print();

        Vector Asigma_loc(sol_loc.GetBlock(0).Size());
        Asigma.GetSubVector(*Local_inds[0], Asigma_loc);

        Vector BTlambda_loc(sol_loc.GetBlock(0).Size());
        BTlambda.GetSubVector(*Local_inds[0], BTlambda_loc);

        Vector v1(Asigma_loc.Size());
        v1 = Asigma_loc;
        v1 += BTlambda_loc;

        if (numblocks > 1)
        {
            Vector DTS_loc(sol_loc.GetBlock(0).Size());
            DTS.GetSubVector(*Local_inds[0], DTS_loc);
            v1 += DTS_loc;
            std::cout << "Asigma_loc  + BTlambda_loc + DTS_loc \n";
        }
        else
        {
            std::cout << "Asigma_loc  + BTlambda_loc \n";
        }
        v1.Print();

        Vector Aloc_sigmaloc(sol_loc.GetBlock(0).Size());
        LocalAE_Matrices(0,0)->Mult(sigma_loc, Aloc_sigmaloc);

        Vector v2(Aloc_sigmaloc.Size());
        v2 = Aloc_sigmaloc;
        v2 -= Asigma_loc;
        std::cout << "Asigma_loc - Aloc * sigmaloc \n";
        std::cout << "Should not be 0 actually \n";
        v2.Print();

        Vector BTloc_lambdaloc(sol_loc.GetBlock(0).Size());
        sub_Constr.MultTranspose(lambda_loc, BTloc_lambdaloc);

        Vector v3(BTloc_lambdaloc.Size());
        v3 = BTloc_lambdaloc;
        v3 -= BTlambda_loc;
        std::cout << "BTlambda_loc - BTloc * lambdaloc \n";
        v3.Print();

        if (numblocks > 1)
        {
            Vector DTS_loc(sol_loc.GetBlock(0).Size());
            DTS.GetSubVector(*Local_inds[0], DTS_loc);

            Vector DTloc_Sloc(sol_loc.GetBlock(0).Size());
            LocalAE_Matrices(0,1)->Mult(S_loc, DTloc_Sloc);

            Vector v4(DTloc_Sloc.Size());
            v4 = DTloc_Sloc;
            v4 -= DTS_loc;
            std::cout << "DTS_loc - DTloc * Sloc \n";
            v4.Print();

            Vector Dloc_sigmaloc(sol_loc.GetBlock(1).Size());
            LocalAE_Matrices(1,0)->Mult(sigma_loc, Dloc_sigmaloc);

            Vector Dsigma_loc(sol_loc.GetBlock(1).Size());
            Dsigma.GetSubVector(*Local_inds[1], Dsigma_loc);

            Vector v5(Dloc_sigmaloc.Size());
            v5 = Dloc_sigmaloc;
            v5 -= Dsigma_loc;
            std::cout << "Dsigma_loc - Dloc * sigmaloc \n";
            v5.Print();

            Vector Cloc_Sloc(sol_loc.GetBlock(1).Size());
            LocalAE_Matrices(1,1)->Mult(S_loc, Cloc_Sloc);

            Vector CS_loc(sol_loc.GetBlock(1).Size());
            CS.GetSubVector(*Local_inds[1], CS_loc);

            Vector v6(sol_loc.GetBlock(1).Size());
            v6 = Dsigma_loc;
            v6 += CS_loc;
            std::cout << "Dsigma_loc + CS_loc \n";
            v6.Print();

        }
        */
#endif

        // solving local problem at the agglomerate element AE
#ifdef COMPUTE_EXACTDISCRETESOL
        SolveLocalProblem(AE, LocalAE_Matrices, sub_Constr, sub_Func, sub_rhsconstr,
                          sol_loc, is_degenerate, &sigma_loc, &S_loc, &lambda_loc);
#else
        SolveLocalProblem(AE, LocalAE_Matrices, sub_Constr, sub_Func, sub_rhsconstr,
                          sol_loc, is_degenerate);
#endif
        // computing solution as a vector at current level
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            sol.GetBlock(blk).AddElementVector
                    (*Local_inds[blk], sol_loc.GetBlock(blk));
        }

    } // end of loop over AEs

    for (int blk = 0; blk < numblocks; ++blk)
        //d_td_blocks[blk]->MultTranspose(1.0, sol_update.GetBlock(blk), 1.0, truesol.GetBlock(blk));
        d_td_blocks[blk]->MultTranspose(sol.GetBlock(blk), truesol.GetBlock(blk));

    return;

}

#ifdef COMPUTE_EXACTDISCRETESOL
void LocalProblemSolver::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate, Vector * sigma_loc, Vector * S_loc, Vector * lambda_loc) const
#else
void LocalProblemSolver::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate) const
#endif
{
    if (optimized_localsolve)
    {
        DenseMatrixInverse * inv_A = LUfactors[AE][0];
        DenseMatrixInverse * inv_Schur = LUfactors[AE][1];
        SolveLocalProblemOpt(inv_A, inv_Schur, B, G, F, sol, is_degenerate);
    }
    else // lazy variant with computations of lu each time
    {
        // creating a Schur complement matrix Binv(A)BT
        //std::cout << "Inverting A: \n";
        //FunctBlks[0].Print();
        DenseMatrixInverse inv_A(*FunctBlks(0,0));

        // invAG = invA * G
        Vector invAG;
        inv_A.Mult(G, invAG);

        DenseMatrix BT(B.Width(), B.Height());
        BT.Transpose(B);

        DenseMatrix invABT;
        inv_A.Mult(BT, invABT);

        // Schur = BinvABT
        DenseMatrix Schur(B.Height(), invABT.Width());
        mfem::Mult(B, invABT, Schur);

        //std::cout << "Inverting Schur: \n";

        // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
        if (is_degenerate)
        {
            Schur.SetRow(0,0);
            Schur.SetCol(0,0);
            Schur(0,0) = 1.;
        }

        //Schur.Print();
        DenseMatrixInverse inv_Schur(Schur);

        // temp = ( B * invA * G - F )
        Vector temp(B.Height());
        B.Mult(invAG, temp);
        temp -= F;

        if (is_degenerate)
            temp(0) = 0;

        // lambda = inv(BinvABT) * ( B * invA * G - F )
        Vector lambda(inv_Schur.Height());
        inv_Schur.Mult(temp, lambda);

        // temp2 = (G - BT * lambda)
        Vector temp2(B.Width());
        B.MultTranspose(lambda,temp2);
        temp2 *= -1;
        temp2 += G;

        // sig = invA * temp2 = invA * (G - BT * lambda)
        inv_A.Mult(temp2, sol.GetBlock(0));

#ifdef COMPUTE_EXACTDISCRETESOL
        std::cout << "rhs func block 0 \n";
        G.GetBlock(0).Print();
        std::cout << "sol block 0 \n";
        sol.GetBlock(0).Print();
#endif

        /*
#ifdef COMPUTE_EXACTDISCRETESOL
        // checking that exact discrete solution satisfies local problems

        double checkval = 0.0;

        Vector res3special(F.Size());
        B.Mult(*sigma_loc, res3special);
        res3special -= F;

        checkval = std::max(1.0e-13 * F.Norml2(), 1.0e-13);
        if (!(res3special.Norml2() < checkval))
        {
            std::cout << "res3special: \n";
            res3special.Print();
            std::cout << "sigma_loc: \n";
            sigma_loc->Print();
            std::cout << "F: \n";
            F.Print();
            std::cout << "norm res3special = " << res3special.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }

        MFEM_ASSERT(res3special.Norml2() < checkval,
                    "Local system is not solved by exact discrete solution, res3special != 0 \n");

        Vector res1special(G.GetBlock(0).Size());
        Vector temp1res1special(G.GetBlock(0).Size());
        FunctBlks(0,0)->Mult(*sigma_loc, temp1res1special);
        Vector temp3res1special(G.GetBlock(0).Size());
        B.MultTranspose(*lambda_loc, temp3res1special);
        res1special = 0.0;
        res1special += temp1res1special;
        res1special += temp3res1special;
        //res1special -= G.GetBlock(0);

        checkval = std::max(1.0e-13 * G.GetBlock(0).Norml2(), 1.0e-13);
        //std::cout << "res1special norm = " << res1special.Norml2() << "\n";
        if (!(res1special.Norml2() < checkval))
        {
            std::cout << "res1special: \n";
            res1special.Print();
            std::cout << "norm res1special = " << res1special.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }
        MFEM_ASSERT(res1special.Norml2() < checkval,
                    "Local system is not solved by exact discrete solution, res1special too large \n");
#endif
        */
        //std::cout << "sol \n";
        //sol.Print();
    }

    return;
}

// Optimized version of SolveLocalProblem where LU factors for the local
// problem's matrices were computed during the setup via SaveLocalLUFactors()
void LocalProblemSolver::SolveLocalProblemOpt(DenseMatrixInverse * inv_A, DenseMatrixInverse * inv_Schur,
                                           DenseMatrix& B, BlockVector &G,
                                           Vector& F, BlockVector &sol, bool is_degenerate) const
{
    // invAG = invA * G
    Vector invAG;
    inv_A->Mult(G, invAG);

    // temp = ( B * invA * G - F )
    Vector temp(B.Height());
    B.Mult(invAG, temp);
    temp -= F;

    if (is_degenerate)
        temp(0) = 0;

    // lambda = inv(BinvABT) * ( B * invA * G - F )
    Vector lambda(B.Height());
    inv_Schur->Mult(temp, lambda);

    // temp2 = (G - BT * lambda)
    Vector temp2(B.Width());
    B.MultTranspose(lambda,temp2);
    temp2 *= -1;
    temp2 += G;

    // sig = invA * temp2 = invA * (G - BT * lambda)
    inv_A->Mult(temp2, sol.GetBlock(0));
}

void LocalProblemSolver::SaveLocalLUFactors() const
{
    if (!optimized_localsolve)
        return;

    int nAE = AE_edofs_L2->Height();
    LUfactors.resize(nAE);

    DenseMatrix sub_Constr;
    DenseMatrix sub_Func;

    SparseMatrix * AE_eintdofs = &(AE_eintdofs_blocks->GetBlock(0,0));
    const SparseMatrix * Op_blk = &(Op_blkspmat.GetBlock(0,0));

    // loop over all AE, computing and saving factorization
    // of local saddle point matrices in each AE
    for( int AE = 0; AE < nAE; ++AE)
    {
        // for each AE we will store A^(-1) and Schur^(-1)
        LUfactors[AE].resize(2);

        //std::cout << "AE = " << AE << "\n";
        bool is_degenerate = true;

        //Array<int> tempview_inds(AE_eintdofs->GetRowColumns(AE), AE_eintdofs->RowSize(AE));
        //Local_inds = new Array<int>;
        //tempview_inds.Copy(Local_inds[0]);
        Array<int> Local_inds(AE_eintdofs->GetRowColumns(AE), AE_eintdofs->RowSize(AE));

        Array<int> Wtmp_j(AE_edofs_L2->GetRowColumns(AE), AE_edofs_L2->RowSize(AE));
        sub_Constr.SetSize(Wtmp_j.Size(), Local_inds.Size());
        Constr_spmat.GetSubMatrix(Wtmp_j, Local_inds, sub_Constr);

        for (int i = 0; i < Local_inds.Size(); ++i)
        {
            if ( (*bdrdofs_blocks[0])[Local_inds[i]] != 0 &&
                 (*essbdrdofs_blocks[0])[Local_inds[i]] == 0)
            {
                //std::cout << "then local problem is non-degenerate \n";
                is_degenerate = false;
                break;
            }
        }

        // Setting size of Dense Matrices
        sub_Func.SetSize(Local_inds.Size());

        // Obtaining submatrices:
        Op_blk->GetSubMatrix(Local_inds, Local_inds, sub_Func);

        LUfactors[AE][0] = new DenseMatrixInverse(sub_Func);

        DenseMatrix sub_ConstrT(sub_Constr.Width(), sub_Constr.Height());
        sub_ConstrT.Transpose(sub_Constr);

        DenseMatrix invABT;
        LUfactors[AE][0]->Mult(sub_ConstrT, invABT);

        // Schur = BinvABT
        DenseMatrix Schur(sub_Constr.Height(), invABT.Width());
        mfem::Mult(sub_Constr, invABT, Schur);

        // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
        if (is_degenerate)
        {
            Schur.SetRow(0,0);
            Schur.SetCol(0,0);
            Schur(0,0) = 1.;
        }

        //Schur.Print();
        LUfactors[AE][1] = new DenseMatrixInverse(Schur);

    } // end of loop over AEs

    compute_AEproblem_matrices = false;
    compute_AEproblem_matrices(numblocks, numblocks) = true;
}

// Returns a pointer to a BlockMatrix which stores
// the relation between agglomerated elements (AEs)
// and fine-grid internal (w.r.t. to AEs) dofs for each block.
// For lowest-order elements all the fine-grid dofs will be
// located at the boundary of fine-grid elements and not inside, but
// for higher order elements there will be two parts,
// one for dofs at fine-grid element faces which belong to the global boundary
// and a different treatment for internal (w.r.t. to fine elements) dofs
BlockMatrix* LocalProblemSolver::Get_AE_eintdofs(const BlockMatrix &el_to_dofs,
                                        const std::vector<Array<int>* > &dof_is_essbdr,
                                        const std::vector<Array<int>* > &dof_is_bdr) const
{
#ifdef DEBUG_INFO
    Vector dofs_check;
#endif

    MPI_Comm comm = d_td_blocks[0]->GetComm();
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    Array<int> res_rowoffsets(numblocks+1);
    res_rowoffsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        res_rowoffsets[blk + 1] = res_rowoffsets[blk] + AE_e.Height();
    Array<int> res_coloffsets(numblocks+1);
    res_coloffsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        res_coloffsets[blk + 1] = res_coloffsets[blk] + el_to_dofs.GetBlock(blk,blk).Width();

    BlockMatrix * res = new BlockMatrix(res_rowoffsets, res_coloffsets);

    for (int blk = 0; blk < numblocks; ++blk)
    {
        // a shortcut
        const SparseMatrix * el_dofs_blk = &(el_to_dofs.GetBlock(blk,blk));

#ifdef IMPROVEMENT
        SparseMatrix d_td_diag;
        SparseMatrix td_d_diag;
        HypreParMatrix * td_d;
        SparseMatrix td_d_offd;
        if (num_procs > 1)
        {
            // FIXME: Memory clean-up needed or not here?
            // things from dof_truedof relation tables required to determine shared dofs
            d_td_blocks[blk]->GetDiag(d_td_diag);
            //SparseMatrix d_td_blocks_offd;
            //HYPRE_Int * cmap_d_td;
            //d_td_blocks[blk]->GetOffd(d_td_blocks_offd, cmap_d_td);

            td_d = d_td_blocks[blk]->Transpose();
            td_d->GetDiag(td_d_diag);
            HYPRE_Int * cmap_td_d;
            td_d->GetOffd(td_d_offd, cmap_td_d);
        }

#endif
        // creating dofs_to_AE relation table
        SparseMatrix * dofs_AE = Transpose(*mfem::Mult(AE_e, *el_dofs_blk));
        int ndofs = dofs_AE->Height();
#ifdef DEBUG_INFO
        if (blk == 0)
        {
            dofs_check.SetSize(ndofs);
            dofs_check = -1.0;
        }
#endif

#ifndef IMPROVEMENT
        int * dofs_AE_i = dofs_AE->GetI();
        int * dofs_AE_j = dofs_AE->GetJ();
        double * dofs_AE_data = dofs_AE->GetData();
#endif
        int * innerdofs_AE_i = new int [ndofs + 1];

        // computing the number of internal degrees of freedom in all AEs
        int nnz = 0;
        //bool increased1, increased2;
        for (int dof = 0; dof < ndofs; ++dof)
        {
            //increased1 = false;
            //increased2 = false;

            //std::cout << "dof = " << dof << "\n";

            innerdofs_AE_i[dof]= nnz;

#ifdef IMPROVEMENT
            bool dof_is_shared = false;

            if (num_procs > 1)
            {
                bool own_tdof = (d_td_diag.RowSize(dof) > 0);
                if (own_tdof) // then it is either a local tdof or a boundary tdof (~ shared dof)
                {
                    int tdof = d_td_diag.GetRowColumns(dof)[0];
                    dof_is_shared = (td_d_offd.RowSize(tdof) > 0);
                }
                else
                    dof_is_shared = true;
            }

            bool dof_on_bdr = ((*dof_is_bdr[blk])[dof]!= 0 );
            bool dof_on_nonessbdr = ( (*dof_is_essbdr[blk])[dof] == 0 && dof_on_bdr);

            //std::cout << "dof_is_shared = " << dof_is_shared << "\n";
            //std::cout << "dof_on_bdr = " << dof_on_bdr << "\n";
            //std::cout << "dof_on_nonessbdr = " << dof_on_nonessbdr << "\n";
            //std::cout << "dofs_AE->RowSize(dof) = " << dofs_AE->RowSize(dof) << "\n";

            if (( (dofs_AE->RowSize(dof) == 1 && !dof_on_bdr) || dof_on_nonessbdr) && (!dof_is_shared) )
            {
                nnz++;
                //std::cout << "new code increases nnz! \n";
                //increased1 = true;
                //nnz--;
            }

            /*
            //bool on_noness_bdr = false;
            bool on_noness_bdr = ( (*dof_is_essbdr[blk])[dof] == 0 &&
                                  (*dof_is_bdr[blk])[dof]!= 0);

            std::cout << "on_noness_bdr = " << on_noness_bdr << "\n";

            for (int j = dofs_AE_i[dof]; j < dofs_AE_i[dof+1]; ++j)
            {
                // if a dof belongs to only one fine-grid element and is not at the domain boundary
                bool inside_finegrid_el = (higher_order &&
                                           (*dof_is_bdr[blk])[dof] == 0 && dofs_AE_data[j] == 1);
                std::cout << "inside finegrid = " << inside_finegrid_el << "\n";

                MFEM_ASSERT( !inside_finegrid_el,
                        "Remove this assert in Get_AE_eintdofs() before using higher-order elements");
                MFEM_ASSERT( ( !inside_finegrid_el || (dofs_AE_i[dof+1] - dofs_AE_i[dof] == 1) ),
                        "A fine-grid dof inside a fine-grid element cannot belong to more than one AE");
                // if a dof is shared by two fine grid elements inside a single AE
                // OR a dof is strictly internal to a fine-grid element,
                // OR a dof belongs to the non-essential part of the domain boundary,
                // then it is an internal dof for this AE

                std::cout << "dofs_AE_data[j] = " << dofs_AE_data[j] << "\n";
                if (dofs_AE_data[j] == 2 || inside_finegrid_el || on_noness_bdr )
                {
                    //std::cout << "old code increases nnz! \n";
                    increased2 = true;
                    nnz++;
                }
            }

            if (increased1 != increased2)
            {
                std::cout << "dof = " << dof << "gives different result for old and new codes \n";
                std::cout << "increase1 = " << increased1 << ", increased2 = " << increased2 << "\n";
                std::cout << "pampam! \n";
            }

            std::cout << "\n";
            */

#else
            for (int j = dofs_AE_i[dof]; j < dofs_AE_i[dof+1]; ++j)
            {
                // if a dof belongs to only one fine-grid element and is not at the domain boundary
                bool inside_finegrid_el = (higher_order &&
                                           (*dof_is_bdr[blk])[dof] == 0 && dofs_AE_data[j] == 1);
                //bool on_noness_bdr = false;
                bool on_noness_bdr = ( (*dof_is_essbdr[blk])[dof] == 0 &&
                                      (*dof_is_bdr[blk])[dof]!= 0);
                MFEM_ASSERT( !inside_finegrid_el,
                        "Remove this assert in Get_AE_eintdofs() before using higher-order elements");
                MFEM_ASSERT( ( !inside_finegrid_el || (dofs_AE_i[dof+1] - dofs_AE_i[dof] == 1) ),
                        "A fine-grid dof inside a fine-grid element cannot belong to more than one AE");
                // if a dof is shared by two fine grid elements inside a single AE
                // OR a dof is strictly internal to a fine-grid element,
                // OR a dof belongs to the non-essential part of the domain boundary,
                // then it is an internal dof for this AE
                if (dofs_AE_data[j] == 2 || inside_finegrid_el || on_noness_bdr )
                    nnz++;
            }
#endif


        }
        innerdofs_AE_i[ndofs] = nnz;

        // allocating j and data arrays for the created relation table
        int * innerdofs_AE_j = new int[nnz];
        double * innerdofs_AE_data = new double[nnz];

        int nnz_count = 0;
        for (int dof = 0; dof < ndofs; ++dof)
        {
#ifdef IMPROVEMENT
            bool dof_is_shared = false;

            if (num_procs > 1)
            {
                bool own_tdof = (d_td_diag.RowSize(dof) > 0);
                if (own_tdof) // then it is either a local tdof or a boundary tdof (~ shared dof)
                {
                    int tdof = d_td_diag.GetRowColumns(dof)[0];
                    dof_is_shared = (td_d_offd.RowSize(tdof) > 0);
                }
                else
                    dof_is_shared = true;
            }

            bool dof_on_bdr = ((*dof_is_bdr[blk])[dof]!= 0 );
            bool dof_on_nonessbdr = ( (*dof_is_essbdr[blk])[dof] == 0 && dof_on_bdr);

            if (( (dofs_AE->RowSize(dof) == 1 && !dof_on_bdr) || dof_on_nonessbdr) && (!dof_is_shared) )
                innerdofs_AE_j[nnz_count++] = dofs_AE->GetRowColumns(dof)[0];

#else
            for (int j = dofs_AE_i[dof]; j < dofs_AE_i[dof+1]; ++j)
            {
#ifdef DEBUG_INFO
                dofs_check[dof] = 0;
#endif
                bool inside_finegrid_el = (higher_order &&
                                           (*dof_is_bdr[blk])[dof] == 0 && dofs_AE_data[j] == 1);
                //bool on_noness_bdr = false;
                bool on_noness_bdr = ( (*dof_is_essbdr[blk])[dof] == 0 &&
                                      (*dof_is_bdr[blk])[dof]!= 0);
                if (dofs_AE_data[j] == 2 || inside_finegrid_el || on_noness_bdr )
                {
                    innerdofs_AE_j[nnz_count++] = dofs_AE_j[j];
#ifdef DEBUG_INFO
                    dofs_check[dof] = 1;
#endif
                }
#ifdef DEBUG_INFO
                if ( (*dof_is_essbdr[blk])[dof] != 0)
                {
                    if (dofs_check[dof] > 0)
                        std::cout << "Error: Smth wrong in dofs \n";
                    else
                        dofs_check[dof] = 2;
                }
                if (dofs_AE_data[j] == 1 && dofs_AE_i[dof+1] - dofs_AE_i[dof] == 2)
                {
                    if (dofs_check[dof] > 0)
                        std::cout << "Error: Smth wrong in dofs \n";
                    else
                        dofs_check[dof] = 3;
                }
#endif
            }
#endif


        }

        std::fill_n(innerdofs_AE_data, nnz, 1);

        // creating a relation between internal fine-grid dofs (w.r.t to AE) and AEs,
        // keeeping zero rows for non-internal dofs
        SparseMatrix * innerdofs_AE = new SparseMatrix(innerdofs_AE_i, innerdofs_AE_j, innerdofs_AE_data,
                                                       dofs_AE->Height(), dofs_AE->Width());
        //std::cout << "dofs_check \n";
        //dofs_check.Print();

        //if (blk == 1)
            //innerdofs_AE->Print();

        delete dofs_AE;

        res->SetBlock(blk, blk, Transpose(*innerdofs_AE));

        //if (blk == 1)
            //res->GetBlock(blk,blk).Print();
        //return Transpose(*innerdofs_AE);
    }

    return res;
}

class LocalProblemSolverWithS : public LocalProblemSolver
{
#ifdef COMPUTE_EXACTDISCRETESOL
    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate, Vector* sigma_loc = NULL, Vector * S_loc = NULL, Vector * lambda_loc = NULL) const override;
#else
    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate) const override;
#endif
    // Optimized version of SolveLocalProblem where LU factors for the local
    // problem's matrices were computed during the setup via SaveLocalLUFactors()
    void SolveLocalProblemOpt(DenseMatrixInverse * inv_A, DenseMatrixInverse * inv_Schur,
                                               std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B, BlockVector &G,
                                               Vector& F, BlockVector &sol, bool is_degenerate) const;
    // an optional routine which can save LU factors for the local problems
    // solved at finer levels if needed. Should be redefined in the inheriting
    // classes in order to speed up iterations
    virtual void SaveLocalLUFactors() const override;
public:
    // main constructor
    LocalProblemSolverWithS(const BlockMatrix& Op_Blksmat,
                       const SparseMatrix& Constr_Spmat,
                       const std::vector<HypreParMatrix*>& D_tD_blks,
                       const SparseMatrix& AE_el,
                       const BlockMatrix& El_to_Dofs_Func,
                       const SparseMatrix& El_to_Dofs_L2,
                       const std::vector<Array<int>* >& BdrDofs_blks,
                       const std::vector<Array<int>* >& EssBdrDofs_blks,
#ifdef COMPUTE_EXACTDISCRETESOL
                       Vector * sigma, Vector * S, Vector * lambda,
#endif
                       bool Optimized_LocalSolve,
                       bool Higher_Order)
        : LocalProblemSolver(Op_Blksmat,
                              Constr_Spmat,
                              D_tD_blks,
                              AE_el,
                              El_to_Dofs_Func, El_to_Dofs_L2,
                              BdrDofs_blks, EssBdrDofs_blks,
#ifdef COMPUTE_EXACTDISCRETESOL
                              sigma, S, lambda,
#endif
                              Optimized_LocalSolve, Higher_Order)
    {
    }

};

// TODO: Implement this after the non-optimized version starts working
void LocalProblemSolverWithS::SaveLocalLUFactors() const
{
    MFEM_ABORT("LocalProblemSolverWithS::SaveLocalLUFactors is not implemented!");
}

#ifdef COMPUTE_EXACTDISCRETESOL
void LocalProblemSolverWithS::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate, Vector * sigma_loc, Vector * S_loc, Vector * lambda_loc) const
#else
void LocalProblemSolverWithS::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate) const
#endif
{
    if (optimized_localsolve)
    {
        MFEM_ABORT("Optimized local problem solving routine was not implemented yet \n");
        //DenseMatrixInverse * inv_A = LUfactors[AE][0];
        //DenseMatrixInverse * inv_Schur = LUfactors[AE][1];
        //SolveLocalProblemOpt(inv_A, inv_Schur, B, G, F, sol, is_degenerate);
    }
    else // an expensive variant with computations of lu each time
    {
        Vector lambda(B.Height());

        //shortcuts
        DenseMatrix * A = FunctBlks(0,0);
        DenseMatrix * C = FunctBlks(1,1);
        DenseMatrix * D = FunctBlks(1,0);
        DenseMatrix * DT = FunctBlks(0,1);

        if (G.GetBlock(1).Size() == 0) // this means no internal dofs for S in the current AE
        {
            // then only a 2x2 block system is to be solved
            // creating a Schur complement matrix Binv(A)BT
            //std::cout << "Inverting A: \n";
            //FunctBlks[0].Print();
            DenseMatrixInverse inv_A(*FunctBlks(0,0));

            // invAG = invA * G
            Vector invAG;
            inv_A.Mult(G, invAG);

            DenseMatrix BT(B.Width(), B.Height());
            BT.Transpose(B);

            DenseMatrix invABT;
            inv_A.Mult(BT, invABT);

            // Schur = BinvABT
            DenseMatrix Schur(B.Height(), invABT.Width());
            mfem::Mult(B, invABT, Schur);

            //std::cout << "Inverting Schur: \n";

            // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
            if (is_degenerate)
            {
                Schur.SetRow(0,0);
                Schur.SetCol(0,0);
                Schur(0,0) = 1.;
            }

            //Schur.Print();
            DenseMatrixInverse inv_Schur(Schur);

            // temp = ( B * invA * G - F )
            Vector temp(B.Height());
            B.Mult(invAG, temp);
            temp -= F;

            if (is_degenerate)
                temp(0) = 0;

            // lambda = inv(BinvABT) * ( B * invA * G - F )
            inv_Schur.Mult(temp, lambda);

            // temp2 = (G - BT * lambda)
            Vector temp2(B.Width());
            B.MultTranspose(lambda,temp2);
            temp2 *= -1;
            temp2 += G;

            // sig = invA * temp2 = invA * (G - BT * lambda)
            inv_A.Mult(temp2, sol.GetBlock(0));
        }
        else // then it is 3x3 block matrix under consideration
        {
            // creating a Schur complement matrix Binv(A)BT

            // creating inv_C
            //std::cout << "inv_C \n";
            //C->Print();
            DenseMatrixInverse inv_C(*C);

            // creating D * inv_C * DT
            DenseMatrix invCD;
            inv_C.Mult(*D, invCD);

            DenseMatrix DTinvCD(DT->Height(), D->Width());
            mfem::Mult(*DT, invCD, DTinvCD);

            // creating inv Atilda = inv(A - D * inv_C * DT)
            DenseMatrix Atilda(A->Height(), A->Width());
            Atilda = *A;
            Atilda -= DTinvCD;

            //std::cout << "inv_Atilda \n";
            //Atilda.Print();
            DenseMatrixInverse inv_Atilda(Atilda);

            // computing Schur = B * inv_Atilda * BT
            DenseMatrix inv_AtildaBT;
            DenseMatrix BT(B.Width(), B.Height());
            BT.Transpose(B);
            inv_Atilda.Mult(BT, inv_AtildaBT);

            DenseMatrix Schur(B.Height(), B.Height());
            mfem::Mult(B, inv_AtildaBT, Schur);

            // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
            if (is_degenerate)
            {
                //std::cout << "degenerate! \n";
                Schur.SetRow(0,0);
                Schur.SetCol(0,0);
                Schur(0,0) = 1.;
            }

            //std::cout << "inv_Schur \n";
            //Schur.Print();
            DenseMatrixInverse inv_Schur(Schur);

            // creating DT * invC * F_S
            Vector invCF2;
            inv_C.Mult(G.GetBlock(1), invCF2);

            Vector DTinvCF2(DT->Height());
            DT->Mult(invCF2, DTinvCF2);

            // creating F1tilda = F_sigma - DT * invC * F_S
            Vector F1tilda(DT->Height());
            F1tilda = G.GetBlock(0);
            F1tilda -= DTinvCF2;

            // creating invAtildaFtilda = inv(Atilda) * Ftilda =
            // = inv(A - D * inv_C * DT) * (F_sigma - DT * invC * F_S)
            Vector invAtildaFtilda(Atilda.Height());
            inv_Atilda.Mult(F1tilda, invAtildaFtilda);

            Vector FinalFlam(B.Height());
            B.Mult(invAtildaFtilda, FinalFlam);
            FinalFlam -= F;

            if (is_degenerate)
                FinalFlam(0) = 0;

            // lambda = inv_Schur * ( F_lam - B * inv(A - D * inv_C * DT) * (F_sigma - DT * invC * F_S) )
            inv_Schur.Mult(FinalFlam, lambda);

            // changing Ftilda so that Ftilda_new = Ftilda_old - BT * lambda
            // = F_sigma - DT * invC * F_S - BT * lambda
            Vector temp(B.Width());
            B.MultTranspose(lambda, temp);
            F1tilda -= temp;

            // sigma = inv_Atilda * Ftilda(new)
            // = inv(A - D * inv_C * DT) * ( F_sigma - DT * invC * F_S - BT * lambda )
            inv_Atilda.Mult(F1tilda, sol.GetBlock(0));

            // temp2 = F_S - D * sigma
            Vector temp2(D->Height());
            D->Mult(sol.GetBlock(0), temp2);
            temp2 *= -1.0;
            temp2 += G.GetBlock(1);

            inv_C.Mult(temp2, sol.GetBlock(1));
        }

        //std::cout << "AE = " << AE << "\n";
        //std::cout << "F (local constraint rhs): \n";
        //F.Print();

#ifdef CHECK_LOCALSOLVE
        //if (is_degenerate)
            //std::cout << "degenerate! \n";
        //else
            //std::cout << "non-degenerate! \n";

        /*
        std::cout << "sol block 0 \n";
        sol.GetBlock(0).Print();
        if (numblocks > 1)
        {
            std::cout << "sol block 1 \n";
            sol.GetBlock(1).Print();
        }
        //temporarily
        if (sol.GetBlock(0).Norml2() / sol.GetBlock(0).Size() > 1.0e-13)
            std::cout << "Caught the bug behavior \n";
        if (G.GetBlock(1).Size() != 0)
            if (sol.GetBlock(1).Norml2() / sol.GetBlock(1).Size() > 1.0e-12)
                std::cout << "Caught the bug behavior \n";
        //MFEM_ASSERT(sol.GetBlock(0).Norml2() / sol.GetBlock(0).Size() < 1.0e-13, "For exact solution init, local sigma updates must be 0 \n");
        //if (G.GetBlock(1).Size() != 0)
            //MFEM_ASSERT(sol.GetBlock(1).Norml2() / sol.GetBlock(1).Size() < 1.0e-12, "For exact solution init, local S updates must be 0 \n");

        std::cout << "lambda \n";
        lambda.Print();

        std::cout << "rhsfunc block 0 \n";
        G.GetBlock(0).Print();
        if (numblocks > 1)
        {
            std::cout << "rhsfunc block 1 \n";
            G.GetBlock(1).Print();
        }
        std::cout << "rhsconstr \n";
        F.Print();
        */
#endif

#ifdef CHECK_LOCALSOLVE
        // checking that the system was solved correctly
        double checkval = 0.0;

        Vector res1(G.GetBlock(0).Size());
        Vector temp1res1(G.GetBlock(0).Size());
        A->Mult(sol.GetBlock(0), temp1res1);
        Vector temp2res1(G.GetBlock(0).Size());
        if (G.GetBlock(1).Size() == 0)
            temp2res1 = 0.0;
        else
            DT->Mult(sol.GetBlock(1), temp2res1);
        Vector temp3res1(G.GetBlock(0).Size());
        B.MultTranspose(lambda, temp3res1);
        res1 = 0.0;
        res1 += temp1res1;
        res1 += temp2res1;
        res1 += temp3res1;
        res1 -= G.GetBlock(0);

        checkval = std::max(1.0e-12 * G.GetBlock(0).Norml2(), 1.0e-12);
        //std::cout << "res1 norm = " << res1.Norml2() << "\n";
        if (!(res1.Norml2() < checkval))
        {
            std::cout << "res1: \n";
            res1.Print();
            std::cout << "norm res1 = " << res1.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }
        MFEM_ASSERT(res1.Norml2() < checkval,
                    "Local system was solved incorrectly, res1 too large \n");

        if (G.GetBlock(1).Size() != 0)
        {
            Vector res2(G.GetBlock(1).Size());
            Vector temp1res2(G.GetBlock(1).Size());
            D->Mult(sol.GetBlock(0), temp1res2);
            Vector temp2res2(G.GetBlock(1).Size());
            C->Mult(sol.GetBlock(1), temp2res2);
            res2 = 0.0;
            res2 += temp1res2;
            res2 += temp2res2;
            res2 -= G.GetBlock(1);

            //std::cout << "res2 norm = " << res2.Norml2() << "\n";
            checkval = std::max(1.0e-12 * G.GetBlock(1).Norml2(), 1.0e-12);
            if (!(res2.Norml2() < checkval ))
            {
                std::cout << "res2: \n";
                res2.Print();
                std::cout << "norm res2 = " << res2.Norml2() <<
                             ", checkval = " << checkval << "\n";
            }
            MFEM_ASSERT(res2.Norml2() < checkval,
                        "Local system was solved incorrectly, res2 too large \n");
        }

        Vector res3(F.Size());
        B.Mult(sol.GetBlock(0), res3);
        res3 -= F;

        checkval = std::max(1.0e-12 * F.Norml2(), 1.0e-12);
        if (!(res3.Norml2() < checkval))
        {
            std::cout << "res3: \n";
            res3.Print();
            std::cout << "norm res3 = " << res3.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }

        MFEM_ASSERT(res3.Norml2() < checkval,
                    "Local system was solved incorrectly, res3 != 0 \n");
#endif

        /*
         * shouldn't work
#ifdef COMPUTE_EXACTDISCRETESOL
        // checking that exact discrete solution satisfies local problems
        Vector res3special(F.Size());
        B.Mult(*sigma_loc, res3special);
        res3special -= F;

        checkval = std::max(1.0e-13 * F.Norml2(), 1.0e-13);
        if (!(res3special.Norml2() < checkval))
        {
            std::cout << "res3special: \n";
            res3special.Print();
            std::cout << "norm res3special = " << res3special.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }

        MFEM_ASSERT(res3special.Norml2() < checkval,
                    "Local system is not solved by exact discrete solution, res3special != 0 \n");

        Vector res2special(G.GetBlock(1).Size());
        Vector temp1res2special(G.GetBlock(1).Size());
        D->Mult(*sigma_loc, temp1res2special);
        Vector temp2res2special(G.GetBlock(1).Size());
        C->Mult(*S_loc, temp2res2special);
        res2special = 0.0;
        res2special += temp1res2special;
        res2special += temp2res2special;
        res2special -= G.GetBlock(1);

        //std::cout << "res2special norm = " << res2special.Norml2() << "\n";
        checkval = std::max(1.0e-13 * G.GetBlock(1).Norml2(), 1.0e-13);
        if (!(res2special.Norml2() < checkval ))
        {
            std::cout << "res2special: \n";
            res2special.Print();
            std::cout << "norm res2special = " << res2special.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }
        MFEM_ASSERT(res2special.Norml2() < checkval,
                    "Local system is not solved by exact discrete solution, res2special too large \n");

        Vector res1special(G.GetBlock(0).Size());
        Vector temp1res1special(G.GetBlock(0).Size());
        A->Mult(*sigma_loc, temp1res1special);
        Vector temp2res1special(G.GetBlock(0).Size());
        DT->Mult(*S_loc, temp2res1special);
        Vector temp3res1special(G.GetBlock(0).Size());
        B.MultTranspose(*lambda_loc, temp3res1special);
        res1special = 0.0;
        res1special += temp1res1special;
        res1special += temp2res1special;
        res1special += temp3res1special;
        res1special -= G.GetBlock(0);

        checkval = std::max(1.0e-13 * G.GetBlock(0).Norml2(), 1.0e-13);
        //std::cout << "res1special norm = " << res1special.Norml2() << "\n";
        if (!(res1special.Norml2() < checkval))
        {
            std::cout << "res1special: \n";
            res1special.Print();
            std::cout << "norm res1special = " << res1special.Norml2() <<
                         ", checkval = " << checkval << "\n";
        }
        MFEM_ASSERT(res1special.Norml2() < checkval,
                    "Local system is not solved by exact discrete solution, res1special too large \n");
#endif
    */
    }

    return;
}


// class for finding a particular solution to a divergence constraint
class DivConstraintSolver : public Solver
{
private:
    // if true, coarsened operators will be constructed from Funct_lvls[0]
    // and Constr_levels[0]; else, the entire hierarchy of coarsened operators
    // must be provided in the constructor call of the solver
    const bool construct_coarseops;

    // a flag which indicates whether the solver setup was called
    // before trying to solve anything
    mutable bool setup_finished;

    mutable int print_level;
protected:
    int num_levels;

    // Relation tables which represent agglomerated elements-to-elements relation at each level
    const Array< SparseMatrix*>& AE_e;

    // Dof_TrueDof relation tables for each level for functional-related
    // variables and the L2 variable (constraint space).
    // Used for assembling the coarsest level problem
    // and for the smoother setup in the general case
    const std::vector<std::vector<HypreParMatrix*> >& dof_trueDof_Func_lvls;
    const std::vector<HypreParMatrix*> & dof_trueDof_L2_lvls;

    const MPI_Comm comm;

    // Projectors for the variables related to the functional and constraint
    const Array< BlockMatrix*>& P_Func; // used only for constructing coarsened operators now
    const Array< BlockOperator*>& TrueP_Func;
    const Array< SparseMatrix*>& P_L2; // used for operators coarsening and in ComputeLocalRhsConstr (via ProjectFinerL2ToCoarser)

    // for each level and for each variable in the functional stores a vector
    // which defines if a dof is at the boundary / essential part of the boundary
    // or not
    const std::vector<std::vector<Array<int>* > > & essbdrtruedofs_Func;

    // parts of block structure which define the Functional at the finest level
    const int numblocks;
    //const Array<int>& block_offsets;
    mutable Array<int> block_trueoffsets;

    // Righthand side of  the divergence contraint on dofs
    // (remains unchanged throughout the solving process)
    const Vector& ConstrRhs;

    const Array<Operator*>& Smoothers_lvls;

    // a given blockvector which satisfies essential bdr conditions
    // imposed for the initial problem
    // on true dofs
    const BlockVector& bdrdata_truedofs;

    // stores Functional matrix on all levels except the finest
    // so that Funct_levels[0] = Functional matrix on level 1 (not level 0!)
    mutable Array<BlockMatrix*> Funct_lvls;
    mutable Array<SparseMatrix*> Constr_lvls;

    mutable Array<BlockVector*> Funct_rhs_lvls;

    // The same as xblock and yblock but on true dofs
    mutable BlockVector* xblock_truedofs;
    mutable BlockVector* yblock_truedofs;
    mutable BlockVector* tempblock_truedofs;

    mutable Array<Array<int>* > trueoffsets_lvls;
    mutable Array<BlockVector*> truetempvec_lvls;
    mutable Array<BlockVector*> truetempvec2_lvls;
    mutable Array<BlockVector*> trueresfunc_lvls;
    mutable Array<BlockVector*> truesolupdate_lvls;

    mutable Array<LocalProblemSolver*> LocalSolvers_lvls;
    mutable CoarsestProblemSolver* CoarseSolver;

    // Allocates current level-related data and computes coarser matrices for the functional
    // and the constraint.
    // Called only during the SetUpSolver()
    virtual void SetUpFinerLvl(int lvl) const;

    virtual void Setup(bool verbose = false) const;

    virtual void ComputeTrueResFunc(const BlockVector& x_l, BlockVector& rhs_l) const;

    virtual void MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector& rhs_l) const;

    // Computes rhs in the constraint for the finer levels (~ Q_l f - Q_lminus1 f)
    // Should be called only during the first solver iterate (since it must be 0 at the next)
    void ComputeLocalRhsConstr(int level, Vector &Qlminus1_f, Vector &rhs_constr, Vector &workfvec) const;

    void ProjectFinerL2ToCoarser(int level, const Vector& in, Vector &ProjTin, Vector &out) const;

    void UpdateTrueResidual(int level, const BlockVector* rhs_l,  const BlockVector& solupd_l, BlockVector& out_l) const;

public:
    DivConstraintSolver(int NumLevels,
                           const Array< SparseMatrix*> &AE_to_e,
                           const std::vector<std::vector<HypreParMatrix*> >& Dof_TrueDof_Func_lvls,
                           const std::vector<HypreParMatrix*>& Dof_TrueDof_L2_lvls,
                           const Array< BlockMatrix*> &Proj_Func,
                           const Array< BlockOperator*>& TrueProj_Func,
                           const Array< SparseMatrix*> &Proj_L2,
                           const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                           const Array<BlockMatrix*> & FunctOp_lvls,
                           const Array<SparseMatrix*> &ConstrOp_lvls,
                           const Array<BlockVector*> & FunctRhs_lvls,
                           const Vector& ConstrRhsVec,
                           const Array<Operator*>& Smoothers_Lvls,
                           const BlockVector& Bdrdata_TrueDofs,
                           Array<LocalProblemSolver*>* LocalSolvers,
                           CoarsestProblemSolver* CoarsestSolver,
                           bool Construct_CoarseOps = true);

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const
    {
        // x will be accessed through xblock as its view
        xblock_truedofs->Update(x.GetData(), block_trueoffsets);
        // y will be accessed through yblock as its view
        yblock_truedofs->Update(y.GetData(), block_trueoffsets);

        FindParticularSolution(*xblock_truedofs, *yblock_truedofs, ConstrRhs, print_level);
    }

    // existence of this method is required by the (abstract) base class Solver
    virtual void SetOperator(const Operator &op) override{}

    void FindParticularSolution(const BlockVector& truestart_guess, BlockVector& particular_solution, const Vector& ConstrRhs, bool verbose) const;

    // have to define these to mimic useful routines from IterativeSolver class
    void SetPrintLevel(int PrintLevel) const {print_level = PrintLevel;}

};

DivConstraintSolver::DivConstraintSolver(int NumLevels,
                       const Array< SparseMatrix*> &AE_to_e,
                       const std::vector<std::vector<HypreParMatrix*> >& Dof_TrueDof_Func_lvls,
                       const std::vector<HypreParMatrix*>& Dof_TrueDof_L2_lvls,
                       const Array< BlockMatrix*> &Proj_Func,
                       const Array< BlockOperator*>& TrueProj_Func,
                       const Array< SparseMatrix*> &Proj_L2,
                       const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                       const Array<BlockMatrix*> & FunctOp_lvls,
                       const Array<SparseMatrix*> &ConstrOp_lvls,
                       const Array<BlockVector*> & FunctRhs_lvls,
                       const Vector& ConstrRhsVec,
                       const Array<Operator*>& Smoothers_Lvls,
                       const BlockVector& Bdrdata_TrueDofs,
                       Array<LocalProblemSolver*>* LocalSolvers,
                       CoarsestProblemSolver* CoarsestSolver,
                       bool Construct_CoarseOps)
     : Solver(FunctOp_lvls[0]->Height(), FunctOp_lvls[0]->Width()),
       construct_coarseops(Construct_CoarseOps),
       setup_finished(false),
       num_levels(NumLevels),
       AE_e(AE_to_e),
       dof_trueDof_Func_lvls(Dof_TrueDof_Func_lvls),
       dof_trueDof_L2_lvls(Dof_TrueDof_L2_lvls),
       comm(Dof_TrueDof_L2_lvls[0]->GetComm()),
       P_Func(Proj_Func), TrueP_Func(TrueProj_Func), P_L2(Proj_L2),
       essbdrtruedofs_Func(EssBdrTrueDofs_Func),
       numblocks(FunctOp_lvls[0]->NumColBlocks()),
       ConstrRhs(ConstrRhsVec),
       Smoothers_lvls(Smoothers_Lvls),
       bdrdata_truedofs(Bdrdata_TrueDofs)
{

    MFEM_ASSERT(FunctOp_lvls[0] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                " Funct operator at the finest level must be given anyway!");
    MFEM_ASSERT(ConstrOp_lvls[0] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                " Constraint operator at the finest level must be given anyway!");
    if (!construct_coarseops)
        for ( int l = 0; l < num_levels; ++l)
        {
            MFEM_ASSERT(FunctOp_lvls[l] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                        " functional operators at all levels must be provided "
                                                        " when construct_curls == false!");
            MFEM_ASSERT(ConstrOp_lvls[l] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                        " constraint operators at all levels must be provided "
                                                        " when construct_curls == false!");
        }

    Funct_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Funct_lvls[l] = FunctOp_lvls[l];

    Funct_rhs_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Funct_rhs_lvls[l] = FunctRhs_lvls[l];

    Constr_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Constr_lvls[l] = ConstrOp_lvls[l];

    block_trueoffsets.SetSize(numblocks + 1);
    block_trueoffsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk )
        block_trueoffsets[blk + 1] = Dof_TrueDof_Func_lvls[0][blk]->Width();
    block_trueoffsets.PartialSum();

    xblock_truedofs = new BlockVector(block_trueoffsets);
    yblock_truedofs = new BlockVector(block_trueoffsets);
    tempblock_truedofs = new BlockVector(block_trueoffsets);

    truesolupdate_lvls.SetSize(num_levels);
    truesolupdate_lvls[0] = new BlockVector(block_trueoffsets);

    trueoffsets_lvls.SetSize(num_levels);
    trueoffsets_lvls[0] = &block_trueoffsets;
    truetempvec_lvls.SetSize(num_levels);
    truetempvec_lvls[0] = new BlockVector(block_trueoffsets);
    truetempvec2_lvls.SetSize(num_levels);
    truetempvec2_lvls[0] = new BlockVector(block_trueoffsets);
    trueresfunc_lvls.SetSize(num_levels);
    trueresfunc_lvls[0] = new BlockVector(block_trueoffsets);

    if (CoarsestSolver)
        CoarseSolver = CoarsestSolver;
    else
        CoarseSolver = NULL;

    SetPrintLevel(0);

    if (LocalSolvers)
    {
        LocalSolvers_lvls.SetSize(num_levels - 1);
        for (int l = 0; l < num_levels - 1; ++l)
            LocalSolvers_lvls[l] = (*LocalSolvers)[l];
    }

    Setup();
}


// the start_guess is on dofs
// (*) returns particular solution as a vector on true dofs!
void DivConstraintSolver::FindParticularSolution(const BlockVector& truestart_guess,
                                                         BlockVector& particular_solution, const Vector &constr_rhs, bool verbose) const
{
    // 3. checking if the given initial vector satisfies the divergence constraint
    BlockVector temp_dofs(Funct_lvls[0]->RowOffsets());
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[0][blk]->Mult(truestart_guess.GetBlock(blk), temp_dofs.GetBlock(blk));
    }

    Vector temp_constr(Constr_lvls[0]->Height());
    Constr_lvls[0]->Mult(temp_dofs.GetBlock(0), temp_constr);
    temp_constr -= constr_rhs;

    // 3.1 if not, computing the particular solution
    if ( ComputeMPIVecNorm(comm, temp_constr,"", verbose) > 1.0e-14 )
    {
        if (verbose)
            std::cout << "Initial vector does not satisfy the divergence constraint. \n";
    }
    else
    {
        if (verbose)
            std::cout << "Initial vector already satisfies divergence constraint. \n";
        particular_solution = truestart_guess;
        return;
    }

    MFEM_ASSERT(CheckBdrError(truestart_guess, bdrdata_truedofs, *essbdrtruedofs_Func[0][0], true),
                              "for the initial guess");

    // variable-size vectors (initialized with the finest level sizes) on dofs
    Vector rhs_constr((Constr_lvls[0]->Height()));     // righthand side (from the divergence constraint) at level l
    Constr_lvls[0]->Mult(temp_dofs.GetBlock(0), rhs_constr);
    rhs_constr *= -1.0;
    rhs_constr += constr_rhs;

    Vector Qlminus1_f(rhs_constr.Size());     // stores P_l^T rhs_constr_l
    Vector workfvec(rhs_constr.Size());       // used only in ComputeLocalRhsConstr()

    // 0. Compute rhs in the functional for the finest level
    ComputeTrueResFunc(truestart_guess, *trueresfunc_lvls[0]);

    Qlminus1_f = rhs_constr;

    // 1. loop over levels finer than the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        // solution updates will always satisfy homogeneous essential boundary conditions
        *truesolupdate_lvls[l] = 0.0;

        ComputeLocalRhsConstr(l, Qlminus1_f, rhs_constr, workfvec);

        // solve local problems at level l
        if (LocalSolvers_lvls[l])
        {
            LocalSolvers_lvls[l]->Mult(*trueresfunc_lvls[l], *truetempvec_lvls[l], &rhs_constr);
            *truesolupdate_lvls[l] += *truetempvec_lvls[l];
        }

        UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );

        // smooth
        if (Smoothers_lvls[l])
        {
            Smoothers_lvls[l]->Mult(*truetempvec_lvls[l], *truetempvec2_lvls[l] );
            *truesolupdate_lvls[l] += *truetempvec2_lvls[l];
            UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );
        }

        *trueresfunc_lvls[l] = *truetempvec_lvls[l];

        // setting up rhs from the functional for the next (coarser) level
        TrueP_Func[l]->MultTranspose(*trueresfunc_lvls[l], *trueresfunc_lvls[l + 1]);

    } // end of loop over finer levels

    // 2. setup and solve the coarse problem
    rhs_constr = Qlminus1_f;

    // imposes boundary conditions and assembles coarsest level's
    // righthand side (from rhsfunc) on true dofs

    // 2.5 solve coarse problem
    CoarseSolver->Mult(*trueresfunc_lvls[num_levels - 1], *truesolupdate_lvls[num_levels - 1], &rhs_constr);

    // 3. assemble the final solution update
    // final sol update (at level 0)  =
    //                   = solupdate[0] + P_0 * (solupdate[1] + P_1 * ( ...) )
    for (int level = num_levels - 1; level > 0; --level)
    {
        // solupdate[level-1] = solupdate[level-1] + P[level-1] * solupdate[level]
        TrueP_Func[level - 1]->Mult(*truesolupdate_lvls[level], *truetempvec_lvls[level - 1] );
        *truesolupdate_lvls[level - 1] += *truetempvec_lvls[level - 1];

    }

    // 4. update the global iterate by the computed update (interpolated to the finest level)
    // setting temporarily tempvec[0] is actually the particular solution on dofs

    particular_solution = truestart_guess;
    particular_solution += *truesolupdate_lvls[0];

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[0][blk]->Mult(particular_solution.GetBlock(blk), temp_dofs.GetBlock(blk));
    }

    MFEM_ASSERT(CheckConstrRes(temp_dofs.GetBlock(0), *Constr_lvls[0],
                &constr_rhs, "for the particular solution"),"");

}

void DivConstraintSolver::Setup(bool verbose) const
{
    if (verbose)
        std::cout << "Starting solver setup \n";

    // 1. copying the given initial vector to the internal variable

    CheckFunctValue(comm, *Funct_lvls[0], dof_trueDof_Func_lvls[0], bdrdata_truedofs,
            "for initial vector at the beginning of solver setup: ", print_level);

    // 2. setting up the required internal data at all levels

    // 2.1 all levels except the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        //sets up the current level and prepares operators for the next one
        SetUpFinerLvl(l);

    } // end of loop over finer levels

    // in the end, part_solution is in any case a valid initial iterate
    // i.e, it satisfies the divergence contraint
    setup_finished = true;

    if (verbose)
        std::cout << "DivConstraintSolver setup completed \n";
}

void DivConstraintSolver::ComputeTrueResFunc(const BlockVector& x_l, BlockVector &rhs_l) const
{
    // FIXME: Get rid of temp1 and temp2
    BlockVector temp1(Funct_lvls[0]->ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[0][blk]->Mult(x_l.GetBlock(blk), temp1.GetBlock(blk));
    }

    BlockVector temp2(Funct_lvls[0]->RowOffsets());
    Funct_lvls[0]->Mult(temp1, temp2);

    temp2 *= -1.0;

    temp2 += *Funct_rhs_lvls[0];

    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[0][blk]->MultTranspose(temp2.GetBlock(blk), rhs_l.GetBlock(blk));
    }
}

void DivConstraintSolver::MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector &rhs_l) const
{
    // FIXME: Get rid of temp1 and temp2
    BlockVector temp1(Funct_lvls[l]->ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[l][blk]->Mult(x_l.GetBlock(blk), temp1.GetBlock(blk));
    }

    BlockVector temp2(Funct_lvls[l]->RowOffsets());
    Funct_lvls[l]->Mult(temp1, temp2);

    temp2 *= coeff;

    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[l][blk]->MultTranspose(temp2.GetBlock(blk), rhs_l.GetBlock(blk));
    }
}

// Computes prerequisites required for solving local problems at level l
// such as relation tables between AEs and internal fine-grid dofs
// and maybe smth else ... ?
void DivConstraintSolver::SetUpFinerLvl(int lvl) const
{
    // Funct_lvls[lvl] stores the Functional matrix on level lvl
    if (construct_coarseops)
    {
        BlockMatrix * Funct_PR;
        BlockMatrix * P_FuncT = Transpose(*P_Func[lvl]);
        Funct_PR = mfem::Mult(*Funct_lvls[lvl],*P_Func[lvl]);

        // checking the difference between coarsened and true
        // (from bilinear form) functional operators
        /*
        std::cout << "level = " << lvl << "\n";
        BlockMatrix * tempdiff = mfem::Mult(*P_FuncT, *Funct_PR);
        for ( int blk = 0; blk < numblocks; blk++)
        {
            std::cout << "blk = " << blk << "\n";
            SparseMatrix * tempdiffblk = new SparseMatrix(tempdiff->GetBlock(blk,blk));
            tempdiffblk->Add(-1.0,Funct_lvls[lvl + 1]->GetBlock(blk,blk));
            std::cout << tempdiffblk->MaxNorm() << "\n";
        }
        */
        Funct_lvls[lvl + 1] = mfem::Mult(*P_FuncT, *Funct_PR);

        SparseMatrix *P_L2T = Transpose(*P_L2[lvl]);
        SparseMatrix *Constr_PR;
        Constr_PR = mfem::Mult(*Constr_lvls[lvl], P_Func[lvl]->GetBlock(0,0));

        // checking the difference between coarsened and true
        // (from bilinear form) constraint operators
        /*
        SparseMatrix * tempdiffsp = mfem::Mult(*P_L2T, *Constr_PR);
        tempdiffsp->Add(-1.0, *Constr_lvls[lvl + 1]);
        std::cout << tempdiffsp->MaxNorm() << "\n";
        */

        Constr_lvls[lvl + 1] = mfem::Mult(*P_L2T, *Constr_PR);

        delete Funct_PR;
        delete Constr_PR;
        delete P_FuncT;
        delete P_L2T;
    }

    //tempvec_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //tempvec2_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //solupdate_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //rhsfunc_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());

    trueoffsets_lvls[lvl + 1] = new Array<int>(numblocks + 1);
    (*trueoffsets_lvls[lvl + 1])[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        (*trueoffsets_lvls[lvl + 1])[blk + 1] = (*trueoffsets_lvls[lvl + 1])[blk] +
                dof_trueDof_Func_lvls[lvl + 1][blk]->Width();
    }

    truetempvec_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    truetempvec2_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    truesolupdate_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    trueresfunc_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
}


// Computes out_l as an updated rhs in the functional part for the given level
//      out_l :=  rhs_l - M_l sol_l
// the same as ComputeUpdatedLvlRhsFunc but on true dofs
void DivConstraintSolver::UpdateTrueResidual(int level, const BlockVector* rhs_l,
                                                          const BlockVector& solupd_l, BlockVector& out_l) const
{
    // out_l = - M_l * solupd_l
    MultTrueFunc(level, -1.0, solupd_l, out_l);

    // out_l = rhs_l - M_l * solupd_l
    if (rhs_l)
        out_l += *rhs_l;
}

void DivConstraintSolver::ProjectFinerL2ToCoarser(int level, const Vector& in,
                                                         Vector& ProjTin, Vector &out) const
{
    const SparseMatrix * Proj = P_L2[level];

    ProjTin.SetSize(Proj->Width());
    Proj->MultTranspose(in, ProjTin);

    const SparseMatrix * AE_e_lvl = AE_e[level];
    for ( int i = 0; i < ProjTin.Size(); ++i)
        ProjTin[i] /= AE_e_lvl->RowSize(i) * 1.;

    out.SetSize(Proj->Height());
    Proj->Mult(ProjTin, out);

    // TODO: We need either to use additional memory for storing
    // result of the previous division in a temporary vector or
    // to multiply the output (ProjTin) back as in the loop below
    // in order to get correct output ProjTin in the end
    for ( int i = 0; i < ProjTin.Size(); ++i)
        ProjTin[i] *= AE_e_lvl->RowSize(i);

    return;
}

// Righthand side at level l is of the form:
//   rhs_l = (Q_l - Q_{l+1}) where Q_k is an orthogonal L2-projector: W -> W_k
// or, equivalently,
//   rhs_l = (I - Q_{l-1,l}) rhs_{l-1},
// where Q_{k,k+1} is an orthogonal L2-projector W_{k+1} -> W_k,
// and rhs_{l-1} = Q_{l-1} f (setting Q_0 = Id)
// Hence,
//   Q_{l-1,l} = P_l * inv(P_l^T P_l) * P_l^T
// where P_l columns compose the basis of the coarser space.
// (*) Uses workfvec as an intermediate buffer
// Input: Qlminus1_f
// Output: Qlminus1_f, rhs_constr
// Buffer: workfvec
// (*) All vectors are on dofs
void DivConstraintSolver::ComputeLocalRhsConstr(int level, Vector& Qlminus1_f, Vector& rhs_constr, Vector& workfvec) const
{
    // 1. rhs_constr = Q_{l-1,l} * Q_{l-1} * f = Q_l * f
    //    workfvec = P_l^T * Q_{l-1} * f
    ProjectFinerL2ToCoarser(level, Qlminus1_f, workfvec, rhs_constr);

    // 2. rhs_constr = Q_l f - Q_{l-1}f
    rhs_constr -= Qlminus1_f;

    // 3. rhs_constr (new) = - rhs_constr(old) = Q_{l-1} f - Q_l f
    rhs_constr *= -1;

    // 3. Q_{l-1} (new) = P_L2T[level] * f
    Qlminus1_f = workfvec;

    return;
}

// TODO: Implement a separate class for Hcurl smoother at level l in the block form, probably abstract
class HcurlGSSSmoother : public BlockOperator
{
private:
    int numblocks;

    // number of GS sweeps for each block
    mutable Array<int> sweeps_num;

    int print_level;

protected:
    const BlockMatrix& Funct_mat;

    // discrete curl operators at all levels;
    const SparseMatrix* Curlh;

#ifdef MEMORY_OPTIMIZED
    mutable Vector* temp_Hdiv_dofs;
    mutable Vector* temp_Hcurl_dofs;
#else
    // global discrete curl operator
    mutable HypreParMatrix* Curlh_global;
#endif

    // Projection of the system matrix onto discrete Hcurl space
    // Curl_hT * A_l * Curlh matrices at all levels
    mutable SparseMatrix* CTMC;

    // global CTMC as HypreParMatrices at all levels;
    mutable HypreParMatrix* CTMC_global;

    // structures used when all dofs are relaxed (via HypreSmoothers):
    mutable Array<Operator*> Smoothers;
    mutable Array<int> trueblock_offsets; //block offsets for H(curl) x other blocks problems on true dofs
    mutable BlockVector* truerhs;  // rhs for H(curl) x other blocks problems on true dofs
    mutable BlockVector* truex;    // sol for H(curl) x other blocks problems on true dofs

    // Dof_TrueDof tables for Hcurl at all levels
    const HypreParMatrix & d_td_Hcurl;

    // Dof_TrueDof tables for Hdiv at all levels
    const std::vector<HypreParMatrix*> & d_td_Funct_blocks;

    // List of essential boundary dofs for Hcurl
    const Array<int> & essbdrdofs_Hcurl;
    // FIXME: Move it to a const constructor argument,
    // but take care about the fact that GetTrueVDofs is different from GetVDofs()
    const Array<int> & essbdrtruedofs_Hcurl;

    mutable Array<int> block_offsets;
    mutable BlockVector * xblock;
    mutable BlockVector * yblock;

public:
    // constructor
    HcurlGSSSmoother (const BlockMatrix& Funct_Mat,
                     const SparseMatrix& Discrete_Curl,
                     const HypreParMatrix& Dof_TrueDof_Hcurl,
                     const std::vector<HypreParMatrix*> & Dof_TrueDof_Funct,
                     const Array<int>& EssBdrdofs_Hcurl, const Array<int> &EssBdrtruedofs_Hcurl,
                     const Array<int> * SweepsNum, const Array<int> &Block_Offsets);

    // FIXME: Implement this
    //void PrintAllOptions() const;

    virtual void Setup() const;

    // Operator application
    virtual void Mult (const Vector & x, Vector & y) const;

    // Action of the transpose operator
    virtual void MultTranspose (const Vector & x, Vector & y) const {Mult(x,y);}

    // service routines
    int GetSweepsNumber(int block) const {return sweeps_num[block];}

};

HcurlGSSSmoother::HcurlGSSSmoother (const BlockMatrix& Funct_Mat,
                                    const SparseMatrix& Discrete_Curl,
                                    const HypreParMatrix& Dof_TrueDof_Hcurl,
                                    const std::vector<HypreParMatrix *> &Dof_TrueDof_Funct,
                                    const Array<int>& EssBdrdofs_Hcurl,
                                    const Array<int>& EssBdrtruedofs_Hcurl,
                                    const Array<int> * SweepsNum,
                                    const Array<int>& Block_Offsets)
    : BlockOperator(Block_Offsets),
      numblocks(Funct_Mat.NumRowBlocks()),
      print_level(0),
      Funct_mat(Funct_Mat),
      d_td_Hcurl(Dof_TrueDof_Hcurl),
      d_td_Funct_blocks(Dof_TrueDof_Funct),
      essbdrdofs_Hcurl(EssBdrdofs_Hcurl),
      essbdrtruedofs_Hcurl(EssBdrtruedofs_Hcurl)
{
    Curlh = &Discrete_Curl;

    block_offsets.SetSize(numblocks + 1);
    for ( int i = 0; i < numblocks + 1; ++i)
        block_offsets[i] = Block_Offsets[i];

    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    trueblock_offsets.SetSize(numblocks + 1);
    trueblock_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            trueblock_offsets[1] = Dof_TrueDof_Hcurl.Width();
        else
            trueblock_offsets[blk + 1] = Dof_TrueDof_Funct[blk]->Width();
    }
    trueblock_offsets.PartialSum();

    Smoothers.SetSize(numblocks);

    sweeps_num.SetSize(numblocks);
    if (SweepsNum)
        for ( int blk = 0; blk < numblocks; ++blk)
            sweeps_num[blk] = (*SweepsNum)[blk];
    else
        sweeps_num = 1;

    Setup();
}

void HcurlGSSSmoother::Mult(const Vector & x, Vector & y) const
{
    if (print_level)
        std::cout << "Smoothing with HcurlGSS smoother \n";

    if (x.GetData() == y.GetData())
        mfem_error("Error in HcurlGSSSmoother::Mult(): x and y can't point to the same data \n");

    // x will be accessed through xblock as its view
    xblock->Update(x.GetData(), block_offsets);
    // y will be accessed through yblock as its view
    yblock->Update(y.GetData(), block_offsets);

    for (int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
        {
#ifdef MEMORY_OPTIMIZED
            SparseMatrix d_td_Hdiv_diag;
            d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
            d_td_Hdiv_diag.Mult(xblock->GetBlock(0), *temp_Hdiv_dofs);

            // rhs_l = CT_l * res_lvl
            Curlh->MultTranspose(*temp_Hdiv_dofs, *temp_Hcurl_dofs);

            d_td_Hcurl.MultTranspose(*temp_Hcurl_dofs, truerhs->GetBlock(0));
#else
            Curlh_global->MultTranspose(xblock->GetBlock(0), truerhs->GetBlock(0));
#endif
        }
        else
        {
            truerhs->GetBlock(blk) = xblock->GetBlock(blk);
        }

    }


    // imposing boundary conditions in Hcurl on the righthand side
    for ( int tdofind = 0; tdofind < essbdrtruedofs_Hcurl.Size(); ++tdofind)
    {
        truerhs->GetBlock(0)[essbdrtruedofs_Hcurl[tdofind]] = 0.0;
    }


    /*
    for ( int tdof = 0; tdof < temp->Size(); ++tdof)
    {
        if ( (*temp)[tdof] != 0)
        {
            truerhs->GetBlock(0)[tdof] = 0.0;
        }
    }
    */

    *truex = 0.0;
    //Operator * id = new IdentityOperator(Smoothers_lvls[level]->Height());
    //id->Mult(*truerhs_lvls[level], *truex_lvls[level]);
    //CTMC_global_lvls[level]->Mult(*truerhs_lvls[level], *truex_lvls[level]);
    //Smoothers_lvls[level]->Mult(*truerhs_lvls[level], *truex_lvls[level]);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Smoothers[blk]->Mult(truerhs->GetBlock(blk), truex->GetBlock(blk));
    }

    // computing the solution update in the H(div) x other blocks space
    // in two steps:

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0) // first component should be transferred from Hcurl to Hdiv
        {
#ifdef MEMORY_OPTIMIZED
            d_td_Hcurl.Mult(truex->GetBlock(0), *temp_Hcurl_dofs);
            Curlh->Mult(*temp_Hcurl_dofs, *temp_Hdiv_dofs);

            SparseMatrix d_td_Hdiv_diag;
            d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
            d_td_Hdiv_diag.MultTranspose(*temp_Hdiv_dofs, yblock->GetBlock(0));
#else
            Curlh_global->Mult(truex->GetBlock(0), yblock->GetBlock(0));
#endif
        }
        else
            yblock->GetBlock(blk) += truex->GetBlock(blk);
    }


}


void HcurlGSSSmoother::Setup() const
{
    MFEM_ASSERT(numblocks <= 2, "HcurlGSSSmoother::Setup was implemented for the "
                                "cases numblocks = 1 or 2 only \n");

    // shortcuts
    SparseMatrix *CurlhT = Transpose(*Curlh);
    const SparseMatrix * M = &(Funct_mat.GetBlock(0,0));

    HypreParMatrix * d_td_Hcurl_T = d_td_Hcurl.Transpose();

    // form CT*M*C as a SparseMatrix
    SparseMatrix *M_Curlh = mfem::Mult(*M, *Curlh);
    CTMC = mfem::Mult(*CurlhT, *M_Curlh);

    delete M_Curlh;
    delete CurlhT;

    // imposing essential boundary conditions
    for ( int dof = 0; dof < essbdrdofs_Hcurl.Size(); ++dof)
    {
        if ( essbdrdofs_Hcurl[dof] != 0)
        {
            CTMC->EliminateRowCol(dof);
        }
    }

    // form CT*M*C as HypreParMatrices
    HypreParMatrix* CTMC_d_td;
    CTMC_d_td = d_td_Hcurl.LeftDiagMult( *CTMC );

    CTMC_global = ParMult(d_td_Hcurl_T, CTMC_d_td);
    CTMC_global->CopyRowStarts();
    CTMC_global->CopyColStarts();

#ifdef MEMORY_OPTIMIZED
    temp_Hdiv_dofs = new Vector(Curlh->Height());
    temp_Hcurl_dofs = new Vector(Curlh->Width());
#else
    // FIXME : RowStarts or ColStarts()?
    HypreParMatrix* C_d_td = d_td_Hcurl.LeftDiagMult(*Curlh, d_td_Funct_blocks[0]->GetRowStarts() );
    SparseMatrix d_td_Hdiv_diag;
    d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
    Curlh_global = C_d_td->LeftDiagMult(*Transpose(d_td_Hdiv_diag), d_td_Funct_blocks[0]->GetColStarts() );
    Curlh_global->CopyRowStarts();
    Curlh_global->CopyColStarts();
    delete C_d_td;
#endif

    Smoothers[0] = new HypreSmoother(*CTMC_global, HypreSmoother::Type::l1GS, sweeps_num[0]);
    if (numblocks > 1)
    {
        int blk = 1;
        const SparseMatrix * Funct_blk = &(Funct_mat.GetBlock(blk,blk));
        HypreParMatrix* Functblk_d_td_blk = d_td_Funct_blocks[blk]->LeftDiagMult(*Funct_blk, d_td_Funct_blocks[blk]->GetRowStarts() );
        HypreParMatrix * d_td_blk_T = d_td_Funct_blocks[blk]->Transpose();
        HypreParMatrix * Functblk_global = ParMult(d_td_blk_T, Functblk_d_td_blk);
        Functblk_global->CopyRowStarts();
        Functblk_global->CopyColStarts();
        delete Functblk_d_td_blk;

        Smoothers[1] = new HypreBoomerAMG(*Functblk_global);
        delete Functblk_global;
    }

    truex = new BlockVector(trueblock_offsets);
    truerhs = new BlockVector(trueblock_offsets);

    // allocating memory for local-to-level vector arrays
    //rhs_lvls[level] = new Vector(Curlh_lvls[level]->Width());
    //tempvec_lvls[level] = new Vector(Curlh_lvls[level]->Width());

    delete CTMC_d_td;
    delete d_td_Hcurl_T;
}

#if 0
class MultilevelHCurlGSSmoother : public MultilevelSmoother
{
private:
    // number of GS sweeps
    int sweeps_num;

    // if true, coarser curl operators will be constructed from Curlh_lvls[0]
    // else, the entire hierarchy of curl operators must be provided in
    // the constructor
    bool construct_curls;

    // if true, HypreSmoother's are constructed and used thereafter for GS
    // relaxation
    // else, some new code will be used (but was not implemented)
    bool relax_all_dofs;
protected:
    const Array<BlockMatrix*> & Funct_mat_lvls;

    // discrete curl operators at all levels;
    mutable Array<SparseMatrix*> Curlh_lvls;

#ifdef MEMORY_OPTIMIZED
    mutable Array<Vector*> temp_Hdiv_dofs_lvls;
    mutable Array<Vector*> temp_Hcurl_dofs_lvls;
#else
    // global discrete curl operators at all levels;
    mutable Array<HypreParMatrix*> Curlh_global_lvls;
#endif

    // Projection of the system matrix onto discrete Hcurl space
    // Curl_hT * A_l * Curlh matrices at all levels
    mutable Array<SparseMatrix*> CTMC_lvls;

    // global CTMC as HypreParMatrices at all levels;
    mutable Array<HypreParMatrix*> CTMC_global_lvls;

    // structures used when all dofs are relaxed (via HypreSmoothers):
    // used when relax_all_dofs = true
    mutable Array<HypreSmoother*> Smoothers_lvls;
    mutable Array<Vector*> truerhs_lvls;  // rhs for H(curl) problems on true dofs
    mutable Array<Vector*> truex_lvls;    // sol for H(curl) problems on true dofs

    // structures to be used when not all dofs are relaxed:
    // stores additionally diagonal entries of global CTMC matrices
    mutable Array<Vector*> CTMC_global_diag_lvls;
    // global CT*M operators at all levels;
    mutable Array<HypreParMatrix*> CTM_global_lvls;

    mutable Array<Vector*> truevec_lvls;  // lives in Hdiv_h on true dofs
    mutable Array<Vector*> truevec2_lvls;
    mutable Array<Vector*> truevec3_lvls; // lives in Hcurl_h on true dofs

    // Dof_TrueDof tables for Hcurl at all levels
    const Array<HypreParMatrix*> & d_td_Hcurl_lvls;

    // Dof_TrueDof tables for Hdiv at all levels
    const Array<HypreParMatrix*> & d_td_Hdiv_lvls;

    // Lists of essential boundary dofs for Hcurl at all levels
    const std::vector<Array<int>* >  & essbdrdofs_lvls;
    // FIXME: Move it to a const constructor argument,
    // but take care about the fact that GetTrueVDofs is different from GetVDofs()
    mutable std::vector<Array<int>* >  essbdrtruedofs_lvls;

public:
    // constructor
    MultilevelHCurlGSSmoother (int Num_Levels,
                     const Array< BlockMatrix*> & Funct_Mat_lvls,
                     const Array< SparseMatrix*> & Discrete_Curls_lvls,
                     //const Array< SparseMatrix*>& Proj_lvls,
                     const Array<HypreParMatrix *>& Dof_TrueDof_Hcurl_lvls,
                     const Array<HypreParMatrix *>& Dof_TrueDof_Hdiv_lvls,
                     const std::vector<Array<int>* > & EssBdrdofs_lvls,
                     int SweepsNum = 1, bool Construct_Curls = false, bool Relax_All_Dofs = true);

    void SetUpSmoother(int level, const BlockMatrix& SysMat_lvl) const override;

    // Computes the righthand side for the local minimization problem
    // solved in MultLevel() from the given residual at level l of the
    // original problem
    //void ComputeRhsLevel(int level, const BlockVector& res_lvl) override;
    void ComputeTrueRhsLevel(int level, const BlockVector& res_lvl) override;

    // Updates the given iterate at level l by solving a minimization
    // problem in H(curl) at level l (using the precomputed righthand side)
    //void MultLevel(int level, Vector& in_lvl, Vector& out_lvl) override;
    void MultTrueLevel(int level, Vector& in_lvl, Vector& out_lvl) override;

    // service routines
    bool WillConstructCurls() const {return construct_curls;}
    bool WillRelaxAllDofs() const {return relax_all_dofs;}
    int GetSweepsNumber() const {return sweeps_num;}
    void SetSweepsNumber(int Number_of_sweeps)
    {
        sweeps_num = Number_of_sweeps;
        for ( int l = 0; l < num_levels; ++l)
            Smoothers_lvls[l]->SetType(HypreSmoother::Type::l1GS, sweeps_num);
    }
    void SetDofsToRelax(bool Relax_all_dofs) {relax_all_dofs = Relax_all_dofs;}

    void PrintAllOptions() const override;

    void Setup() const override;
};

void MultilevelHCurlGSSmoother::Setup() const
{
    for (int l = 0; l < num_levels; ++l)
    {
        SetUpSmoother(l, *Funct_mat_lvls[l]);
    }
}


void MultilevelHCurlGSSmoother::PrintAllOptions() const
{
    MultilevelSmoother::PrintAllOptions();
    std::cout << "HcurlGSS smoother options: \n";
    std::cout << "construct_curls: " << construct_curls << "\n";
    std::cout << "relax_all_dofs:" << relax_all_dofs << "\n";
    std::cout << "sweeps_num: " << sweeps_num << "\n";
    std::cout << "\n";
}


MultilevelHCurlGSSmoother::MultilevelHCurlGSSmoother (int Num_Levels,
                                  const Array< BlockMatrix*> & Funct_Mat_lvls,
                                  const Array< SparseMatrix*> & Discrete_Curls_lvls,
                                  const Array<HypreParMatrix*>& Dof_TrueDof_Hcurl_lvls,
                                  const Array<HypreParMatrix*>& Dof_TrueDof_Hdiv_lvls,
                                  const std::vector<Array<int>* > & EssBdrdofs_lvls,
                                  int SweepsNum, bool Construct_Curls, bool Relax_All_Dofs)
    : MultilevelSmoother(Num_Levels),
      sweeps_num(SweepsNum),
      construct_curls(Construct_Curls),
      relax_all_dofs(Relax_All_Dofs),
      Funct_mat_lvls(Funct_Mat_lvls),
      d_td_Hcurl_lvls(Dof_TrueDof_Hcurl_lvls),
      d_td_Hdiv_lvls(Dof_TrueDof_Hdiv_lvls),
      essbdrdofs_lvls(EssBdrdofs_lvls)
{
    std::cout << "Calling constructor of the MultilevelHCurlGSSmoother \n";
    MFEM_ASSERT(Discrete_Curls_lvls[0] != NULL, "MultilevelHCurlGSSmoother::MultilevelHCurlGSSmoother()"
                                                " Curl operator at the finest level must be given anyway!");
    MFEM_ASSERT(!construct_curls, "Construction of discrete curls using projectors is not possible for now,"
                                  "canonical projectors are required!" );
    if (!construct_curls)
        for ( int l = 0; l < num_levels; ++l)
            MFEM_ASSERT(Discrete_Curls_lvls[l] != NULL, "MultilevelHCurlGSSmoother::MultilevelHCurlGSSmoother()"
                                                        " curl operators at all levels must be provided "
                                                        " when construct_curls == false!");
    MFEM_ASSERT(relax_all_dofs, "Case relax-all_dofs = false is not implemented!");

    Curlh_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        Curlh_lvls[l] = Discrete_Curls_lvls[l];

    if (relax_all_dofs)
    {
        Smoothers_lvls.SetSize(num_levels);
        for ( int l = 0; l < num_levels; ++l)
            Smoothers_lvls[l] = NULL;

        truex_lvls.SetSize(num_levels);
    }
    else // relax_all_dofs = false
    {
        CTM_global_lvls.SetSize(num_levels);
        for ( int l = 0; l < num_levels; ++l)
            CTM_global_lvls[l] = NULL;

        CTMC_global_diag_lvls.SetSize(num_levels);
        for ( int l = 0; l < num_levels; ++l)
            CTMC_global_diag_lvls[l] = NULL;

    }

#ifdef MEMORY_OPTIMIZED
    temp_Hdiv_dofs_lvls.SetSize(num_levels);
    temp_Hcurl_dofs_lvls.SetSize(num_levels);
#else
    Curlh_global_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        Curlh_global_lvls[l] = NULL;
#endif

    CTMC_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        CTMC_lvls[l] = NULL;

    CTMC_global_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        CTMC_global_lvls[l] = NULL;

    truerhs_lvls.SetSize(num_levels);
    essbdrtruedofs_lvls.resize(num_levels);

    truevec_lvls.SetSize(num_levels);
    truevec2_lvls.SetSize(num_levels);
    truevec3_lvls.SetSize(num_levels);

    Setup();
}

void MultilevelHCurlGSSmoother::SetUpSmoother(int level, const BlockMatrix& SysMat_blk) const
{
    if ( !finalized_lvls[level] ) // if level was not set up before
    {
        MFEM_ASSERT(Curlh_lvls[level], "MultilevelHCurlGSSmoother::SetUpSmoother():"
                                       " curl operator must have been set already at this level \n");
        // shortcuts
        SparseMatrix *Curlh = Curlh_lvls[level];
        SparseMatrix *CurlhT = Transpose(*Curlh);
        const SparseMatrix * SysMat_lvl = &(SysMat_blk.GetBlock(0,0));
        //std::cout << "SysMat_lvl, lvl = " << level << "--------------------------------------\n";
        //SysMat_lvl->Print();
        //ofstream ofs("newsolver_out.txt");
        //res_constr.Print(ofs,1);
        Array<int> * essbdr = essbdrdofs_lvls[level];

        HypreParMatrix * d_td = d_td_Hcurl_lvls[level];
        d_td->SetOwnerFlags(3,3,1);
        HypreParMatrix * d_td_T = d_td->Transpose();

        if (!relax_all_dofs)
        {
            MFEM_ABORT("Not implemented yet");
        }

        // form CT*M*C as a SparseMatrix
        SparseMatrix *SysMat_Curlh = mfem::Mult(*SysMat_lvl, *Curlh_lvls[level]);
        CTMC_lvls[level] = mfem::Mult(*CurlhT, *SysMat_Curlh);

        delete SysMat_Curlh;
        delete CurlhT;

        // imposing essential boundary conditions
        for ( int dof = 0; dof < essbdr->Size(); ++dof)
        {
            if ( (*essbdr)[dof] != 0)
            {
                CTMC_lvls[level]->EliminateRowCol(dof);
            }
        }

        // form CT*M*C as HypreParMatrices
        HypreParMatrix* CTMC_d_td;
        CTMC_d_td = d_td_Hcurl_lvls[level]->LeftDiagMult( *CTMC_lvls[level] );

        CTMC_global_lvls[level] = ParMult(d_td_T, CTMC_d_td);

        CTMC_global_lvls[level]->CopyRowStarts();
        CTMC_global_lvls[level]->CopyColStarts();

#ifdef MEMORY_OPTIMIZED
        temp_Hdiv_dofs_lvls[level] = new Vector(Curlh_lvls[level]->Height());
        temp_Hcurl_dofs_lvls[level] = new Vector(Curlh_lvls[level]->Width());
#else
        // FIXME : RowStarts or ColStarts()?
        HypreParMatrix* C_d_td = d_td->LeftDiagMult(*Curlh, d_td_Hdiv_lvls[level]->GetRowStarts() );
        SparseMatrix d_td_Hdiv_diag;
        d_td_Hdiv_lvls[level]->GetDiag(d_td_Hdiv_diag);
        Curlh_global_lvls[level] = C_d_td->LeftDiagMult(*Transpose(d_td_Hdiv_diag), d_td_Hdiv_lvls[level]->GetColStarts() );
        Curlh_global_lvls[level]->CopyRowStarts();
        Curlh_global_lvls[level]->CopyColStarts();
        delete C_d_td;
#endif

        if (relax_all_dofs)
        {
            Smoothers_lvls[level] = new HypreSmoother(*CTMC_global_lvls[level],
                                                      HypreSmoother::Type::l1GS, sweeps_num);

            truex_lvls[level] = new Vector(CTMC_global_lvls[level]->Height());
        }
        else
        {
            MFEM_ABORT("Case relax_all_dofs = false was not tested \n");
            CTMC_global_diag_lvls[level] = new Vector();
            CTMC_global_lvls[level]->GetDiag(*CTMC_global_diag_lvls[level]);

            truevec_lvls[level] = new Vector(CTMC_global_lvls[level]->Height());
            truevec2_lvls[level] = new Vector(CTMC_global_lvls[level]->Height());
            truevec3_lvls[level] = new Vector(CTMC_global_lvls[level]->Width());
        }

        truerhs_lvls[level] = new Vector(d_td_Hcurl_lvls[level]->Width());
        // creating essbdrtruedofs list at level l
        essbdrtruedofs_lvls[level] = new Array<int>(d_td_T->Height());
        *essbdrtruedofs_lvls[level] = 0.0;
        d_td_T->BooleanMult(1.0, essbdrdofs_lvls[level]->GetData(),
                            0.0, essbdrtruedofs_lvls[level]->GetData());

        // allocating memory for local-to-level vector arrays
        //rhs_lvls[level] = new Vector(Curlh_lvls[level]->Width());
        //tempvec_lvls[level] = new Vector(Curlh_lvls[level]->Width());

        delete CTMC_d_td;
        delete d_td_T;

        finalized_lvls[level] = true;

    }// end of if level wasn't finelized already before the call
}

/*
// Computes the residual for the "smoother equation"
// from the given residual of the basic minimization process:
//      rhs_l = CT_l * res_l
void MultilevelHCurlGSSmoother::ComputeRhsLevel(int level, const BlockVector& res_lvl)
{
    // rhs_l = CT_l * res_lvl
    Curlh_lvls[level]->MultTranspose(res_lvl.GetBlock(0), *rhs_lvls[level]);
}
*/

// (*) same as ComputeRhsLevel but on true dofs
// Computes the residual for the "smoother equation"
// from the given residual of the basic minimization process:
//      rhs_l = CT_l * res_l
void MultilevelHCurlGSSmoother::ComputeTrueRhsLevel(int level, const BlockVector& res_lvl)
{
#ifdef MEMORY_OPTIMIZED
    SparseMatrix d_td_Hdiv_diag;
    d_td_Hdiv_lvls[level]->GetDiag(d_td_Hdiv_diag);
    d_td_Hdiv_diag.Mult(res_lvl.GetBlock(0), *temp_Hdiv_dofs_lvls[level]);

    // rhs_l = CT_l * res_lvl
    Curlh_lvls[level]->MultTranspose(*temp_Hdiv_dofs_lvls[level], *temp_Hcurl_dofs_lvls[level]);

    d_td_Hcurl_lvls[level]->MultTranspose(*temp_Hcurl_dofs_lvls[level], *truerhs_lvls[level]);
#else
    Curlh_global_lvls[level]->MultTranspose(res_lvl, *truerhs_lvls[level]);
#endif
}

/*
// Solves the minimization problem in the div-free subspace
// Takes the current iterate in_lvl
// and returns the updated iterate
//      out_lvl = in_lvl + Curl_l * sol_l (all assembled on dofs)
// where sol_l is obtained by a several GS sweeps
// for the system
//      CurlT_l M Curl_l sol_l = rhs_l
// and rhs_l is computed using the residual of the original problem
// during the call to SetUpRhs() before MultLevel
void MultilevelHCurlGSSmoother::MultLevel(int level, Vector& in_lvl, Vector& out_lvl)
{
    MFEM_ASSERT(finalized_lvls[level] == true,
                "MultLevel() must not be called for a non-finalized level");

    if (print_level)
        std::cout << "Smoothing with GSS smoother at level " << level << "\n";

    if (relax_all_dofs)
    {
        // imposing boundary conditions on the righthand side
        Array<int> * temp = essbdrdofs_lvls[level];
        for ( int dof = 0; dof < temp->Size(); ++dof)
        {
            if ( (*temp)[dof] != 0)
            {
                (*rhs_lvls[level])[dof] = 0.0;
            }
        }

        // assemble righthand side on the true dofs
        d_td_Hcurl_lvls[level]->MultTranspose(*rhs_lvls[level], *truerhs_lvls[level]);

        *truex_lvls[level] = 0.0;
        Smoothers_lvls[level]->Mult(*truerhs_lvls[level], *truex_lvls[level]);

        // distributing the solution from true dofs to dofs
        // temp_l = truex_l, but on dofs
        d_td_Hcurl_lvls[level]->Mult(*truex_lvls[level], *tempvec_lvls[level]);

        // computing the solution update in the H(div)_h space
        // in two steps:


        if (out_lvl.GetData() == in_lvl.GetData())
        {
            mfem_error("Error: out_lvl and in_lvl can't point to the same datas \n");
        }
        else
        {
            // 1. out = Curlh_l * temp_l = Curlh_l * x_l
            Curlh_lvls[level]->Mult( *tempvec_lvls[level], out_lvl);
            // 2. out_lvl = in_lvl + Curlh_l * x_l
            out_lvl += in_lvl;
        }

    }
    else
    {
        MFEM_ABORT ("MultilevelHCurlGSSmoother::MultLevel(): This case was not implemented!");
    }

}
*/

// (*) same as MultLevel but on true dofs
// Solves the minimization problem in the div-free subspace
// Takes the current iterate in_lvl
// and returns the updated iterate
//      out_lvl = in_lvl + Curl_l * sol_l (all assembled on dofs)
// where sol_l is obtained by a several GS sweeps
// for the system
//      CurlT_l M Curl_l sol_l = rhs_l
// and rhs_l is computed using the residual of the original problem
// during the call to SetUpRhs() before MultLevel
void MultilevelHCurlGSSmoother::MultTrueLevel(int level, Vector& in_lvl, Vector& out_lvl)
{
    MFEM_ASSERT(finalized_lvls[level] == true,
                "MultLevel() must not be called for a non-finalized level");

    if (print_level)
        std::cout << "Smoothing with GSS smoother at level " << level << "\n";

    if (relax_all_dofs)
    {
        // imposing boundary conditions on the righthand side
        Array<int> * temp = essbdrtruedofs_lvls[level];

        for ( int tdof = 0; tdof < temp->Size(); ++tdof)
        {
            if ( (*temp)[tdof] != 0)
            {
                (*truerhs_lvls[level])[tdof] = 0.0;
            }
        }

        *truex_lvls[level] = 0.0;
        //Operator * id = new IdentityOperator(Smoothers_lvls[level]->Height());
        //id->Mult(*truerhs_lvls[level], *truex_lvls[level]);
        //CTMC_global_lvls[level]->Mult(*truerhs_lvls[level], *truex_lvls[level]);
        Smoothers_lvls[level]->Mult(*truerhs_lvls[level], *truex_lvls[level]);

        // computing the solution update in the H(div)_h space
        // in two steps:

        if (out_lvl.GetData() == in_lvl.GetData())
        {
            mfem_error("Error: out_lvl and in_lvl can't point to the same datas \n");
        }
        else
        {
#ifdef MEMORY_OPTIMIZED
            d_td_Hcurl_lvls[level]->Mult(*truex_lvls[level], *temp_Hcurl_dofs_lvls[level]);

            // rhs_l = CT_l * res_lvl
            Curlh_lvls[level]->Mult(*temp_Hcurl_dofs_lvls[level], *temp_Hdiv_dofs_lvls[level]);

            SparseMatrix d_td_Hdiv_diag;
            d_td_Hdiv_lvls[level]->GetDiag(d_td_Hdiv_diag);
            d_td_Hdiv_diag.MultTranspose(*temp_Hdiv_dofs_lvls[level], out_lvl);
#else
            Curlh_global_lvls[level]->Mult(*truex_lvls[level], out_lvl);
#endif

            out_lvl += in_lvl;

        }

    }
    else
    {
        MFEM_ABORT ("MultilevelHCurlGSSmoother::MultTrueLevel(): This case was not implemented!");
    }

}
#endif

#if 0

// Implements a multilevelel smoother which can update the solution x = (x_l)
// at each level l by solving a minimization problem
//      J ( x_l + Curl_l * z_l) -> min over z_l
// where z_l is from discrete Hcurl space.
// The output of one action of the smoother is
//      y_l = x_l + Curl_l * z_l
// The functional J(x_l) at level l is defined as
//      J(x_l) = (M_l x_l, x_l)
// where M_l is a matrix provided as an external argument during the call to SetUpSmoother()
class HCurlSmoother : public MultilevelSmoother
{
    using MultilevelSmoother::SetUpSmoother;
protected:

    mutable double abs_tol;
    mutable double rel_tol;
    mutable int max_iter_internal;

    // Projection matrices for Hcurl at all levels
    const Array< SparseMatrix*>& P_lvls;

    // Discrete curl operators at all levels;
    mutable Array<SparseMatrix*> Curlh_lvls;

    // Curl_hT * A_l * Curlh matrices at all levels
    mutable Array<SparseMatrix*> CTMC_lvls;

    // Projection of the system matrix onto discrete Hcurl space
    // stored as HypreParMatrices at all levels;
    mutable Array<HypreParMatrix*> CTMC_global_lvls;

    mutable Array<Solver*> prec_global_lvls;

    // dof_Truedof tables at all levels;
    const Array<HypreParMatrix*> & d_td_Hcurl_lvls;

    const std::vector<Array<int>* >  & essbdrdofs_lvls;

    // temporary storage variables
    mutable Array<Vector*> rhs_lvls;      // rhs for the problems in H(curl)
    mutable Array<Vector*> tempvec_lvls;  // lives in H(curl)_h
    mutable Array<Vector*> tempvec2_lvls; // lives in H(div)_h
    mutable Array<Vector*> truerhs_lvls;  // rhs for H(curl) problems on true dofs
    mutable Array<Vector*> truex_lvls;    // sol for H(curl) problems on true dofs

public:
    // constructor
    HCurlSmoother (int Num_Levels, SparseMatrix *DiscreteCurl,
                   const Array< SparseMatrix*>& Proj_lvls, const Array<HypreParMatrix *>& Dof_TrueDof_lvls,
                   const std::vector<Array<int>* > & EssBdrdofs_lvls);

    // SparseMatrix version of SetUpSmoother()
    void SetUpSmoother(int level, const SparseMatrix& SysMat_lvl,
                       const SparseMatrix* Proj_lvl = NULL, const HypreParMatrix* D_tD_lvl = NULL) override;

    // BlockMatrix version of SetUpSmoother()
    void SetUpSmoother(int level, const BlockMatrix& SysMat_lvl,
                       const BlockMatrix* Proj_lvl = NULL, const std::vector<HypreParMatrix*>* D_tD_lvl = NULL) override;

    // Computes the righthand side for the local minimization problem
    // solved in MultLevel() from the given residual at level l of the
    // original problem
    //void ComputeRhsLevel(int level, const BlockVector& res_lvl) override;

    // Updates the given iterate at level l by solving a minimization
    // problem in H(curl) at level l (using the precomputed righthand side)
    //void MultLevel(int level, Vector& in_lvl, Vector& out_lvl) override;

    void SetAbsTol(double AbsTol) const {abs_tol = AbsTol;}
    void SetRelTol(double RelTol) const {rel_tol = RelTol;}
    void SetMaxIterInt(double MaxIterInt) const {max_iter_internal = MaxIterInt;}
};

HCurlSmoother::HCurlSmoother (int Num_Levels, SparseMatrix* DiscreteCurl,
                              const Array< SparseMatrix*>& Proj_lvls,
                              const Array<HypreParMatrix*>& Dof_TrueDof_lvls,
                              const std::vector<Array<int>* > & EssBdrdofs_lvls) :
    MultilevelSmoother(Num_Levels),
    abs_tol(1.0e-12), rel_tol(1.0e-12), max_iter_internal(20000),
    P_lvls(Proj_lvls), d_td_Hcurl_lvls(Dof_TrueDof_lvls), essbdrdofs_lvls(EssBdrdofs_lvls)
{
    std::cout << "Calling constructor of the HCurlSmoother \n";
    Curlh_lvls.SetSize(num_levels);
    Curlh_lvls[0] = DiscreteCurl;
    CTMC_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        CTMC_lvls[l] = NULL;
    CTMC_global_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        CTMC_global_lvls[l] = NULL;
    prec_global_lvls.SetSize(num_levels);
    for ( int l = 0; l < num_levels; ++l)
        prec_global_lvls[l] = NULL;
    rhs_lvls.SetSize(num_levels);
    tempvec2_lvls.SetSize(num_levels);
    tempvec_lvls.SetSize(num_levels);
    truerhs_lvls.SetSize(num_levels);
    truex_lvls.SetSize(num_levels);
}

void HCurlSmoother::SetUpSmoother(int level, const BlockMatrix& SysMat_lvl,
                                  const BlockMatrix* Proj_lvl, const std::vector<HypreParMatrix *> *D_tD_lvl)
{
    MFEM_ABORT("HcurlSmoother: BlockMatrix arguments are not supported\n");
}

void HCurlSmoother::SetUpSmoother(int level, const SparseMatrix& SysMat_lvl,
                                  const SparseMatrix* Proj_lvl, const HypreParMatrix* D_tD_lvl)
{
    if ( !finalized_lvls[level] ) // if level was not set up before
    {
        // for level 0 the curl SparseMatrix is already known after the constructor has been called
        // otherwise one needs to compute it from the previous level
        if (level != 0)
        {
            MFEM_ASSERT(Proj_lvl != NULL, "For finer level a projection matrix should be given"
                                          "to construct the coarser operator!\n");
            // computing Curlh as SparseMatrix for the current level using the previous one
            // Curlh[level] = PT * Curlh[level] P
            // FIXME: Can one avoid allocation of projector transpose and intermediate matrix product?
            SparseMatrix *P_T = Transpose(*Proj_lvl);
            SparseMatrix *Curlh_P;
            Curlh_P = mfem::Mult(*Curlh_lvls[level - 1], *P_lvls[level - 1]);
            Curlh_lvls[level] = mfem::Mult(*P_T, *Curlh_P);

            delete P_T;
            delete Curlh_P;
        }

        // form CT*M*C as SparseMatrices
        SparseMatrix *CurlhT = Transpose( *Curlh_lvls[level]);
        SparseMatrix *SysMat_Curlh = mfem::Mult(SysMat_lvl, *Curlh_lvls[level]);
        CTMC_lvls[level] = mfem::Mult(*CurlhT, *SysMat_Curlh);
        // FIXME: Is sorting necessary?
        //CTMC_lvls[level]->SortColumnIndices();

        delete SysMat_Curlh;
        delete CurlhT;

        // imposing boundary conditions
        Array<int> * temp = essbdrdofs_lvls[level];
        for ( int dof = 0; dof < temp->Size(); ++dof)
        {
            if ( (*temp)[dof] != 0)
            {
                CTMC_lvls[level]->EliminateRowCol(dof);
            }
        }

        // form CT*M*C as HypreParMatrices
        // FIXME: Can one avoid allocation of intermediate matrices?
        HypreParMatrix* CTMC_d_td;
        d_td_Hcurl_lvls[level]->SetOwnerFlags(3,3,1);
        CTMC_d_td = d_td_Hcurl_lvls[level]->LeftDiagMult( *CTMC_lvls[level] );
        HypreParMatrix * d_td_T = d_td_Hcurl_lvls[level]->Transpose();
        //d_td_T->CopyRowStarts();
        //d_td_T->CopyColStarts();
        //d_td_T->SetOwnerFlags(3,3,1);

        // this is wrong in global sens but lives as a debugging check here
        // so something wrong is with CTMC_d_td
        //CTMC_global_lvls[level] = ParMult(d_td_T, d_td_Hcurl_lvls[level]);

        // and this line segfaults!
        //d_td_T->SetOwnerFlags(3,3,1); // - even this doesn't help
        CTMC_global_lvls[level] = ParMult(d_td_T, CTMC_d_td);

        CTMC_global_lvls[level]->CopyRowStarts();
        CTMC_global_lvls[level]->CopyColStarts();

        delete CTMC_d_td;
        delete d_td_T;

        prec_global_lvls[level] = new HypreSmoother(*CTMC_global_lvls[level]);
        //prec_global_lvls[level]->iterative_mode = false;

        // resizing local-to-level vector arrays
        rhs_lvls[level] = new Vector(Curlh_lvls[level]->Width());
        tempvec_lvls[level] = new Vector(Curlh_lvls[level]->Width());
        tempvec2_lvls[level] = new Vector(Curlh_lvls[level]->Height());
        truerhs_lvls[level] = new Vector(CTMC_global_lvls[level]->Height());
        truex_lvls[level] = new Vector(CTMC_global_lvls[level]->Height());
        finalized_lvls[level] = true;
    }
}

/*
void HCurlSmoother::ComputeRhsLevel(int level, const BlockVector& res_lvl)
{
    // rhs_l = CT_l * res_lvl
    Curlh_lvls[level]->MultTranspose(res_lvl.GetBlock(0), *rhs_lvls[level]);
}


// Solves the minimization problem in the div-free subspace
// Takes the current iterate in_lvl
// and returns the updated iterate
//      out_lvl = in_lvl + Curl_l * sol_l
// where
//      CurlT_l M Curl_l sol_l = rhs_l
// rhs_l is computed using the residual of the original problem
// during the call to SetUpRhs() before MultLevel
void HCurlSmoother::MultLevel(int level, Vector& in_lvl, Vector& out_lvl)
{
    MFEM_ASSERT(finalized_lvls[level] == true,
                "MultLevel() must not be called for the non-finalized level");

    // for now we are smoothing in Hcurl only at the finest level
    // because we don't have canonical projectors to ensure that
    // coarsened curl will be in the kernel of coarsened divergence
    if (level != 0)
    {
        std::cout << "HCurlSmoother::MultLevel(): For now we are smoothing in "
                     "Hcurl only at the finest level \ndue to the absence of"
                     "canonical projector. Thus, returning out = in! \n";
        out_lvl = in_lvl;
        return;
    }

    if (print_level)
        std::cout << "Solving the minimization problem in Hcurl at level " << level << "\n";

    // 1. imposing boundary conditions on the righthand side
    Array<int> * temp = essbdrdofs_lvls[level];
    for ( int dof = 0; dof < temp->Size(); ++dof)
    {
        if ( (*temp)[dof] != 0)
        {
            (*rhs_lvls[level])[dof] = 0.0;
        }
    }

    *truex_lvls[level] = 0.0;

    // 2. assemble righthand side on the true dofs
    d_td_Hcurl_lvls[level]->MultTranspose(*rhs_lvls[level], *truerhs_lvls[level]);

    // 3. setting up the iterative CG solver
    HypreParMatrix * matrix_shortcut = CTMC_global_lvls[level];
    Solver * prec_shortcut = prec_global_lvls[level];

    //int maxIter(70000);
    //double rtol(1.e-12);
    //double atol(1.e-12);

    //std::cout << "Calling the PCG solver \n";
    //PCG(*matrix_shortcut, *prec_shortcut, *truerhs_lvls[level], *truex_lvls[level], 0, maxIter, rtol, atol );

    CGSolver solver(MPI_COMM_WORLD);
    //solver.SetAbsTol(abs_tol);
    //solver.SetRelTol(rel_tol);
    solver.SetAbsTol(sqrt(abs_tol));
    solver.SetRelTol(sqrt(rel_tol));
    solver.SetMaxIter(max_iter_internal);
    solver.SetOperator(*matrix_shortcut);
    solver.SetPreconditioner(*prec_shortcut);
    solver.SetPrintLevel(0);

    // 4. solving the linear system with preconditioned MINRES
    // on true dofs:
    // CT*M*C truex_l = truerhs_l
    if (print_level)
        std::cout << "Calling the CG solver for global Hcurl level problem \n";

    solver.Mult(*truerhs_lvls[level], *truex_lvls[level]);

    // temp_l = truex_l, but on dofs
    d_td_Hcurl_lvls[level]->Mult(*truex_lvls[level], *tempvec_lvls[level]);

    // 5. computing the solution update in the H(div)_h space

    // out = Curlh_l * temp_l = Curlh_l * x_l
    Curlh_lvls[level]->Mult( *tempvec_lvls[level], out_lvl);

    // out_lvl = in_lvl + Curlh_l * x_l
    out_lvl += in_lvl;
}
*/

#endif

// TODO: Add as an option using blas and lapack versions for solving local problems
// TODO: Test after all  with nonzero boundary conditions for sigma
// TODO: Check the timings and make it faster
// TODO: Add destructors for the new classes which deallocates all the memory
// TODO: Run a valgrind check
// TODO: Clean up the function descriptions
// TODO: Clean up the variables names
// TODO: Update HcurlSmoother class
// TODO: Maybe, local matrices can also be stored as an improvement (see SolveLocalProblems())?
// TODO: Make dof_truedof an optional data member so that either dof_truedof or
// TODO: global funct matrices (and offsets) are given at all level, maybe via two different constructors
// TODO: In the latter case ComputeTrueRes can be rewritten using global matrices and dof_truedof
// TODO: can remain unused at all.

// Solver and not IterativeSolver is the right choice for the base class
class GeneralMinConstrSolver : public Solver
{
private:
    // if true, coarsened operators will be constructed from Funct_lvls[0]
    // and Constr_levels[0]; else, the entire hierarchy of coarsened operators
    // must be provided in the constructor call of the solver
    const bool construct_coarseops;

    // if 0, relative change for consecutive iterations is checked
    // if 1, relative value is checked
    mutable int stopcriteria_type;

    // a flag which indicates whether the solver setup was called
    // before trying to solve anything
    mutable bool setup_finished;

    // makes changes if the solver is used as a preconditioner
    // changes MFEM_ASSERT checks for residual constraint
    // and sets init_guess to zero in Mult()
    mutable bool preconditioner_mode;

    // defines if the solver is symmetrized (default is yes)
    mutable bool symmetric;

    mutable int print_level;
    mutable double rel_tol;
    mutable int max_iter;
    mutable int converged;

protected:
    int num_levels;

    // iteration counter (solver behavior is different for the first iteration)
    mutable int current_iteration;

    // stores the functional values on the consecutive iterations
    // (needed for a variant of stopping criteria, type = 0)
    mutable double funct_prevnorm;
    mutable double funct_currnorm;
    mutable double funct_firstnorm;

    // used for stopping criteria (type = 1) based on solution updates
    mutable double solupdate_prevnorm;
    mutable double solupdate_currnorm;
    mutable double sol_firstitnorm;

    // used for stopping criteria (type = 2) based on solution updates
    // in a special mg norm
    mutable double solupdate_prevmgnorm;
    mutable double solupdate_currmgnorm;
    mutable double solupdate_firstmgnorm;

    // Dof_TrueDof relation tables for each level for functional-related
    // variables and the L2 variable (constraint space).
    // Used for assembling the coarsest level problem
    // and for the smoother setup in the general case
    const std::vector<std::vector<HypreParMatrix*> >& dof_trueDof_Func_lvls;

    const MPI_Comm comm;

    // Projectors for the variables related to the functional and constraint
    const Array< BlockMatrix*>& P_Func; // used only for operator coarsening
    const Array< BlockOperator*>& TrueP_Func;
    const Array< SparseMatrix*>& P_L2; // used only for operators coarsening

    // for each level and for each variable in the functional stores a vector
    // which defines if a dof is at the boundary / essential part of the boundary
    // or not
    const std::vector<std::vector<Array<int>* > > & essbdrtruedofs_Func; // can be removed since it is used only for debugging

    // parts of block structure which define the Functional at the finest level
    const int numblocks;
    //const Array<int>& block_offsets;
    //mutable Array<int> block_trueoffsets;

    // Righthand side of  the divergence contraint on dofs
    // (remains unchanged throughout the solving process)
    const Vector& ConstrRhs; // can be removed since it used only for debugging

    const Array<Operator*>& Smoothers_lvls;

    // a given blockvector which satisfies essential bdr conditions
    // imposed for the initial problem
    // on true dofs
    const BlockVector& bdrdata_truedofs;

    // stores Functional matrix on all levels except the finest
    // so that Funct_levels[0] = Functional matrix on level 1 (not level 0!)
    mutable Array<BlockMatrix*> Funct_lvls;
    mutable Array<SparseMatrix*> Constr_lvls; // can be removed since it's used only for debugging

    mutable Array<BlockVector*> Funct_rhs_lvls;

    const BlockOperator& Funct_global;
    // a required input since MFEM cannot give out offsets out of the const BlockOperator which is ugly imo
    // FIXME: Because of this one cannot check the compatibility of these two _global variables though they must be compatible
    const Array<int>& offsets_global;

    // The same as xblock and yblock but on true dofs
    mutable BlockVector* xblock_truedofs;
    mutable BlockVector* yblock_truedofs;
    mutable BlockVector* tempblock_truedofs;
    mutable BlockVector * Funct_rhsglobal_truedofs; // only for CheckFunctValue, only at the finest level

    // stores the initial guess for the solver
    // which satisfies the divergence contraint
    // if not specified in the constructor.
    // it is 0 by default
    // Muts be defined on true dofs
    mutable BlockVector* init_guess;

    mutable Array<Array<int>* > trueoffsets_lvls;
    mutable Array<BlockVector*> truetempvec_lvls;
    mutable Array<BlockVector*> truetempvec2_lvls;
    mutable Array<BlockVector*> trueresfunc_lvls;
    mutable Array<BlockVector*> truesolupdate_lvls;

    mutable Array<Operator*> LocalSolvers_lvls;
    mutable Operator* CoarseSolver;

protected:
    virtual void MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector& rhs_l) const;

    // Allocates current level-related data and computes coarser matrices for the functional
    // and the constraint.
    // Called only during the Setup()
    virtual void SetUpFinerLvl(int lvl) const;

    // Computes out_l as updated rhs in the functional for the current level
    //      out_l := rhs_l - M_l * solupd_l
    // Routine is used to update righthand side before and after the smoother call
    void UpdateTrueResidual(int level, const BlockVector* rhs_l,  const BlockVector& solupd_l, BlockVector& out_l) const;

    // constructs hierarchy of all objects requires
    // if necessary, builds the particular solution which satisfies
    // the given contraint and sets it as the initial iterate.
    // at each finer level also computes factorization of the local problems
    // matrices and stores them
    void Setup(bool verbose = false) const;

    // main solver iteration routine
    void Solve(const BlockVector &righthand_side, const BlockVector &previous_sol, BlockVector &next_sol) const;

public:
    // constructor
    GeneralMinConstrSolver(int NumLevels,
                           const std::vector<std::vector<HypreParMatrix *> > &Dof_TrueDof_Func_lvls,
                           const Array< BlockMatrix*> &Proj_Func,
                           const Array< BlockOperator*>& TrueProj_Func,
                           const Array< SparseMatrix*> &Proj_L2,
                           const std::vector<std::vector<Array<int>* > > &EssBdrTrueDofs_Func,
                           const Array<BlockMatrix *> &FunctOp_lvls,
                           const Array<SparseMatrix *> &ConstrOp_lvls,
                           const Array<BlockVector *> &FunctRhs_lvls,
                           const Vector& ConstrRhsVec,
                           const BlockOperator& Funct_Global,
                           const Array<int>& Offsets_Global,
                           const Array<Operator*>& Smoothers_Lvls,
                           const BlockVector& Bdrdata_TrueDofs,
                           Array<Operator*>* LocalSolvers = NULL,
                           Operator* CoarseSolver = NULL,
#ifdef COMPUTE_EXACTDISCRETESOL
                           Vector * sigma = NULL, Vector * S = NULL, Vector * lambda = NULL,
#endif
                           bool Construct_CoarseOps = true,
                           int StopCriteria_Type = 1);

    GeneralMinConstrSolver() = delete;

    // external calling routine (as in any IterativeSolver) which takes care of convergence
    virtual void Mult(const Vector & x, Vector & y) const override;

    // existence of this method is required by the (abstract) base class Solver
    virtual void SetOperator(const Operator &op) override{}

    bool StoppingCriteria(int type, double value_curr, double value_prev, double value_scalefactor,
                          double stop_tol, bool monotone_check = true, char const * name = NULL,
                          bool print = false) const;

    int GetStopCriteriaType () {return stopcriteria_type;} const
    void SetStopCriteriaType (int StopCriteria_Type) const {stopcriteria_type = StopCriteria_Type;}

    void SetAsPreconditioner(bool yes_or_no) const
    {preconditioner_mode = yes_or_no; if(preconditioner_mode) SetMaxIter(1); }
    bool IsSymmetric() const {return symmetric;}
    void SetSymmetric() const {symmetric = true;}
    void SetUnSymmetric() const {symmetric = false;}

    void SetInitialGuess(Vector& InitGuess) const;

    // have to define these to mimic useful routines from IterativeSolver class
    void SetRelTol(double RelTol) const {rel_tol = RelTol;}
    void SetMaxIter(int MaxIter) const {max_iter = MaxIter;}
    void SetPrintLevel(int PrintLevel) const {print_level = PrintLevel;}

    virtual void PrintAllOptions() const;

    void SetLocalSolvers(Array<Operator*> &LocalSolvers) const
    {
        LocalSolvers_lvls.SetSize(num_levels - 1);
        for (int l = 0; l < num_levels - 1; ++l)
            LocalSolvers_lvls[l] = LocalSolvers[l];
    }
};

void GeneralMinConstrSolver::PrintAllOptions() const
{
    std::cout << "GeneralMinConstrSolver options: \n";
    std::cout << "num_levels: " << num_levels << "\n";
    std::cout << "numblocks:" << numblocks << "\n";
    std::cout << "construct_coarseops: " << construct_coarseops << "\n";
    std::cout << "setup_finished: " << setup_finished << "\n";
    std::cout << "symmetric: " << symmetric << "\n";
    std::cout << "print_level: " << print_level << "\n";
    std::cout << "preconditioner_mode: " << preconditioner_mode << "\n";
    std::cout << "stop_criteria_type: " << stopcriteria_type << "\n";
    std::cout << "rel_tol: " << rel_tol << "\n";
    std::cout << "max_iter: " <<  max_iter << "\n";
    std::cout << "\n";
}

// The input must be defined on true dofs
void GeneralMinConstrSolver::SetInitialGuess(Vector& InitGuess) const
{
    init_guess->Update(InitGuess.GetData(), offsets_global);
}

bool GeneralMinConstrSolver::StoppingCriteria(int type, double value_curr, double value_prev,
                                                  double value_scalefactor, double stop_tol,
                                                  bool monotone_check, char const * name,
                                                  bool print) const
{
    bool already_printed = false;
    if (monotone_check)
        if (value_curr > value_prev && fabs(value_prev - value_curr) / value_scalefactor > 1.0e-10 )
        {
            std::cout << "criteria: " << name << " is increasing! \n";
            std::cout << "current " << name << ": " << value_curr << "\n";
            std::cout << "previous " << name << ": " << value_prev << "\n";
            std::cout << "rel change = " << (value_prev - value_curr) / value_scalefactor
                      << " (rel.tol = " << stop_tol << ")\n";
            already_printed = true;
        }

    switch(type)
    {
    case 0:
    {
        if (print && !already_printed)
        {

            std::cout << "current " << name << ": " << value_curr << "\n";
            std::cout << "previous " << name << ": " << value_prev << "\n";
            std::cout << "rel change = " << (value_prev - value_curr) / value_scalefactor
                      << " (rel.tol = " << stop_tol << ")\n";
        }

        if ( fabs(value_prev - value_curr) / value_scalefactor < stop_tol )
            return true;
        else
            return false;
    }
        break;
    case 1:
    case 2:
    {
        if (print && !already_printed)
        {

            std::cout << "current " << name << ": " << value_curr << "\n";
            std::cout << "rel = " << value_curr / value_scalefactor
                      << " (rel.tol = " << stop_tol << ")\n";
        }

        if ( fabs(value_curr) / value_scalefactor < stop_tol )
            return true;
        else
            return false;

    }
        break;
    default:
        MFEM_ABORT("Unknown value of type in StoppingCriteria() \n");
        return false;
        break;
    }
}


GeneralMinConstrSolver::GeneralMinConstrSolver(int NumLevels,
                       const std::vector<std::vector<HypreParMatrix*> >& Dof_TrueDof_Func_lvls,
                       const Array< BlockMatrix*> &Proj_Func,
                       const Array< BlockOperator*>& TrueProj_Func,
                       const Array< SparseMatrix*> &Proj_L2,
                       const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                       const Array<BlockMatrix*> & FunctOp_lvls,
                       const Array<SparseMatrix*> &ConstrOp_lvls,
                       const Array<BlockVector *> &FunctRhs_lvls,
                       const Vector& ConstrRhsVec,
                       const BlockOperator& Funct_Global,
                       const Array<int>& Offsets_Global,
                       const Array<Operator*>& Smoothers_Lvls,
                       const BlockVector& Bdrdata_TrueDofs,
                       Array<Operator*>* LocalSolvers,
                       Operator *CoarsestSolver,
#ifdef COMPUTE_EXACTDISCRETESOL
                       Vector * sigma, Vector * S, Vector * lambda,
#endif

                       bool Construct_CoarseOps, int StopCriteria_Type)
     : Solver(FunctOp_lvls[0]->Height(), FunctOp_lvls[0]->Width()), // FIXME: Wrong sizes may affect smth, should be true sizes
       construct_coarseops(Construct_CoarseOps),
       stopcriteria_type(StopCriteria_Type),
       setup_finished(false),
       num_levels(NumLevels),
       current_iteration(0),
       dof_trueDof_Func_lvls(Dof_TrueDof_Func_lvls),
       comm(Dof_TrueDof_Func_lvls[0][0]->GetComm()),
       P_Func(Proj_Func), TrueP_Func(TrueProj_Func), P_L2(Proj_L2),
       essbdrtruedofs_Func(EssBdrTrueDofs_Func),
       numblocks(FunctOp_lvls[0]->NumColBlocks()),
       ConstrRhs(ConstrRhsVec),
       Smoothers_lvls(Smoothers_Lvls),
       bdrdata_truedofs(Bdrdata_TrueDofs),
       Funct_global(Funct_Global),
       offsets_global(Offsets_Global)
{

    MFEM_ASSERT(FunctOp_lvls[0] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                " Funct operator at the finest level must be given anyway!");
    MFEM_ASSERT(ConstrOp_lvls[0] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                " Constraint operator at the finest level must be given anyway!");

    if (!construct_coarseops)
        for ( int l = 0; l < num_levels; ++l)
        {
            MFEM_ASSERT(FunctOp_lvls[l] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                        " functional operators at all levels must be provided "
                                                        " when construct_curls == false!");
            MFEM_ASSERT(ConstrOp_lvls[l] != NULL, "GeneralMinConstrSolver::GeneralMinConstrSolver()"
                                                        " constraint operators at all levels must be provided "
                                                        " when construct_curls == false!");
        }

    Funct_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Funct_lvls[l] = FunctOp_lvls[l];

    Constr_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Constr_lvls[l] = ConstrOp_lvls[l];

    Funct_rhs_lvls.SetSize(num_levels);
    for (int l = 0; l < num_levels; ++l)
        Funct_rhs_lvls[l] = FunctRhs_lvls[l];

    xblock_truedofs = new BlockVector(offsets_global);
    yblock_truedofs = new BlockVector(offsets_global);
    tempblock_truedofs = new BlockVector(offsets_global);
    Funct_rhsglobal_truedofs = new BlockVector(offsets_global);

    // FIXME: Probably eliminate by changing the interface and constructor parameters of the class
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        // assembly
        dof_trueDof_Func_lvls[0][blk]->MultTranspose(Funct_rhs_lvls[0]->GetBlock(blk), Funct_rhsglobal_truedofs->GetBlock(blk));

        for ( int tdofind = 0; tdofind < essbdrtruedofs_Func[0][blk]->Size(); ++tdofind )
        {
            int tdof = (*essbdrtruedofs_Func[0][blk])[tdofind];
            Funct_rhsglobal_truedofs->GetBlock(blk)[tdof] = 0.0;
        }

        //SparseMatrix d_td_blk_diag;
        //dof_trueDof_Func_lvls[0][blk]->GetDiag(d_td_blk_diag);
        //d_td_blk_diag.MultTranspose(Funct_rhs_lvls[0]->GetBlock(blk), Funct_rhsglobal_truedofs->GetBlock(blk));
    }


    truesolupdate_lvls.SetSize(num_levels);
    truesolupdate_lvls[0] = new BlockVector(offsets_global);

    trueoffsets_lvls.SetSize(num_levels);
    trueoffsets_lvls[0] = NULL;//&offsets_global;
    truetempvec_lvls.SetSize(num_levels);
    truetempvec_lvls[0] = new BlockVector(offsets_global);
    truetempvec2_lvls.SetSize(num_levels);
    truetempvec2_lvls[0] = new BlockVector(offsets_global);
    trueresfunc_lvls.SetSize(num_levels);
    trueresfunc_lvls[0] = new BlockVector(offsets_global);

    if (CoarsestSolver)
        CoarseSolver = CoarsestSolver;
    else
        CoarseSolver = NULL;

    SetRelTol(1.0e-12);
    SetMaxIter(1000);
    SetPrintLevel(0);
    SetSymmetric();
    SetAsPreconditioner(false);
    converged = 0;

    funct_prevnorm = 0.0;
    funct_currnorm = 0.0;
    funct_firstnorm = 0.0;

    solupdate_prevnorm = 0.0;
    solupdate_currnorm = 0.0;
    sol_firstitnorm = 0.0;

    solupdate_prevmgnorm = 0.0;
    solupdate_currmgnorm = 0.0;
    solupdate_firstmgnorm = 0.0;

    init_guess = new BlockVector(offsets_global);
    *init_guess = 0.0;

    if (LocalSolvers)
    {
        LocalSolvers_lvls.SetSize(num_levels - 1);
        for (int l = 0; l < num_levels - 1; ++l)
            LocalSolvers_lvls[l] = (*LocalSolvers)[l];
    }

    Setup();
}

void GeneralMinConstrSolver::Setup(bool verbose) const
{
    if (verbose)
        std::cout << "Starting solver setup \n";

    // 1. copying the given initial vector to the internal variable

    // old interface
    //CheckFunctValue(comm, *Funct_lvls[0], dof_trueDof_Func_lvls[0], *init_guess,
            //"for the initial guess during solver setup: ", print_level);
    CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs, offsets_global, *init_guess,
            "for the initial guess during solver setup (no rhs provided): ", print_level);

    // 2. setting up the required internal data at all levels
    // including smoothers

    // 2.1 loop over all levels except the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        //sets up the current level and prepares operators for the next one
        SetUpFinerLvl(l);
    } // end of loop over finer levels

    setup_finished = true;

    if (verbose)
        std::cout << "Solver setup completed \n";
}

// The top-level wrapper for the solver which overrides Solver::Mult()
// Works on true dof vectors
// x is the righthand side
// y is the output
void GeneralMinConstrSolver::Mult(const Vector & x, Vector & y) const
{
    MFEM_ASSERT(setup_finished, "Solver setup must have been called before Mult() \n");
#ifdef MFEM_DEBUG
    BlockVector temp_dofs(Funct_lvls[0]->RowOffsets());
#endif

    // start iteration
    current_iteration = 0;
    converged = 0;

    // x will be accessed through xblock_truedofs as its view
    xblock_truedofs->Update(x.GetData(), offsets_global);
    // y will be accessed through yblock_truedofs as its view
    yblock_truedofs->Update(y.GetData(), offsets_global);

    if (preconditioner_mode)
        *init_guess = 0.0;
    else
        funct_firstnorm = CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs, offsets_global, *init_guess,
                                 "for the initial guess: ", print_level);


    // tempblock is the initial guess (on true dofs)
    *tempblock_truedofs = *init_guess;

    int itnum = 0;
    for (int i = 0; i < max_iter; ++i )
    {
        MFEM_ASSERT(i == current_iteration, "Iteration counters mismatch!");

        MFEM_ASSERT(CheckBdrError(*tempblock_truedofs, bdrdata_truedofs,
                                  *essbdrtruedofs_Func[0][0], true), "before the iteration");

#ifdef MFEM_DEBUG
        for ( int blk = 0; blk < numblocks; ++blk)
            dof_trueDof_Func_lvls[0][blk]->Mult(tempblock_truedofs->GetBlock(blk), temp_dofs.GetBlock(blk));

        if (!preconditioner_mode)
        {
            MFEM_ASSERT(CheckConstrRes(tempblock_truedofs->GetBlock(0), *Constr_lvls[0], &ConstrRhs,
                                       "before the iteration"),"");
            MFEM_ASSERT(CheckConstrRes(temp_dofs.GetBlock(0), *Constr_lvls[0], &ConstrRhs,
                                       "before the iteration"),"");
        }
        else
            MFEM_ASSERT(CheckConstrRes(temp_dofs.GetBlock(0), *Constr_lvls[0], NULL, "before the iteration"),"");
#endif

        Solve(*xblock_truedofs, *tempblock_truedofs, *yblock_truedofs);

        // monitoring convergence
        bool monotone_check = (i != 0);
        if (!preconditioner_mode)
        {
            if (i == 0)
                StoppingCriteria(1, funct_currnorm, funct_prevnorm, funct_firstnorm, rel_tol,
                                 monotone_check, "functional", print_level);
            else
                StoppingCriteria(0, funct_currnorm, funct_prevnorm, funct_firstnorm, rel_tol,
                                 monotone_check, "functional", print_level);

            StoppingCriteria(stopcriteria_type, solupdate_currnorm, solupdate_prevnorm,
                             sol_firstitnorm,  rel_tol, monotone_check, "sol_update", print_level);
        }
        StoppingCriteria(stopcriteria_type, solupdate_currmgnorm, solupdate_prevmgnorm,
                         solupdate_firstmgnorm, rel_tol, monotone_check, "sol_update in mg ", print_level);

        bool stopped;
        switch(stopcriteria_type)
        {
        case 0:
            if (i == 0)
                stopped = StoppingCriteria(1, funct_currnorm, funct_prevnorm, funct_firstnorm, rel_tol,
                                                   false, "functional", 0);
            else
                stopped = StoppingCriteria(0, funct_currnorm, funct_prevnorm, funct_firstnorm, rel_tol,
                                                   false, "functional", 0);
            break;
        case 1:
            stopped = StoppingCriteria(1, solupdate_currnorm, solupdate_prevnorm,
                                       sol_firstitnorm,  rel_tol, monotone_check, "sol_update", 0);
            break;
        case 2:
            stopped = StoppingCriteria(2, solupdate_currmgnorm, solupdate_prevmgnorm,
                                       solupdate_firstmgnorm, rel_tol, monotone_check, "sol_update in mg ", 0);
            break;
        default:
            MFEM_ABORT("Unknown stopping criteria type \n");
        }

        if (stopped)
        {
            converged = 1;
            itnum = i;
            break;
        }
        else
        {
            if (i == max_iter - 1)
            {
                converged = -1;
                itnum = max_iter;
                break;
            }
            funct_prevnorm = funct_currnorm;
            solupdate_prevnorm = solupdate_currnorm;
            solupdate_prevmgnorm = solupdate_currmgnorm;

            // resetting the input and output vectors for the next iteration

            *tempblock_truedofs = *yblock_truedofs;
        }

    } // end of main iterative loop

    // describing the reason for the stop:
    if (print_level)
    {
        if (converged == 1)
            std::cout << "Solver converged in " << itnum << " iterations. \n";
        else // -1
            std::cout << "Solver didn't converge in " << itnum << " iterations. \n";
    }

}

void GeneralMinConstrSolver::MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector &rhs_l) const
{
    // FIXME: Get rid of temp1 and temp2
    BlockVector temp1(Funct_lvls[l]->ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[l][blk]->Mult(x_l.GetBlock(blk), temp1.GetBlock(blk));
    }

    BlockVector temp2(Funct_lvls[l]->RowOffsets());
    Funct_lvls[l]->Mult(temp1, temp2);

    temp2 *= coeff;

    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[l][blk]->MultTranspose(temp2.GetBlock(blk), rhs_l.GetBlock(blk));
    }
}

// Computes out_l as an updated rhs in the functional part for the given level
//      out_l :=  rhs_l - M_l sol_l
// the same as ComputeUpdatedLvlRhsFunc but on true dofs
void GeneralMinConstrSolver::UpdateTrueResidual(int level, const BlockVector* rhs_l,
                                                          const BlockVector& solupd_l, BlockVector& out_l) const
{
    // out_l = - M_l * solupd_l
    MultTrueFunc(level, -1.0, solupd_l, out_l);

    // out_l = rhs_l - M_l * solupd_l
    if (rhs_l)
        out_l += *rhs_l;
}

// Computes one iteration of the new solver
// Input: previous_sol (and all the setup)
// Output: next_sol
// All parameters are defined as vectors on true dofs
void GeneralMinConstrSolver::Solve(const BlockVector& righthand_side,
                                       const BlockVector& previous_sol, BlockVector& next_sol) const
{
    if (print_level)
        std::cout << "Starting iteration " << current_iteration << " ... \n";

#ifndef CHECK_SPDSOLVER
    MFEM_ASSERT(CheckBdrError(previous_sol, bdrdata_truedofs, *essbdrtruedofs_Func[0][0], true),
            "at the start of Solve()");
#endif

    next_sol = previous_sol;

    // FIXME: Remove
    CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs, offsets_global, next_sol,
                             "at the beginning of Solve: ", print_level);

    UpdateTrueResidual(0, &righthand_side, previous_sol, *trueresfunc_lvls[0] );
    /*
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        for (int i = 0; i < essbdrtruedofs_Func[0][blk]->Size(); ++i)
        {
            int ind = (*essbdrtruedofs_Func[0][blk])[i];
            trueresfunc_lvls[0]->GetBlock(blk)[ind] = 0.0;
        }
    }
    */

    /*
    // why this is not zero?
    {
        BlockVector temp(offsets_global);
        temp = righthand_side;
        temp -= *Funct_rhsglobal_truedofs;

        std::cout << "norm of righthand_side in Solve() - Funct_rhsglobal_truedofs = " << temp.Norml2() / sqrt (temp.Size()) << "\n";
    }

    // a check
    {
        BlockVector temp(offsets_global);
        MultTrueFunc(0,-1.0, next_sol, temp);
        temp += *Funct_rhsglobal_truedofs;
        //temp += righthand_side;
        //Funct_rhsglobal_truedofs->Print();
        temp -= *trueresfunc_lvls[0];
        //temp.Print();
        std::cout << "norm of (trueresfunc[0] - (F - Funct * current sol)), "
                     "before the finest level local solve = " << temp.Norml2() / sqrt (temp.Size());
    }
    */


    /*
    // FIXME: Comment
    Funct_global.Mult(previous_sol, *trueresfunc_lvls[0]);
    *trueresfunc_lvls[0] *= -1.0;
    *trueresfunc_lvls[0] += righthand_side;
    * */

    //trueresfunc_lvls[0]->GetBlock(0).Print();
    //trueresfunc_lvls[0]->GetBlock(1).Print();

    /*
    *truesolupdate_lvls[0] = 0.0;

    if (LocalSolvers_lvls[0])
    {
        LocalSolvers_lvls[0]->Mult(*trueresfunc_lvls[0], *truetempvec_lvls[0]);
        *truesolupdate_lvls[0] += *truetempvec_lvls[0];
    }
    */

    /*
    // debugging only thing
    BlockVector lvlrhs_func(Funct_lvls[0]->ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
    {
        dof_trueDof_Func_lvls[0][blk]->Mult(trueresfunc_lvls[0]->GetBlock(blk), lvlrhs_func.GetBlock(blk));
    }
    ofstream ofs3("Gblock1_insideSolve.txt");
    ofs3 << lvlrhs_func.GetBlock(1).Size() << "\n";
    lvlrhs_func.GetBlock(1).Print(ofs3,1);
    ofs3.close();
    */

    // DOWNWARD loop: from finest to coarsest
    // 1. loop over levels finer than the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        // solution updates will always satisfy homogeneous essential boundary conditions
        *truesolupdate_lvls[l] = 0.0;

        if (LocalSolvers_lvls[l])
        {
            std::cout << "Temporarily not solving the local problems! \n";
            LocalSolvers_lvls[l]->Mult(*trueresfunc_lvls[l], *truetempvec_lvls[l]);
            // FIXME: comment this
            *truetempvec_lvls[l] = 0.0;
            *truesolupdate_lvls[l] += *truetempvec_lvls[l];
        }

        // FIXME: Remove
        //MFEM_ASSERT(CheckConstrRes(truesolupdate_lvls[l]->GetBlock(0), *Constr_lvls[0], NULL,
                                   //"for the local update"),"");

        UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );
        /*
        for ( int blk = 0; blk < numblocks; ++blk)
        {
            for (int i = 0; i < essbdrtruedofs_Func[l][blk]->Size(); ++i)
            {
                int ind = (*essbdrtruedofs_Func[l][blk])[i];
                trueresfunc_lvls[l]->GetBlock(blk)[ind] = 0.0;
            }
        }
        */

        // smooth
        if (Smoothers_lvls[l])
        {
            Smoothers_lvls[l]->Mult(*truetempvec_lvls[l], *truetempvec2_lvls[l] );
            *truesolupdate_lvls[l] += *truetempvec2_lvls[l];
            UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );
        }

        *trueresfunc_lvls[l] = *truetempvec_lvls[l];

        // imposing bnd condititons and projecting rhs from the functional to the next (coarser) level
        for ( int blk = 0; blk < numblocks; ++blk)
        {
            for (int i = 0; i < essbdrtruedofs_Func[l][blk]->Size(); ++i)
            {
                int ind = (*essbdrtruedofs_Func[l][blk])[i];
                trueresfunc_lvls[l]->GetBlock(blk)[ind] = 0.0;
            }
        }
        TrueP_Func[l]->MultTranspose(*trueresfunc_lvls[l], *trueresfunc_lvls[l + 1]);

    } // end of loop over finer levels

    // FIXME: Only for debugging, the functional in blocked case is not being minimized
    // 4. update the global iterate by the resulting update at the finest level

    std::cout << "level 0 update norm = " << truesolupdate_lvls[0]->Norml2() / sqrt (truesolupdate_lvls[0]->Size());

    next_sol += *truesolupdate_lvls[0];
    funct_currnorm = CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs,  offsets_global, next_sol,
                             "after local solve at level 0: ", print_level);

    // a check of the rhs for the coarsest level problem
    {
        BlockVector temp(offsets_global);
        MultTrueFunc(0,-1.0, next_sol, temp);
        temp += *Funct_rhsglobal_truedofs;
        temp -= *trueresfunc_lvls[0];
        //trueresfunc_lvls[0]->Print();
        //std::cout << "res at level 0 norm = " << trueresfunc_lvls[0]->Norml2() / sqrt(trueresfunc_lvls[0]->Size()) << "\n";
        //std::cout << "after projector below \n";
        //trueresfunc_lvls[1]->Print();
        //std::cout << "P * res at level 1 norm = " << trueresfunc_lvls[1]->Norml2() / sqrt(trueresfunc_lvls[1]->Size()) << "\n";
        //temp.Print();
        for ( int blk = 0; blk < numblocks; ++blk)
        {
            for (int i = 0; i < essbdrtruedofs_Func[0][blk]->Size(); ++i)
            {
                int ind = (*essbdrtruedofs_Func[0][blk])[i];
                temp.GetBlock(blk)[ind] = 0.0;
            }
        }

        std::cout << "norm of (trueresfunc[0] - (F - Funct * current sol)) = " << temp.Norml2() / sqrt (temp.Size()) << "\n";
    }

    next_sol -= *truesolupdate_lvls[0];

    // BOTTOM: solve the global problem at the coarsest level
    // imposes boundary conditions and assembles the coarsests level's
    // righthand side  (from rhsfunc) on true dofs

    // trying to organize the coarsest solver right here
    // manually creating the operator at the coarsest level from Funct_lvls[0]
    SparseMatrix P_RT = P_Func[0]->GetBlock(0,0);
    SparseMatrix P_H1;
    if (numblocks > 1)
        P_H1 = P_Func[0]->GetBlock(1,1);
    SparseMatrix * Funct_00_h = new SparseMatrix(Funct_lvls[0]->GetBlock(0,0));
    SparseMatrix * Funct_01_h, * Funct_11_h;
    if (numblocks > 1)
    {
        Funct_01_h = new SparseMatrix(Funct_lvls[0]->GetBlock(0,1));
        Funct_11_h = new SparseMatrix(Funct_lvls[0]->GetBlock(1,1));
    }

    SparseMatrix * Funct_00_H;
    SparseMatrix * Funct_01_H, * Funct_11_H;

    SparseMatrix * temp00 = mfem::Mult(*Transpose(P_RT), *Funct_00_h);
    Funct_00_H = mfem::Mult(*temp00, P_RT);
    Funct_00_H->SortColumnIndices();

    if (numblocks > 1)
    {
        SparseMatrix * temp11 = mfem::Mult(*Transpose(P_H1), *Funct_11_h);
        Funct_11_H = mfem::Mult(*temp11, P_H1);
        Funct_11_H->SortColumnIndices();

        SparseMatrix * temp01 = mfem::Mult(*Transpose(P_RT), *Funct_01_h);
        Funct_01_H = mfem::Mult(*temp01, P_H1);
        Funct_01_H->SortColumnIndices();
    }

    SparseMatrix * Constr_H;
    SparseMatrix * tempconstr = mfem::Mult(*Transpose(*P_L2[0]), *Constr_lvls[0]);
    Constr_H = mfem::Mult(*tempconstr, P_RT);
    Constr_H->SortColumnIndices();

    Array<int> tempblock_offsets(numblocks + 2);
    tempblock_offsets[0] = 0;
    tempblock_offsets[1] = Funct_00_H->Width();
    if (numblocks > 1)
        tempblock_offsets[2] = Funct_11_H->Width();
    tempblock_offsets[numblocks + 1] = Constr_H->Height();
    tempblock_offsets.PartialSum();

    std::cout << "tempblock_offsets \n";
    tempblock_offsets.Print();

    BlockMatrix newcoarsemat(tempblock_offsets);
    newcoarsemat.SetBlock(0,0, Funct_00_H);
    if (numblocks > 1)
    {
        newcoarsemat.SetBlock(0,1, Funct_01_H);
        newcoarsemat.SetBlock(1,1, Funct_11_H);
        newcoarsemat.SetBlock(1,0, Transpose(*Funct_01_H));
    }
    newcoarsemat.SetBlock(0, numblocks, Transpose(*Constr_H));
    newcoarsemat.SetBlock(numblocks, 0, Constr_H);

    // latest version of the code
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
    {
        const Array<int> * temp1 = essbdrtruedofs_Func[1][blk1];
        Array<int> temp1_another(newcoarsemat.GetBlock(blk1,blk1).Width());
        temp1_another = 0;
        for ( int i = 0; i < temp1->Size(); ++i)
        {
            int ind = (*temp1)[i];
            temp1_another[ind] = 1;
        }

        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            const Array<int> * temp2 = essbdrtruedofs_Func[1][blk2];
            Array<int> temp2_another(newcoarsemat.GetBlock(blk2,blk2).Width());
            temp2_another = 0;
            for ( int i = 0; i < temp2->Size(); ++i)
            {
                int ind = (*temp2)[i];
                temp2_another[ind] = 1;
            }

            /*
            for ( int i = 0; i < temp2->Size(); ++i)
            {
                int ind = (*temp2)[i];
                newcoarsemat.GetBlock(blk1,blk2).EliminateCol(ind);
            }
            */
            newcoarsemat.GetBlock(blk1,blk2).EliminateCols(temp2_another);

            for ( int dof1 = 0; dof1 < temp1_another.Size(); ++dof1)
            {
                /*
            for ( int j = 0; j < temp1_another.Size(); ++j)
            {
                int ind = (*temp1)[j];
                if (blk1 == blk2)
                    newcoarsemat.GetBlock(blk1,blk2).EliminateRow(ind, 1.0);
                else // doesn't set diagonal entry to 1
                    newcoarsemat.GetBlock(blk1,blk2).EliminateRow(ind);
                */

                if ( temp1_another[dof1] != 0)
                {
                    if (blk1 == blk2)
                        newcoarsemat.GetBlock(blk1,blk2).EliminateRow(dof1, 1.0);
                    else // doesn't set diagonal entry to 1
                        newcoarsemat.GetBlock(blk1,blk2).EliminateRow(dof1);
                }
            }

        }
    }

    // manually creating a standard iterative solver
    MINRESSolver * coarseSolver = new MINRESSolver(comm);
    coarseSolver->SetAbsTol(1.0e-20);
    coarseSolver->SetRelTol(1.0e-20);
    coarseSolver->SetMaxIter(20000);
    coarseSolver->SetOperator(newcoarsemat);

    const Array<int> * temp1 = essbdrtruedofs_Func[1][0];
    Array<int> temp1_another(newcoarsemat.GetBlock(0,0).Width());
    temp1_another = 0;
    for ( int i = 0; i < temp1->Size(); ++i)
    {
        int ind = (*temp1)[i];
        temp1_another[ind] = 1;
    }

    newcoarsemat.GetBlock(numblocks,0).EliminateCols(temp1_another);
    for ( int dof1 = 0; dof1 < temp1_another.Size(); ++dof1)
    {
        if ( temp1_another[dof1] != 0)
        {
            newcoarsemat.GetBlock(0, numblocks).EliminateRow(dof1);
        }
    }

    // solving
    BlockVector coarserhs(tempblock_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        coarserhs.GetBlock(blk) = trueresfunc_lvls[1]->GetBlock(blk);
        // plus boundary conditions
        const Array<int> * temp1 = essbdrtruedofs_Func[1][blk];
        for ( int i = 0; i < temp1->Size(); ++i)
        {
            int ind = (*temp1)[i];
            coarserhs.GetBlock(blk)[ind] = 0.0;
        }
    }

    coarserhs.GetBlock(numblocks) = 0.0;

    BlockVector coarsesol(tempblock_offsets);
    coarsesol = 0.0;

    coarseSolver->Mult(coarserhs, coarsesol);

    ofstream ofs("coarserhs_outside solver.txt");
    ofs << coarserhs.Size() << "\n";
    coarserhs.Print(ofs,1);
    ofs.close();

    for ( int blk = 0; blk < numblocks; ++blk)
        truesolupdate_lvls[num_levels - 1]->GetBlock(blk) = coarsesol.GetBlock(blk);

    *truetempvec_lvls[1] = *truesolupdate_lvls[num_levels - 1];



    CoarseSolver->Mult(*trueresfunc_lvls[num_levels - 1], *truesolupdate_lvls[num_levels - 1]);

    *truetempvec2_lvls[1] = *truetempvec_lvls[1];
    *truetempvec2_lvls[1] -= *truesolupdate_lvls[1];
    std::cout << "norm of difference between coarsest updates = " << truetempvec2_lvls[1]->Norml2() / sqrt (truetempvec2_lvls[1]->Size()) << "\n";
    truetempvec2_lvls[1]->Print(); // must be 0

    //*truesolupdate_lvls[1] = *truetempvec_lvls[1];
    //*truesolupdate_lvls[num_levels - 1] = 0.0;

    TrueP_Func[0]->Mult(*truesolupdate_lvls[1], *truetempvec_lvls[0] );
    std::cout << "coarsest level update norm = " << truesolupdate_lvls[1]->Norml2() / sqrt (truesolupdate_lvls[1]->Size());
    *truesolupdate_lvls[0] += *truetempvec_lvls[0];
    next_sol += *truesolupdate_lvls[0];

    funct_currnorm = CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs, offsets_global, next_sol,
                             "after local solve plus coarsest level solve: ", print_level);

    next_sol -= *truesolupdate_lvls[0];
    *truesolupdate_lvls[0] -= *truetempvec_lvls[0];

    //truesolupdate_lvls[num_levels - 1]->Print();

    // UPWARD loop: from coarsest to finest
    if (symmetric) // then also smoothing and solving local problems on the way up
    {
        for (int l = num_levels - 1; l > 0; --l)
        {
            // interpolate back to the finer level
            TrueP_Func[l - 1]->Mult(*truesolupdate_lvls[l], *truetempvec_lvls[l - 1]);

            *truesolupdate_lvls[l - 1] += *truetempvec_lvls[l - 1];

            UpdateTrueResidual(l - 1, trueresfunc_lvls[l - 1], *truetempvec_lvls[l - 1], *truetempvec2_lvls[l - 1] );
            *trueresfunc_lvls[l - 1] = *truetempvec2_lvls[l - 1];

            // smooth at the finer level
            if (Smoothers_lvls[l - 1])
            {
                Smoothers_lvls[l - 1]->MultTranspose(*truetempvec2_lvls[l - 1], *truetempvec_lvls[l - 1] );
                *truesolupdate_lvls[l - 1] += *truetempvec_lvls[l - 1];
                UpdateTrueResidual(l - 1, trueresfunc_lvls[l - 1], *truetempvec_lvls[l - 1], *truetempvec_lvls[l - 1] );
            }

            if (LocalSolvers_lvls[l - 1])
            {
                LocalSolvers_lvls[l - 1]->Mult(*truetempvec_lvls[l - 1], *truetempvec2_lvls[l - 1]);
                *truesolupdate_lvls[l - 1] += *truetempvec2_lvls[l - 1];
            }

        }

    }
    else // then simply interpolating and adding updates
    {
        // assemble the final solution update from all levels
        // final sol update (at level 0)  =
        //                   = solupdate[0] + P_0 * (solupdate[1] + P_1 * ( ...) )
        for (int level = num_levels - 1; level > 0; --level)
        {
            // solupdate[level-1] = solupdate[level-1] + P[level-1] * solupdate[level]
            TrueP_Func[level - 1]->Mult(*truesolupdate_lvls[level], *truetempvec_lvls[level - 1] );
            *truesolupdate_lvls[level - 1] += *truetempvec_lvls[level - 1];
        }

    }

    // 4. update the global iterate by the resulting update at the finest level
    next_sol += *truesolupdate_lvls[0];

    if (print_level)
    {
        if (!preconditioner_mode)
        {
            MFEM_ASSERT(CheckBdrError(next_sol, bdrdata_truedofs, *essbdrtruedofs_Func[0][0], true),
                    "after all levels update");
        }

    }

    if (print_level > 10)
    {
        std::cout << "sol_update norm: " << truesolupdate_lvls[0]->GetBlock(0).Norml2() /
                  sqrt(truesolupdate_lvls[0]->GetBlock(0).Size()) << "\n";
    }

    // some monitoring service calls
    if (!preconditioner_mode)
        if (print_level || stopcriteria_type == 0)
        {
            //funct_currnorm = CheckFunctValue(comm, *Funct_lvls[0], dof_trueDof_Func_lvls[0], next_sol,
                                             //"at the end of iteration: ", print_level);
            funct_currnorm = CheckFunctValue(comm, Funct_global, Funct_rhsglobal_truedofs, offsets_global, next_sol,
                                     "at the end of iteration: ", print_level);

            //if (fabs(funct_currnorm - funct_currnorm2) < 1.0e-14)
                //MFEM_ABORT("");
            //MFEM_ASSERT(fabs(funct_currnorm - funct_currnorm2) < 1.0e-14, "Functional values are computed differently! \n");
        }

    if (!preconditioner_mode)
        if (print_level || stopcriteria_type == 1)
            solupdate_currnorm = ComputeMPIVecNorm(comm, *truesolupdate_lvls[0],
                                                    "of the update: ", print_level);

    if (print_level || stopcriteria_type == 2)
    {
        if (!preconditioner_mode)
        {
            UpdateTrueResidual(0, &righthand_side, previous_sol, *trueresfunc_lvls[0] );
            solupdate_currmgnorm = sqrt(ComputeMPIDotProduct(comm, *truesolupdate_lvls[0], *trueresfunc_lvls[0]));
        }
        else
        {
            // FIXME: is this correct?
            solupdate_currmgnorm = sqrt(ComputeMPIDotProduct(comm, *truesolupdate_lvls[0], righthand_side));
        }
    }

    if (current_iteration == 0)
        solupdate_firstmgnorm = solupdate_currmgnorm;

    ++current_iteration;

    return;
}

// Computes prerequisites required for solving local problems at level l
// such as relation tables between AEs and internal fine-grid dofs
// and maybe smth else ... ?
void GeneralMinConstrSolver::SetUpFinerLvl(int lvl) const
{
    // Funct_lvls[lvl] stores the Functional matrix on level lvl
    if (construct_coarseops)
    {
        BlockMatrix * Funct_PFunc;
        BlockMatrix * P_FuncT = Transpose(*P_Func[lvl]);
        Funct_PFunc = mfem::Mult(*Funct_lvls[lvl],*P_Func[lvl]);

        // checking the difference between coarsened and true
        // (from bilinear form) functional operators
        /*
        std::cout << "level = " << lvl << "\n";
        BlockMatrix * tempdiff = mfem::Mult(*P_FuncT, *Funct_PFunc);
        for ( int blk = 0; blk < numblocks; blk++)
        {
            std::cout << "blk = " << blk << "\n";
            SparseMatrix * tempdiffblk = new SparseMatrix(tempdiff->GetBlock(blk,blk));
            tempdiffblk->Add(-1.0,Funct_lvls[lvl + 1]->GetBlock(blk,blk));
            std::cout << tempdiffblk->MaxNorm() << "\n";
        }
        */
        Funct_lvls[lvl + 1] = mfem::Mult(*P_FuncT, *Funct_PFunc);

        SparseMatrix *P_L2T = Transpose(*P_L2[lvl]);
        SparseMatrix *Constr_PR;
        Constr_PR = mfem::Mult(*Constr_lvls[lvl], P_Func[lvl]->GetBlock(0,0));

        // checking the difference between coarsened and true
        // (from bilinear form) constraint operators
        /*
        SparseMatrix * tempdiffsp = mfem::Mult(*P_L2T, *Constr_PR);
        tempdiffsp->Add(-1.0, *Constr_lvls[lvl + 1]);
        std::cout << tempdiffsp->MaxNorm() << "\n";
        */

        Constr_lvls[lvl + 1] = mfem::Mult(*P_L2T, *Constr_PR);

        delete Funct_PFunc;
        delete Constr_PR;
        delete P_FuncT;
        delete P_L2T;
    }

    //tempvec_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //tempvec2_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //solupdate_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());
    //rhsfunc_lvls[lvl + 1] = new BlockVector(Funct_lvls[lvl + 1]->RowOffsets());

    trueoffsets_lvls[lvl + 1] = new Array<int>(numblocks + 1);
    (*trueoffsets_lvls[lvl + 1])[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        (*trueoffsets_lvls[lvl + 1])[blk + 1] = (*trueoffsets_lvls[lvl + 1])[blk] +
                dof_trueDof_Func_lvls[lvl + 1][blk]->Width();
    }

    truetempvec_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    truetempvec2_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    truesolupdate_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
    trueresfunc_lvls[lvl + 1] = new BlockVector(*trueoffsets_lvls[lvl + 1]);
}

#if 0
class MinConstrSolverWithS : private GeneralMinConstrSolver
{
private:
    const int strategy;

protected:
    virtual void SolveLocalProblem (std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B, BlockVector &G, Vector& F, BlockVector &sol) const;
    virtual void SolveCoarseProblem(BlockVector& rhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const;
    virtual void ComputeRhsFunc(BlockVector &rhs_func, const Vector& x) const;
    virtual void SetUpFinerLvl(int level) const
    { GeneralMinConstrSolver::SetUpFinerLvl(level);}
public:
    // constructor
    MinConstrSolverWithS(int NumLevels, const Array< SparseMatrix*> &AE_to_e,
                         const Array< BlockMatrix*> &El_to_dofs_Func, const Array< SparseMatrix*> &El_to_dofs_L2,
                         const std::vector<HypreParMatrix*>& Dof_TrueDof_Func,
                         const HypreParMatrix& Dof_TrueDof_L2,
                         const Array< BlockMatrix*> &Proj_Func, const Array< SparseMatrix*> &Proj_L2,
                         const std::vector<std::vector<Array<int>* > > &BdrDofs_Func,
                         const BlockMatrix& FunctBlockMat,
                         const SparseMatrix& ConstrMat, const Vector& ConstrRhsVec,
                         const BlockVector& Bdrdata_Finest,
                         bool Higher_Order_Elements = false, int Strategy = 0)
        : GeneralMinConstrSolver(NumLevels, AE_to_e, El_to_dofs_Func, El_to_dofs_L2,
                         Dof_TrueDof_Func, Dof_TrueDof_L2, Proj_Func, Proj_L2, BdrDofs_Func,
                         FunctBlockMat, ConstrMat, ConstrRhsVec,
                         Bdrdata_Finest,
                         Higher_Order_Elements),
         strategy(Strategy)
         {}

    virtual void Mult(const Vector & x, Vector & y) const;
};

void MinConstrSolverWithS::Mult(const Vector & x, Vector & y) const
{
    std::cout << "Mult() for (sigma,S) formulation is not implemented! \n";
    y = x;
}

// Computes rhs coming from the last iterate sigma
// rhs_func = - A * x, where A is the matrix arising
// from the local minimization functional, and x is the
// minimzed variables (sigma or (sigma,S)).
void MinConstrSolverWithS::ComputeRhsFunc(BlockVector &rhs_func, const Vector& x) const
{
    // if we going to minimize only sigma
    if (strategy != 0)
    {
        xblock->Update(x.GetData(), block_offsets);
        Funct.GetBlock(0,0).Mult(xblock->GetBlock(0), rhs_func);
    }
    else
    {
        xblock->Update(x.GetData(), block_offsets);
        Funct.Mult(*xblock, rhs_func);
        rhs_func *= -1;
    }
}

// Solves a local linear system of the form
// [ A  DT  BT ] [ sig ] = [ Gsig ]
// [ D  0   0  ] [  s  ] = [ GS   ]
// [ B  0   0  ] [ lam ] = [ F    ]
// as
// [s, lam]^T = inv ( [D B]^T invA [DT BT] ) * ( [D B]^T invA * Gsig - [GS F]^T )
// s = [s, lam]_1
// sig = invA * (Gsig - [DT BT] * [s, lam]^T)
void MinConstrSolverWithS::SolveLocalProblem (std::vector<DenseMatrix> &FunctBlks, DenseMatrix& B, BlockVector &G, Vector& F, BlockVector &sol) const
{
    std::cout << "MinConstrSolverWithS::SolveLocalProblem() is not implemented!";
    // FIXME: rewrite the routine

    /*

    Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = GS.Size();
    offsets[2] = F.Size();
    offsets.PartialSum();

    BlockVector s_lam(offsets);

    BlockDenseMatrix D_B(offsets);
    D_B.SetBlock(0,0,D);
    D_B.SetBlock(1,0,B);

    DenseMatrixInverse inv_A(A);
    BlockDenseMatrix invA_D_B;
    inv_A.Mult(D_B, invA_D_B);

    BlockDenseMatrix Schur;
    Mult(D_B, inv_A_DT_BT, Schur);

    DenseBlockMatrixInverse inv_Schur(Schur);

    s = s_lam.GetBlock(0);

    // computing sig
    // temp2 = Gsig - [DT BT] * [s, lam]^T
    Vector temp2;
    D_B.MultTranspose(s_lam, temp2);
    temp2 *= -1;
    temp2 += Gsig;

    // sig = invA * temp2
    inv_A.Mult(temp2, sig);
    */

    return;
}

void MinConstrSolverWithS::SolveCoarseProblem(BlockVector& rhs_func, Vector& rhs_constr, BlockVector& sol_coarse) const
{
    std::cout << "SolveCoarseProblem is not implemented! \n";
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

namespace mfem
{

// self-written copy routine for HypreParMatrices
// faces the issues with LeftDiagMult and ParMult combination
// My guess is that offd.num_rownnz != 0 is the bug
// but no proof for now
HypreParMatrix * CopyHypreParMatrix (HypreParMatrix& inputmat)
{
    MPI_Comm comm = inputmat.GetComm();
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    HYPRE_Int global_num_rows = inputmat.M();
    HYPRE_Int global_num_cols = inputmat.N();

    int size_starts = num_procs;
    if (num_procs > 1) // in thi case offd exists
    {
        //int myid;
        //MPI_Comm_rank(comm,&myid);

        HYPRE_Int * row_starts_in = inputmat.GetRowStarts();
        HYPRE_Int * col_starts_in = inputmat.GetColStarts();

        HYPRE_Int * row_starts = new HYPRE_Int[num_procs];
        memcpy(row_starts, row_starts_in, size_starts * sizeof(HYPRE_Int));
        HYPRE_Int * col_starts = new HYPRE_Int[num_procs];
        memcpy(col_starts, col_starts_in, size_starts * sizeof(HYPRE_Int));

        //std::cout << "memcpy calls finished \n";

        SparseMatrix diag_in;
        inputmat.GetDiag(diag_in);
        SparseMatrix * diag_out = new SparseMatrix(diag_in);

        //std::cout << "diag copied \n";

        SparseMatrix offdiag_in;
        HYPRE_Int * offdiag_cmap_in;
        inputmat.GetOffd(offdiag_in, offdiag_cmap_in);

        int size_offdiag_cmap = offdiag_in.Width();

        SparseMatrix * offdiag_out = new SparseMatrix(offdiag_in);
        HYPRE_Int * offdiag_cmap_out = new HYPRE_Int[size_offdiag_cmap];

        memcpy(offdiag_cmap_out, offdiag_cmap_in, size_offdiag_cmap * sizeof(int));


        return new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                  row_starts, col_starts,
                                  diag_out, offdiag_out, offdiag_cmap_out);

        //std::cout << "constructor called \n";
    }
    else // in this case offd doesn't exist and we have to use a different constructor
    {
        HYPRE_Int * row_starts = new HYPRE_Int[2];
        row_starts[0] = 0;
        row_starts[1] = global_num_rows;
        HYPRE_Int * col_starts = new HYPRE_Int[2];
        col_starts[0] = 0;
        col_starts[1] = global_num_cols;

        SparseMatrix diag_in;
        inputmat.GetDiag(diag_in);
        SparseMatrix * diag_out = new SparseMatrix(diag_in);

        return new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                  row_starts, col_starts, diag_out);
    }

}

// faces the same issues as CopyHypreParMatrix
HypreParMatrix * CopyRAPHypreParMatrix (HypreParMatrix& inputmat)
{
    MPI_Comm comm = inputmat.GetComm();
    int num_procs;
    MPI_Comm_size(comm, &num_procs);

    HYPRE_Int global_num_rows = inputmat.M();
    HYPRE_Int global_num_cols = inputmat.N();

    int size_starts = 2;

    HYPRE_Int * row_starts_in = inputmat.GetRowStarts();
    HYPRE_Int * col_starts_in = inputmat.GetColStarts();

    HYPRE_Int * row_starts = new HYPRE_Int[num_procs];
    memcpy(row_starts, row_starts_in, size_starts * sizeof(HYPRE_Int));
    HYPRE_Int * col_starts = new HYPRE_Int[num_procs];
    memcpy(col_starts, col_starts_in, size_starts * sizeof(HYPRE_Int));

    int num_local_rows = row_starts[1] - row_starts[0];
    int num_local_cols = col_starts[1] - col_starts[0];
    int * ia_id = new int[num_local_rows + 1];
    ia_id[0] = 0;
    for ( int i = 0; i < num_local_rows; ++i)
        ia_id[i + 1] = ia_id[i] + 1;

    int id_nnz = num_local_rows;
    int * ja_id = new int[id_nnz];
    double * a_id = new double[id_nnz];
    for ( int i = 0; i < id_nnz; ++i)
    {
        ja_id[i] = i;
        a_id[i] = 1.0;
    }

    SparseMatrix * id_diag = new SparseMatrix(ia_id, ja_id, a_id, num_local_rows, num_local_cols);

    HypreParMatrix * id = new HypreParMatrix(comm, global_num_rows, global_num_cols,
                                             row_starts, col_starts, id_diag);

    return RAP(&inputmat,id);
}

} // end of namespace mfem
