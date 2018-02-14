#include "mfem.hpp"
#include "linalg/linalg.hpp"

#include <iterator>

using namespace mfem;
using namespace std;
using std::unique_ptr;

// PLAN:
// 1) ...

#define MEMORY_OPTIMIZED

// activates a check for the correctness of local problem solve for the blocked case (with S)
//#define CHECK_LOCALSOLVE

// activates some additional checks
//#define DEBUG_INFO

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
double CheckFunctValue(MPI_Comm comm, const Operator& Funct, const Vector* truefunctrhs, const Vector& truevec, char const * string, bool print)
{
    Vector trueres(truevec.Size());
    Funct.Mult(truevec, trueres);
    double local_func_norm;
    if (truefunctrhs)
    {
        trueres.Add(-2.0, *truefunctrhs);
        local_func_norm = truevec * trueres / sqrt (trueres.Size()); // incorrect, different f should be used + (*truefunctrhs) * (*truefunctrhs) / sqrt(truefunctrhs->Size());
    }
    else // NULL case assumed to denote zero righthand side
        local_func_norm = truevec * trueres / sqrt (trueres.Size());
    double global_func_norm = 0;
    MPI_Allreduce(&local_func_norm, &global_func_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (print)
        std::cout << "Functional norm " << string << global_func_norm << " ... \n";
    return global_func_norm;
}

// Computes and prints the norm of || Constr * sigma - ConstrRhs ||_2,h, everything on true dofs
bool CheckConstrRes(const Vector& sigma, const HypreParMatrix& Constr, const Vector* ConstrRhs,
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
        //res_constr.Print();
        std::cout << "Constraint residual norm " << string << ": "
                  << constr_norm << " ... \n";
        passed = false;
    }

    return passed;
}

// true for truedofs, false for dofs
bool CheckBdrError (const Vector& Candidate, const Vector* Given_bdrdata, const Array<int>& ess_bdr, bool dof_or_truedof)
{
    bool passed = true;
    double max_bdr_error = 0;
    if (dof_or_truedof) // for true dofs
    {
        for ( int i = 0; i < ess_bdr.Size(); ++i)
        {
            int tdof = ess_bdr[i];
            double bdr_error_dof;
            if (Given_bdrdata)
                bdr_error_dof = fabs((*Given_bdrdata)[tdof] - Candidate[tdof]);
            else
                bdr_error_dof = fabs(Candidate[tdof]);
            if ( bdr_error_dof > max_bdr_error )
                max_bdr_error = bdr_error_dof;
        }
    }
    else // for dofs
    {
        for ( int dof = 0; dof < Candidate.Size(); ++dof)
        {
            if (ess_bdr[dof] != 0.0)
            {
                double bdr_error_dof;
                if (Given_bdrdata)
                    bdr_error_dof = fabs((*Given_bdrdata)[dof] - Candidate[dof]);
                else
                    bdr_error_dof = fabs(Candidate[dof]);
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

class CoarsestProblemHcurlSolver : public Operator
{
private:
    const int numblocks;

    mutable int sweeps_num;

    MPI_Comm comm;

    mutable bool finalized;

    const Array2D<HypreParMatrix*> & Funct_global;

    const HypreParMatrix& Divfreeop;
    mutable HypreParMatrix * Divfreeop_T;

    const std::vector<Array<int>* > & essbdrdofs_blocks;
    const std::vector<Array<int>* > & essbdrtruedofs_blocks;
    const Array<int> & essbdrdofs_Hcurl;
    const Array<int> & essbdrtruedofs_Hcurl;

    mutable Array<int> coarse_offsets;
    mutable BlockOperator* coarse_matrix;
    mutable BlockDiagonalPreconditioner * coarse_prec;
    mutable BlockVector * coarsetrueX;
    mutable BlockVector * coarsetrueRhs;
    mutable IterativeSolver * coarseSolver;

    // all on true dofs
    mutable Array<int> block_offsets;
    mutable BlockVector* xblock;
    mutable BlockVector* yblock;

protected:
    mutable int maxIter;
    mutable double rtol;
    mutable double atol;

    void Setup() const;

public:
    ~CoarsestProblemHcurlSolver();
    CoarsestProblemHcurlSolver(int Size,
                               const Array2D<HypreParMatrix*> & Funct_Global,
                               const HypreParMatrix& DivfreeOp,
                               const std::vector<Array<int>* >& EssBdrDofs_blks,
                               const std::vector<Array<int> *> &EssBdrTrueDofs_blks, const Array<int> &EssBdrDofs_Hcurl,
                               const Array<int>& EssBdrTrueDofs_Hcurl);

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const;

    void SetMaxIter(int MaxIter) const {maxIter = MaxIter;}
    void SetAbsTol(double AbsTol) const {atol = AbsTol;}
    void SetRelTol(double RelTol) const {rtol = RelTol;}
    void ResetSolverParams() const
    {
        coarseSolver->SetAbsTol(atol);
        coarseSolver->SetRelTol(rtol);
        coarseSolver->SetMaxIter(maxIter);
    }
    void PrintSolverParams() const
    {
        std::cout << "maxIter: " << maxIter << "\n";
        std::cout << "rtol: " << rtol << "\n";
        std::cout << "atol: " << atol << "\n";
        std::cout << std::flush;
    }
};

CoarsestProblemHcurlSolver::CoarsestProblemHcurlSolver(int Size,
                                                       const Array2D<HypreParMatrix*> & Funct_Global,
                                                       const HypreParMatrix& DivfreeOp,
                                                       const std::vector<Array<int>* >& EssBdrDofs_blks,
                                                       const std::vector<Array<int> *> &EssBdrTrueDofs_blks,
                                                       const Array<int>& EssBdrDofs_Hcurl,
                                                       const Array<int>& EssBdrTrueDofs_Hcurl)
    : Operator(Size),
      numblocks(Funct_Global.NumRows()),
      comm(DivfreeOp.GetComm()),
      Funct_global(Funct_Global),
      Divfreeop(DivfreeOp),
      essbdrdofs_blocks(EssBdrDofs_blks),
      essbdrtruedofs_blocks(EssBdrTrueDofs_blks),
      essbdrdofs_Hcurl(EssBdrDofs_Hcurl),
      essbdrtruedofs_Hcurl(EssBdrTrueDofs_Hcurl)
{
    finalized = false;

    block_offsets.SetSize(numblocks + 1);
    block_offsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        block_offsets[blk + 1] = Funct_global(blk,blk)->Height();
    block_offsets.PartialSum();

    coarse_offsets.SetSize(numblocks + 1);

    maxIter = 50;
    rtol = 1.e-4;
    atol = 1.e-4;

    sweeps_num = 1;
    Setup();

}


CoarsestProblemHcurlSolver::~CoarsestProblemHcurlSolver()
{
    delete xblock;
    delete yblock;

    delete coarsetrueRhs;
    delete coarsetrueX;
    for ( int blk = 0; blk < coarse_prec->NumBlocks(); ++blk)
            if (&(coarse_prec->GetDiagonalBlock(blk)))
                delete &(coarse_prec->GetDiagonalBlock(blk));
    delete coarse_prec;
    for ( int blk1 = 0; blk1 < coarse_matrix->NumRowBlocks(); ++blk1)
        for ( int blk2 = 0; blk2 < coarse_matrix->NumColBlocks(); ++blk2)
            if ( !(blk1 == 1 && blk2 == 1) && coarse_matrix->IsZeroBlock(blk1, blk2) == false)
                delete &(coarse_matrix->GetBlock(blk1, blk2));
    delete coarse_matrix;

    delete coarseSolver;

    delete Divfreeop_T;
}

void CoarsestProblemHcurlSolver::Setup() const
{
    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    Divfreeop_T = Divfreeop.Transpose();

    Array2D<HypreParMatrix*> HcurlFunct_global(numblocks, numblocks);
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
    {
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            HypreParMatrix * Funct_blk = Funct_global(blk1,blk2);

            if (Funct_blk)
            {
                if (blk1 == 0)
                {
                    HypreParMatrix * temp1 = ParMult(Divfreeop_T, Funct_blk);
                    temp1->CopyRowStarts();
                    temp1->CopyColStarts();

                    if (blk2 == 0)
                    {
                        HcurlFunct_global(blk1, blk2) = RAP(&Divfreeop, Funct_blk, &Divfreeop);

                        //HcurlFunct_global(blk1, blk2) = ParMult(temp1, &Divfreeop);

                        HcurlFunct_global(blk1, blk2)->CopyRowStarts();
                        HcurlFunct_global(blk1, blk2)->CopyColStarts();

                        delete temp1;
                    }
                    else
                        HcurlFunct_global(blk1, blk2) = temp1;

                }
                else if (blk2 == 0)
                {
                    HcurlFunct_global(blk1, blk2) = ParMult(Funct_blk,
                                                                  &Divfreeop);
                    HcurlFunct_global(blk1, blk2)->CopyRowStarts();
                    HcurlFunct_global(blk1, blk2)->CopyColStarts();
                }
                else
                {
                    HcurlFunct_global(blk1, blk2)  = Funct_blk;
                }
            } // else of if Funct_blk != NULL

        }
    }

    coarse_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_offsets[blk + 1] = HcurlFunct_global(blk, blk)->Height();
    coarse_offsets.PartialSum();

    //coarse_offsets.Print();

    coarse_matrix = new BlockOperator(coarse_offsets);
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
            coarse_matrix->SetBlock(blk1, blk2, HcurlFunct_global(blk1, blk2));

    // coarse solution and righthand side vectors
    coarsetrueX = new BlockVector(coarse_offsets);
    coarsetrueRhs = new BlockVector(coarse_offsets);

    // preconditioner for the coarse problem
    std::vector<Operator*> Prec_blocks(numblocks);
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(numblocks <= 2, "Current implementation of coarsest level solver "
                                   "knows preconditioners only for sigma or (sigma,S) setups \n");
        //if (blk == 0)
        //{
            //Prec_blocks[0] = new HypreDiagScale(*HcurlFunct_global(blk,blk));
        //}
        //else
            Prec_blocks[blk] = new HypreSmoother(*HcurlFunct_global(blk,blk),
                                           HypreSmoother::Type::l1GS, sweeps_num);
    }

    coarse_prec = new BlockDiagonalPreconditioner(coarse_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_prec->SetDiagonalBlock(blk, Prec_blocks[blk]);

    // coarse solver
    coarseSolver = new CGSolver(comm);
    coarseSolver->SetAbsTol(atol);
    coarseSolver->SetRelTol(rtol);
    coarseSolver->SetMaxIter(maxIter);
    coarseSolver->SetOperator(*coarse_matrix);
    if (coarse_prec)
        coarseSolver->SetPreconditioner(*coarse_prec);
    coarseSolver->SetPrintLevel(0);

    finalized = true;
}

void CoarsestProblemHcurlSolver::Mult(const Vector &x, Vector &y) const
{
    MFEM_ASSERT(finalized, "Mult() must not be called before the coarse solver was finalized \n");

    // x and y will be accessed through xblock and yblock as viewing structures
    xblock->Update(x.GetData(), block_offsets);
    yblock->Update(y.GetData(), block_offsets);

    // 1. set up solution and righthand side vectors
    *coarsetrueX = 0.0;
    *coarsetrueRhs = 0.0;


    //Divfreeop.MultTranspose(xblock->GetBlock(0), coarsetrueRhs->GetBlock(0));
    //Divfreeop_T->Mult(xblock->GetBlock(0), coarsetrueRhs->GetBlock(0));
    //Divfreeop.Mult(coarsetrueRhs->GetBlock(0), yblock->GetBlock(0));

    //yblock->GetBlock(1) = xblock->GetBlock(1);

    //return;

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
        {
            const Array<int> * temp;
            temp = essbdrtruedofs_blocks[blk];

            /*
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                xblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            }
            */

#ifdef CHECK_BNDCND
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                if ( fabs(xblock->GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                    std::cout << "bnd cnd is violated for xblock! blk = " << blk << ", value = "
                              << xblock->GetBlock(blk)[(*temp)[tdofind]]
                              << ", index = " << (*temp)[tdofind] << "\n";
            }
#endif
        }


        if (blk == 0)
            Divfreeop_T->Mult(xblock->GetBlock(blk), coarsetrueRhs->GetBlock(blk));
        else
        {
            MFEM_ASSERT(coarsetrueRhs->GetBlock(blk).Size() == xblock->GetBlock(blk).Size(),
                        "Sizes mismatch when finalizing rhs at the coarsest level!\n");
            coarsetrueRhs->GetBlock(blk) = xblock->GetBlock(blk);
        }

    }

    // 2. solve the linear system with preconditioned CG.

    /*
    if (coarsetrueRhs->Norml2() / sqrt(coarsetrueRhs->Size()) < atol * atol)
    {
        std::cout << "norm = " << coarsetrueRhs->Norml2() / sqrt(coarsetrueRhs->Size()) << ", atol^2 = " << atol * atol << "\n";
        std::cout << "Resetting max iter num to 1 \n";
        coarseSolver->SetMaxIter(1);
        coarseSolver->SetAbsTol(1.0e-15);
    }
    */

    // imposing bnd conditions on the internal solver input vector
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> * temp;
        if (blk == 0)
            temp = &essbdrtruedofs_Hcurl;
        else
            temp = essbdrtruedofs_blocks[blk];

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            coarsetrueRhs->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
        }
    }
    //*coarsetrueX = *coarsetrueRhs;
    //coarse_prec->Mult(*coarsetrueRhs, *coarsetrueX);
    //coarse_matrix->Mult(*coarsetrueRhs, *coarsetrueX);
    coarseSolver->Mult(*coarsetrueRhs, *coarsetrueX);
    // imposing bnd conditions on the internal solver output vector
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> * temp;
        if (blk == 0)
            temp = &essbdrtruedofs_Hcurl;
        else
            temp = essbdrtruedofs_blocks[blk];

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            coarsetrueX->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
        }
    }

    //*coarsetrueX = *coarsetrueRhs;



    //if (coarseSolver->GetConverged())
        //std::cout << "coarseSolver converged in " << coarseSolver->GetNumIterations()
                  //<< " iterations with a residual norm of " << coarseSolver->GetFinalNorm() << ".\n";

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            Divfreeop.Mult(coarsetrueX->GetBlock(blk), yblock->GetBlock(blk));
        else
            yblock->GetBlock(blk) = coarsetrueX->GetBlock(blk);
    }

    // imposing/checking bnd conditions on the resulting output vector
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> * temp;
        temp = essbdrtruedofs_blocks[blk];

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            //yblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
#ifdef CHECK_BNDCND
            // checking bnd conditions for the resulting output vector
            if ( fabs(yblock->GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for yblock! blk = " << blk << ", value = "
                          << yblock->GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
#endif
        }
    }

    return;
}

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

    mutable HypreParMatrix *Schur;
protected:
    mutable int maxIter;
    mutable double rtol;
    mutable double atol;

    void Setup() const;

public:
    ~CoarsestProblemSolver();
    CoarsestProblemSolver(int Size, BlockMatrix& Op_Blksmat, SparseMatrix& Constr_Spmat,
                          const std::vector<HypreParMatrix*>& D_tD_blks,
                          const HypreParMatrix& D_tD_L2,
                          const std::vector<Array<int>* >& EssBdrDofs_blks,
                          const std::vector<Array<int> *> &EssBdrTrueDofs_blks);

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const { Mult(x,y, NULL); }

    void DebugMult(Vector& rhs, Vector &sol) const;

    void Mult(const Vector &x, Vector &y, Vector* rhs_constr) const;

    void SetMaxIter(int MaxIter) const {maxIter = MaxIter;}
    void SetAbsTol(double AbsTol) const {atol = AbsTol;}
    void SetRelTol(double RelTol) const {rtol = RelTol;}
    void ResetSolverParams() const
    {
        coarseSolver->SetAbsTol(atol);
        coarseSolver->SetRelTol(rtol);
        coarseSolver->SetMaxIter(maxIter);
    }
    void PrintSolverParams() const
    {
        std::cout << "maxIter: " << maxIter << "\n";
        std::cout << "rtol: " << rtol << "\n";
        std::cout << "atol: " << atol << "\n";
        std::cout << std::flush;
    }
};

CoarsestProblemSolver::CoarsestProblemSolver(int Size, BlockMatrix& Op_Blksmat,
                                             SparseMatrix& Constr_Spmat,
                                             const std::vector<HypreParMatrix*>& D_tD_blks,
                                             const HypreParMatrix& D_tD_L2,
                                             const std::vector<Array<int>* >& EssBdrDofs_blks,
                                             const std::vector<Array<int>* >& EssBdrTrueDofs_blks)
    : Operator(Size),
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

    maxIter = 50;
    rtol = 1.e-4;
    atol = 1.e-4;

    Setup();
}

CoarsestProblemSolver::~CoarsestProblemSolver()
{
    delete xblock;
    delete yblock;

    delete coarseSolver;
    delete coarsetrueRhs;
    delete coarsetrueX;
    delete coarse_rhsfunc;
    for ( int blk = 0; blk < coarse_prec->NumBlocks(); ++blk)
            if (&(coarse_prec->GetDiagonalBlock(blk)))
                delete &(coarse_prec->GetDiagonalBlock(blk));
    delete coarse_prec;
    for ( int blk1 = 0; blk1 < coarse_matrix->NumRowBlocks(); ++blk1)
        for ( int blk2 = 0; blk2 < coarse_matrix->NumColBlocks(); ++blk2)
            if (coarse_matrix->IsZeroBlock(blk1, blk2) == false)
                delete &(coarse_matrix->GetBlock(blk1, blk2));
    delete coarse_matrix;

    delete Schur;
}

void CoarsestProblemSolver::Setup() const
{
    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    // 1. eliminating boundary conditions at coarse level
    const Array<int> * temp = essbdrdofs_blocks[0];

    Constr_spmat->EliminateCols(*temp);

    // latest version of the bnd conditions imposing code
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
    HYPRE_Int glob_num_rows = dof_trueDof_L2.M();
    HYPRE_Int glob_num_cols = dof_trueDof_blocks[0]->M();
    HYPRE_Int * row_starts = dof_trueDof_L2.GetRowStarts();
    HYPRE_Int * col_starts = dof_trueDof_blocks[0]->GetRowStarts();;
    HypreParMatrix * temphpmat = new HypreParMatrix(comm, glob_num_rows, glob_num_cols, row_starts, col_starts, Constr_spmat);
    //temp->CopyRowStarts();
    //temp->CopyColStarts();
    HypreParMatrix * Constr_global = RAP(&dof_trueDof_L2, temphpmat, dof_trueDof_blocks[0]);

    // old way
    //HypreParMatrix * Constr_d_td = dof_trueDof_blocks[0]->LeftDiagMult(
                //*Constr_spmat, dof_trueDof_L2.GetColStarts());
    //HypreParMatrix * d_td_L2_T = dof_trueDof_L2.Transpose();
    //HypreParMatrix * Constr_global = ParMult(d_td_L2_T, Constr_d_td);

    Constr_global->CopyRowStarts();
    Constr_global->CopyColStarts();

    HypreParMatrix *ConstrT_global = Constr_global->Transpose();

    //delete Constr_d_td;
    //delete d_td_L2_T;
    delete temphpmat;

    Array2D<HypreParMatrix*> Funct_global(numblocks, numblocks);
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            // alternative way
            HYPRE_Int glob_num_rows = dof_trueDof_blocks[blk1]->M();
            HYPRE_Int glob_num_cols = dof_trueDof_blocks[blk2]->M();
            HYPRE_Int * row_starts = dof_trueDof_blocks[blk1]->GetRowStarts();
            HYPRE_Int * col_starts = dof_trueDof_blocks[blk2]->GetRowStarts();;
            //std::cout << "row_starts: " << row_starts[0] << ", " << row_starts[1] << "\n";
            //std::cout << "col_starts: " << col_starts[0] << ", " << col_starts[1] << "\n";
            //HypreParMatrix * temp = new HypreParMatrix(MPI_COMM_WORLD, glob_size, row_starts, &(Op_blkspmat->GetBlock(blk1, blk2)));
            HypreParMatrix * temphpmat = new HypreParMatrix(comm, glob_num_rows, glob_num_cols, row_starts, col_starts, &(Op_blkspmat->GetBlock(blk1, blk2)));
            //temp->CopyRowStarts();
            //temp->CopyColStarts();
            Funct_global(blk1, blk2) = RAP(dof_trueDof_blocks[blk1], temphpmat, dof_trueDof_blocks[blk2]);

            // old way
            //auto Funct_d_td = dof_trueDof_blocks[blk2]->LeftDiagMult(Op_blkspmat->GetBlock(blk1,blk2),
            //                                                         dof_trueDof_blocks[blk1]->GetRowStarts() );
            //auto d_td_T = dof_trueDof_blocks[blk1]->Transpose();
            //Funct_global(blk1, blk2) = ParMult(d_td_T, Funct_d_td);

            Funct_global(blk1, blk2)->CopyRowStarts();
            Funct_global(blk1, blk2)->CopyColStarts();

            //SparseMatrix diag_debug;
            //Funct_global(blk1, blk2)->GetDiag(diag_debug);
            //diag_debug.Print();

            delete temphpmat;
            //delete Funct_d_td;
            //delete d_td_T;
        }

    coarse_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_offsets[blk + 1] = Funct_global(blk, blk)->Height();
    coarse_offsets[numblocks + 1] = Constr_global->Height();
    coarse_offsets.PartialSum();

    coarse_rhsfunc_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_rhsfunc_offsets[blk + 1] = coarse_offsets[blk + 1];
    coarse_rhsfunc_offsets.PartialSum();

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

    //IdentityOperator * invSchur = new IdentityOperator(Constr_global->Height());

    
    HypreParMatrix *MinvBt = Constr_global->Transpose();
    HypreParVector *Md = new HypreParVector(comm, Funct_global(0,0)->GetGlobalNumRows(),
                                            Funct_global(0,0)->GetRowStarts());
    Funct_global(0,0)->GetDiag(*Md);
    MinvBt->InvScaleRows(*Md);
    Schur = ParMult(Constr_global, MinvBt);
    Schur->CopyRowStarts();
    Schur->CopyColStarts();

   
    delete MinvBt;
    delete Md;
    
    
    HypreBoomerAMG * invSchur = new HypreBoomerAMG(*Schur);
    invSchur->SetPrintLevel(0);
    invSchur->iterative_mode = false;
    

    //MPI_Barrier(comm);
    //std::cout << "I have " << Schur->GetNumRows() << " rows \n" << std::flush;
    //MPI_Barrier(comm);

    coarse_prec = new BlockDiagonalPreconditioner(coarse_offsets);
    for ( int blk = 0; blk < numblocks; ++blk)
        coarse_prec->SetDiagonalBlock(blk, Funct_prec[blk]);
    coarse_prec->SetDiagonalBlock(numblocks, invSchur);

    // coarse solver
    coarseSolver = new MINRESSolver(comm);
    coarseSolver->SetAbsTol(atol);
    coarseSolver->SetRelTol(rtol);
    coarseSolver->SetMaxIter(maxIter);
    coarseSolver->SetOperator(*coarse_matrix);
    if (coarse_prec)
        coarseSolver->SetPreconditioner(*coarse_prec);
    //std::cout << "no prec \n" << std::flush;
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

    // 1. set up solution and righthand side vectors
    *coarsetrueX = 0.0;
    *coarsetrueRhs = 0.0;

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(coarsetrueRhs->GetBlock(blk).Size() == xblock->GetBlock(blk).Size(),
                    "Sizes mismatch when finalizing rhs at the coarsest level!\n");
        coarsetrueRhs->GetBlock(blk) = xblock->GetBlock(blk);

        // imposing boundary conditions on true dofs
        Array<int> * temp = essbdrtruedofs_blocks[blk];

        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            coarsetrueRhs->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
        }
    }

    if (rhs_constr)
    {
        MFEM_ASSERT(coarsetrueRhs->GetBlock(numblocks).Size() == rhs_constr->Size(),
                    "Sizes mismatch when finalizing rhs at the coarsest level!\n");
        coarsetrueRhs->GetBlock(numblocks) = *rhs_constr;
    }

    //std::cout << "Block 0 \n";
    //coarsetrueRhs->GetBlock(0).Print();
    //std::cout << "Block 1 \n";
    //coarsetrueRhs->GetBlock(1).Print();

    // 2. solve the linear system with preconditioned MINRES.
    coarseSolver->Mult(*coarsetrueRhs, *coarsetrueX);

    for ( int blk = 0; blk < numblocks; ++blk)
        yblock->GetBlock(blk) = coarsetrueX->GetBlock(blk);

    return;
}

void CoarsestProblemSolver::DebugMult(Vector& rhs, Vector &sol) const
{
    coarseSolver->Mult(rhs, sol);
}

// ~ Non-overlapping Schwarz smoother based on agglomerated elements
// which provides zeros at the interfaces in the output
class LocalProblemSolver : public Operator
{
private:
    mutable bool finalized;

protected:
    int numblocks;

    const BlockMatrix& Op_blkspmat;
    const SparseMatrix& Constr_spmat;

    const std::vector<HypreParMatrix*>& d_td_blocks;

    // Relation tables which represent agglomerated elements-to-elements relation
    const SparseMatrix& AE_e;

    const BlockMatrix& el_to_dofs_Op;
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

    mutable BlockVector* temprhs_func;
    mutable BlockVector* tempsol;

protected:
    mutable bool optimized_localsolve;

    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate) const;

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
    ~LocalProblemSolver();
    // main constructor
    LocalProblemSolver(const BlockMatrix& Op_Blksmat,
                       const SparseMatrix& Constr_Spmat,
                       const std::vector<HypreParMatrix*>& D_tD_blks,
                       const SparseMatrix& AE_el,
                       const BlockMatrix& El_to_Dofs_Op,
                       const SparseMatrix& El_to_Dofs_L2,
                       const std::vector<Array<int>* >& BdrDofs_blks,
                       const std::vector<Array<int>* >& EssBdrDofs_blks,
                       bool Optimized_LocalSolve)
        : Operator(),
          numblocks(Op_Blksmat.NumRowBlocks()),
          Op_blkspmat(Op_Blksmat), Constr_spmat(Constr_Spmat),
          d_td_blocks(D_tD_blks),
          AE_e(AE_el),
          el_to_dofs_Op(El_to_Dofs_Op),
          el_to_dofs_L2(El_to_Dofs_L2),
          bdrdofs_blocks(BdrDofs_blks),
          essbdrdofs_blocks(EssBdrDofs_blks)
    {
        finalized = 0;
        optimized_localsolve = Optimized_LocalSolve;
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

        Setup();
    }

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const override { Mult(x,y, NULL); }

    // considers x as the righthand side
    // and returns y as a solution to all the local problems
    // (*) both x and y are vectors on true dofs
    virtual void Mult(const Vector &x, Vector &y, Vector* rhs_constr) const;

    // is public since one might want to use that to compute particular solution witn nonzero righthand side in the constraint
    void SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol_update, Vector* localrhs_constr) const;
};

LocalProblemSolver::~LocalProblemSolver()
{
    delete tempsol;
    delete temprhs_func;

    delete AE_edofs_L2;
    for (int blk = 0; blk < AE_eintdofs_blocks->NumRowBlocks(); ++blk)
        delete &(AE_eintdofs_blocks->GetBlock(blk,blk));
    delete AE_eintdofs_blocks;

    delete xblock;
    delete yblock;

    if (optimized_localsolve)
        for (unsigned int i = 0; i < LUfactors.size(); ++i)
            for (unsigned int j = 0; j < LUfactors[i].size(); ++j)
                if (LUfactors[i][j])
                    delete LUfactors[i][j];
}

void LocalProblemSolver::Mult(const Vector &x, Vector &y, Vector * rhs_constr) const
{
    // x will be accessed through xblock as its view
    xblock->Update(x.GetData(), block_offsets);
    // y will be accessed through yblock as its view
    yblock->Update(y.GetData(), block_offsets);

    SolveTrueLocalProblems( *xblock, *yblock, rhs_constr);
}


void LocalProblemSolver::Setup()
{
    AE_edofs_L2 = mfem::Mult(AE_e, el_to_dofs_L2);
    AE_eintdofs_blocks = Get_AE_eintdofs(el_to_dofs_Op, essbdrdofs_blocks, bdrdofs_blocks);

    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

    temprhs_func = new BlockVector(Op_blkspmat.ColOffsets());
    tempsol = new BlockVector(Op_blkspmat.RowOffsets());
    //std:cout << "sol size = " << sol->Size() << "\n";

    // (optionally) saves LU factors related to the local problems to be solved
    // for each agglomerate element
    if (optimized_localsolve)
        SaveLocalLUFactors();

    finalized = true;

}

void LocalProblemSolver::SolveTrueLocalProblems(BlockVector& truerhs_func, BlockVector& truesol, Vector* localrhs_constr) const
{
    //BlockVector lrhs_func(Op_blkspmat.ColOffsets());
    for (int blk = 0; blk < numblocks; ++blk)
        d_td_blocks[blk]->Mult(truerhs_func.GetBlock(blk), temprhs_func->GetBlock(blk));
    //BlockVector lsol(Op_blkspmat.RowOffsets());
    *tempsol = 0.0;

    DenseMatrix sub_Constr;
    Vector sub_rhsconstr;
    Array<int> sub_Func_offsets(numblocks + 1);

    Array2D<const SparseMatrix *> Op_blks(numblocks, numblocks);
    Array2D<DenseMatrix*> LocalAE_Matrices(numblocks, numblocks);
    std::vector<SparseMatrix*> AE_eintdofs_blks(numblocks);
    std::vector<Array<int>*> Local_inds(numblocks);

    for ( int blk = 0; blk < numblocks; ++blk )
    {
        AE_eintdofs_blks[blk] = &(AE_eintdofs_blocks->GetBlock(blk,blk));
        Local_inds[blk] = new Array<int>();
    }

    // loop over all AE, solving a local problem in each AE
    int nAE = AE_edofs_L2->Height();
    for( int AE = 0; AE < nAE; ++AE)
    {
        //std::cout << "AE = " << AE << "\n";
        bool is_degenerate = true;
        sub_Func_offsets[0] = 0;
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            // no memory allocation here, it's just a viewer which is created. no valgrind suggests allocation here
            //Local_inds[blk] = new Array<int>(AE_eintdofs_blks[blk]->GetRowColumns(AE),
                                             //AE_eintdofs_blks[blk]->RowSize(AE));
            Local_inds[blk]->MakeRef(AE_eintdofs_blks[blk]->GetRowColumns(AE),
                                             AE_eintdofs_blks[blk]->RowSize(AE));

            if (blk == 0) // degeneracy comes from Constraint matrix which involves only sigma = the first block
            {
                for (int i = 0; i < Local_inds[blk]->Size(); ++i)
                {
                    if ( (*bdrdofs_blocks[blk])[(*Local_inds[blk])[i]] != 0 &&
                         (*essbdrdofs_blocks[blk])[(*Local_inds[blk])[i]] == 0)
                    {
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

                } // end of special treatment of the first block involved into constraint

                sub_Func_offsets[blk1 + 1] = sub_Func_offsets[blk1] + Local_inds[blk1]->Size();

                if (compute_AEproblem_matrices(blk1,blk2))
                    Op_blks(blk1,blk2) = &(Op_blkspmat.GetBlock(blk1,blk2));

                if (compute_AEproblem_matrices(blk1,blk2))
                {
                    // Extracting local problem matrices:
                    LocalAE_Matrices(blk1,blk2) = new DenseMatrix(Local_inds[blk1]->Size(), Local_inds[blk2]->Size());
                    Op_blks(blk1,blk2)->GetSubMatrix(*Local_inds[blk1], *Local_inds[blk2], *LocalAE_Matrices(blk1,blk2));

                } // end of the block for non-optimized version


            }
        } // end of loop over all blocks in the functional

        BlockVector sub_Func(sub_Func_offsets);

        for ( int blk = 0; blk < numblocks; ++blk )
            temprhs_func->GetBlock(blk).GetSubVector(*Local_inds[blk], sub_Func.GetBlock(blk));

        BlockVector sol_loc(sub_Func_offsets);
        sol_loc = 0.0;

        // solving local problem at the agglomerate element AE
        SolveLocalProblem(AE, LocalAE_Matrices, sub_Constr, sub_Func, sub_rhsconstr,
                          sol_loc, is_degenerate);
        // computing solution as a vector at current level
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            tempsol->GetBlock(blk).AddElementVector
                    (*Local_inds[blk], sol_loc.GetBlock(blk));
        }

        for ( int blk1 = 0; blk1 < numblocks; ++blk1 )
            for ( int blk2 = 0; blk2 < numblocks; ++blk2 )
                if (compute_AEproblem_matrices(blk1,blk2))
                    delete LocalAE_Matrices(blk1,blk2);

    } // end of loop over AEs

    for (int blk = 0; blk < numblocks; ++blk)
        d_td_blocks[blk]->MultTranspose(tempsol->GetBlock(blk), truesol.GetBlock(blk));

    for ( int blk = 0; blk < numblocks; ++blk )
        delete Local_inds[blk];

    return;

}

void LocalProblemSolver::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate) const
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

        // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
        if (is_degenerate)
        {
            Schur.SetRow(0,0);
            Schur.SetCol(0,0);
            Schur(0,0) = 1.;
        }

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
    MFEM_ASSERT(optimized_localsolve,
                "Error: Unnecessary saving of the LU factors for the local problems"
                " with optimized_localsolve deactivated \n");

    int nAE = AE_edofs_L2->Height();
    LUfactors.resize(nAE);

    DenseMatrix sub_Constr;
    DenseMatrix sub_Func;

    SparseMatrix * AE_eintdofs = &(AE_eintdofs_blocks->GetBlock(0,0));
    const SparseMatrix * Op_blk = &(Op_blkspmat.GetBlock(0,0));

    Array<int> * Local_inds = new Array<int>();

    // loop over all AE, computing and saving factorization
    // of local saddle point matrices in each AE
    for( int AE = 0; AE < nAE; ++AE)
    {
        // for each AE we will store A^(-1) and Schur^(-1)
        LUfactors[AE].resize(2);

        //std::cout << "AE = " << AE << "\n";
        bool is_degenerate = true;

        //Array<int> Local_inds(AE_eintdofs->GetRowColumns(AE), AE_eintdofs->RowSize(AE));
        Local_inds->MakeRef(AE_eintdofs->GetRowColumns(AE), AE_eintdofs->RowSize(AE));

        Array<int> Wtmp_j(AE_edofs_L2->GetRowColumns(AE), AE_edofs_L2->RowSize(AE));
        sub_Constr.SetSize(Wtmp_j.Size(), Local_inds->Size());
        Constr_spmat.GetSubMatrix(Wtmp_j, *Local_inds, sub_Constr);

        for (int i = 0; i < Local_inds->Size(); ++i)
        {
            if ( (*bdrdofs_blocks[0])[(*Local_inds)[i]] != 0 &&
                 (*essbdrdofs_blocks[0])[(*Local_inds)[i]] == 0)
            {
                //std::cout << "then local problem is non-degenerate \n";
                is_degenerate = false;
                break;
            }
        }

        // Setting size of Dense Matrices
        sub_Func.SetSize(Local_inds->Size());

        // Obtaining submatrices:
        Op_blk->GetSubMatrix(*Local_inds, *Local_inds, sub_Func);

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

    delete Local_inds;

    //compute_AEproblem_matrices = false;
    //compute_AEproblem_matrices(numblocks, numblocks) = true;
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

        SparseMatrix d_td_diag;
        SparseMatrix td_d_diag;
        HypreParMatrix * td_d;
        SparseMatrix td_d_offd;
        if (num_procs > 1)
        {
            // things from dof_truedof relation tables required to determine shared dofs
            d_td_blocks[blk]->GetDiag(d_td_diag);

            td_d = d_td_blocks[blk]->Transpose();
            td_d->GetDiag(td_d_diag);
            HYPRE_Int * cmap_td_d;
            td_d->GetOffd(td_d_offd, cmap_td_d);
        }

        // creating dofs_to_AE relation table
        SparseMatrix * tempprod = mfem::Mult(AE_e, *el_dofs_blk);
        SparseMatrix * dofs_AE = Transpose(*tempprod);
        delete tempprod;

        int ndofs = dofs_AE->Height();

        int * innerdofs_AE_i = new int [ndofs + 1];

        // computing the number of internal degrees of freedom in all AEs
        int nnz = 0;
        for (int dof = 0; dof < ndofs; ++dof)
        {
            innerdofs_AE_i[dof]= nnz;

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
            {
                nnz++;
            }

        }
        innerdofs_AE_i[ndofs] = nnz;

        // allocating j and data arrays for the created relation table
        int * innerdofs_AE_j = new int[nnz];
        double * innerdofs_AE_data = new double[nnz];

        int nnz_count = 0;
        for (int dof = 0; dof < ndofs; ++dof)
        {
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
        }

        std::fill_n(innerdofs_AE_data, nnz, 1);

        // creating a relation between internal fine-grid dofs (w.r.t to AE) and AEs,
        // keeeping zero rows for non-internal dofs
        SparseMatrix * innerdofs_AE = new SparseMatrix(innerdofs_AE_i, innerdofs_AE_j, innerdofs_AE_data,
                                                       dofs_AE->Height(), dofs_AE->Width());
        delete dofs_AE;

        res->SetBlock(blk, blk, Transpose(*innerdofs_AE));

        if (num_procs > 1)
            delete td_d;

        delete innerdofs_AE;

    } // end of the loop over blocks

    return res;
}

class LocalProblemSolverWithS : public LocalProblemSolver
{
    virtual void SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                                   BlockVector &G, Vector& F, BlockVector &sol,
                                   bool is_degenerate) const override;
    // Optimized version of SolveLocalProblem where LU factors for the local
    // problem's matrices were computed during the setup via SaveLocalLUFactors()
    void SolveLocalProblemOpt(DenseMatrixInverse * inv_AorAtilda,
                              DenseMatrixInverse * inv_C, DenseMatrixInverse * inv_Schur,
                              DenseMatrix& B, DenseMatrix &D, BlockVector &G, Vector& F,
                              BlockVector &sol, bool is_degenerate) const;
    // an optional routine which can save LU factors for the local problems
    // solved at finer levels if needed. Should be redefined in the inheriting
    // classes in order to speed up iterations
    virtual void SaveLocalLUFactors() const override;
public:
    // ~LocalProblemSolverWithS() : ~LocalProblemSolver() {} will call LocalProblemSolver destructor as I understand

    // main constructor
    LocalProblemSolverWithS(const BlockMatrix& Op_Blksmat,
                       const SparseMatrix& Constr_Spmat,
                       const std::vector<HypreParMatrix*>& D_tD_blks,
                       const SparseMatrix& AE_el,
                       const BlockMatrix& El_to_Dofs_Op,
                       const SparseMatrix& El_to_Dofs_L2,
                       const std::vector<Array<int>* >& BdrDofs_blks,
                       const std::vector<Array<int>* >& EssBdrDofs_blks,
                       bool Optimized_LocalSolve)
        : LocalProblemSolver(Op_Blksmat,
                              Constr_Spmat,
                              D_tD_blks,
                              AE_el,
                              El_to_Dofs_Op, El_to_Dofs_L2,
                              BdrDofs_blks, EssBdrDofs_blks,
                              false)
    {
        optimized_localsolve = Optimized_LocalSolve;
        compute_AEproblem_matrices.SetSize(numblocks + 1, numblocks + 1);
        compute_AEproblem_matrices = true;

        if (optimized_localsolve)
        {
            compute_AEproblem_matrices = false;
            compute_AEproblem_matrices(1,0) = true;
            compute_AEproblem_matrices(numblocks, numblocks) = true;

            SaveLocalLUFactors();
        }


    }

};

void LocalProblemSolverWithS::SaveLocalLUFactors() const
{
    MFEM_ASSERT(optimized_localsolve,
                "Error: Unnecessary saving of the LU factors for the local problems"
                " with optimized_localsolve deactivated \n");

    DenseMatrix sub_Constr;
    Array<int> sub_Func_offsets(numblocks + 1);

    Array2D<const SparseMatrix *> Op_blks(numblocks, numblocks);
    Array2D<DenseMatrix*> LocalAE_Matrices(numblocks, numblocks);
    std::vector<SparseMatrix*> AE_eintdofs_blks(numblocks);
    std::vector<Array<int>*> Local_inds(numblocks);

    for ( int blk = 0; blk < numblocks; ++blk )
    {
        AE_eintdofs_blks[blk] = &(AE_eintdofs_blocks->GetBlock(blk,blk));
        Local_inds[blk] = new Array<int>();
    }

    // loop over all AE, solving a local problem in each AE
    int nAE = AE_edofs_L2->Height();
    LUfactors.resize(nAE);

    for( int AE = 0; AE < nAE; ++AE)
    {
        // for each AE we will store A^(-1), Schur^(-1) or Atilda^(-1), C^(-1) and Schur^(-1)
        LUfactors[AE].resize(3);

        //std::cout << "AE = " << AE << "\n";
        bool is_degenerate = true;
        sub_Func_offsets[0] = 0;
        for ( int blk = 0; blk < numblocks; ++blk )
        {
            // no memory allocation here, it's just a viewer which is created. no valgrind suggests allocation here
            //Local_inds[blk] = new Array<int>(AE_eintdofs_blks[blk]->GetRowColumns(AE),
                                             //AE_eintdofs_blks[blk]->RowSize(AE));
            Local_inds[blk]->MakeRef(AE_eintdofs_blks[blk]->GetRowColumns(AE),
                                             AE_eintdofs_blks[blk]->RowSize(AE));

            if (blk == 0) // degeneracy comes from Constraint matrix which involves only sigma = the first block
            {
                for (int i = 0; i < Local_inds[blk]->Size(); ++i)
                {
                    if ( (*bdrdofs_blocks[blk])[(*Local_inds[blk])[i]] != 0 &&
                         (*essbdrdofs_blocks[blk])[(*Local_inds[blk])[i]] == 0)
                    {
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

                } // end of special treatment of the first block involved into constraint

                sub_Func_offsets[blk1 + 1] = sub_Func_offsets[blk1] + Local_inds[blk1]->Size();

                Op_blks(blk1,blk2) = &(Op_blkspmat.GetBlock(blk1,blk2));

                // Extracting local problem matrices:
                LocalAE_Matrices(blk1,blk2) = new DenseMatrix(Local_inds[blk1]->Size(), Local_inds[blk2]->Size());
                Op_blks(blk1,blk2)->GetSubMatrix(*Local_inds[blk1], *Local_inds[blk2], *LocalAE_Matrices(blk1,blk2));

            }
        } // end of loop over all blocks in the functional


        if (Local_inds[1]->Size() == 0) // this means no internal dofs for S in the current AE
        {
            // then only a 2x2 block system is to be solved

            LUfactors[AE][0] = new DenseMatrixInverse(*LocalAE_Matrices(0,0));
            LUfactors[AE][1] = NULL;

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

            LUfactors[AE][2] = new DenseMatrixInverse(Schur);

        }
        else // then it is 3x3 block matrix under consideration
        {
            LUfactors[AE][1] = new DenseMatrixInverse(*LocalAE_Matrices(1,1));

            // creating D * inv_C * DT
            DenseMatrix invCD;
            LUfactors[AE][1]->Mult(*LocalAE_Matrices(1,0), invCD);

            DenseMatrix DTinvCD(Local_inds[0]->Size(), Local_inds[0]->Size());
            mfem::Mult(*LocalAE_Matrices(0,1), invCD, DTinvCD);

            // creating inv Atilda = inv(A - D * inv_C * DT)
            DenseMatrix Atilda(Local_inds[0]->Size(), Local_inds[0]->Size());
            Atilda = *LocalAE_Matrices(0,0);
            Atilda -= DTinvCD;

            LUfactors[AE][0] = new DenseMatrixInverse(Atilda);

            // computing Schur = B * inv_Atilda * BT
            DenseMatrix inv_AtildaBT;
            DenseMatrix sub_ConstrT(sub_Constr.Width(), sub_Constr.Height());
            sub_ConstrT.Transpose(sub_Constr);
            LUfactors[AE][0]->Mult(sub_ConstrT, inv_AtildaBT);

            DenseMatrix Schur(sub_Constr.Height(), sub_Constr.Height());
            mfem::Mult(sub_Constr, inv_AtildaBT, Schur);

            // getting rid of the one-dimensional kernel which exists
            // for lambda if the problem is degenerate
            if (is_degenerate)
            {
                Schur.SetRow(0,0);
                Schur.SetCol(0,0);
                Schur(0,0) = 1.;
            }

            LUfactors[AE][2] = new DenseMatrixInverse(Schur);
        }

        for ( int blk1 = 0; blk1 < numblocks; ++blk1 )
            for ( int blk2 = 0; blk2 < numblocks; ++blk2 )
                delete LocalAE_Matrices(blk1, blk2);

    } // end of loop over AEs

    for ( int blk = 0; blk < numblocks; ++blk )
        delete Local_inds[blk];

    return;
}

void LocalProblemSolverWithS::SolveLocalProblem(int AE, Array2D<DenseMatrix*> &FunctBlks, DenseMatrix& B,
                               BlockVector &G, Vector& F, BlockVector &sol,
                               bool is_degenerate) const
{
    if (optimized_localsolve)
    {
        //MFEM_ABORT("Optimized local problem solving routine was not implemented yet \n");
        DenseMatrixInverse * inv_AorAtilda = LUfactors[AE][0];
        DenseMatrixInverse * inv_C = LUfactors[AE][1];
        DenseMatrixInverse * inv_Schur = LUfactors[AE][2];
        DenseMatrix * D = FunctBlks(1,0);
        SolveLocalProblemOpt(inv_AorAtilda, inv_C, inv_Schur, B, *D, G, F, sol, is_degenerate);
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

            // getting rid of the one-dimensional kernel which exists for lambda if the problem is degenerate
            if (is_degenerate)
            {
                Schur.SetRow(0,0);
                Schur.SetCol(0,0);
                Schur(0,0) = 1.;
            }

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

    }

    return;
}

// Optimized version of SolveLocalProblem where LU factors for the local
// problem's matrices were computed during the setup via SaveLocalLUFactors()
void LocalProblemSolverWithS::SolveLocalProblemOpt(DenseMatrixInverse * inv_AorAtilda,
                                                   DenseMatrixInverse * inv_C, DenseMatrixInverse * inv_Schur,
                                                   DenseMatrix& B, DenseMatrix& D, BlockVector &G,
                                                   Vector& F, BlockVector &sol, bool is_degenerate) const
{
    Vector lambda(B.Height());

    if (G.GetBlock(1).Size() == 0) // this means no internal dofs for S in the current AE
    {
        // invAG = invA * G
        Vector invAG;
        inv_AorAtilda->Mult(G, invAG);

        DenseMatrix BT(B.Width(), B.Height());
        BT.Transpose(B);

        DenseMatrix invABT;
        inv_AorAtilda->Mult(BT, invABT);

        // temp = ( B * invA * G - F )
        Vector temp(B.Height());
        B.Mult(invAG, temp);
        temp -= F;

        if (is_degenerate)
            temp(0) = 0;

        // lambda = inv(BinvABT) * ( B * invA * G - F )
        inv_Schur->Mult(temp, lambda);

        // temp2 = (G - BT * lambda)
        Vector temp2(B.Width());
        B.MultTranspose(lambda,temp2);
        temp2 *= -1;
        temp2 += G;

        // sig = invA * temp2 = invA * (G - BT * lambda)
        inv_AorAtilda->Mult(temp2, sol.GetBlock(0));
    }
    else // then it is 3x3 block matrix under consideration
    {

        /*
        // creating D * inv_C * DT
        DenseMatrix invCD;
        inv_C->Mult(*D, invCD);

        DenseMatrix DTinvCD(DT->Height(), D->Width());
        mfem::Mult(*DT, invCD, DTinvCD);

        // computing Schur = B * inv_Atilda * BT
        DenseMatrix inv_AtildaBT;
        DenseMatrix BT(B.Width(), B.Height());
        BT.Transpose(B);
        inv_Atilda->Mult(BT, inv_AtildaBT);
        */


        // creating DT * invC * F_S
        Vector invCF2;
        inv_C->Mult(G.GetBlock(1), invCF2);

        Vector DTinvCF2(D.Width());
        D.MultTranspose(invCF2, DTinvCF2);

        // creating F1tilda = F_sigma - DT * invC * F_S
        Vector F1tilda(D.Width());
        F1tilda = G.GetBlock(0);
        F1tilda -= DTinvCF2;

        // creating invAtildaFtilda = inv(Atilda) * Ftilda =
        // = inv(A - D * inv_C * DT) * (F_sigma - DT * invC * F_S)
        Vector invAtildaFtilda(G.GetBlock(0).Size());
        inv_AorAtilda->Mult(F1tilda, invAtildaFtilda);

        Vector FinalFlam(B.Height());
        B.Mult(invAtildaFtilda, FinalFlam);
        FinalFlam -= F;

        if (is_degenerate)
            FinalFlam(0) = 0;

        // lambda = inv_Schur * ( F_lam - B * inv(A - D * inv_C * DT) * (F_sigma - DT * invC * F_S) )
        inv_Schur->Mult(FinalFlam, lambda);

        // changing Ftilda so that Ftilda_new = Ftilda_old - BT * lambda
        // = F_sigma - DT * invC * F_S - BT * lambda
        Vector temp(B.Width());
        B.MultTranspose(lambda, temp);
        F1tilda -= temp;

        // sigma = inv_Atilda * Ftilda(new)
        // = inv(A - D * inv_C * DT) * ( F_sigma - DT * invC * F_S - BT * lambda )
        inv_AorAtilda->Mult(F1tilda, sol.GetBlock(0));

        // temp2 = F_S - D * sigma
        Vector temp2(D.Height());
        D.Mult(sol.GetBlock(0), temp2);
        temp2 *= -1.0;
        temp2 += G.GetBlock(1);

        inv_C->Mult(temp2, sol.GetBlock(1));
    }
}


// class for finding a particular solution to a divergence constraint
class DivConstraintSolver : public Solver
{
private:
    // a flag which indicates whether the solver setup was called
    // before trying to solve anything
    mutable bool setup_finished;

protected:
    mutable int print_level;

    int num_levels;

    // Relation tables which represent agglomerated elements-to-elements relation at each level
    const Array< SparseMatrix*>& AE_e;

    const MPI_Comm comm;

    // Projectors for the variables related to the functional and constraint
    const Array< BlockOperator*>& TrueP_Func;
    const Array< SparseMatrix*>& P_L2; // used for operators coarsening and in ComputeLocalRhsConstr (via ProjectFinerL2ToCoarser)

    // for each level and for each variable in the functional stores a vector
    // which defines if a dof is at the boundary / essential part of the boundary
    // or not
    const std::vector<std::vector<Array<int>* > > & essbdrtruedofs_Func;

    // parts of block structure which define the Functional at the finest level
    const int numblocks;

    // righthand side of  the divergence contraint on dofs (= on true dofs for L2)
    const Vector& ConstrRhs;

    const Array<Operator*>& Smoothers_lvls;

    // a given blockvector (on true dofs) which satisfies
    // essential bdr conditions imposed for the initial problem
    const BlockVector& bdrdata_truedofs;

    const std::vector<Operator*> & Func_global_lvls;
    const HypreParMatrix & Constr_global;

    // stores Functional matrix on all levels except the finest
    // so that Funct_levels[0] = Functional matrix on level 1 (not level 0!)

    // The same as xblock and yblock but on true dofs
    mutable BlockVector* xblock_truedofs;
    mutable BlockVector* yblock_truedofs;
    mutable BlockVector* tempblock_truedofs;

    mutable Array<BlockVector*> truetempvec_lvls;
    mutable Array<BlockVector*> truetempvec2_lvls;
    mutable Array<BlockVector*> trueresfunc_lvls;
    mutable Array<BlockVector*> truesolupdate_lvls;

    mutable Array<LocalProblemSolver*> LocalSolvers_lvls;
    mutable CoarsestProblemSolver* CoarseSolver;

#ifdef CHECK_CONSTR
    mutable Vector * Constr_rhs_global;
#endif


    // Allocates current level-related data and computes coarser matrices for the functional
    // and the constraint.
    // Called only during the SetUpSolver()
    virtual void SetUpFinerLvl(int lvl) const;

    virtual void Setup(bool verbose = false) const;

    virtual void MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector& rhs_l) const;

    // Computes rhs in the constraint for the finer levels (~ Q_l f - Q_lminus1 f)
    // Should be called only during the first solver iterate (since it must be 0 at the next)
    void ComputeLocalRhsConstr(int level, Vector &Qlminus1_f, Vector &rhs_constr, Vector &workfvec) const;

    void ProjectFinerL2ToCoarser(int level, const Vector& in, Vector &ProjTin, Vector &out) const;

    void UpdateTrueResidual(int level, const BlockVector* rhs_l,  const BlockVector& solupd_l, BlockVector& out_l) const;

public:
    ~DivConstraintSolver();
    DivConstraintSolver(MPI_Comm Comm, int NumLevels,
                           const Array< SparseMatrix*> &AE_to_e,
                           const Array< BlockOperator*>& TrueProj_Func,
                           const Array< SparseMatrix*> &Proj_L2,
                           const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                           const std::vector<Operator*> & Func_Global_lvls,
                           const HypreParMatrix & Constr_Global,
                           const Vector& ConstrRhsVec,
                           const Array<Operator*>& Smoothers_Lvls,
                           const BlockVector& Bdrdata_TrueDofs,
#ifdef CHECK_CONSTR
                           Vector & Constr_Rhs_global,
#endif
                           Array<LocalProblemSolver*>* LocalSolvers,
                           CoarsestProblemSolver* CoarsestSolver);

    // Operator application: `y=A(x)`.
    virtual void Mult(const Vector &x, Vector &y) const
    {
        // x and y will be accessed through its viewers
        xblock_truedofs->Update(x.GetData(), TrueP_Func[0]->RowOffsets());
        yblock_truedofs->Update(y.GetData(), TrueP_Func[0]->RowOffsets());

        FindParticularSolution(*xblock_truedofs, *yblock_truedofs, ConstrRhs, print_level);
    }

    // existence of this method is required by the (abstract) base class Solver
    virtual void SetOperator(const Operator &op) override{}

    void FindParticularSolution(const BlockVector& truestart_guess, BlockVector& particular_solution, const Vector& ConstrRhs, bool verbose) const;

    // have to define these to mimic useful routines from IterativeSolver class
    void SetPrintLevel(int PrintLevel) const {print_level = PrintLevel;}

};

DivConstraintSolver::~DivConstraintSolver()
{
    delete xblock_truedofs;
    delete yblock_truedofs;
    delete tempblock_truedofs;

    for (int i = 0; i < truetempvec_lvls.Size(); ++i)
        delete truetempvec_lvls[i];
    for (int i = 0; i < truetempvec2_lvls.Size(); ++i)
        delete truetempvec2_lvls[i];
    for (int i = 0; i < trueresfunc_lvls.Size(); ++i)
        delete trueresfunc_lvls[i];
    for (int i = 0; i < truesolupdate_lvls.Size(); ++i)
        delete truesolupdate_lvls[i];
}

DivConstraintSolver::DivConstraintSolver(MPI_Comm Comm, int NumLevels,
                       const Array< SparseMatrix*> &AE_to_e,
                       const Array< BlockOperator*>& TrueProj_Func,
                       const Array< SparseMatrix*> &Proj_L2,
                       const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                       const std::vector<Operator*> & Func_Global_lvls,
                       const HypreParMatrix & Constr_Global,
                       const Vector& ConstrRhsVec,
                       const Array<Operator*>& Smoothers_Lvls,
                       const BlockVector& Bdrdata_TrueDofs,
#ifdef CHECK_CONSTR
                       Vector & Constr_Rhs_global,
#endif
                       Array<LocalProblemSolver*>* LocalSolvers,
                       CoarsestProblemSolver* CoarsestSolver)
     : Solver(Func_Global_lvls[0]->Height(), Func_Global_lvls[0]->Width()),
       setup_finished(false),
       num_levels(NumLevels),
       AE_e(AE_to_e),
       comm(Comm),
       TrueP_Func(TrueProj_Func), P_L2(Proj_L2),
       essbdrtruedofs_Func(EssBdrTrueDofs_Func),
       numblocks(TrueProj_Func[0]->NumRowBlocks()),
       ConstrRhs(ConstrRhsVec),
       Smoothers_lvls(Smoothers_Lvls),
       bdrdata_truedofs(Bdrdata_TrueDofs)
       , Func_global_lvls(Func_Global_lvls),
       Constr_global(Constr_Global)
#ifdef CHECK_CONSTR
       ,Constr_rhs_global(&Constr_Rhs_global)
#endif
{

    xblock_truedofs = new BlockVector(TrueP_Func[0]->RowOffsets());
    yblock_truedofs = new BlockVector(TrueP_Func[0]->RowOffsets());
    tempblock_truedofs = new BlockVector(TrueP_Func[0]->RowOffsets());

    truesolupdate_lvls.SetSize(num_levels);
    truesolupdate_lvls[0] = new BlockVector(TrueP_Func[0]->RowOffsets());

    truetempvec_lvls.SetSize(num_levels);
    truetempvec_lvls[0] = new BlockVector(TrueP_Func[0]->RowOffsets());
    truetempvec2_lvls.SetSize(num_levels);
    truetempvec2_lvls[0] = new BlockVector(TrueP_Func[0]->RowOffsets());
    trueresfunc_lvls.SetSize(num_levels);
    trueresfunc_lvls[0] = new BlockVector(TrueP_Func[0]->RowOffsets());

    if (CoarsestSolver)
        CoarseSolver = CoarsestSolver;
    else
        CoarseSolver = NULL;

    SetPrintLevel(0);

    LocalSolvers_lvls.SetSize(num_levels - 1);
    for (int l = 0; l < num_levels - 1; ++l)
        if (LocalSolvers)
            LocalSolvers_lvls[l] = (*LocalSolvers)[l];
        else
            LocalSolvers_lvls[l] = NULL;

    Setup();
}


// the start_guess is on dofs
// (*) returns particular solution as a vector on true dofs!
void DivConstraintSolver::FindParticularSolution(const BlockVector& truestart_guess,
                                                         BlockVector& particular_solution, const Vector &constr_rhs, bool verbose) const
{
    // checking if the given initial vector satisfies the divergence constraint
    Vector rhs_constr(Constr_global.Height());
    Constr_global.Mult(truestart_guess.GetBlock(0), rhs_constr);
    rhs_constr -= constr_rhs;
    rhs_constr *= -1.0;
    // 3.1 if not, computing the particular solution
    if ( ComputeMPIVecNorm(comm, rhs_constr,"", verbose) > 1.0e-14 )
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

#ifdef CHECK_BNDCND
    for (int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(CheckBdrError(truestart_guess.GetBlock(blk), &(bdrdata_truedofs.GetBlock(blk)), *essbdrtruedofs_Func[0][blk], true),
                              "for the initial guess");
    }
#endif

    // variable-size vectors (initialized with the finest level sizes) on dofs
    Vector Qlminus1_f(rhs_constr.Size());     // stores P_l^T rhs_constr_l
    Vector workfvec(rhs_constr.Size());       // used only in ComputeLocalRhsConstr()

    // 0. Compute rhs in the functional for the finest level
    UpdateTrueResidual(0, NULL, truestart_guess, *trueresfunc_lvls[0] );

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
            //std::cout << "l = " << l << "\n";
            //std::cout << "tempvec_l = " << truetempvec_lvls[l] << ", tempvec2_l = " << truetempvec2_lvls[l] << "\n";
            MPI_Barrier(comm);
            Smoothers_lvls[l]->Mult(*truetempvec_lvls[l], *truetempvec2_lvls[l] );

            //truetempvec2_lvls[l]->Print();

            *truesolupdate_lvls[l] += *truetempvec2_lvls[l];
#ifdef CHECK_CONSTR
            if (l == 0)
                CheckConstrRes(truetempvec2_lvls[l]->GetBlock(0), Constr_global, NULL, "for the smoother level 0 update");
#endif
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

    //trueresfunc_lvls[num_levels - 1]->Print();

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

#ifdef CHECK_CONSTR
    CheckConstrRes(particular_solution.GetBlock(0), Constr_global,
                    Constr_rhs_global, "for the particular solution inside in the end");
#endif

}

void DivConstraintSolver::Setup(bool verbose) const
{
    if (verbose)
        std::cout << "Starting solver setup \n";

    // 1. copying the given initial vector to the internal variable
    CheckFunctValue(comm, *Func_global_lvls[0], NULL, bdrdata_truedofs,
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

void DivConstraintSolver::MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector &rhs_l) const
{
    Func_global_lvls[l]->Mult(x_l, rhs_l);
    rhs_l *= coeff;
}

// Computes prerequisites required for solving local problems at level l
// such as relation tables between AEs and internal fine-grid dofs
// and maybe smth else ... ?
void DivConstraintSolver::SetUpFinerLvl(int lvl) const
{
    truetempvec_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    truetempvec2_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    truesolupdate_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    trueresfunc_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
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

    // We need either to use additional memory for storing
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

class HcurlGSSSmoother : public BlockOperator
{
private:
    int numblocks;

    // number of GS sweeps for each block
    mutable Array<int> sweeps_num;

    int print_level;

protected:
    const MPI_Comm comm;

#if defined NEW_SMOOTHERSETUP
    mutable Array2D<HypreParMatrix*> * Funct_hpmat;
    mutable HypreParMatrix* Divfree_hpmat_nobnd;
#endif

    const BlockMatrix* Funct_mat;

    // discrete curl operator;
    const SparseMatrix* Curlh;

#ifdef MEMORY_OPTIMIZED
    mutable Vector* temp_Hdiv_dofs;
    mutable Vector* temp_Hcurl_dofs;
#else
    // global discrete curl operator
    mutable HypreParMatrix* Divfree_hpmat;
#endif

#ifdef DEBUG_SMOOTHER
    mutable HypreParMatrix * Constr_global;
#endif

#ifdef TIMING
    mutable StopWatch chrono;
    mutable StopWatch chrono2;
    mutable double time_beforeintmult;
    mutable double time_intmult;
    mutable double time_afterintmult;
    mutable double time_globalmult;
#endif

    // Projection of the system matrix M onto discrete Hcurl space
    // stores Curl_hT * M * Curlh
    //mutable SparseMatrix* CTMC;

    // global CTMC as HypreParMatrix
    mutable HypreParMatrix* CTMC_global;

    // stores global Funct blocks for all blocks except row = 0 or col = 0
    mutable Array2D<HypreParMatrix *> Funct_restblocks_global;

    // structures used when all dofs are relaxed (via HypreSmoothers):
    mutable Array<Operator*> Smoothers;
    mutable Array<int> trueblock_offsets; //block offsets for H(curl) x other blocks problems on true dofs
    mutable BlockVector* truerhs;  // rhs for H(curl) x other blocks problems on true dofs
    mutable BlockVector* truex;    // sol for H(curl) x other blocks problems on true dofs

    // Dof_TrueDof table for Hcurl
    const HypreParMatrix * d_td_Hcurl;

    // Dof_TrueDof table for the functional blocks
    const std::vector<HypreParMatrix*> d_td_Funct_blocks;

    // Lists of essential boundary dofs and true dofs for Hcurl
    const Array<int> * essbdrdofs_Hcurl;
    const Array<int> & essbdrtruedofs_Hcurl;

    // Lists of essential boundary dofs and true dofs for functional variables
    const std::vector<Array<int>*> essbdrdofs_Funct;
    const std::vector<Array<int>*> & essbdrtruedofs_Funct;

    // block structure (on true dofs)
    mutable Array<int> block_offsets;
    mutable BlockVector * xblock;
    mutable BlockVector * yblock;

public:
    ~HcurlGSSSmoother();
#ifdef NEW_SMOOTHERSETUP
    HcurlGSSSmoother (Array2D<HypreParMatrix*> & Funct_HpMat,
                                        HypreParMatrix& Divfree_HpMat_nobnd,
                                        const Array<int>& EssBdrtruedofs_Hcurl,
                                        const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                        const Array<int> * SweepsNum,
                                        const Array<int>& Block_Offsets);
#endif
    // constructor
    HcurlGSSSmoother (const BlockMatrix& Funct_Mat,
                     const SparseMatrix& Discrete_Curl,
                     const HypreParMatrix& Dof_TrueDof_Hcurl,
                     const std::vector<HypreParMatrix*> & Dof_TrueDof_Funct,
                     const Array<int>& EssBdrdofs_Hcurl, const Array<int> &EssBdrtruedofs_Hcurl,
                     const std::vector<Array<int>* >& EssBdrDofs_Funct,
                     const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                     const Array<int> * SweepsNum, const Array<int> &Block_Offsets);

    virtual void Setup() const;

    // Operator application
    virtual void Mult (const Vector & x, Vector & y) const;

    // Action of the transpose operator
    virtual void MultTranspose (const Vector & x, Vector & y) const {Mult(x,y);}

    // service routines
    int GetSweepsNumber(int block) const {return sweeps_num[block];}
#ifdef TIMING
    void ResetInternalTimings() const;

    double GetGlobalMultTime() {return time_globalmult;}
    double GetInternalMultTime() {return time_intmult;}
    double GetBeforeIntMultTime() {return time_beforeintmult;}
    double GetAfterIntMultTime() {return time_afterintmult;}
#endif

};

#ifdef TIMING
void HcurlGSSSmoother::ResetInternalTimings() const
{
    time_beforeintmult = 0.0;
    time_intmult = 0.0;
    time_afterintmult = 0.0;
    time_globalmult = 0.0;
}
#endif

HcurlGSSSmoother::~HcurlGSSSmoother()
{
    delete xblock;
    delete yblock;
    delete truerhs;
    delete truex;

    //delete CTMC;
    delete CTMC_global;

#if defined NEW_SMOOTHERSETUP
#else
    for (int rowblk = 0; rowblk < Funct_restblocks_global.NumRows(); ++rowblk)
        for (int colblk = 0; colblk < Funct_restblocks_global.NumCols(); ++colblk)
            if (Funct_restblocks_global(rowblk,colblk))
                delete Funct_restblocks_global(rowblk,colblk);
#ifdef MEMORY_OPTIMIZED
    delete temp_Hdiv_dofs;
    delete temp_Hcurl_dofs;
#else
    delete Curlh_global;
#endif

#endif

    for (int i = 0; i < Smoothers.Size(); ++i)
        delete Smoothers[i];

}

HcurlGSSSmoother::HcurlGSSSmoother (const BlockMatrix& Funct_Mat,
                                    const SparseMatrix& Discrete_Curl,
                                    const HypreParMatrix& Dof_TrueDof_Hcurl,
                                    const std::vector<HypreParMatrix *> &Dof_TrueDof_Funct,
                                    const Array<int>& EssBdrdofs_Hcurl,
                                    const Array<int>& EssBdrtruedofs_Hcurl,
                                    const std::vector<Array<int>* >& EssBdrDofs_Funct,
                                    const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                    const Array<int> * SweepsNum,
                                    const Array<int>& Block_Offsets)
    : BlockOperator(Block_Offsets),
      numblocks(Funct_Mat.NumRowBlocks()),
      print_level(0),
      comm(Dof_TrueDof_Hcurl.GetComm()),
      Funct_mat(&Funct_Mat),
      d_td_Hcurl(&Dof_TrueDof_Hcurl),
      d_td_Funct_blocks(Dof_TrueDof_Funct),
      essbdrdofs_Hcurl(&EssBdrdofs_Hcurl),
      essbdrtruedofs_Hcurl(EssBdrtruedofs_Hcurl),
      essbdrdofs_Funct(EssBdrDofs_Funct),
      essbdrtruedofs_Funct(EssBdrTrueDofs_Funct)
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

    Funct_restblocks_global.SetSize(numblocks, numblocks);
    for (int rowblk = 0; rowblk < numblocks; ++rowblk)
        Funct_restblocks_global(rowblk,0) = NULL;
    for (int colblk = 0; colblk < numblocks; ++colblk)
        Funct_restblocks_global(0,colblk) = NULL;

    Setup();
}

#ifdef NEW_SMOOTHERSETUP

HcurlGSSSmoother::HcurlGSSSmoother (Array2D<HypreParMatrix*> & Funct_HpMat,
                                    HypreParMatrix& Divfree_HpMat_nobnd,
                                    const Array<int>& EssBdrtruedofs_Hcurl,
                                    const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                    const Array<int> * SweepsNum,
                                    const Array<int>& Block_Offsets)
    : BlockOperator(Block_Offsets),
      numblocks(Funct_HpMat.NumRows()),
      print_level(0),
      comm(Divfree_HpMat_nobnd.GetComm()),
      Funct_hpmat (&Funct_HpMat),
      Divfree_hpmat_nobnd (&Divfree_HpMat_nobnd),
      essbdrtruedofs_Hcurl(EssBdrtruedofs_Hcurl),
      essbdrtruedofs_Funct(EssBdrTrueDofs_Funct)
{

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
            trueblock_offsets[1] = Divfree_hpmat_nobnd->Width();
        else
            trueblock_offsets[blk + 1] = (*Funct_hpmat)(blk,blk)->Height();
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
#endif

void HcurlGSSSmoother::Mult(const Vector & x, Vector & y) const
{
#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Clear();
    chrono2.Start();
#endif

    if (print_level)
        std::cout << "Smoothing with HcurlGSS smoother \n";

    //if (x.GetData() == y.GetData() && x.GetData() != NULL)
        //mfem_error("Error in HcurlGSSSmoother::Mult(): x and y can't point to the same data \n");

    // x will be accessed through xblock as its view
    // y will be accessed through yblock as its view
    xblock->Update(x.GetData(), block_offsets);
    yblock->Update(y.GetData(), block_offsets);

    //xblock = new BlockVector(x.GetData(), block_offsets);
    //yblock = new BlockVector(y.GetData(), block_offsets);

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp = essbdrtruedofs_Funct[blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            xblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
    }

#ifdef CHECK_BNDCND
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp = essbdrtruedofs_Funct[blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(xblock->GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for xblock! blk = " << blk << ", value = "
                          << xblock->GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }
#endif


#ifdef NEW_SMOOTHERSETUP
    for (int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            Divfree_hpmat_nobnd->MultTranspose(xblock->GetBlock(0), truerhs->GetBlock(0));
        else
            truerhs->GetBlock(blk) = xblock->GetBlock(blk);

        // imposing boundary conditions on the righthand side
        if (blk == 0) // in Hcurl
            for ( int tdofind = 0; tdofind < essbdrtruedofs_Hcurl.Size(); ++tdofind)
            {
                int tdof = essbdrtruedofs_Hcurl[tdofind];
                truerhs->GetBlock(0)[tdof] = 0.0;
#ifdef CHECK_BNDCND
                if (fabs(truerhs->GetBlock(blk)[tdof]) > 1.0e-14 )
                    std::cout << "bnd cnd is violated for truerhs! blk = " << blk << ", value = "
                              << truerhs->GetBlock(blk)[tdof]
                              << ", index = " << tdof << "\n";
#endif
                //truerhs->GetBlock(0)[tdof] = 0.0;
            }
#ifdef CHECK_BNDCND
        else
            for ( int tdofind = 0; tdofind < essbdrtruedofs_Funct[blk]->Size(); ++tdofind)
            {
                int tdof = (*essbdrtruedofs_Funct[blk])[tdofind];
                if (fabs(truerhs->GetBlock(blk)[tdof]) > 1.0e-14 )
                    std::cout << "bnd cnd is violated for truerhs! blk = " << blk << ", value = "
                              << truerhs->GetBlock(blk)[tdof]
                              << ", index = " << tdof << "\n";
                //truerhs->GetBlock(blk)[tdof] = 0.0;
            }
#endif
    }

    *truex = 0.0;

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_beforeintmult += chrono.RealTime();
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    //truerhs->GetBlock(0).Print();

    for ( int blk = 0; blk < numblocks; ++blk)
        Smoothers[blk]->Mult(truerhs->GetBlock(blk), truex->GetBlock(blk));

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_intmult += chrono.RealTime();
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    for (int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0) // in Hcurl
            for ( int tdofind = 0; tdofind < essbdrtruedofs_Hcurl.Size(); ++tdofind)
            {
                int tdof = essbdrtruedofs_Hcurl[tdofind];
                truex->GetBlock(blk)[tdof] = 0.0;
#ifdef CHECK_BNDCND
                if (fabs(truex->GetBlock(blk)[tdof]) > 1.0e-14 )
                    std::cout << "bnd cnd is violated for truex! blk = " << blk << ", value = "
                              << truex->GetBlock(blk)[tdof]
                              << ", index = " << tdof << "\n";
#endif
            }
#ifdef CHECK_BNDCND
        else
            for ( int tdofind = 0; tdofind < essbdrtruedofs_Funct[blk]->Size(); ++tdofind)
            {
                int tdof = (*essbdrtruedofs_Funct[blk])[tdofind];
                if (fabs(truex->GetBlock(blk)[tdof]) > 1.0e-14 )
                    std::cout << "bnd cnd is violated for truex! blk = " << blk << ", value = "
                              << truex->GetBlock(blk)[tdof]
                              << ", index = " << tdof << "\n";
            }
#endif
    }
    //truex->Print();
    //std::cout << "debug print \n";

    // computing the solution update in the H(div) x other blocks space
    // in two steps:

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0) // first component should be transferred from Hcurl to Hdiv
            Divfree_hpmat_nobnd->Mult(truex->GetBlock(0), yblock->GetBlock(0));
        else
            yblock->GetBlock(blk) = truex->GetBlock(blk);
    }

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp = essbdrtruedofs_Funct[blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            yblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
#ifdef CHECK_BNDCND
            if ( fabs(yblock->GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for yblock! blk = " << blk << ", value = "
                          << yblock->GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
#endif
        }
    }

    //yblock->GetBlock(0).Print();


#else // for NEW_SMOOTHERSETUP
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

            d_td_Hcurl->MultTranspose(*temp_Hcurl_dofs, truerhs->GetBlock(0));
#else
            std::cout << "xblock(0) size = " << xblock->GetBlock(0).Size() << "\n";
            Curlh_global->MultTranspose(xblock->GetBlock(0), truerhs->GetBlock(0));
#endif
        }
        else
        {
            truerhs->GetBlock(blk) = xblock->GetBlock(blk);
        }

    }

    // imposing boundary conditions in Hcurl on the righthand side (block 0)
    for ( int tdofind = 0; tdofind < essbdrtruedofs_Hcurl.Size(); ++tdofind)
    {
        int tdof = essbdrtruedofs_Hcurl[tdofind];
        truerhs->GetBlock(0)[tdof] = 0.0;
    }

    // imposing boundary conditions for the rest of the blocks in the righthand side
    for ( int blk = 1; blk < numblocks; ++blk)
        for ( int tdofind = 0; tdofind < essbdrtruedofs_Funct[blk]->Size(); ++tdofind)
        {
            int tdof = (*essbdrtruedofs_Funct[blk])[tdofind];
            truerhs->GetBlock(blk)[tdof] = 0.0;
        }

    *truex = 0.0;

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_beforeintmult += chrono.RealTime();
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        Smoothers[blk]->Mult(truerhs->GetBlock(blk), truex->GetBlock(blk));
    }

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_intmult += chrono.RealTime();
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    // imposing boundary conditions in Hcurl on the internal solver solution (block 0)
    for ( int tdofind = 0; tdofind < essbdrtruedofs_Hcurl.Size(); ++tdofind)
    {
        int tdof = essbdrtruedofs_Hcurl[tdofind];
        truex->GetBlock(0)[tdof] = 0.0;
    }

    // imposing boundary conditions for the rest of the blocks in the righthand side
    for ( int blk = 1; blk < numblocks; ++blk)
        for ( int tdofind = 0; tdofind < essbdrtruedofs_Funct[blk]->Size(); ++tdofind)
        {
            int tdof = (*essbdrtruedofs_Funct[blk])[tdofind];
            truex->GetBlock(blk)[tdof] = 0.0;
        }

    // computing the solution update in the H(div) x other blocks space
    // in two steps:


    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0) // first component should be transferred from Hcurl to Hdiv
        {
#ifdef MEMORY_OPTIMIZED
            d_td_Hcurl->Mult(truex->GetBlock(0), *temp_Hcurl_dofs);
            Curlh->Mult(*temp_Hcurl_dofs, *temp_Hdiv_dofs);

            SparseMatrix d_td_Hdiv_diag;
            d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
            d_td_Hdiv_diag.MultTranspose(*temp_Hdiv_dofs, yblock->GetBlock(0));
#else
            Curlh_global->Mult(truex->GetBlock(0), yblock->GetBlock(0));
#endif
        }
        else
            yblock->GetBlock(blk) = truex->GetBlock(blk);
    }

#ifdef CHECK_BNDCND
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp = essbdrtruedofs_Funct[blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            if ( fabs(yblock->GetBlock(blk)[(*temp)[tdofind]]) > 1.0e-14 )
                std::cout << "bnd cnd is violated for yblock! blk = " << blk << ", value = "
                          << yblock->GetBlock(blk)[(*temp)[tdofind]]
                          << ", index = " << (*temp)[tdofind] << "\n";
        }
    }
#endif

#endif // for NEW_SMOOTHERSETUP


#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_afterintmult += chrono.RealTime();

    MPI_Barrier(comm);
    chrono2.Stop();
    time_globalmult += chrono2.RealTime();
#endif

}

void HcurlGSSSmoother::Setup() const
{
    MFEM_ASSERT(numblocks <= 2, "HcurlGSSSmoother::Setup was implemented for the "
                                "cases numblocks = 1 or 2 only \n");

#ifdef NEW_SMOOTHERSETUP
    CTMC_global = RAP(Divfree_hpmat_nobnd, (*Funct_hpmat)(0,0), Divfree_hpmat_nobnd);
    CTMC_global->CopyRowStarts();
    CTMC_global->CopyColStarts();

    /*
    SparseMatrix diagg;
    CTMC_global->GetDiag(diagg);
    diagg.EliminateZeroRows();
    */

    Smoothers[0] = new HypreSmoother(*CTMC_global, HypreSmoother::Type::l1GS, sweeps_num[0]);
    if (numblocks > 1) // i.e. if S exists in the functional
    {
        //Smoothers[1] = new HypreBoomerAMG(*Funct_restblocks_global(1,1));
        //((HypreBoomerAMG*)(Smoothers[1]))->SetPrintLevel(0);
        //((HypreBoomerAMG*)(Smoothers[1]))->iterative_mode = false;
        Smoothers[1] = new HypreSmoother( *(*Funct_hpmat)(1,1), HypreSmoother::Type::l1GS, sweeps_num[1]);
    }

    truex = new BlockVector(trueblock_offsets);
    truerhs = new BlockVector(trueblock_offsets);
#else // for NEW_SMOOTHERSETUP
    // shortcuts
    SparseMatrix *CurlhT = Transpose(*Curlh);
    const SparseMatrix * M = &(Funct_mat->GetBlock(0,0));

    HypreParMatrix * d_td_Hcurl_T = d_td_Hcurl->Transpose();

    // form CT*M*C as a SparseMatrix
    SparseMatrix *M_Curlh = mfem::Mult(*M, *Curlh);
    SparseMatrix * CTMC = mfem::Mult(*CurlhT, *M_Curlh);

    delete M_Curlh;
    delete CurlhT;

    // imposing essential boundary conditions
    for ( int dof = 0; dof < essbdrdofs_Hcurl->Size(); ++dof)
    {
        if ( (*essbdrdofs_Hcurl)[dof] != 0)
        {
            CTMC->EliminateRowCol(dof);
        }
    }

    // form CT*M*C as HypreParMatrices
    HypreParMatrix* CTMC_d_td;
    CTMC_d_td = d_td_Hcurl->LeftDiagMult( *CTMC );

    CTMC_global = ParMult(d_td_Hcurl_T, CTMC_d_td);
    CTMC_global->CopyRowStarts();
    CTMC_global->CopyColStarts();

    delete CTMC;

#ifdef MEMORY_OPTIMIZED
    temp_Hdiv_dofs = new Vector(Curlh->Height());
    temp_Hcurl_dofs = new Vector(Curlh->Width());
#else
    // Cannot use RAP directly because Curl should be multiplied obly by a diagonal of the d_td_Hdiv
    // FIXME : RowStarts or ColStarts()? Probably wrong row_starts n LeftDiagMult, thus a replacement code has been written but not tested

    /*
    // old variant
    HypreParMatrix* C_d_td = d_td_Hcurl.LeftDiagMult(*Curlh, d_td_Funct_blocks[0]->GetRowStarts() );
    SparseMatrix d_td_Hdiv_diag;
    d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
    Curlh_global = C_d_td->LeftDiagMult(*Transpose(d_td_Hdiv_diag), d_td_Funct_blocks[0]->GetColStarts() );
    Curlh_global->CopyRowStarts();
    Curlh_global->CopyColStarts();
    delete C_d_td;
    */

    // alternative way
    HypreParMatrix * temphpmat;
    SparseMatrix * Curlh_copy;
    {
        HYPRE_Int glob_num_rows = d_td_Funct_blocks[0]->M();
        HYPRE_Int glob_num_cols = d_td_Hcurl.M();
        HYPRE_Int * row_starts = d_td_Funct_blocks[0]->GetRowStarts();
        HYPRE_Int * col_starts = d_td_Hcurl.GetRowStarts();
        Curlh_copy = new SparseMatrix(*Curlh);
        temphpmat = new HypreParMatrix(comm, glob_num_rows, glob_num_cols, row_starts, col_starts, Curlh_copy);
    }

    HypreParMatrix * d_td_hdiv_diaghpmat;
    {
        HYPRE_Int glob_size = d_td_Funct_blocks[0]->M();
        HYPRE_Int * row_starts = d_td_Funct_blocks[0]->GetRowStarts();
        SparseMatrix d_td_Hdiv_diag;
        d_td_Funct_blocks[0]->GetDiag(d_td_Hdiv_diag);
        d_td_hdiv_diaghpmat = new HypreParMatrix(comm, glob_size, row_starts, &d_td_Hdiv_diag) ;
        d_td_hdiv_diaghpmat->CopyRowStarts();
        d_td_hdiv_diaghpmat->CopyColStarts();
    }

    //HypreParMatrix * temphp = d_td_hdiv_diaghpmat->Transpose();
    //HypreParMatrix * C_d_td = ParMult(temphpmat, &d_td_Hcurl);
    //Curlh_global = ParMult(temphp, C_d_td);

    Curlh_global = RAP(d_td_hdiv_diaghpmat, temphpmat, &d_td_Hcurl);
    Curlh_global->CopyRowStarts();
    Curlh_global->CopyColStarts();

    std::cout << "Curlh_global = " << Curlh_global->Height() << " x " << Curlh_global->Width() << "\n";

    delete Curlh_copy;
    delete temphpmat;
    //delete C_d_td;
    delete d_td_hdiv_diaghpmat;
    //delete temphp;
#endif

    //SparseMatrix diagg;
    //CTMC_global->GetDiag(diagg);
    //diagg.EliminateZeroRows();
    //diagg.SetDiagIdentity();

    Smoothers[0] = new HypreSmoother(*CTMC_global, HypreSmoother::Type::l1GS, sweeps_num[0]);
    if (numblocks > 1)
    {
        int blk = 1;
        // FIXME: Unnecessary memory allocation, if one can provide the functional with bnd dofs eliminated in the input
        SparseMatrix * Funct_blk = new SparseMatrix(Funct_mat->GetBlock(blk,blk));

        for ( int dof = 0; dof < essbdrdofs_Funct[blk]->Size(); ++dof)
        {
            if ( (*essbdrdofs_Funct[blk])[dof] != 0)
            {
                Funct_blk->EliminateRowCol(dof);
            }
        }

        // alternative way
        HYPRE_Int glob_size = d_td_Funct_blocks[blk]->M();
        HYPRE_Int * row_starts = d_td_Funct_blocks[blk]->GetRowStarts();
        HypreParMatrix * temphpmat = new HypreParMatrix(comm, glob_size, row_starts, Funct_blk);
        Funct_restblocks_global(1,1) = RAP(temphpmat, d_td_Funct_blocks[blk]);

        // old way
        //HypreParMatrix* Functblk_d_td_blk = d_td_Funct_blocks[blk]->LeftDiagMult(*Funct_blk, d_td_Funct_blocks[blk]->GetRowStarts() );
        //HypreParMatrix * d_td_blk_T = d_td_Funct_blocks[blk]->Transpose();
        //Funct_restblocks_global(1,1) = ParMult(d_td_blk_T, Functblk_d_td_blk);
        Funct_restblocks_global(1,1)->CopyRowStarts();
        Funct_restblocks_global(1,1)->CopyColStarts();

        //delete Functblk_d_td_blk;
        //delete Funct_blk;

        delete Funct_blk;
        delete temphpmat;

        //Smoothers[1] = new HypreBoomerAMG(*Funct_restblocks_global(1,1));
        //((HypreBoomerAMG*)(Smoothers[1]))->SetPrintLevel(0);
        //((HypreBoomerAMG*)(Smoothers[1]))->iterative_mode = false;
        Smoothers[1] = new HypreSmoother(*Funct_restblocks_global(1,1), HypreSmoother::Type::l1GS, sweeps_num[1]);
    }

    truex = new BlockVector(trueblock_offsets);
    truerhs = new BlockVector(trueblock_offsets);

    delete CTMC_d_td;
    delete d_td_Hcurl_T;
#endif // for #else to #ifdef NEW_SMOOTHERSETUP

#ifdef TIMING
    ResetInternalTimings();
#endif
}

// TODO: Add as an option using blas and lapack versions for solving local problems
// TODO: Test after all with nonzero boundary conditions for sigma
// TODO: Check the timings and make it faster
// TODO: Clean up the function descriptions
// TODO: Clean up the variables names
// TODO: Maybe, local matrices can also be stored as an improvement (see SolveLocalProblems())?
// TODO: Make dof_truedof an optional data member so that either dof_truedof or
// TODO: global funct matrices (and offsets) are given at all level, maybe via two different constructors
// TODO: In the latter case ComputeTrueRes can be rewritten using global matrices and dof_truedof
// TODO: can remain unused at all.

class GeneralMinConstrSolver : public Solver
{
private:
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

    const MPI_Comm comm;

    // Projectors for the variables related to the functional and constraint
    const Array< BlockOperator*>& TrueP_Func;

    // for each level and for each variable in the functional stores a vector
    // which defines if a dof is at the boundary / essential part of the boundary
    // or not
    const std::vector<std::vector<Array<int>* > > & essbdrtruedofs_Func; // can be removed since it is used only for debugging

    // parts of block structure which define the Functional at the finest level
    const int numblocks;

    const Array<Operator*>& Smoothers_lvls;

    // a given blockvector which satisfies essential bdr conditions
    // imposed for the initial problem
    // on true dofs
    const BlockVector& bdrdata_truedofs;

    const std::vector<Operator*> & Func_global_lvls;


#ifdef CHECK_CONSTR
    mutable HypreParMatrix * Constr_global;
    mutable Vector * Constr_rhs_global;
#endif

#ifdef TIMING
private:
    mutable StopWatch chrono;
    mutable StopWatch chrono2;
    mutable StopWatch chrono3;
    mutable StopWatch chrono4;
    mutable double time_solve;
    mutable double time_mult;
    mutable double time_localsolve;
    mutable double* time_localsolve_lvls;
    mutable double time_smoother;
    mutable double* time_smoother_lvls;
    mutable double time_coarsestproblem;
    mutable double time_resupdate;
    mutable double time_fw;
    mutable double time_up;
    mutable std::list<double>* times_mult;
    mutable std::list<double>* times_solve;
    mutable std::list<double>* times_localsolve;
    mutable std::list<double>* times_localsolve_lvls;
    mutable std::list<double>* times_smoother;
    mutable std::list<double>* times_smoother_lvls;
    mutable std::list<double>* times_coarsestproblem;
    mutable std::list<double>* times_resupdate;
    mutable std::list<double>* times_fw;
    mutable std::list<double>* times_up;
#endif

protected:
    const BlockVector& Functrhs_global; // used only for FunctCheck (hence, it is not used in the preconditioner mode at all)

    // The same as xblock and yblock but on true dofs
    mutable BlockVector* xblock_truedofs;
    mutable BlockVector* yblock_truedofs;
    mutable BlockVector* tempblock_truedofs;

    // stores the initial guess for the solver
    // which satisfies the divergence contraint
    // if not specified in the constructor.
    // it is 0 by default
    // Muts be defined on true dofs
    mutable BlockVector* init_guess;

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
    ~GeneralMinConstrSolver();
    // constructor
    GeneralMinConstrSolver(
                           MPI_Comm Comm,
                           int NumLevels,
                           const Array< BlockOperator*>& TrueProj_Func,
                           const std::vector<std::vector<Array<int>* > > &EssBdrTrueDofs_Func,
                           const BlockVector& Functrhs_Global,
                           const Array<Operator*>& Smoothers_Lvls,
                           const BlockVector& Bdrdata_TrueDofs,
                           const std::vector<Operator*> & Func_Global_lvls,
#ifdef CHECK_CONSTR
                           HypreParMatrix & Constr_Global,
                           Vector & Constr_Rhs_global,
#endif

#ifdef TIMING
                            std::list<double>* Times_mult,
                            std::list<double>* Times_solve,
                            std::list<double>* Times_localsolve,
                            std::list<double>* Times_localsolve_lvls,
                            std::list<double>* Times_smoother,
                            std::list<double>* Times_smoother_lvls,
                            std::list<double>* Times_coarsestproblem,
                            std::list<double>* Times_resupdate,
                            std::list<double>* Times_fw,
                            std::list<double>* Times_up,
#endif
                           Array<Operator*>* LocalSolvers = NULL,
                           Operator* CoarseSolver = NULL,
                           int StopCriteria_Type = 1);

    GeneralMinConstrSolver() = delete;

    // external calling routine (as in any IterativeSolver) which takes care of convergence
    virtual void Mult(const Vector & x, Vector & y) const override;

    // existence of this method is required by the (abstract) base class Solver
    virtual void SetOperator(const Operator &op) override{}

    bool StoppingCriteria(int type, double value_curr, double value_prev, double value_scalefactor,
                          double stop_tol, bool monotone_check = true, char const * name = NULL,
                          bool print = false) const;

    int GetStopCriteriaType () const {return stopcriteria_type;}
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

GeneralMinConstrSolver::~GeneralMinConstrSolver()
{
    delete xblock_truedofs;
    delete yblock_truedofs;
    delete tempblock_truedofs;
    delete init_guess;

    for (int i = 0; i < truetempvec_lvls.Size(); ++i)
        delete truetempvec_lvls[i];
    for (int i = 0; i < truetempvec2_lvls.Size(); ++i)
        delete truetempvec2_lvls[i];
    for (int i = 0; i < trueresfunc_lvls.Size(); ++i)
        delete trueresfunc_lvls[i];
    for (int i = 0; i < truesolupdate_lvls.Size(); ++i)
        delete truesolupdate_lvls[i];

#ifdef TIMING
    delete time_localsolve_lvls;
    delete time_smoother_lvls;
#endif
}

void GeneralMinConstrSolver::PrintAllOptions() const
{
    std::cout << "GeneralMinConstrSolver options: \n";
    std::cout << "num_levels: " << num_levels << "\n";
    std::cout << "numblocks:" << numblocks << "\n";
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
    init_guess->Update(InitGuess.GetData(), TrueP_Func[0]->RowOffsets());
}

bool GeneralMinConstrSolver::StoppingCriteria(int type, double value_curr, double value_prev,
                                                  double value_scalefactor, double stop_tol,
                                                  bool monotone_check, char const * name,
                                                  bool print) const
{
    bool already_printed = false;
    if (monotone_check)
        if (value_curr > value_prev && fabs(value_prev - value_curr) / fabs(value_scalefactor) > 1.0e-10 )
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

        if ( fabs(value_prev - value_curr) / fabs(value_scalefactor) < stop_tol )
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

        if ( fabs(value_curr) / fabs(value_scalefactor) < stop_tol )
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


GeneralMinConstrSolver::GeneralMinConstrSolver(
                        MPI_Comm Comm,
                        int NumLevels,
                       const Array< BlockOperator*>& TrueProj_Func,
                       const std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                       const BlockVector& Functrhs_Global,
                       const Array<Operator*>& Smoothers_Lvls,
                       const BlockVector& Bdrdata_TrueDofs,
                       const std::vector<Operator*> & Func_Global_lvls,
#ifdef CHECK_CONSTR
                       HypreParMatrix & Constr_Global,
                       Vector & Constr_Rhs_global,
#endif
#ifdef TIMING
                        std::list<double>* Times_mult,
                        std::list<double>* Times_solve,
                        std::list<double>* Times_localsolve,
                        std::list<double>* Times_localsolve_lvls,
                        std::list<double>* Times_smoother,
                        std::list<double>* Times_smoother_lvls,
                        std::list<double>* Times_coarsestproblem,
                        std::list<double>* Times_resupdate,
                        std::list<double>* Times_fw,
                        std::list<double>* Times_up,
#endif
                       Array<Operator*>* LocalSolvers,
                       Operator *CoarsestSolver,
                       int StopCriteria_Type)
     : Solver(TrueProj_Func[0]->Height()),
       stopcriteria_type(StopCriteria_Type),
       setup_finished(false),
       num_levels(NumLevels),
       current_iteration(0),
       comm(Comm),
       TrueP_Func(TrueProj_Func),
       essbdrtruedofs_Func(EssBdrTrueDofs_Func),
       numblocks(TrueProj_Func[0]->NumRowBlocks()),
       Smoothers_lvls(Smoothers_Lvls),
       bdrdata_truedofs(Bdrdata_TrueDofs),
       Func_global_lvls(Func_Global_lvls),

#ifdef CHECK_CONSTR
       Constr_global(&Constr_Global),
       Constr_rhs_global(&Constr_Rhs_global),
#endif
#ifdef TIMING
       times_mult(Times_mult),
       times_solve(Times_solve),
       times_localsolve(Times_localsolve),
       times_localsolve_lvls(Times_localsolve_lvls),
       times_smoother(Times_smoother),
       times_smoother_lvls(Times_smoother_lvls),
       times_coarsestproblem(Times_coarsestproblem),
       times_resupdate(Times_resupdate),
       times_fw(Times_fw),
       times_up(Times_up),
#endif
       Functrhs_global(Functrhs_Global)
{

    TrueProj_Func[0]->RowOffsets();

    xblock_truedofs = new BlockVector(TrueProj_Func[0]->RowOffsets());
    yblock_truedofs = new BlockVector(TrueProj_Func[0]->RowOffsets());
    tempblock_truedofs = new BlockVector(TrueProj_Func[0]->RowOffsets());

    truesolupdate_lvls.SetSize(num_levels);
    truesolupdate_lvls[0] = new BlockVector(TrueProj_Func[0]->RowOffsets());

    truetempvec_lvls.SetSize(num_levels);
    truetempvec_lvls[0] = new BlockVector(TrueProj_Func[0]->RowOffsets());
    truetempvec2_lvls.SetSize(num_levels);
    truetempvec2_lvls[0] = new BlockVector(TrueProj_Func[0]->RowOffsets());
    trueresfunc_lvls.SetSize(num_levels);
    trueresfunc_lvls[0] = new BlockVector(TrueProj_Func[0]->RowOffsets());

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

    init_guess = new BlockVector(TrueProj_Func[0]->RowOffsets());
    *init_guess = 0.0;

    LocalSolvers_lvls.SetSize(num_levels - 1);
    for (int l = 0; l < num_levels - 1; ++l)
        if (LocalSolvers)
            LocalSolvers_lvls[l] = (*LocalSolvers)[l];
        else
            LocalSolvers_lvls[l] = NULL;

#ifdef TIMING
    time_mult = 0.0;
    time_solve = 0.0;
    time_localsolve = 0.0;
    time_localsolve_lvls = new double[num_levels - 1];
    for (int l = 0; l < num_levels - 1; ++l)
        time_localsolve_lvls[l] = 0.0;
    time_smoother = 0.0;
    time_smoother_lvls = new double[num_levels - 1];
    for (int l = 0; l < num_levels - 1; ++l)
        time_smoother_lvls[l] = 0.0;
    time_coarsestproblem = 0.0;
    time_resupdate = 0.0;
    time_fw = 0.0;
    time_up = 0.0;
#endif

    Setup();
}

void GeneralMinConstrSolver::Setup(bool verbose) const
{
    if (verbose)
        std::cout << "Starting solver setup \n";

    CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, *init_guess,
            "for the initial guess during solver setup (no rhs provided): ", print_level);
    // 2. setting up the required internal data at all levels

    // 2.1 loop over all levels except the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        // sets up the current level and prepares operators for the next one
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

#ifdef TIMING
    time_mult = 0.0;
    time_solve = 0.0;
    time_localsolve = 0.0;
    for (int l = 0; l < num_levels - 1; ++l)
        time_localsolve_lvls[l] = 0.0;
    time_smoother = 0.0;
    for (int l = 0; l < num_levels - 1; ++l)
        time_smoother_lvls[l] = 0.0;
    time_coarsestproblem = 0.0;
    time_resupdate = 0.0;
    time_fw = 0.0;
    time_up = 0.0;
#endif

#ifdef TIMING
    MPI_Barrier(comm);
    chrono4.Clear();
    chrono4.Start();
#endif

    // start iteration
    current_iteration = 0;
    converged = 0;

    // x will be accessed through xblock_truedofs as its view
    xblock_truedofs->Update(x.GetData(), TrueP_Func[0]->RowOffsets());
    // y will be accessed through yblock_truedofs as its view
    yblock_truedofs->Update(y.GetData(), TrueP_Func[0]->RowOffsets());

    if (preconditioner_mode)
        *init_guess = 0.0;
    else
    {
        funct_firstnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, *init_guess,
                                 "for the initial guess: ", print_level);
    }
    // tempblock is the initial guess (on true dofs)
    *tempblock_truedofs = *init_guess;
#ifdef CHECK_CONSTR
     if (!preconditioner_mode)
     {
        MFEM_ASSERT(CheckConstrRes(tempblock_truedofs->GetBlock(0), *Constr_global, Constr_rhs_global, "for the initial guess"),"");
     }
     else
     {
         MFEM_ASSERT(CheckConstrRes(tempblock_truedofs->GetBlock(0), *Constr_global, NULL, "for the initial guess"),"");
     }
#endif


    int itnum = 0;
    for (int i = 0; i < max_iter; ++i )
    {
#ifdef DEBUG_INFO
        std::cout << "i = " << i << " (iter) \n";
#endif
        MFEM_ASSERT(i == current_iteration, "Iteration counters mismatch!");

#ifdef CHECK_BNDCND
        for (int blk = 0; blk < numblocks; ++blk)
        {
            MFEM_ASSERT(CheckBdrError(tempblock_truedofs->GetBlock(blk), &(bdrdata_truedofs.GetBlock(blk)), *essbdrtruedofs_Func[0][blk], true),
                                  "before the iteration");
        }
#endif

#ifdef CHECK_CONSTR
        if (!preconditioner_mode)
        {
            MFEM_ASSERT(CheckConstrRes(tempblock_truedofs->GetBlock(0), *Constr_global, Constr_rhs_global, "before the iteration"),"");
        }
        else
            MFEM_ASSERT(CheckConstrRes(tempblock_truedofs->GetBlock(0), *Constr_global, NULL, "before the iteration"),"");
#endif

#ifdef TIMING
        MPI_Barrier(comm);
        chrono3.Clear();
        chrono3.Start();
#endif

        Solve(*xblock_truedofs, *tempblock_truedofs, *yblock_truedofs);

#ifdef TIMING
        MPI_Barrier(comm);
        chrono3.Stop();
        time_solve  += chrono3.RealTime();
#endif

#ifdef CHECK_CONSTR
        if (!preconditioner_mode)
        {
           MFEM_ASSERT(CheckConstrRes(yblock_truedofs->GetBlock(0), *Constr_global, Constr_rhs_global, "for the initial guess"),"");
        }
        else
        {
            MFEM_ASSERT(CheckConstrRes(yblock_truedofs->GetBlock(0), *Constr_global, NULL, "for the initial guess"),"");
        }
#endif
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
    if (!preconditioner_mode)
    {
        int myrank;
        MPI_Comm_rank(comm, &myrank);
        if (myrank == 0)
        {
            if (converged == 1)
                std::cout << "Solver converged in " << itnum << " iterations. \n" << std::flush;
            else // -1
                std::cout << "Solver didn't converge in " << itnum << " iterations. \n" << std::flush;
        }
    }

#ifdef TIMING
    MPI_Barrier(comm);
    chrono4.Stop();
    time_mult += chrono4.RealTime();
#endif

#ifdef TIMING
    times_mult->push_back(time_mult);
    times_solve->push_back(time_solve);
    times_localsolve->push_back(time_localsolve);
    for (int l = 0; l < num_levels - 1; ++l)
        times_localsolve_lvls[l].push_back(time_localsolve_lvls[l]);
    times_smoother->push_back(time_smoother);
    for (int l = 0; l < num_levels - 1; ++l)
        times_smoother_lvls[l].push_back(time_smoother_lvls[l]);
    times_coarsestproblem->push_back(time_coarsestproblem);
    times_resupdate->push_back(time_resupdate);
    times_fw->push_back(time_fw);
    times_up->push_back(time_up);
#endif

}

void GeneralMinConstrSolver::MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector &rhs_l) const
{
    Func_global_lvls[l]->Mult(x_l, rhs_l);
    rhs_l *= coeff;
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
#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Clear();
    chrono2.Start();
#endif

    if (print_level)
        std::cout << "Starting iteration " << current_iteration << " ... \n";

#ifdef CHECK_BNDCND
    for (int blk = 0; blk < numblocks; ++blk)
    {
        MFEM_ASSERT(CheckBdrError(previous_sol.GetBlock(blk), &(bdrdata_truedofs.GetBlock(blk)), *essbdrtruedofs_Func[0][blk], true),
                              "at the start of Solve()");
    }
#endif

#ifdef CHECK_CONSTR
    if (!preconditioner_mode)
    {
        MFEM_ASSERT(CheckConstrRes(previous_sol.GetBlock(0), *Constr_global, Constr_rhs_global, "for previous_sol"),"");
    }
    else
    {
        MFEM_ASSERT(CheckConstrRes(previous_sol.GetBlock(0), *Constr_global, NULL, "for previous_sol"),"");
    }
#endif

    next_sol = previous_sol;

    if (!preconditioner_mode && print_level)
        CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                             "at the beginning of Solve: ", print_level);

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif
    UpdateTrueResidual(0, &righthand_side, previous_sol, *trueresfunc_lvls[0] );

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_resupdate += chrono.RealTime();
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    // DOWNWARD loop: from finest to coarsest
    // 1. loop over levels finer than the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {

#ifdef TIMING
        MPI_Barrier(comm);
        chrono.Clear();
        chrono.Start();
#endif

        // solution updates will always satisfy homogeneous essential boundary conditions
        *truesolupdate_lvls[l] = 0.0;

        if (LocalSolvers_lvls[l])
        {
            LocalSolvers_lvls[l]->Mult(*trueresfunc_lvls[l], *truetempvec_lvls[l]);
            *truesolupdate_lvls[l] += *truetempvec_lvls[l];

#ifdef DEBUG_INFO
            next_sol += *truesolupdate_lvls[0];
            if (!preconditioner_mode)
                funct_currnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                                     "after the localsolve update: ", 1);
            next_sol -= *truesolupdate_lvls[0];
#endif
        }

#ifdef TIMING
        MPI_Barrier(comm);
        chrono.Stop();
        time_localsolve_lvls[l] += chrono.RealTime();
        time_localsolve += chrono.RealTime();
        MPI_Barrier(comm);
        chrono.Clear();
        chrono.Start();
#endif

        UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );

#ifdef TIMING
        MPI_Barrier(comm);
        chrono.Stop();
        time_resupdate += chrono.RealTime();
        MPI_Barrier(comm);
        chrono.Clear();
        chrono.Start();
#endif

        // smooth
        if (Smoothers_lvls[l])
        {
#ifdef DEBUG_INFO
            next_sol += *truesolupdate_lvls[0];
            funct_currnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                                     "before the smoother update: ", 1);
            next_sol -= *truesolupdate_lvls[0];
#endif
            //std::cout << "l = " << l << "\n";
            //std::cout << "tempvec_l = " << truetempvec_lvls[l] << ", tempvec2_l = " << truetempvec2_lvls[l] << "\n";
            Smoothers_lvls[l]->Mult(*truetempvec_lvls[l], *truetempvec2_lvls[l] );
#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Stop();
            time_smoother_lvls[l] += chrono.RealTime();
            time_smoother += chrono.RealTime();
#endif
            *truesolupdate_lvls[l] += *truetempvec2_lvls[l];
#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif

            UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Stop();
            time_resupdate += chrono.RealTime();
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif


#ifdef DEBUG_INFO
            next_sol += *truesolupdate_lvls[0];
            funct_currnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                                     "after the smoother update: ", 1);
            next_sol -= *truesolupdate_lvls[0];
#endif
        }


        *trueresfunc_lvls[l] = *truetempvec_lvls[l];

        TrueP_Func[l]->MultTranspose(*trueresfunc_lvls[l], *trueresfunc_lvls[l + 1]);

        // manually setting the boundary conditions (requried for S from H1 at least) at the coarser level
        for (int blk = 0; blk < numblocks; ++blk)
        {
            const Array<int> * temp;
            temp = essbdrtruedofs_Func[l + 1][blk];

            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                trueresfunc_lvls[l + 1]->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            }
        }


    } // end of loop over finer levels

#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Stop();
    time_fw  += chrono2.RealTime();
#endif

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Clear();
    chrono.Start();
#endif

    // BOTTOM: solve the global problem at the coarsest level
    CoarseSolver->Mult(*trueresfunc_lvls[num_levels - 1], *truesolupdate_lvls[num_levels - 1]);
    //*truesolupdate_lvls[num_levels - 1] = 0.0;

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    //std::cout << "before: " << time_coarsestproblem << "\n";
    time_coarsestproblem += chrono.RealTime();
    //std::cout << "after: " << time_coarsestproblem << "\n";
#endif

#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Clear();
    chrono2.Start();
#endif

#ifdef CHECK_CONSTR
    TrueP_Func[0]->Mult(*truesolupdate_lvls[1], *truetempvec_lvls[0]);
    MFEM_ASSERT(CheckConstrRes(truetempvec_lvls[0]->GetBlock(0), *Constr_global, NULL, "for interpolated coarsest level update"),"");

    *truesolupdate_lvls[0] += *truetempvec_lvls[0];
    next_sol += *truesolupdate_lvls[0];
    MFEM_ASSERT(CheckConstrRes(truetempvec_lvls[0]->GetBlock(0), *Constr_global, NULL, "after fw and bottom updates"),"");
    next_sol -= *truesolupdate_lvls[0];
    *truesolupdate_lvls[0] -= *truetempvec_lvls[0];
#endif

#ifdef DEBUG_INFO
    TrueP_Func[0]->Mult(*truesolupdate_lvls[1], *truetempvec_lvls[0]);


    *truesolupdate_lvls[0] += *truetempvec_lvls[0];
    next_sol += *truesolupdate_lvls[0];
    funct_currnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                             "after the coarsest level update: ", 1);
    next_sol -= *truesolupdate_lvls[0];
    *truesolupdate_lvls[0] -= *truetempvec_lvls[0];
#endif

    // UPWARD loop: from coarsest to finest
    if (symmetric) // then also smoothing and solving local problems on the way up
    {
        for (int l = num_levels - 1; l > 0; --l)
        {
            // interpolate back to the finer level
            TrueP_Func[l - 1]->Mult(*truesolupdate_lvls[l], *truetempvec_lvls[l - 1]);

            *truesolupdate_lvls[l - 1] += *truetempvec_lvls[l - 1];

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif
            UpdateTrueResidual(l - 1, trueresfunc_lvls[l - 1], *truetempvec_lvls[l - 1], *truetempvec2_lvls[l - 1] );
#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Stop();
            time_resupdate += chrono.RealTime();
#endif
            *trueresfunc_lvls[l - 1] = *truetempvec2_lvls[l - 1];

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif
            // smooth at the finer level
            if (Smoothers_lvls[l - 1])
            {
                Smoothers_lvls[l - 1]->MultTranspose(*truetempvec2_lvls[l - 1], *truetempvec_lvls[l - 1] );
#ifdef TIMING
                MPI_Barrier(comm);
                chrono.Stop();
                time_smoother_lvls[l - 1] += chrono.RealTime();
                time_smoother += chrono.RealTime();
#endif

                *truesolupdate_lvls[l - 1] += *truetempvec_lvls[l - 1];

#ifdef TIMING
                MPI_Barrier(comm);
                chrono.Clear();
                chrono.Start();
#endif
                UpdateTrueResidual(l - 1, trueresfunc_lvls[l - 1], *truetempvec_lvls[l - 1], *truetempvec2_lvls[l - 1] );

#ifdef TIMING
                MPI_Barrier(comm);
                chrono.Stop();
                time_resupdate += chrono.RealTime();
#endif
            }

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif

            if (LocalSolvers_lvls[l - 1])
            {
                LocalSolvers_lvls[l - 1]->Mult(*truetempvec2_lvls[l - 1], *truetempvec_lvls[l - 1]);
                *truesolupdate_lvls[l - 1] += *truetempvec_lvls[l - 1];
            }

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Stop();
            time_localsolve_lvls[l - 1] += chrono.RealTime();
            time_localsolve += chrono.RealTime();
#endif
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

#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Stop();
    time_up  += chrono2.RealTime();
#endif

#ifdef CHECK_CONSTR
    MFEM_ASSERT(CheckConstrRes(truesolupdate_lvls[0]->GetBlock(0), *Constr_global, NULL, "for update after full V-cycle"),"");
#endif

    // 4. update the global iterate by the resulting update at the finest level
    next_sol += *truesolupdate_lvls[0];

#ifdef CHECK_CONSTR
    if (!preconditioner_mode)
    {
        MFEM_ASSERT(CheckConstrRes(next_sol.GetBlock(0), *Constr_global, Constr_rhs_global, "for next_sol"),"");
    }
    else
    {
        MFEM_ASSERT(CheckConstrRes(next_sol.GetBlock(0), *Constr_global, NULL, "for next_sol"),"");
    }
#endif

#ifdef CHECK_BNDCND
    if (print_level && !preconditioner_mode)
    {
        for (int blk = 0; blk < numblocks; ++blk)
        {
            MFEM_ASSERT(CheckBdrError(next_sol.GetBlock(blk), &(bdrdata_truedofs.GetBlock(blk)), *essbdrtruedofs_Func[0][blk], true),
                                  "after all levels update");
        }
    }
#endif

    if (print_level > 10)
    {
        std::cout << "sol_update norm: " << truesolupdate_lvls[0]->GetBlock(0).Norml2() /
                  sqrt(truesolupdate_lvls[0]->GetBlock(0).Size()) << "\n";
    }

    // some monitoring service calls
    if (!preconditioner_mode)
        if (print_level || stopcriteria_type == 0)
        {
            funct_currnorm = CheckFunctValue(comm, *Func_global_lvls[0], &Functrhs_global, next_sol,
                                     "at the end of iteration: ", print_level);
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
    truetempvec_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    truetempvec2_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    truesolupdate_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
    trueresfunc_lvls[lvl + 1] = new BlockVector(TrueP_Func[lvl]->ColOffsets());
}

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

            delete R_t;
            delete W_t;

            // 2. For RT elements, we impose boundary condition equal zero,
            //   see the function: GetInternalDofs2AE to obtained them

            SparseMatrix intDofs_R_AE;
            GetInternalDofs2AE(*R_AE,intDofs_R_AE);

            //  AE elements x localDofs stored in AE_R & AE_W
            SparseMatrix *AE_R =  Transpose(intDofs_R_AE);
            SparseMatrix *AE_W = Transpose(*W_AE);

            delete W_AE;
            delete R_AE;


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

            delete P_WT;
            Vector Diag(P_WTxP_W->Size());
            Vector invDiag(P_WTxP_W->Size());
            P_WTxP_W->GetDiag(Diag);

            for(int m=0; m < P_WTxP_W->Size(); m++)
            {
                //std::cout << "Diag(m) = " << Diag(m) << "\n";
                invDiag(m) = comp(m)/Diag(m);
            }

            delete P_WTxP_W;

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

                    delete M_PR;
                }

                delete B_PR;
                delete P_WT2;
                if (M_fine)
                    delete P_RT2;
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

            delete AE_R;
            delete AE_W;

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

            delete M_PR;

            for ( int k = 0; k < ess_dof_coarsestlvl_list.Size(); ++k)
                if (ess_dof_coarsestlvl_list[k] !=0)
                    M_coarse->EliminateRowCol(k);
        }

        delete P_WT2;
        if (M_fine)
            delete P_RT2;
        delete B_PR;

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
            S->CopyColStarts();
            S->CopyRowStarts();

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

            for ( int blk = 0; blk < darcyPr->NumBlocks(); ++blk)
                    if (&(darcyPr->GetDiagonalBlock(blk)))
                        delete &(darcyPr->GetDiagonalBlock(blk));
            delete MinvBt;
            delete Md;
            delete darcyPr;
            delete M_Global;
            delete BT;
            delete S;
            delete d_td_T;
            delete d_td_M;
        }
        else
        {
            int maxIter(50000);
            double rtol(1.e-16);
            double atol(1.e-16);

            HypreParMatrix *MinvBt = B_Global->Transpose();
            HypreParMatrix *S = ParMult(B_Global, MinvBt);
            S->CopyColStarts();
            S->CopyRowStarts();

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

            delete MinvBt;
            delete invS;
            delete S;
        }

        delete B_coarse;
        if (M_fine)
            delete M_coarse;
        delete B_Global;

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

        delete R_T;
        delete Dofs_AE;
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

            B00 = new HypreSmoother(A00, HypreSmoother::Type::l1GS, 1);
            B11 = new HypreSmoother(A11, HypreSmoother::Type::l1GS, 1);

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
        }

    private:
        HypreSmoother *B00;
        HypreSmoother *B11;
        HypreParMatrix &A01;
        HypreParMatrix &A10;

        const Array<int> &offsets;
        mutable BlockVector xblock;
        mutable BlockVector yblock;
        mutable Vector tmp01;
        mutable Vector tmp02;
        mutable Vector tmp1;
    };

public:
    MonolithicMultigrid(BlockOperator &Op,
                        const Array<BlockOperator*> &P,
                        Solver* Coarse_Prec = NULL)
        :
          Solver(Op.RowOffsets().Last()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarsePrec_(Coarse_Prec),
          built_prec(false)
    {
        Operators_.Last() = &Op;

        for (int l = Operators_.Size()-1; l >= 0; l--)
        {
            Array<int>& Offsets = Operators_[l]->RowOffsets();

            if (l < Operators_.Size() - 1)
                correction[l] = new Vector(Offsets.Last());
            else // exist because of SetDataAndSize call to correction.Last()
                 //in MonolithicMultigrid::Mult which drops the data (= memory leak)
                 // (if allocated here, i.e. if no if-clause)
                correction[l] = new Vector();

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

        CoarseSolver = new CGSolver(((HypreParMatrix&)Op.GetBlock(0,0)).GetComm() );
        CoarseSolver->SetAbsTol(sqrt(1e-32));
        CoarseSolver->SetRelTol(sqrt(1e-16));
        CoarseSolver->SetMaxIter(100);
        CoarseSolver->SetPrintLevel(0);
        CoarseSolver->SetOperator(*Operators_[0]);

        if (!CoarsePrec_)
        {
            built_prec = true;

            CoarsePrec_ = new BlockDiagonalPreconditioner(Operators_[0]->ColOffsets());

            HypreParMatrix &A00 = (HypreParMatrix&)Operators_[0]->GetBlock(0,0);
            HypreParMatrix &A11 = (HypreParMatrix&)Operators_[0]->GetBlock(1,1);

            HypreSmoother * precU = new HypreSmoother(A00, HypreSmoother::Type::l1GS, 1);
            HypreSmoother * precS = new HypreSmoother(A11, HypreSmoother::Type::l1GS, 1);

            ((BlockDiagonalPreconditioner*)CoarsePrec_)->SetDiagonalBlock(0, precU);
            ((BlockDiagonalPreconditioner*)CoarsePrec_)->SetDiagonalBlock(1, precS);
        }

        CoarseSolver->SetPreconditioner(*CoarsePrec_);
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

            if (l < Operators_.Size() - 1)
                delete Operators_[l];
        }
        delete CoarseSolver;
        if (built_prec)
            delete CoarsePrec_;
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

    mutable bool built_prec;
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

        CoarseSolver->Mult(residual_l, cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(cor_cor, help);
        residual_l -= help;
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
              Solver *CoarsePrec = NULL)
        :
          Solver(Operator.GetNumRows()),
          P_(P),
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarsePrec_(CoarsePrec),
          built_prec(false)
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
            Smoothers_[l] = new HypreSmoother(*Operators_[l], HypreSmoother::Type::l1GS, 1);
            residual[l] = new Vector(Operators_[l]->GetNumRows());
            if (l < Operators_.Size() - 1)
                correction[l] = new Vector(Operators_[l]->GetNumRows());
            else // exist because of SetDataAndSize call to correction.Last() in Multigrid::Mult
                 // which drops the  data (if allocated here, i.e. if no if-clause)
                correction[l] = new Vector();
        }

        CoarseSolver = new CGSolver(Operators_[0]->GetComm());
        CoarseSolver->SetAbsTol(sqrt(1e-32));
        CoarseSolver->SetRelTol(sqrt(1e-16));
        CoarseSolver->SetMaxIter(100);
        CoarseSolver->SetPrintLevel(0);
        CoarseSolver->SetOperator(*Operators_[0]);

        if (!CoarsePrec_)
        {
            built_prec = true;

            HypreParMatrix &A_c = (HypreParMatrix&)(*Operators_[0]);

            CoarsePrec_ = new HypreSmoother(A_c, HypreSmoother::Type::l1GS, 1);
        }

        CoarseSolver->SetPreconditioner(*CoarsePrec_);

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
            if (l < Operators_.Size() - 1)
                delete Operators_[l];

        }
        delete CoarseSolver;
        if (built_prec)
            delete CoarsePrec_;
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

    mutable CGSolver *CoarseSolver;
    Solver * CoarsePrec_;

    mutable bool built_prec;
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

        CoarseSolver->Mult(residual_l, cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
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


// Eliminates all entries in the Operator acting in a pair of spaces,
// assembled as a HypreParMatrix, which connect internal dofs to boundary dofs
// Used to modife the Curl and Divskew operator for the new multigrid solver
void Eliminate_ib_block(HypreParMatrix& Op_hpmat, const Array<int>& EssBdrTrueDofs_dom, const Array<int>& EssBdrTrueDofs_range )
{
    MPI_Comm comm = Op_hpmat.GetComm();

    int ntdofs_dom = Op_hpmat.Width();
    Array<int> btd_flags(ntdofs_dom);
    btd_flags = 0;
    //if (verbose)
        //std::cout << "EssBdrTrueDofs_dom \n";
    //EssBdrTrueDofs_dom.Print();

    for ( int i = 0; i < EssBdrTrueDofs_dom.Size(); ++i )
    {
        int tdof = EssBdrTrueDofs_dom[i];
        btd_flags[tdof] = 1;
    }

    int * td_btd_i = new int[ ntdofs_dom + 1];
    td_btd_i[0] = 0;
    for (int i = 0; i < ntdofs_dom; ++i)
        td_btd_i[i + 1] = td_btd_i[i] + 1;

    int * td_btd_j = new int [td_btd_i[ntdofs_dom]];
    double * td_btd_data = new double [td_btd_i[ntdofs_dom]];
    for (int i = 0; i < ntdofs_dom; ++i)
    {
        td_btd_j[i] = i;
        if (btd_flags[i] != 0)
            td_btd_data[i] = 1.0;
        else
            td_btd_data[i] = 0.0;
    }

    SparseMatrix * td_btd_diag = new SparseMatrix(td_btd_i, td_btd_j, td_btd_data, ntdofs_dom, ntdofs_dom);

    HYPRE_Int * row_starts = Op_hpmat.GetColStarts();

    HypreParMatrix * td_btd_hpmat = new HypreParMatrix(comm, Op_hpmat.N(),
            row_starts, td_btd_diag);
    td_btd_hpmat->CopyColStarts();
    td_btd_hpmat->CopyRowStarts();

    HypreParMatrix * C_td_btd = ParMult(&Op_hpmat, td_btd_hpmat);

    // processing local-to-process block of the Divfree matrix
    SparseMatrix C_td_btd_diag;
    C_td_btd->GetDiag(C_td_btd_diag);

    //C_td_btd_diag.Print();

    SparseMatrix C_diag;
    Op_hpmat.GetDiag(C_diag);

    //C_diag.Print();

    int ntdofs_range = Op_hpmat.Height();

    Array<int> btd_flags_range(ntdofs_range);
    btd_flags_range = 0;
    for ( int i = 0; i < EssBdrTrueDofs_range.Size(); ++i )
    {
        int tdof = EssBdrTrueDofs_range[i];
        btd_flags_range[tdof] = 1;
    }

    //if (verbose)
        //std::cout << "EssBdrTrueDofs_range \n";
    //EssBdrTrueDofs_range.Print();

    for (int row = 0; row < C_td_btd_diag.Height(); ++row)
    {
        if (btd_flags_range[row] == 0)
        {
            for (int colind = 0; colind < C_td_btd_diag.RowSize(row); ++colind)
            {
                int nnz_ind = C_td_btd_diag.GetI()[row] + colind;
                int col = C_td_btd_diag.GetJ()[nnz_ind];
                double fabs_entry = fabs(C_td_btd_diag.GetData()[nnz_ind]);

                if (fabs_entry > 1.0e-10)
                {
                    for (int j = 0; j < C_diag.RowSize(row); ++j)
                    {
                        int colorig = C_diag.GetJ()[C_diag.GetI()[row] + j];
                        if (colorig == col && colorig != row)
                        {
                            //std::cout << "Changes made in row = " << row << ", col = " << colorig << "\n";
                            C_diag.GetData()[C_diag.GetI()[row] + j] = 0.0;

                        }
                    }
                } // else of if fabs_entry is large enough

            }
        } // end of if row corresponds to the non-boundary Hdiv dof
    }

    //C_diag.Print();

    // processing the off-diagonal block of the Divfree matrix
    SparseMatrix C_td_btd_offd;
    HYPRE_Int * C_td_btd_cmap;
    C_td_btd->GetOffd(C_td_btd_offd, C_td_btd_cmap);

    SparseMatrix C_offd;
    HYPRE_Int * C_cmap;
    Op_hpmat.GetOffd(C_offd, C_cmap);

    for (int row = 0; row < C_td_btd_offd.Height(); ++row)
    {
        if (btd_flags_range[row] == 0)
        {
            for (int colind = 0; colind < C_td_btd_offd.RowSize(row); ++colind)
            {
                int nnz_ind = C_td_btd_offd.GetI()[row] + colind;
                int truecol = C_td_btd_cmap[C_td_btd_offd.GetJ()[nnz_ind]];
                double fabs_entry = fabs(C_td_btd_offd.GetData()[nnz_ind]);

                if (fabs_entry > 1.0e-10)
                {
                    for (int j = 0; j < C_offd.RowSize(row); ++j)
                    {
                        int truecolorig = C_cmap[C_offd.GetJ()[C_offd.GetI()[row] + j]];
                        if (truecolorig == truecol && truecolorig != row)
                        {
                            //std::cout << "Changes made in row = " << row << ", col = " << colorig << "\n";
                            C_offd.GetData()[C_offd.GetI()[row] + j] = 0.0;

                        }
                    }
                } // else of if fabs_entry is large enough

            }
        } // end of if row corresponds to the non-boundary Hdiv dof

    }
}
