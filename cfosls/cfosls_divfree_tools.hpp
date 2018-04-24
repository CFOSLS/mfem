#include <iostream>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_DIVFREETOOLS
#define MFEM_CFOSLS_DIVFREETOOLS

using namespace std;
using namespace mfem;

namespace mfem
{

#define BND_FOR_MULTIGRID

#define MEMORY_OPTIMIZED

#define CHECK_CONSTR

// activates a check for the correctness of local problem solve for the blocked case (with S)
//#define CHECK_LOCALSOLVE

// activates some additional checks
//#define DEBUG_INFO

#ifdef TIMING
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#undef DEBUG_INFO
#endif

#ifndef MFEM_DEBUG
#undef CHECK_LOCALSOLVE
#undef CHECK_CONSTR
#undef CHECK_BNDCND
#undef DEBUG_INFO
#endif

// shouldn't be used anymore, we need a monolithic smoother
//#define BLKDIAG_SMOOTHER


// Checking routines used for debugging
// Vector dot product assembled over MPI
double ComputeMPIDotProduct(MPI_Comm comm, const Vector& vec1, const Vector& vec2);

// Vector norm assembled over MPI
double ComputeMPIVecNorm(MPI_Comm comm, const Vector& bvec, char const * string, bool print);

// Computes and prints the norm of ( Funct * y, y )_2,h, assembled over all processes, input vectors on true dofs, matrix on dofs
// w/o proper assembly over shared dofs
double CheckFunctValue(MPI_Comm comm, const BlockMatrix& Funct,
                       const std::vector<HypreParMatrix*> Dof_TrueDof, const BlockVector& truevec,
                       char const * string, bool print);

// Computes and prints the norm of ( Funct * y, y )_2,h, assembled over all processes, everything on truedofs
double CheckFunctValue(MPI_Comm comm, const Operator& Funct, const Vector* truefunctrhs,
                       const Vector& truevec, char const * string, bool print);

// Computes and prints the norm of || Constr * sigma - ConstrRhs ||_2,h, everything on true dofs
bool CheckConstrRes(const Vector& sigma, const HypreParMatrix& Constr, const Vector* ConstrRhs,
                                                char const* string);
// true for truedofs, false for dofs
bool CheckBdrError (const Vector& Candidate, const Vector* Given_bdrdata,
                    const Array<int>& ess_bdr, bool dof_or_truedof);

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

#ifdef COMPARE_MG
    void SetCoarseOperator(Array2D<HypreParMatrix*> & CoarseOperator)
    {
        for ( int blk1 = 0; blk1 < numblocks; ++blk1)
            for ( int blk2 = 0; blk2 < numblocks; ++blk2)
                coarse_matrix->SetBlock(blk1, blk2, CoarseOperator(blk1, blk2));
    }
#endif
};

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

class HcurlGSSSmoother : public BlockOperator
{
private:
    int numblocks;

    // number of GS sweeps for each block
    mutable Array<int> sweeps_num;

    int print_level;

protected:
    const MPI_Comm comm;

    mutable Array2D<HypreParMatrix*> * Funct_hpmat;
    mutable HypreParMatrix* Divfree_hpmat_nobnd;

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

    // stores global HcurlFunct blocks
    // this means that first row and first column are modified using the Divfree operator
    // and the rest are the same as in Funct
    mutable Array2D<HypreParMatrix *> HcurlFunct_global;

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

#ifndef BLKDIAG_SMOOTHER
    mutable Vector * tmp1;
    mutable Vector * tmp2;
#endif

public:
    ~HcurlGSSSmoother();
    HcurlGSSSmoother (Array2D<HypreParMatrix*> & Funct_HpMat,
                                        HypreParMatrix& Divfree_HpMat_nobnd,
                                        const Array<int>& EssBdrtruedofs_Hcurl,
                                        const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                        const Array<int> * SweepsNum,
                                        const Array<int>& Block_Offsets);
    // constructor, deprecated
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
#ifndef BLKDIAG_SMOOTHER
    virtual void MultTranspose (const Vector & x, Vector & y) const;
#else
    virtual void MultTranspose (const Vector & x, Vector & y) const {Mult(x,y);}
#endif

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

// TODO: Fix the block case - boundary conditions for the vectors and matrices in the geometric multigrid
// TODO: Implement the Gauss-Seidel smoother for the block case for the new multigrid (i.e. add off-diagonal blocks)
// TODO: so that finally for the block case geometric MG and new MG w/o Schwarz smoother give the same result
// TODO: Test after all with nonzero boundary conditions for sigma
// TODO: Check the timings and make it faster
// TODO: Clean up the function descriptions
// TODO: Clean up the variables names
// TODO: Maybe, local matrices can also be stored as an improvement (see SolveLocalProblems())?

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
#ifdef BLKDIAG_SMOOTHER
            B11->Mult(xblock.GetBlock(1), yblock.GetBlock(1));
#else
            tmp1 = xblock.GetBlock(1);
            A10.Mult(-1.0, yblock.GetBlock(0), 1.0, tmp1);
            B11->Mult(tmp1, yblock.GetBlock(1));
#endif
        }

        virtual void MultTranspose(const Vector & x, Vector & y) const
        {
            yblock.Update(y.GetData(), offsets);
            xblock.Update(x.GetData(), offsets);

            yblock.GetBlock(1) = 0.0;
            B11->Mult(xblock.GetBlock(1), yblock.GetBlock(1));

#ifdef BLKDIAG_SMOOTHER
            B00->Mult(xblock.GetBlock(0), yblock.GetBlock(0));
#else
            tmp01 = xblock.GetBlock(0);
            A01.Mult(-1.0, yblock.GetBlock(1), 1.0, tmp01);
            B00->Mult(tmp01, yblock.GetBlock(0));
#endif
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
#ifdef BND_FOR_MULTIGRID
                        const std::vector<std::vector<Array<int>*> > & EssBdrTDofs_lvls,
#endif
                        Solver* Coarse_Prec = NULL)
        :
          Solver(Op.RowOffsets().Last()),
          P_(P),
#ifdef BND_FOR_MULTIGRID
          essbdrtdofs_lvls(EssBdrTDofs_lvls),
#endif
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
          CoarsePrec_(Coarse_Prec),
          built_prec(false)
    {
        Operators_.Last() = &Op;

#ifdef BND_FOR_MULTIGRID
        block_offsets.resize(Operators_.Size());
        block_viewers.resize(Operators_.Size());
#endif

        for (int l = Operators_.Size()-1; l >= 0; l--)
        {
            Array<int>& Offsets = Operators_[l]->RowOffsets();
#ifdef BND_FOR_MULTIGRID
            block_viewers[l] = new BlockVector;
            block_offsets[l] = new Array<int>(Offsets.Size());
            for (int i = 0; i < Offsets.Size(); ++i)
                (*block_offsets[l])[i] = Offsets[i];
#endif

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

                /*
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
                */

                HypreParMatrix * A00_c = RAP(&P0, &A00, &P0);
                {
                    Eliminate_ib_block(*A00_c, *essbdrtdofs_lvls[Operators_.Size() - l][0], *essbdrtdofs_lvls[Operators_.Size() - l][0] );
                    HypreParMatrix * temphpmat = A00_c->Transpose();
                    Eliminate_ib_block(*temphpmat, *essbdrtdofs_lvls[Operators_.Size() - l][0], *essbdrtdofs_lvls[Operators_.Size() - l][0] );
                    A00_c = temphpmat->Transpose();
                    A00_c->CopyColStarts();
                    A00_c->CopyRowStarts();
                    SparseMatrix diag;
                    A00_c->GetDiag(diag);
                    diag.MoveDiagonalFirst();
                    delete temphpmat;
                    //Eliminate_bb_block(*A00_c, *essbdrtdofs_lvls[Operators_.Size() - l]);
                }

                HypreParMatrix * A11_c = RAP(&P1, &A11, &P1);
                {
                    Eliminate_ib_block(*A11_c, *essbdrtdofs_lvls[Operators_.Size() - l][1], *essbdrtdofs_lvls[Operators_.Size() - l][1] );
                    HypreParMatrix * temphpmat = A11_c->Transpose();
                    Eliminate_ib_block(*temphpmat, *essbdrtdofs_lvls[Operators_.Size() - l][1], *essbdrtdofs_lvls[Operators_.Size() - l][1] );
                    A11_c = temphpmat->Transpose();
                    A11_c->CopyColStarts();
                    A11_c->CopyRowStarts();
                    SparseMatrix diag;
                    A11_c->GetDiag(diag);
                    diag.MoveDiagonalFirst();
                    delete temphpmat;
                    //Eliminate_bb_block(*A11_c, *essbdrtdofs_lvls[Operators_.Size() - l][1]);
                }

                HypreParMatrix * A01_c = RAP(&P0, &A01, &P1);
                {
                    Eliminate_ib_block(*A01_c, *essbdrtdofs_lvls[Operators_.Size() - l][1], *essbdrtdofs_lvls[Operators_.Size() - l][0] );
                    HypreParMatrix * temphpmat = A01_c->Transpose();
                    Eliminate_ib_block(*temphpmat, *essbdrtdofs_lvls[Operators_.Size() - l][0], *essbdrtdofs_lvls[Operators_.Size() - l][1] );
                    A01_c = temphpmat->Transpose();
                    A01_c->CopyColStarts();
                    A01_c->CopyRowStarts();
                    delete temphpmat;
                }

                HypreParMatrix * A10_c = A01_c->Transpose();


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
        CoarseSolver->SetRelTol(sqrt(1e-12));
#ifdef COMPARE_MG
        CoarseSolver->SetMaxIter(NCOARSEITER);
#else
        CoarseSolver->SetMaxIter(100);
#endif
        CoarseSolver->SetPrintLevel(0);
        CoarseSolver->SetOperator(*Operators_[0]);
        CoarseSolver->iterative_mode = false;

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

#ifdef BND_FOR_MULTIGRID
    const std::vector<std::vector<Array<int>*> > & essbdrtdofs_lvls;
    mutable std::vector<Array<int>*>  block_offsets;
    mutable std::vector<BlockVector*> block_viewers;
#endif

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

class Multigrid : public Solver
{
public:
    Multigrid(HypreParMatrix &Op,
              const Array<HypreParMatrix*> &P,
#ifdef BND_FOR_MULTIGRID
              const std::vector<Array<int>*> & EssBdrTDofs_lvls,
#endif
#ifdef COARSEPREC_AMS
              ParFiniteElementSpace * C_space_coarsest = NULL,
#endif
              Solver *CoarsePrec = NULL)
        :
          Solver(Op.GetNumRows()),
          P_(P),
#ifdef BND_FOR_MULTIGRID
          essbdrtdofs_lvls(EssBdrTDofs_lvls),
#endif
          Operators_(P.Size()+1),
          Smoothers_(Operators_.Size()),
          current_level(Operators_.Size()-1),
          correction(Operators_.Size()),
          residual(Operators_.Size()),
#ifdef COARSEPREC_AMS
          curl_space_coarsest(C_space_coarsest),
#endif
          CoarsePrec_(CoarsePrec),
          built_prec(false)
    {
        Operators_.Last() = &Op;
        for (int l = Operators_.Size()-1; l > 0; l--)
        {
            /*
            // Two steps RAP
            unique_ptr<HypreParMatrix> AP( ParMult(Operators_[l], P[l-1]) );
            Operators_[l-1] = ParMult(PT.get(), AP.get());
            Operators_[l-1]->CopyRowStarts();
            */


            Operators_[l-1] = RAP(P[l-1], Operators_[l], P[l-1]);
//#ifdef COMPARE_MG
            Eliminate_ib_block(*Operators_[l-1], *essbdrtdofs_lvls[Operators_.Size() - l], *essbdrtdofs_lvls[Operators_.Size() - l] );
            HypreParMatrix * temphpmat = Operators_[l-1]->Transpose();
            Eliminate_ib_block(*temphpmat, *essbdrtdofs_lvls[Operators_.Size() - l], *essbdrtdofs_lvls[Operators_.Size() - l] );
            Operators_[l-1] = temphpmat->Transpose();
            Operators_[l-1]->CopyColStarts();
            Operators_[l-1]->CopyRowStarts();
            SparseMatrix diag;
            Operators_[l-1]->GetDiag(diag);
            diag.MoveDiagonalFirst();
            /*
#if defined(COARSEPREC_AMS) && defined(WITH_PENALTY)
            if (l - 1 == 0)
            {
                diag.
            }
#endif
            */
            delete temphpmat;
            //Eliminate_bb_block(*Operators_[l-1], *essbdrtdofs_lvls[Operators_.Size() - l]);
//#else
            Operators_[l-1]->CopyColStarts();
            Operators_[l-1]->CopyRowStarts();
//#endif
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

        //CoarseSolver = new CGSolver_mod(Operators_[0]->GetComm(), *essbdrtdofs_lvls[Operators_.Size() - 1]);
        CoarseSolver = new CGSolver(Operators_[0]->GetComm());
        CoarseSolver->SetAbsTol(sqrt(1e-32));
        CoarseSolver->SetRelTol(sqrt(1e-12));
#ifdef COMPARE_MG
        CoarseSolver->SetMaxIter(NCOARSEITER);
#else
        CoarseSolver->SetMaxIter(100);
#endif
        CoarseSolver->SetPrintLevel(0);
        CoarseSolver->SetOperator(*Operators_[0]);
        CoarseSolver->iterative_mode = false;

        if (!CoarsePrec_)
        {
            built_prec = true;

            /*
            essbdrtdofs_lvls[Operators_.Size() - 1]->Print();

            SparseMatrix diag;
            Operators_[0]->GetDiag(diag);
            //diag.Print();

            Vector vecdiag;
            diag.GetDiag(vecdiag);

            std::cout << "zero entries which appear at the diagonal of coarse matrix \n";
            for (int i = 0; i < vecdiag.Size(); ++i)
            {
                //std::cout << vecdiag[i] << " ";
                //if ( fabs(vecdiag[i]) < 1.0e-14)
                    //std::cout << " is zero! i = " << i << " ";
                //std::cout << "\n";
                if ( fabs(vecdiag[i]) < 1.0e-14)
                    std::cout << i << " ";
            }
            std::cout << "\n";
            */

            //vecdiag.Print();

            //std::cout << "diag unsymmetry measure = " << diag.IsSymmetric() << "\n";

#ifdef COARSEPREC_AMS
            CoarsePrec_ = new HypreAMS(*Operators_[0], curl_space_coarsest);
#ifndef WITH_PENALTY
            ((HypreAMS*)CoarsePrec_)->SetSingularProblem();
#endif
#else
            CoarsePrec_ = new HypreSmoother(*Operators_[0], HypreSmoother::Type::l1GS, 1);
#endif
            //CoarsePrec_ = new HypreSmoother(*Operators_[0], HypreSmoother::Type::l1Jacobi, 1);
            //CoarsePrec_ = new HypreSmoother(*Operators_[0], HypreSmoother::Type::Jacobi, 1);

            /*
            Vector testvec1(CoarsePrec_->Width());
            testvec1 = 1.0;
            const Array<int> *temp = essbdrtdofs_lvls[Operators_.Size() - 1];
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
                testvec1[(*temp)[tdofind]] = 0.0;
            Vector testvec2(CoarsePrec_->Height());
            CoarsePrec_->Mult(testvec1, testvec2);
            testvec2.Print();
            */
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
#ifdef COMPARE_MG
    CGSolver * GetCoarseSolver() const {return CoarseSolver;}
    HypreParMatrix * GetCoarseOp() const {return Operators_[0];}
#endif

private:
    void MG_Cycle() const;

    const Array<HypreParMatrix*> &P_;

#ifdef BND_FOR_MULTIGRID
    const std::vector<Array<int>*> & essbdrtdofs_lvls;
#endif

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

#ifdef COARSEPREC_AMS
    mutable ParFiniteElementSpace * curl_space_coarsest;
#endif


    mutable bool built_prec;
};


} // for namespace mfem

#endif