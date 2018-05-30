#include <iostream>
#include "testhead.hpp"

using namespace std;

namespace mfem
{

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

CoarsestProblemHcurlSolver::CoarsestProblemHcurlSolver(int Size,
                                                       Array2D<HypreParMatrix*> & Funct_Global,
                                                       const HypreParMatrix& DivfreeOp,
                                                       const std::vector<Array<int>* >& EssBdrDofs_blks,
                                                       const std::vector<Array<int> *> &EssBdrTrueDofs_blks,
                                                       const Array<int>& EssBdrDofs_Hcurl,
                                                       const Array<int>& EssBdrTrueDofs_Hcurl)
    : Operator(Size),
      numblocks(Funct_Global.NumRows()),
      comm(DivfreeOp.GetComm()),
      Funct_global(&Funct_Global),
      using_blockop(false),
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
        block_offsets[blk + 1] = (*Funct_global)(blk,blk)->Height();
    block_offsets.PartialSum();

    coarse_offsets.SetSize(numblocks + 1);

    maxIter = 50;
    rtol = 1.e-4;
    atol = 1.e-4;

    sweeps_num = 1;
    Setup();

}

CoarsestProblemHcurlSolver::CoarsestProblemHcurlSolver(int Size,
                                                       BlockOperator& Funct_BlockOp,
                                                       const HypreParMatrix& DivfreeOp,
                                                       const std::vector<Array<int>* >& EssBdrDofs_blks,
                                                       const std::vector<Array<int> *> &EssBdrTrueDofs_blks,
                                                       const Array<int>& EssBdrDofs_Hcurl,
                                                       const Array<int>& EssBdrTrueDofs_Hcurl)
    : Operator(Size),
      numblocks(Funct_BlockOp.NumRowBlocks()),
      comm(DivfreeOp.GetComm()),
      Funct_op(&Funct_BlockOp),
      using_blockop(true),
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
        block_offsets[blk + 1] = Funct_op->GetBlock(blk,blk).Height();
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
            HypreParMatrix * Funct_blk;
            if (using_blockop)
            {
                Funct_blk = dynamic_cast<HypreParMatrix*>(&(Funct_op->GetBlock(blk1,blk2)));
                if (blk1 == blk2)
                    MFEM_ASSERT(Funct_blk, "Unsuccessful cast of diagonal block into HypreParMatrix* \n");
            }
            else
                Funct_blk = (*Funct_global)(blk1,blk2);

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

//#ifdef COMPARE_MG
                const Array<int> *temp_range;
                const Array<int> *temp_dom;
                if (blk1 == 0)
                    temp_range = &essbdrtruedofs_Hcurl;
                else
                    temp_range = essbdrtruedofs_blocks[blk1];

                if (blk2 == 0)
                    temp_dom = &essbdrtruedofs_Hcurl;
                else
                    temp_dom = essbdrtruedofs_blocks[blk2];

                Eliminate_ib_block(*HcurlFunct_global(blk1, blk2), *temp_dom, *temp_range );
                HypreParMatrix * temphpmat = HcurlFunct_global(blk1, blk2)->Transpose();
                Eliminate_ib_block(*temphpmat, *temp_range, *temp_dom );
                HcurlFunct_global(blk1, blk2) = temphpmat->Transpose();
                if (blk1 == blk2)
                {
                    Eliminate_bb_block(*HcurlFunct_global(blk1, blk2), *temp_dom);
                    SparseMatrix diag;
                    HcurlFunct_global(blk1, blk2)->GetDiag(diag);
                    diag.MoveDiagonalFirst();
                }

                HcurlFunct_global(blk1, blk2)->CopyColStarts();
                HcurlFunct_global(blk1, blk2)->CopyRowStarts();
                delete temphpmat;
//#endif
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
    coarseSolver->iterative_mode = false;

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

    /*
#ifdef COMPARE_MG
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            Divfreeop_T->Mult(xblock->GetBlock(blk), coarsetrueRhs->GetBlock(blk));
        else
        {
            coarsetrueRhs->GetBlock(blk) = xblock->GetBlock(blk);
        }
    }

    coarseSolver->Mult(*coarsetrueRhs, *coarsetrueX);
    // imposing bnd conditions on the internal solver output vector

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            Divfreeop.Mult(coarsetrueX->GetBlock(blk), yblock->GetBlock(blk));
        else
            yblock->GetBlock(blk) = coarsetrueX->GetBlock(blk);
    }

    return;
#endif
    */

    //Divfreeop.MultTranspose(xblock->GetBlock(0), coarsetrueRhs->GetBlock(0));
    //Divfreeop_T->Mult(xblock->GetBlock(0), coarsetrueRhs->GetBlock(0));
    //Divfreeop.Mult(coarsetrueRhs->GetBlock(0), yblock->GetBlock(0));

    //yblock->GetBlock(1) = xblock->GetBlock(1);

    //return;

    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
        {
            /*
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                xblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            }
            */

#ifdef CHECK_BNDCND
            const Array<int> * temp;
            temp = essbdrtruedofs_blocks[blk];

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
      dof_trueDof_blocks(&D_tD_blks),
      d_td_funct(NULL),
      using_blockop(false),
      dof_trueDof_L2(D_tD_L2),
      essbdrdofs_blocks(EssBdrDofs_blks),
      essbdrtruedofs_blocks(EssBdrTrueDofs_blks)
{
    finalized = false;

    block_offsets.SetSize(numblocks + 1);
    block_offsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        block_offsets[blk + 1] = (*dof_trueDof_blocks)[blk]->Width();
    block_offsets.PartialSum();

    coarse_rhsfunc_offsets.SetSize(numblocks + 1);
    coarse_offsets.SetSize(numblocks + 2);

    maxIter = 50;
    rtol = 1.e-4;
    atol = 1.e-4;

    Setup();
}

CoarsestProblemSolver::CoarsestProblemSolver(int Size, BlockMatrix& Op_Blksmat,
                                             SparseMatrix& Constr_Spmat,
                                             BlockOperator * D_tD_blkop,
                                             const HypreParMatrix& D_tD_L2,
                                             const std::vector<Array<int>* >& EssBdrDofs_blks,
                                             const std::vector<Array<int>* >& EssBdrTrueDofs_blks)
    : Operator(Size),
      numblocks(Op_Blksmat.NumRowBlocks()),
      comm(D_tD_L2.GetComm()),
      Op_blkspmat(&Op_Blksmat),
      Constr_spmat(&Constr_Spmat),
      dof_trueDof_blocks(NULL),
      d_td_funct(D_tD_blkop),
      using_blockop(true),
      dof_trueDof_L2(D_tD_L2),
      essbdrdofs_blocks(EssBdrDofs_blks),
      essbdrtruedofs_blocks(EssBdrTrueDofs_blks)
{
    finalized = false;

    block_offsets.SetSize(numblocks + 1);
    block_offsets[0] = 0;
    for (int blk = 0; blk < numblocks; ++blk)
        block_offsets[blk + 1] = d_td_funct->GetBlock(blk,blk).Width();
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
    HYPRE_Int glob_num_cols;
    if (using_blockop)
        glob_num_cols = ((HypreParMatrix&)d_td_funct->GetBlock(0,0)).M();
    else
        glob_num_cols = (*dof_trueDof_blocks)[0]->M();
    HYPRE_Int * row_starts = dof_trueDof_L2.GetRowStarts();
    HYPRE_Int * col_starts;
    if (using_blockop)
        col_starts = ((HypreParMatrix&)d_td_funct->GetBlock(0,0)).GetRowStarts();
    else
        col_starts = (*dof_trueDof_blocks)[0]->GetRowStarts();
    HypreParMatrix * temphpmat = new HypreParMatrix(comm, glob_num_rows, glob_num_cols, row_starts, col_starts, Constr_spmat);
    HypreParMatrix * Constr_global;
    if (using_blockop)
        Constr_global = RAP(&dof_trueDof_L2, temphpmat, (HypreParMatrix*)(&d_td_funct->GetBlock(0,0)));
    else
        Constr_global = RAP(&dof_trueDof_L2, temphpmat, (*dof_trueDof_blocks)[0]);

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
            HYPRE_Int glob_num_rows;
            HYPRE_Int glob_num_cols;
            HYPRE_Int * row_starts;
            HYPRE_Int * col_starts;

            // alternative way
            if (using_blockop)
            {
                glob_num_rows = ((HypreParMatrix&)d_td_funct->GetBlock(blk1,blk1)).M();
                glob_num_cols = ((HypreParMatrix&)d_td_funct->GetBlock(blk2,blk2)).M();
                row_starts = ((HypreParMatrix&)d_td_funct->GetBlock(blk1,blk1)).GetRowStarts();
                col_starts = ((HypreParMatrix&)d_td_funct->GetBlock(blk2,blk2)).GetRowStarts();
            }
            else
            {
                glob_num_rows = (*dof_trueDof_blocks)[blk1]->M();
                glob_num_cols = (*dof_trueDof_blocks)[blk2]->M();
                row_starts = (*dof_trueDof_blocks)[blk1]->GetRowStarts();
                col_starts = (*dof_trueDof_blocks)[blk2]->GetRowStarts();;
            }


            HypreParMatrix * temphpmat = new HypreParMatrix(comm, glob_num_rows, glob_num_cols, row_starts, col_starts, &(Op_blkspmat->GetBlock(blk1, blk2)));
            if (using_blockop)
                Funct_global(blk1, blk2) = RAP((HypreParMatrix*)&d_td_funct->GetBlock(blk1,blk1), temphpmat,
                                               (HypreParMatrix*)&d_td_funct->GetBlock(blk2,blk2));
            else
                Funct_global(blk1, blk2) = RAP((*dof_trueDof_blocks)[blk1], temphpmat, (*dof_trueDof_blocks)[blk2]);

            Funct_global(blk1, blk2)->CopyRowStarts();
            Funct_global(blk1, blk2)->CopyColStarts();

            delete temphpmat;
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
        std::cout << "AE = " << AE << " (nAE = " << nAE << ") \n";
        // we don't need to solve any local problem if AE coincides with a single fine grid element
        if (AE_e.RowSize(AE) > 1)
        {
            std::cout << "main case AE > e \n" << std::flush;
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
        } // end of if AE is bigger than single fine grid element
        else
            std::cout << "side case AE == e \n" << std::flush; //why side case is happening in multigrid example? looks like AE_e is not what I thought

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
        // we don't need to store anything if AE coincides with a single fine grid element
        if (AE_e.RowSize(AE) > 1)
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
        } // end of if AE is bigger than a single fine grid element
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
        //AE_eintdofs_blks[blk]->Print();
        Local_inds[blk] = new Array<int>();
    }

    // loop over all AE, solving a local problem in each AE
    int nAE = AE_edofs_L2->Height();
    LUfactors.resize(nAE);

    for( int AE = 0; AE < nAE; ++AE)
    {
        // we need to consider only AE's which are bigger than a single fine grid element
        // this matters, e.g., in AMR setting
        if (AE_e.RowSize(AE) > 1)
        //somehow this breaks the parallel example cfosls_hyperbolic_multigrid.cpp How?
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
                            //Wtmp_j.Print();
                            //std::cout << " Local_inds[blk1] size = " << Local_inds[blk1]->Size() << "\n";
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
                //std::cout << "fails in the next line when AE = 1 \n";
                //std::cout << "idea of the failure is that for AMR example mesh is refined only locally,"
                             //" so some more degenrate local systems might happen and they are handled"
                             //" incorrectly by the current code \n";
                //sub_Constr.Print();
                //invABT.Print();
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
        } // end of if AE is bigger than one finer element
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

DivConstraintSolver::~DivConstraintSolver()
{
    for (int i = 0; i < truetempvec_lvls.Size(); ++i)
        delete truetempvec_lvls[i];
    for (int i = 0; i < truetempvec2_lvls.Size(); ++i)
        delete truetempvec2_lvls[i];
    for (int i = 0; i < trueresfunc_lvls.Size(); ++i)
        delete trueresfunc_lvls[i];
    for (int i = 0; i < truesolupdate_lvls.Size(); ++i)
        delete truesolupdate_lvls[i];

    if (own_data)
    {
        for (int i = 0; i < AE_e.Size(); ++i)
            delete AE_e[i];

        for (unsigned int i = 0; i < essbdr_tdofs_funct_coarse.size(); ++i)
            delete essbdr_tdofs_funct_coarse[i];

        for (unsigned int i = 0; i < essbdr_dofs_funct_coarse.size(); ++i)
            delete essbdr_dofs_funct_coarse[i];

        for (unsigned int i = 0; i < el2dofs_row_offsets.size(); ++i)
            delete el2dofs_row_offsets[i];

        for (unsigned int i = 0; i < el2dofs_col_offsets.size(); ++i)
            delete el2dofs_col_offsets[i];

        for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
            delete fullbdr_attribs[i];

        for (unsigned int i = 0; i < offsets_funct.size(); ++i)
            delete offsets_funct[i];

        for (int i = 0; i < BlockOps_lvls.Size(); ++i)
            delete BlockOps_lvls[i];

        for (unsigned int i = 0; i < offsets_sp_funct.size(); ++i)
            delete offsets_sp_funct[i];

        for (int i = 0; i < Funct_mat_lvls.Size(); ++i)
            delete Funct_mat_lvls[i];

        for (int i = 0; i < Constraint_mat_lvls.Size(); ++i)
            delete Constraint_mat_lvls[i];

        for (int i = 0; i < Smoothers_lvls.Size(); ++i)
            delete Smoothers_lvls[i];

        for (int i = 0; i < LocalSolvers_lvls.Size(); ++i)
            delete LocalSolvers_lvls[i];
    }
}

DivConstraintSolver::DivConstraintSolver(FOSLSProblem& problem_, GeneralHierarchy& hierarchy_,
                                         bool optimized_localsolvers_, bool verbose_)
    : problem(&problem_),
      hierarchy(&hierarchy_),
      optimized_localsolvers(optimized_localsolvers_),
      update_counter(hierarchy->GetUpdateCounter()),
      own_data(true),
      num_levels(hierarchy->Nlevels()),
      comm(problem->GetComm()),
      numblocks(problem->GetFEformulation().Nblocks()),
      verbose(verbose_)
{
    MFEM_ASSERT(problem->GetParMesh()->GetNE() == hierarchy->GetPmesh(0)->GetNE(),
                "Given FOSLS problem must be defined on the finest level of the "
                "hierarchy in the current implementation");
    //int num = ( num_levels > 1 ? num_levels - 1 : 1);

    P_L2.SetSize(num_levels - 1);
    AE_e.SetSize(num_levels - 1);

    const Array<SpaceName>* space_names_funct =
            problem->GetFEformulation().GetFormulation()->GetFunctSpacesDescriptor();

    int numblocks_funct = space_names_funct->Size();

    const Array<SpaceName>* space_names_problem =
            problem->GetFEformulation().GetFormulation()->GetSpacesDescriptor();

    TrueP_Func.SetSize(num_levels - 1);
    offsets_funct.resize(num_levels);
    offsets_funct[0] = hierarchy->ConstructTrueOffsetsforFormul(0, *space_names_funct);

    size = (*offsets_funct[0])[numblocks_funct];

    offsets_sp_funct.resize(num_levels);
    offsets_sp_funct[0] = hierarchy->ConstructOffsetsforFormul(0, *space_names_funct);

    Funct_mat_lvls.SetSize(num_levels);
    Funct_mat_lvls[0] = problem->ConstructFunctBlkMat(*offsets_sp_funct[0]);

    Constraint_mat_lvls.SetSize(num_levels);
    ParMixedBilinearForm *Divblock = new ParMixedBilinearForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                                              hierarchy->GetSpace(SpaceName::L2, 0));
    Divblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    Divblock->Assemble();
    Divblock->Finalize();
    Constraint_mat_lvls[0] = Divblock->LoseMat();
    delete Divblock;

    Smoothers_lvls.SetSize(num_levels - 1);
    Array<int> SweepsNum(numblocks_funct);

    const Array<int> &essbdr_attribs_Hcurl = problem->GetBdrConditions().GetBdrAttribs(0);
    std::vector<Array<int>*>& essbdr_attribs = problem->GetBdrConditions().GetAllBdrAttribs();

    BlockOps_lvls.SetSize(num_levels);
    BlockOps_lvls[0] = problem->GetFunctOp(*offsets_funct[0]);
    Func_global_lvls.resize(num_levels);
    Func_global_lvls[0] = BlockOps_lvls[0];

    LocalSolvers_lvls.SetSize(num_levels - 1);
    el2dofs_row_offsets.resize(num_levels - 1);
    el2dofs_col_offsets.resize(num_levels - 1);

    fullbdr_attribs.resize(numblocks_funct);
    for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
    {
        fullbdr_attribs[i] = new Array<int>(problem->GetParMesh()->bdr_attributes.Max());
        (*fullbdr_attribs[i]) = 1;
    }

    truesolupdate_lvls.SetSize(num_levels);
    truetempvec_lvls  .SetSize(num_levels);
    truetempvec2_lvls .SetSize(num_levels);
    trueresfunc_lvls  .SetSize(num_levels);

    truesolupdate_lvls[0] = new BlockVector(*offsets_funct[0]);
    truetempvec_lvls[0]   = new BlockVector(*offsets_funct[0]);
    truetempvec2_lvls[0]  = new BlockVector(*offsets_funct[0]);
    trueresfunc_lvls[0]   = new BlockVector(*offsets_funct[0]);

    for (int l = 0; l < num_levels - 1; ++l)
    {
        P_L2[l] = hierarchy->GetPspace(SpaceName::L2, l);
        AE_e[l] = Transpose(*P_L2[l]);

        offsets_funct[l + 1] = hierarchy->ConstructTrueOffsetsforFormul(l + 1, *space_names_funct);
        TrueP_Func[l] = hierarchy->ConstructTruePforFormul(l, *space_names_funct,
                                                           *offsets_funct[l], *offsets_funct[l + 1]);

        BlockOps_lvls[l + 1] = new RAPBlockHypreOperator(*TrueP_Func[l],
                *BlockOps_lvls[l], *TrueP_Func[l], *offsets_funct[l + 1]);

        Func_global_lvls[l + 1] = BlockOps_lvls[l + 1];


        std::vector<Array<int>* > &essbdr_tdofs_funct =
                hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, l + 1);
        EliminateBoundaryBlocks(*BlockOps_lvls[l + 1], essbdr_tdofs_funct);

        SweepsNum = ipow(1, l); // = 1
        Smoothers_lvls[l] = new HcurlGSSSmoother(*BlockOps_lvls[l],
                                                 *hierarchy->GetDivfreeDop(l),
                                                 hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL,
                                                                           essbdr_attribs_Hcurl, l),
                                                 hierarchy->GetEssBdrTdofsOrDofs("tdof",
                                                                           *space_names_funct,
                                                                           essbdr_attribs, l),
                                                 &SweepsNum, *offsets_funct[l]);

        offsets_sp_funct[l + 1] = hierarchy->ConstructOffsetsforFormul(l + 1, *space_names_funct);

        Constraint_mat_lvls[l + 1] = RAP(*hierarchy->GetPspace(SpaceName::L2, l),
                                        *Constraint_mat_lvls[l], *hierarchy->GetPspace(SpaceName::HDIV, l));

        BlockMatrix * P_Funct = hierarchy->ConstructPforFormul
                (l, *space_names_funct, *offsets_sp_funct[l], *offsets_sp_funct[l + 1]);
        Funct_mat_lvls[l + 1] = RAP(*P_Funct, *Funct_mat_lvls[l], *P_Funct);

        delete P_Funct;

        el2dofs_row_offsets[l] = new Array<int>();
        el2dofs_col_offsets[l] = new Array<int>();

        if (numblocks_funct == 2) // both sigma and S are present -> Hdiv-H1 formulation
        {
            LocalSolvers_lvls[l] = new LocalProblemSolverWithS(BlockOps_lvls[l]->Height(), *Funct_mat_lvls[l],
                                                               *Constraint_mat_lvls[l],
                                                               hierarchy->GetDofTrueDof(*space_names_funct, l),
                                                               *AE_e[l],
                                                               *hierarchy->GetElementToDofs(*space_names_funct, l,
                                                                                            *el2dofs_row_offsets[l],
                                                                                            *el2dofs_col_offsets[l]),
                                                               *hierarchy->GetElementToDofs(SpaceName::L2, l),
                                                               hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct,
                                                                                               fullbdr_attribs, l),
                                                               hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct,
                                                                              essbdr_attribs, l),
                                                               optimized_localsolvers);
        }
        else // no S -> Hdiv-L2 formulation
        {
            LocalSolvers_lvls[l] = new LocalProblemSolver(BlockOps_lvls[l]->Height(), *Funct_mat_lvls[l],
                                                              *Constraint_mat_lvls[l],
                                                              hierarchy->GetDofTrueDof(*space_names_funct, l),
                                                              *AE_e[l],
                                                              *hierarchy->GetElementToDofs(*space_names_funct, l,
                                                                                          *el2dofs_row_offsets[l],
                                                                                          *el2dofs_col_offsets[l]),
                                                              *hierarchy->GetElementToDofs(SpaceName::L2, l),
                                                              hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct,
                                                                                       fullbdr_attribs, l),
                                                              hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct,
                                                                                       essbdr_attribs, l),
                                                              optimized_localsolvers);
        }

        truesolupdate_lvls[l + 1] = new BlockVector(*offsets_funct[l + 1]);
        truetempvec_lvls[l + 1]   = new BlockVector(*offsets_funct[l + 1]);
        truetempvec2_lvls[l + 1]  = new BlockVector(*offsets_funct[l + 1]);
        trueresfunc_lvls[l + 1]   = new BlockVector(*offsets_funct[l + 1]);
    }

    essbdr_tdofs_funct_coarse = hierarchy->GetEssBdrTdofsOrDofs
            ("tdof", *space_names_funct, essbdr_attribs, num_levels - 1);
    essbdr_dofs_funct_coarse = hierarchy->GetEssBdrTdofsOrDofs
            ("dof", *space_names_funct, essbdr_attribs, num_levels - 1);

    int coarse_size = 0;
    for (int i = 0; i < space_names_problem->Size(); ++i)
        coarse_size += hierarchy->GetSpace((*space_names_problem)[i], num_levels - 1)->TrueVSize();

    CoarseSolver =  new CoarsestProblemSolver(coarse_size,
                                              *Funct_mat_lvls[num_levels - 1],
            *Constraint_mat_lvls[num_levels - 1],
            hierarchy->GetDofTrueDof(*space_names_funct, num_levels - 1,
                                     row_offsets_coarse, col_offsets_coarse),
            *hierarchy->GetDofTrueDof(SpaceName::L2, num_levels - 1),
            essbdr_dofs_funct_coarse,
            essbdr_tdofs_funct_coarse);

    CoarseSolver->SetMaxIter(70000);
    CoarseSolver->SetAbsTol(1.0e-18);
    CoarseSolver->SetRelTol(1.0e-18);
    CoarseSolver->ResetSolverParams();

    Constr_global = (HypreParMatrix*)(&problem->GetOp_nobnd()->GetBlock(numblocks_funct,0));
}


DivConstraintSolver::DivConstraintSolver(MPI_Comm Comm, int NumLevels,
                       Array< SparseMatrix*> &AE_to_e,
                       Array< BlockOperator*>& TrueProj_Func,
                       Array< SparseMatrix*> &Proj_L2,
                       std::vector<std::vector<Array<int> *> > &EssBdrTrueDofs_Func,
                       std::vector<Operator*> & Func_Global_lvls,
                       HypreParMatrix &Constr_Global,
                       Vector& ConstrRhsVec,
                       Array<Operator*>& Smoothers_Lvls,
                       Array<LocalProblemSolver*>* LocalSolvers,
                       CoarsestProblemSolver* CoarsestSolver, bool verbose_)
     : size(Func_Global_lvls[0]->Height()),
       problem(NULL),
       hierarchy(NULL),
       update_counter(0),
       own_data(false),
       num_levels(NumLevels),
       //AE_e(AE_to_e),
       comm(Comm),
       //TrueP_Func(TrueProj_Func), P_L2(Proj_L2),
       essbdrtruedofs_Func(EssBdrTrueDofs_Func),
       numblocks(TrueProj_Func[0]->NumRowBlocks()),
       ConstrRhs(&ConstrRhsVec),
       //Smoothers_lvls(Smoothers_Lvls),
       //Func_global_lvls(Func_Global_lvls),
       Constr_global(&Constr_Global),
       verbose(verbose_)
{
    Func_global_lvls.resize(Func_Global_lvls.size());
    for (unsigned int i = 0; i < Func_global_lvls.size(); ++i)
        Func_global_lvls[i] = Func_Global_lvls[i];

    AE_e.SetSize(AE_to_e.Size());
    for (int i = 0; i < AE_e.Size(); ++i)
        AE_e[i] = AE_to_e[i];

    TrueP_Func.SetSize(TrueProj_Func.Size());
    for (int i = 0; i < TrueP_Func.Size(); ++i)
        TrueP_Func[i] = TrueProj_Func[i];

    P_L2.SetSize(Proj_L2.Size());
    for (int i = 0; i < P_L2.Size(); ++i)
        P_L2[i] = Proj_L2[i];

    Smoothers_lvls.SetSize(Smoothers_Lvls.Size());
    for (int i = 0; i < Smoothers_lvls.Size(); ++i)
        Smoothers_lvls[i] = Smoothers_Lvls[i];

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

    LocalSolvers_lvls.SetSize(num_levels - 1);
    for (int l = 0; l < num_levels - 1; ++l)
        if (LocalSolvers)
            LocalSolvers_lvls[l] = (*LocalSolvers)[l];
        else
            LocalSolvers_lvls[l] = NULL;

    // for all levels except the coarsest
    for (int l = 0; l < num_levels - 1; ++l)
    {
        truetempvec_lvls[l + 1] = new BlockVector(TrueP_Func[l]->ColOffsets());
        truetempvec2_lvls[l + 1] = new BlockVector(TrueP_Func[l]->ColOffsets());
        truesolupdate_lvls[l + 1] = new BlockVector(TrueP_Func[l]->ColOffsets());
        trueresfunc_lvls[l + 1] = new BlockVector(TrueP_Func[l]->ColOffsets());
    } // end of loop over finer levels
}

void DivConstraintSolver::Update(bool recoarsen)
{
    // Update() is meaningless if the solver is not based on the hierarchy
    if (!hierarchy)
        return;

    MFEM_ASSERT(problem->GetParMesh()->GetNE() == hierarchy->GetPmesh(0)->GetNE(),
                "Given FOSLS problem must be defined on the finest level of the "
                "hierarchy in the current implementation. Probably it was not updated "
                "by a call to DivConstraintSolver::UpdateProblem() after the hierarchy was updated \n");

    int hierarchy_upd_cnt = hierarchy->GetUpdateCounter();
    if (update_counter != hierarchy_upd_cnt)
    {
        MFEM_ASSERT(update_counter == hierarchy_upd_cnt - 1,
                    "Current implementation allows the update counters to differ no more than by one");

        // prepending one more level
        SparseMatrix * P_L2_new = hierarchy->GetPspace(SpaceName::L2, 0);
        P_L2.Prepend(P_L2_new);

        SparseMatrix * AE_e_new = Transpose(*P_L2[0]);
        AE_e.Prepend(AE_e_new);

        const Array<SpaceName>* space_names_funct =
                problem->GetFEformulation().GetFormulation()->GetFunctSpacesDescriptor();

        int numblocks_funct = space_names_funct->Size();

        const Array<int> * offsets_funct_new = hierarchy->ConstructTrueOffsetsforFormul(0, *space_names_funct);
        offsets_funct.push_front(offsets_funct_new);

        const Array<int> * offsets_sp_funct_new = hierarchy->ConstructOffsetsforFormul(0, *space_names_funct);
        offsets_sp_funct.push_front(offsets_sp_funct_new);

        BlockMatrix * Funct_mat_new = problem->ConstructFunctBlkMat(*offsets_sp_funct[0]);
        Funct_mat_lvls.Prepend(Funct_mat_new);

        ParMixedBilinearForm *Divblock = new ParMixedBilinearForm(hierarchy->GetSpace(SpaceName::HDIV, 0),
                                                                  hierarchy->GetSpace(SpaceName::L2, 0));
        Divblock->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
        Divblock->Assemble();
        Divblock->Finalize();
        SparseMatrix * Constraint_mat_new = Divblock->LoseMat();
        delete Divblock;
        Constraint_mat_lvls.Prepend(Constraint_mat_new);

        Array<int> SweepsNum(numblocks_funct);
        SweepsNum = 1;

        const Array<int> &essbdr_attribs_Hcurl = problem->GetBdrConditions().GetBdrAttribs(0);
        std::vector<Array<int>*>& essbdr_attribs = problem->GetBdrConditions().GetAllBdrAttribs();

        BlockOperator * BlockOps_new = problem->GetFunctOp(*offsets_funct[0]);
        BlockOps_lvls.Prepend(BlockOps_new);

        Func_global_lvls.push_front(BlockOps_new);

        size = (*offsets_funct[0])[numblocks_funct];

        fullbdr_attribs.resize(numblocks_funct);
        for (unsigned int i = 0; i < fullbdr_attribs.size(); ++i)
        {
            fullbdr_attribs[i] = new Array<int>(problem->GetParMesh()->bdr_attributes.Max());
            (*fullbdr_attribs[i]) = 1;
        }

        BlockOperator * TrueP_Func_new = hierarchy->ConstructTruePforFormul(0, *space_names_funct,
                                                           *offsets_funct[0], *offsets_funct[1]);

        TrueP_Func.Prepend(TrueP_Func_new);

        BlockVector * truesolupdate_new = new BlockVector(TrueP_Func[0]->RowOffsets());
        truesolupdate_lvls.Prepend(truesolupdate_new);
        BlockVector * truetempvec_new = new BlockVector(TrueP_Func[0]->RowOffsets());
        truetempvec_lvls.Prepend(truetempvec_new);
        BlockVector * truetempvec2_new = new BlockVector(TrueP_Func[0]->RowOffsets());
        truetempvec2_lvls.Prepend(truetempvec2_new);
        BlockVector * trueresfunc_new = new BlockVector(TrueP_Func[0]->RowOffsets());
        trueresfunc_lvls.Prepend(trueresfunc_new);

        HcurlGSSSmoother * Smoother_new = new HcurlGSSSmoother(*BlockOps_lvls[0], *hierarchy->GetDivfreeDop(0),
                hierarchy->GetEssBdrTdofsOrDofs("tdof", SpaceName::HCURL, essbdr_attribs_Hcurl, 0),
                hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, 0),
                &SweepsNum, *offsets_funct[0]);
        Smoothers_lvls.Prepend(Smoother_new);

        Array<int>* el2dofs_row_offsets_new = new Array<int>();
        Array<int>* el2dofs_col_offsets_new = new Array<int>();

        LocalProblemSolver* LocalSolver_new;
        if (numblocks_funct == 2) // both sigma and S are present -> Hdiv-H1 formulation
        {
            LocalSolver_new = new LocalProblemSolverWithS(BlockOps_lvls[0]->Height(),
                    *Funct_mat_lvls[0], *Constraint_mat_lvls[0],
                    hierarchy->GetDofTrueDof(*space_names_funct, 0), *AE_e[0],
                    *hierarchy->GetElementToDofs(*space_names_funct, 0,
                                                 *el2dofs_row_offsets_new,  *el2dofs_col_offsets_new),
                    *hierarchy->GetElementToDofs(SpaceName::L2, 0),
                    hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct, fullbdr_attribs, 0),
                    hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct, essbdr_attribs, 0),
                    optimized_localsolvers);
        }
        else // no S -> Hdiv-L2 formulation
        {
            LocalSolver_new = new LocalProblemSolver(BlockOps_lvls[0]->Height(),
                    *Funct_mat_lvls[0], *Constraint_mat_lvls[0],
                    hierarchy->GetDofTrueDof(*space_names_funct, 0), *AE_e[0],
                    *hierarchy->GetElementToDofs(*space_names_funct, 0,
                                                 *el2dofs_row_offsets_new, *el2dofs_col_offsets_new),
                    *hierarchy->GetElementToDofs(SpaceName::L2, 0),
                    hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct, fullbdr_attribs, 0),
                    hierarchy->GetEssBdrTdofsOrDofs("dof", *space_names_funct, essbdr_attribs, 0),
                    optimized_localsolvers);
        }
        LocalSolvers_lvls.Prepend(LocalSolver_new);

        el2dofs_row_offsets.push_front(el2dofs_row_offsets_new);
        el2dofs_col_offsets.push_front(el2dofs_col_offsets_new);

        Constr_global = (HypreParMatrix*)(&problem->GetOp_nobnd()->GetBlock(numblocks_funct,0));

        // recoarsening local and global matrices
        if (recoarsen)
        {
            for (int l = 0; l < num_levels - 1; ++l)
            {
                BlockOps_lvls[l + 1] = new RAPBlockHypreOperator(*TrueP_Func[l],
                        *BlockOps_lvls[l], *TrueP_Func[l], *offsets_funct[l + 1]);

                std::vector<Array<int>* > &essbdr_tdofs_funct =
                        hierarchy->GetEssBdrTdofsOrDofs("tdof", *space_names_funct, essbdr_attribs, l + 1);
                EliminateBoundaryBlocks(*BlockOps_lvls[l + 1], essbdr_tdofs_funct);

                Func_global_lvls[l + 1] = BlockOps_lvls[l + 1];

                Constraint_mat_lvls[l + 1] = RAP(*hierarchy->GetPspace(SpaceName::L2, l),
                                                *Constraint_mat_lvls[l], *hierarchy->GetPspace(SpaceName::HDIV, l));

                BlockMatrix * P_Funct = hierarchy->ConstructPforFormul
                        (l, *space_names_funct, *offsets_sp_funct[l], *offsets_sp_funct[l + 1]);
                Funct_mat_lvls[l + 1] = RAP(*P_Funct, *Funct_mat_lvls[l], *P_Funct);

                delete P_Funct;

            }

            //MFEM_ABORT("Not implemented \n");
        }

        update_counter = hierarchy_upd_cnt;
    }

}


// the start_guess is on dofs
// (*) returns particular solution as a vector on true dofs!
void DivConstraintSolver::FindParticularSolution(const Vector& start_guess,
                                                         Vector& partsol, const Vector &constr_rhs, bool verbose) const
{
    MFEM_ASSERT(start_guess.Size() == size && partsol.Size() == size,
                "Sizes of all arguments must be equal to the size of the solver");

    const Array<int>* offsets;
    if (own_data)
        offsets = offsets_funct[0];
    else
    {
        MFEM_ASSERT(num_levels > 1, "Old interface works only if number of levels"
                                    " was more than 1 in the constructor");
        offsets = &TrueP_Func[0]->RowOffsets();
    }
    const BlockVector start_guess_viewer(start_guess.GetData(), *offsets);
    BlockVector partsol_viewer(partsol.GetData(), *offsets);

    // old variant, doesn't work when there is only one level and thus TrueP_Func is empty.
    //const BlockVector start_guess_viewer(start_guess.GetData(), TrueP_Func[0]->RowOffsets());
    //BlockVector partsol_viewer(partsol.GetData(), TrueP_Func[0]->RowOffsets());

    // checking if the given initial vector satisfies the divergence constraint
    Vector rhs_constr(Constr_global->Height());
    Constr_global->Mult(start_guess_viewer.GetBlock(0), rhs_constr);
    rhs_constr -= constr_rhs;
    rhs_constr *= -1.0;
    // 3.1 if not, computing the particular solution
    if ( ComputeMPIVecNorm(comm, rhs_constr,"", verbose) > 1.0e-14 )
    {
        if (verbose)
            std::cout << "Initial vector does not satisfy the divergence constraint. "
                         "A V-cycle will be performed to find the particular solution. \n";
    }
    else
    {
        if (verbose)
            std::cout << "Initial vector already satisfies divergence constraint. "
                         "No need to perform a V-cycle \n";
        partsol = start_guess;
        return;
    }

    // variable-size vectors (initialized with the finest level sizes) on dofs
    Vector Qlminus1_f(rhs_constr.Size());     // stores P_l^T rhs_constr_l
    Vector workfvec(rhs_constr.Size());       // used only in ComputeLocalRhsConstr()

    // 0. Compute rhs in the functional for the finest level
    UpdateTrueResidual(0, NULL, start_guess_viewer, *trueresfunc_lvls[0] );

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

    partsol = start_guess;
    partsol += *truesolupdate_lvls[0];

#ifdef CHECK_CONSTR
    CheckConstrRes(partsol_viewer.GetBlock(0), *Constr_global,
                    &constr_rhs, "for the particular solution inside in the end");
#endif

}

void DivConstraintSolver::MultTrueFunc(int l, double coeff, const BlockVector& x_l, BlockVector &rhs_l) const
{
    Func_global_lvls[l]->Mult(x_l, rhs_l);
    rhs_l *= coeff;
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

#ifndef BLKDIAG_SMOOTHER
    delete tmp1;
    if (numblocks > 1)
        delete tmp2;
#endif

    //delete CTMC;
    //delete CTMC_global;
    for (int blk1 = 0; blk1 < numblocks; ++blk1)
        for (int blk2 = 0; blk2 < numblocks; ++blk2)
            if (blk1 == 0 || blk2 == 0)
                delete HcurlFunct_global(blk1,blk2);

    for (int i = 0; i < Smoothers.Size(); ++i)
        delete Smoothers[i];
}

// FIXME: Remove this, too old and unused
// deprecated
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

    Setup();
}

HcurlGSSSmoother::HcurlGSSSmoother (BlockOperator& Funct_HpBlockMat,
                                    const HypreParMatrix& Divfree_HpMat_nobnd,
                                    const Array<int>& EssBdrtruedofs_Hcurl,
                                    const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                    const Array<int> * SweepsNum,
                                    const Array<int>& Block_Offsets)
    : BlockOperator(Block_Offsets),
      numblocks(Funct_HpBlockMat.NumRowBlocks()),
      print_level(0),
      comm(Divfree_HpMat_nobnd.GetComm()),
      Funct_op (&Funct_HpBlockMat),
      using_blockop(true),
      Divfree_hpmat_nobnd (&Divfree_HpMat_nobnd),
      essbdrtruedofs_Hcurl(EssBdrtruedofs_Hcurl),
      essbdrtruedofs_Funct(EssBdrTrueDofs_Funct)
{
    block_offsets.SetSize(numblocks + 1);
    for ( int i = 0; i < numblocks + 1; ++i)
        block_offsets[i] = Block_Offsets[i];

    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

#ifndef BLKDIAG_SMOOTHER
    tmp1 = new Vector(Divfree_hpmat_nobnd->Width());
    if (numblocks > 1)
        tmp2 = new Vector(xblock->GetBlock(1).Size());
#endif

    trueblock_offsets.SetSize(numblocks + 1);
    trueblock_offsets[0] = 0;
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            trueblock_offsets[1] = Divfree_hpmat_nobnd->Width();
        else
            trueblock_offsets[blk + 1] = Funct_op->GetBlock(blk,blk).Height();
    }
    trueblock_offsets.PartialSum();

    Smoothers.SetSize(numblocks);

    sweeps_num.SetSize(numblocks);
    if (SweepsNum)
        for ( int blk = 0; blk < numblocks; ++blk)
            sweeps_num[blk] = (*SweepsNum)[blk];
    else
        sweeps_num = 1;

    HcurlFunct_global.SetSize(numblocks, numblocks);
    for (int rowblk = 0; rowblk < numblocks; ++rowblk)
        for (int colblk = 0; colblk < numblocks; ++colblk)
            HcurlFunct_global(rowblk, colblk) = NULL;

    Setup();
}

HcurlGSSSmoother::HcurlGSSSmoother (Array2D<HypreParMatrix*> & Funct_HpMat,
                                    const HypreParMatrix& Divfree_HpMat_nobnd,
                                    const Array<int>& EssBdrtruedofs_Hcurl,
                                    const std::vector<Array<int>* >& EssBdrTrueDofs_Funct,
                                    const Array<int> * SweepsNum,
                                    const Array<int>& Block_Offsets)
    : BlockOperator(Block_Offsets),
      numblocks(Funct_HpMat.NumRows()),
      print_level(0),
      comm(Divfree_HpMat_nobnd.GetComm()),
      Funct_hpmat (&Funct_HpMat),
      using_blockop(false),
      Divfree_hpmat_nobnd (&Divfree_HpMat_nobnd),
      essbdrtruedofs_Hcurl(EssBdrtruedofs_Hcurl),
      essbdrtruedofs_Funct(EssBdrTrueDofs_Funct)
{

    block_offsets.SetSize(numblocks + 1);
    for ( int i = 0; i < numblocks + 1; ++i)
        block_offsets[i] = Block_Offsets[i];

    xblock = new BlockVector(block_offsets);
    yblock = new BlockVector(block_offsets);

#ifndef BLKDIAG_SMOOTHER
    tmp1 = new Vector(Divfree_hpmat_nobnd->Width());
    if (numblocks > 1)
        tmp2 = new Vector(xblock->GetBlock(1).Size());
#endif

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

    HcurlFunct_global.SetSize(numblocks, numblocks);
    for (int rowblk = 0; rowblk < numblocks; ++rowblk)
        for (int colblk = 0; colblk < numblocks; ++colblk)
            HcurlFunct_global(rowblk, colblk) = NULL;

    Setup();
}


#ifndef BLKDIAG_SMOOTHER
void HcurlGSSSmoother::MultTranspose(const Vector & x, Vector & y) const
{
#ifdef TIMING
    MPI_Barrier(comm);
    chrono2.Clear();
    chrono2.Start();
#endif

    if (print_level)
        std::cout << "Smoothing with HcurlGSS smoother \n";

    xblock->Update(x.GetData(), block_offsets);
    yblock->Update(y.GetData(), block_offsets);

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
            }
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

    if (numblocks == 1)
    {
        Smoothers[0]->Mult(truerhs->GetBlock(0), truex->GetBlock(0));
    }
    else // numblocks = 2
    {
        Smoothers[1]->Mult(truerhs->GetBlock(1), truex->GetBlock(1));
        *tmp1 = truerhs->GetBlock(0);
        HcurlFunct_global(0,1)->Mult(-1.0, truex->GetBlock(1), 1.0, *tmp1);
        Smoothers[0]->Mult(*tmp1, truex->GetBlock(0));
    }

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
            }
    }

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
        }
    }

#ifdef TIMING
    MPI_Barrier(comm);
    chrono.Stop();
    time_afterintmult += chrono.RealTime();

    MPI_Barrier(comm);
    chrono2.Stop();
    time_globalmult += chrono2.RealTime();
#endif

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

    /*
#ifdef COMPARE_MG
    for ( int blk = 0; blk < numblocks; ++blk)
    {
        const Array<int> *temp = essbdrtruedofs_Funct[blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            xblock->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
    }

    for (int blk = 0; blk < numblocks; ++blk)
    {
        if (blk == 0)
            Divfree_hpmat_nobnd->MultTranspose(xblock->GetBlock(0), truerhs->GetBlock(0));
        else
            truerhs->GetBlock(blk) = xblock->GetBlock(blk);
    }

    for ( int blk = 0; blk < numblocks; ++blk)
        Smoothers[blk]->Mult(truerhs->GetBlock(blk), truex->GetBlock(blk));

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
        }
    }


    return;
#endif
    */

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
    //std::cout << "input to Smoothers mult \n";
    //truerhs->Print();

#ifdef BLKDIAG_SMOOTHER
    for ( int blk = 0; blk < numblocks; ++blk)
        Smoothers[blk]->Mult(truerhs->GetBlock(blk), truex->GetBlock(blk));
#else
    if (numblocks == 1)
    {
        Smoothers[0]->Mult(truerhs->GetBlock(0), truex->GetBlock(0));
    }
    else // numblocks = 2
    {
        Smoothers[0]->Mult(truerhs->GetBlock(0), truex->GetBlock(0));
        *tmp2 = truerhs->GetBlock(1);
        HcurlFunct_global(1,0)->Mult(-1.0, truex->GetBlock(0), 1.0, *tmp2);
        Smoothers[1]->Mult(*tmp2, truex->GetBlock(1));
    }
#endif // for #else to #ifdef BLKDIAG

    //std::cout << "output to Smoothers mult \n";
    //truex->Print();

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

    HypreParMatrix * Divfree_hpmat_nobnd_T = Divfree_hpmat_nobnd->Transpose();
    for ( int blk1 = 0; blk1 < numblocks; ++blk1)
    {
        for ( int blk2 = 0; blk2 < numblocks; ++blk2)
        {
            HypreParMatrix * Funct_blk;
            if (using_blockop)
            {
                Funct_blk = dynamic_cast<HypreParMatrix*>(&(Funct_op->GetBlock(blk1,blk2)));
                if (blk1 == blk2)
                    MFEM_ASSERT(Funct_blk, "Unsuccessful cast of diagonal block into HypreParMatrix* \n");
            }
            else
                Funct_blk = (*Funct_hpmat)(blk1,blk2);

            if (Funct_blk)
            {
                if (blk1 == 0)
                {
                    HypreParMatrix * temp1 = ParMult(Divfree_hpmat_nobnd_T, Funct_blk);
                    temp1->CopyRowStarts();
                    temp1->CopyColStarts();

                    if (blk2 == 0)
                    {
                        HcurlFunct_global(blk1, blk2) = RAP(Divfree_hpmat_nobnd, Funct_blk, Divfree_hpmat_nobnd);

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
                                                                  Divfree_hpmat_nobnd);
                    HcurlFunct_global(blk1, blk2)->CopyRowStarts();
                    HcurlFunct_global(blk1, blk2)->CopyColStarts();
                }
                else
                {
                    HcurlFunct_global(blk1, blk2)  = Funct_blk;
                }

//#ifdef COMPARE_MG
                const Array<int> *temp_range;
                const Array<int> *temp_dom;
                if (blk1 == 0)
                    temp_range = &essbdrtruedofs_Hcurl;
                else
                    temp_range = essbdrtruedofs_Funct[blk1];

                if (blk2 == 0)
                    temp_dom = &essbdrtruedofs_Hcurl;
                else
                    temp_dom = essbdrtruedofs_Funct[blk2];

                Eliminate_ib_block(*HcurlFunct_global(blk1, blk2), *temp_dom, *temp_range );
                HypreParMatrix * temphpmat = HcurlFunct_global(blk1, blk2)->Transpose();
                Eliminate_ib_block(*temphpmat, *temp_range, *temp_dom );
                HcurlFunct_global(blk1, blk2) = temphpmat->Transpose();
                if (blk1 == blk2)
                {
                    Eliminate_bb_block(*HcurlFunct_global(blk1, blk2), *temp_dom);
                    SparseMatrix diag;
                    HcurlFunct_global(blk1, blk2)->GetDiag(diag);
                    diag.MoveDiagonalFirst();
                }

                HcurlFunct_global(blk1, blk2)->CopyColStarts();
                HcurlFunct_global(blk1, blk2)->CopyRowStarts();
                delete temphpmat;
//#endif
            } // else of if Funct_blk != NULL

        }
    }


    /*
    CTMC_global = RAP(Divfree_hpmat_nobnd, (*Funct_hpmat)(0,0), Divfree_hpmat_nobnd);

    const Array<int> *temp_dom = &essbdrtruedofs_Hcurl;

    Eliminate_ib_block(*CTMC_global, *temp_dom, *temp_dom );
    HypreParMatrix * temphpmat = CTMC_global->Transpose();
    Eliminate_ib_block(*temphpmat, *temp_dom, *temp_dom );
    CTMC_global = temphpmat->Transpose();
    Eliminate_bb_block(*CTMC_global, *temp_dom);
    SparseMatrix diag;
    CTMC_global->GetDiag(diag);
    diag.MoveDiagonalFirst();

    CTMC_global->CopyRowStarts();
    CTMC_global->CopyColStarts();
    delete temphpmat;

    HcurlFunct_global(0,0) = CTMC_global;


    Smoothers[0] = new HypreSmoother(*CTMC_global, HypreSmoother::Type::l1GS, sweeps_num[0]);
    */

    Smoothers[0] = new HypreSmoother(*HcurlFunct_global(0,0), HypreSmoother::Type::l1GS, sweeps_num[0]);
    if (numblocks > 1) // i.e. if S exists in the functional
    {
        //Smoothers[1] = new HypreBoomerAMG(*Funct_restblocks_global(1,1));
        //((HypreBoomerAMG*)(Smoothers[1]))->SetPrintLevel(0);
        //((HypreBoomerAMG*)(Smoothers[1]))->iterative_mode = false;
        Smoothers[1] = new HypreSmoother( *HcurlFunct_global(1,1), HypreSmoother::Type::l1GS, sweeps_num[1]);
    }

    truex = new BlockVector(trueblock_offsets);
    truerhs = new BlockVector(trueblock_offsets);

#ifdef TIMING
    ResetInternalTimings();
#endif
}

GeneralMinConstrSolver::~GeneralMinConstrSolver()
{
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

    // x and y will be accessed through these viewers as BlockVectors
    const BlockVector x_viewer(x.GetData(), TrueP_Func[0]->RowOffsets());
    BlockVector y_viewer(y.GetData(), TrueP_Func[0]->RowOffsets());

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

        //std::cout << "righthand side on the entrance to Solve() \n";
        //xblock_truedofs->Print();

        Solve(x_viewer, *tempblock_truedofs, y_viewer);

#ifdef TIMING
        MPI_Barrier(comm);
        chrono3.Stop();
        time_solve  += chrono3.RealTime();
#endif

#ifdef CHECK_CONSTR
        if (!preconditioner_mode)
        {
           MFEM_ASSERT(CheckConstrRes(y_viewer.GetBlock(0), *Constr_global, Constr_rhs_global, "after the iteration"),"");
        }
        else
        {
            MFEM_ASSERT(CheckConstrRes(y_viewer.GetBlock(0), *Constr_global, NULL, "after the iteration"),"");
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

            *tempblock_truedofs = y_viewer;
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

        //std::cout << "residual before smoothing, old GenMinConstr, "
                     //"norm = " << trueresfunc_lvls[l]->Norml2() / sqrt (trueresfunc_lvls[l]->Size()) << "\n";

        // solution updates will always satisfy homogeneous essential boundary conditions
        *truesolupdate_lvls[l] = 0.0;

        if (LocalSolvers_lvls[l])
        {
            LocalSolvers_lvls[l]->Mult(*trueresfunc_lvls[l], *truetempvec_lvls[l]);

            //std::cout << "LocalSmoother * r, x, norm = " << truetempvec_lvls[l]->Norml2() / sqrt(truetempvec_lvls[l]->Size()) << "\n";

            *truesolupdate_lvls[l] += *truetempvec_lvls[l];
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

        //std::cout << "residual after LocalSmoother, r - A Smoo1 * r, norm = " << truetempvec_lvls[l]->Norml2() / sqrt(truetempvec_lvls[l]->Size()) << "\n";

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
#ifdef NO_PRESMOOTH
            *truetempvec2_lvls[l] = 0.0;
#else
            //std::cout << "l = " << l << "\n";
            //std::cout << "tempvec_l = " << truetempvec_lvls[l] << ", tempvec2_l = " << truetempvec2_lvls[l] << "\n";

            //std::cout << "input to smoother inside new mg \n";
            //truetempvec_lvls[l]->Print();

            Smoothers_lvls[l]->Mult(*truetempvec_lvls[l], *truetempvec2_lvls[l] );

            //std::cout << "HcurlSmoother * updated residual, Smoo2 * (r - A Smoo1 * r), norm = " << truetempvec2_lvls[l]->Norml2() / sqrt(truetempvec2_lvls[l]->Size()) << "\n";

            //std::cout << "correction after Hcurl smoothing, old GenMinConstr, "
                         //"norm = " << truetempvec2_lvls[l]->Norml2() / sqrt (truetempvec2_lvls[l]->Size())
                      //<< "\n";


            //std::cout << "output to smoother inside new mg \n";
            //truetempvec2_lvls[l]->Print();

#endif

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

            //std::cout << "residual again before smoothing, old GenMinConstr, "
                         //"norm = " << trueresfunc_lvls[l]->Norml2() / sqrt (trueresfunc_lvls[l]->Size()) << "\n";

            UpdateTrueResidual(l, trueresfunc_lvls[l], *truesolupdate_lvls[l], *truetempvec_lvls[l] );

            //std::cout << "new residual inside new mg \n";
            //truetempvec_lvls[l]->Print();

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Stop();
            time_resupdate += chrono.RealTime();
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif

        }

        //std::cout << "correction after smoothing, old GenMinConstr, "
                     //"norm = " << truesolupdate_lvls[l]->Norml2() / sqrt (truesolupdate_lvls[l]->Size()) << "\n";


        *trueresfunc_lvls[l] = *truetempvec_lvls[l];

        //std::cout << "residual after smoothing, old GenMinConstr, "
                     //"norm = " << trueresfunc_lvls[l]->Norml2() / sqrt (trueresfunc_lvls[l]->Size()) << "\n";


        TrueP_Func[l]->MultTranspose(*trueresfunc_lvls[l], *trueresfunc_lvls[l + 1]);

        // manually setting the boundary conditions (requried for S from H1 at least) at the coarser level
        //int shift = 0;
        for (int blk = 0; blk < numblocks; ++blk)
        {
            const Array<int> * temp;
            temp = essbdrtruedofs_Func[l + 1][blk];

            //std::cout << "blk = " << blk << "\n";

            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                //std::cout << "tdof = " << shift + (*temp)[tdofind] << "\n";
                trueresfunc_lvls[l + 1]->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            }

            //shift += trueresfunc_lvls[l + 1]->GetBlock(blk).Size();
        }

        //std::cout << "residual after coarsening, old GenMinConstr, "
                     //"norm = " << trueresfunc_lvls[l + 1]->Norml2() / sqrt (trueresfunc_lvls[l + 1]->Size())
                //<< "\n";



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

    //std::cout << "residual at the coarsest level, old GenMinConstr, "
                 //"norm = " << trueresfunc_lvls[num_levels - 1]->Norml2() / sqrt (trueresfunc_lvls[num_levels - 1]->Size()) << "\n";


#ifdef NO_COARSESOLVE
    *truesolupdate_lvls[num_levels - 1] = 0.0;
#else
    // BOTTOM: solve the global problem at the coarsest level
    CoarseSolver->Mult(*trueresfunc_lvls[num_levels - 1], *truesolupdate_lvls[num_levels - 1]);
#endif

    //std::cout << "coarsest grid correction, old GenMinConstr, "
                 //"norm = " << truesolupdate_lvls[num_levels - 1]->Norml2() / sqrt (truesolupdate_lvls[num_levels - 1]->Size()) << "\n";

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

            //std::cout << "residual before post-smoothing, old GenMinConstr, "
                         //"norm = " << trueresfunc_lvls[l - 1]->Norml2() / sqrt (trueresfunc_lvls[l - 1]->Size()) << "\n";

#ifdef TIMING
            MPI_Barrier(comm);
            chrono.Clear();
            chrono.Start();
#endif

            // smooth at the finer level
            if (Smoothers_lvls[l - 1])
            {
#ifdef NO_POSTSMOOTH
                *truetempvec_lvls[l - 1] = 0.0;
#else
                //std::cout << "input to smoother inside new mg at the upward loop \n";
                //truetempvec2_lvls[l - 1]->Print();

                Smoothers_lvls[l - 1]->MultTranspose(*truetempvec2_lvls[l - 1], *truetempvec_lvls[l - 1] );
#endif

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

    //truesolupdate_lvls[0]->Print();
    //next_sol.Print();

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

MonolithicGSBlockSmoother::MonolithicGSBlockSmoother(BlockOperator &Op, const Array<int>& Offsets, bool IsDiagonal,
                                                     HypreSmoother::Type type, int nsweeps)
    : BlockOperator(Offsets), nblocks(Op.NumRowBlocks()), op(Op), offsets(Offsets), is_diagonal(IsDiagonal)
{
    op_blocks.SetSize(nblocks, nblocks);
    for (int i = 0; i < nblocks; ++i)
        for (int j = 0; j < nblocks; ++j)
            if ( (i != j && !is_diagonal) || i == j)
            {
                op_blocks(i,j) = dynamic_cast<HypreParMatrix*>(&op.GetBlock(i,j));
                if (!op_blocks(i,j))
                {
                    MFEM_ABORT ("Unsuccessful cast \n");
                }
            }
            else
                op_blocks(i,j) = NULL;

    diag_smoothers.SetSize(nblocks);
    for (int i = 0; i < nblocks; ++i)
        diag_smoothers[i] = new HypreSmoother(*op_blocks(i,i), type, nsweeps);

    tmp = new BlockVector(offsets);
}

void MonolithicGSBlockSmoother::Mult(const Vector & x, Vector & y) const
{
    yblock.Update(y.GetData(), offsets);
    xblock.Update(x.GetData(), offsets);

    for (int i = 0; i < nblocks; ++i)
    {
        tmp->GetBlock(i) = xblock.GetBlock(i);

        if (!is_diagonal)
        {
            for (int j = 0; j < i; ++j)
            {
                op_blocks(i,j)->Mult(-1.0, yblock.GetBlock(j), 1.0, tmp->GetBlock(i));
            }
        }
        diag_smoothers[i]->Mult(tmp->GetBlock(i), yblock.GetBlock(i));

    }
}

void MonolithicGSBlockSmoother::MultTranspose(const Vector & x, Vector & y) const
{
    yblock.Update(y.GetData(), offsets);
    xblock.Update(x.GetData(), offsets);

    for (int i = nblocks - 1; i >=0; --i)
    {
        tmp->GetBlock(i) = xblock.GetBlock(i);
        if (!is_diagonal)
            for (int j = i + 1; j < nblocks; ++j)
            {
                op_blocks(i,j)->Mult(-1.0, yblock.GetBlock(j), 1.0, tmp->GetBlock(i));
            }
        diag_smoothers[i]->Mult(tmp->GetBlock(i), yblock.GetBlock(i));
    }
}


BlockSmoother::BlockSmoother(BlockOperator &Op)
    : BlockOperator(Op.RowOffsets()),
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

void BlockSmoother::Mult(const Vector & x, Vector & y) const
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

void BlockSmoother::MultTranspose(const Vector & x, Vector & y) const
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

MonolithicMultigrid::MonolithicMultigrid(BlockOperator &Op, const Array<BlockOperator*> &P,
#ifdef BND_FOR_MULTIGRID
                    const std::vector<std::vector<Array<int>*> > & EssBdrTDofs_lvls,
#endif
                    Solver* Coarse_Prec)
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

            /*
            if (l == 2)
            {
                std::cout << "A11_c size = " << A11_c->Height() << "\n";

                SparseMatrix diag;
                A11_c->GetDiag(diag);
                std::cout << "diag of 11 block in MonolithicMultigrid = " << diag.MaxNorm() << "\n";

                //diag.Print();

                essbdrtdofs_lvls[Operators_.Size() - l][1]->Print();

                //SparseMatrix diag_P1;
                //P1->GetDiag(diag_P1);
                //std::cout << "diag of 11 block in MonolithicMultigrid = " << diag.MaxNorm() << "\n";
            }
            */

            {
                Eliminate_ib_block(*A11_c, *essbdrtdofs_lvls[Operators_.Size() - l][1], *essbdrtdofs_lvls[Operators_.Size() - l][1] );

                /*
                if (l == 2)
                {
                    std::cout << "A11_c size = " << A11_c->Height() << "\n";

                    essbdrtdofs_lvls[Operators_.Size() - l][1]->Print();

                    SparseMatrix diag;
                    A11_c->GetDiag(diag);

                    std::cout << "diag in A11_c \n";
                    diag.Print();
                }
                */

                HypreParMatrix * temphpmat = A11_c->Transpose();
                Eliminate_ib_block(*temphpmat, *essbdrtdofs_lvls[Operators_.Size() - l][1], *essbdrtdofs_lvls[Operators_.Size() - l][1] );

                /*
                if (l == 2)
                {
                    SparseMatrix diag;
                    temphpmat->GetDiag(diag);

                    std::cout << "diag in temphpmat in monolithic multigrid \n";
                    diag.Print();
                }
                */

                A11_c = temphpmat->Transpose();
                A11_c->CopyColStarts();
                A11_c->CopyRowStarts();

                SparseMatrix diag;
                A11_c->GetDiag(diag);
                diag.MoveDiagonalFirst();

                delete temphpmat;
#if 0

                /*
                if (l == 2)
                {
                    SparseMatrix diag;
                    temphpmat->GetDiag(diag);

                    std::cout << "diag in A11_c afterwards in monolithic multigrid \n";
                    diag.Print();
                }
                */


#endif
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

void MonolithicMultigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;

    /*
#ifdef BND_FOR_MULTIGRID
    block_viewers[Operators_.Size() - 1]->Update((*residual.Last()).GetData(), *block_offsets[Operators_.Size() - 1]);
    for (unsigned int blk = 0; blk < block_offsets.size() - 1; ++blk)
    {
        const Array<int> *temp = essbdrtdofs_lvls[0][blk];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            //std::cout << "tdof = " << (*temp)[tdofind] << "\n";
            block_viewers[Operators_.Size() - 1]->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
        }
    }
#endif
    */

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

#ifndef NO_PRESMOOTH
    // PreSmoothing
    if (current_level > 0)
    {
        //std::cout << "residual before presmoothing \n";
        //residual_l.Print();

        //std::cout << "residual before smoothing, old MG, "
                     //"norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";

        Smoother_l.Mult(residual_l, correction_l);

        //std::cout << "correction after presmoothing \n";
        //correction_l.Print();

        //std::cout << "correction after smoothing, old MG, "
                     //"norm = " << correction_l.Norml2() / sqrt (correction_l.Size()) << "\n";

        Operator_l.Mult(correction_l, help);
        residual_l -= help;

        //std::cout << "new residual after presmoothing, old MG, "
                     //"norm = " << residual_l.Norml2() / sqrt (residual_l.Size()) << "\n";
        //residual_l.Print();
    }
#endif

    // Coarse grid correction
    if (current_level > 0)
    {
        const BlockOperator& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

        //std::cout << "residual after projecting onto coarser level\n";
        //residual[current_level-1]->Print();

#ifdef BND_FOR_MULTIGRID
        block_viewers[current_level-1]->Update((*residual[current_level-1]).GetData(), *block_offsets[current_level-1]);
        for (int blk = 0; blk < block_offsets[current_level-1]->Size() - 1; ++blk)
        {
            //std::cout << "blk_shift = " << (*block_offsets[current_level-1])[blk] << "\n";
            const Array<int> *temp = essbdrtdofs_lvls[Operators_.Size() - current_level][blk];
            for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
            {
                //std::cout << "tdof = " << (*temp)[tdofind] << "\n";
                block_viewers[current_level-1]->GetBlock(blk)[(*temp)[tdofind]] = 0.0;
            }
        }
#endif

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
#ifdef NO_COARSESOLVE
        correction_l = 0.0;
#else
        CoarseSolver->Mult(residual_l, correction_l);
#endif
        /*
        cor_cor.SetSize(residual_l.Size());

        CoarseSolver->Mult(residual_l, cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(cor_cor, help);
        residual_l -= help;
        */
    }

#ifndef NO_POSTSMOOTH
    // PostSmoothing
    if (current_level > 0)
    {
        Smoother_l.MultTranspose(residual_l, cor_cor);
        correction_l += cor_cor;
    }
#endif

}

void Multigrid::Mult(const Vector & x, Vector & y) const
{
    *residual.Last() = x;

    //std::cout << "inout x \n";
    //residual.Last()->Print();

#ifdef BND_FOR_MULTIGRID
    //std::cout << "x size = " << x.Size() << "\n";
    const Array<int> *temp = essbdrtdofs_lvls[0];
    for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
    {
        //std::cout << "tdof = " << (*temp)[tdofind] << "\n";
        (*residual[Operators_.Size() - 1])[(*temp)[tdofind]] = 0.0;
    }
#endif

    correction.Last()->SetDataAndSize(y.GetData(), y.Size());
    MG_Cycle();
}

void Multigrid::MG_Cycle() const
{
    const HypreParMatrix& Operator_l = *Operators_[current_level];
    const HypreSmoother& Smoother_l = *Smoothers_[current_level];

    Vector& residual_l = *residual[current_level];
    Vector& correction_l = *correction[current_level];

#ifndef NO_PRESMOOTH
    // PreSmoothing
    if (current_level > 0)
    {
        Smoother_l.Mult(residual_l, correction_l);
        Operator_l.Mult(-1.0, correction_l, 1.0, residual_l);
    }
#endif

    // Coarse grid correction
    if (current_level > 0)
    {

        const HypreParMatrix& P_l = *P_[current_level-1];

        P_l.MultTranspose(residual_l, *residual[current_level-1]);

#ifdef BND_FOR_MULTIGRID
        const Array<int> *temp = essbdrtdofs_lvls[Operators_.Size() - current_level];
        //std::cout << "level of essbdrdofs = " << Operators_.Size() - current_level << "\n";
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            (*residual[current_level - 1])[(*temp)[tdofind]] = 0.0;
        }
#endif
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
#ifdef BND_FOR_MULTIGRID
        const Array<int> *temp = essbdrtdofs_lvls[Operators_.Size() - current_level - 1];
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            residual_l[(*temp)[tdofind]] = 0.0;
        }
#endif

#ifdef NO_COARSESOLVE
        correction_l = 0.0;
#else
        CoarseSolver->Mult(residual_l, correction_l);
#endif

#ifdef BND_FOR_MULTIGRID
        for ( int tdofind = 0; tdofind < temp->Size(); ++tdofind)
        {
            correction_l[(*temp)[tdofind]] = 0.0;
        }
#endif

        /*
        cor_cor.SetSize(residual_l.Size());

        CoarseSolver->Mult(residual_l, cor_cor);
        correction_l += cor_cor;
        Operator_l.Mult(-1.0, cor_cor, 1.0, residual_l);
        */
    }

#ifndef NO_POSTSMOOTH
    // PostSmoothing
    if (current_level > 0)
    {
        Smoother_l.Mult(residual_l, cor_cor);
        correction_l += cor_cor;
    }
#endif

}

} // for namespace mfem
