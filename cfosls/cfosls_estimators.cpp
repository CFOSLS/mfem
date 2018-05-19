#include <iostream>
#include "testhead.hpp"

using namespace std;

namespace mfem
{

// old implementation, now is a simplified form of the blocked case
// here FOSLS functional is given as a bilinear form(sigma, sigma)
double FOSLSErrorEstimator(BilinearFormIntegrator &blfi, GridFunction &sigma, Vector &error_estimates)
{
    FiniteElementSpace * fes = sigma.FESpace();
    int ne = fes->GetNE();
    error_estimates.SetSize(ne);

    double total_error = 0.0;
    for (int i = 0; i < ne; ++i)
    {
        const FiniteElement * fe = fes->GetFE(i);
        ElementTransformation * eltrans = fes->GetElementTransformation(i);
        DenseMatrix elmat;
        blfi.AssembleElementMatrix(*fe, *eltrans, elmat);

        Array<int> eldofs;
        fes->GetElementDofs(i, eldofs);
        Vector localv;
        sigma.GetSubVector(eldofs, localv);

        Vector localAv(localv.Size());
        elmat.Mult(localv, localAv);

        //std::cout << "sigma linf norm = " << sigma.Normlinf() << "\n";
        //sigma.Print();
        //eldofs.Print();
        //localv.Print();
        //localAv.Print();

        double err = localAv * localv;


        error_estimates(i) = std::sqrt(err);
        total_error += err;
    }

    //std::cout << "error estimates linf norm = " << error_estimates.Normlinf() << "\n";

    return std::sqrt(total_error);
}

// here FOSLS functional is given as a symmetric block matrix with bilinear forms for
// different grid functions (each for all solution and rhs components)
double FOSLSErrorEstimator(Array2D<BilinearFormIntegrator*> &blfis, Array<ParGridFunction*> & grfuns, Vector &error_estimates)
{
    /*
     * using a simpler version
    if  (sols.Size() == 1)
    {
        return FOSLSErrorEstimator(*blfis(0,0), *grfuns[0], error_estimates);
    }
    */

    Array<FiniteElementSpace*> fess(grfuns.Size());
    for (int i = 0; i < grfuns.Size(); ++i)
        fess[i] = grfuns[i]->FESpace();

    int ne = fess[0]->GetNE();
    error_estimates.SetSize(ne);

    double total_error = 0.0;
    for (int i = 0; i < ne; ++i)
    {
        double err = 0.0;
        for (int rowblk = 0; rowblk < blfis.NumRows(); ++rowblk)
        {
            for (int colblk = rowblk; colblk < blfis.NumCols(); ++colblk)
            {
                if (rowblk == colblk)
                {
                    if (blfis(rowblk,colblk))
                    {
                        const FiniteElement * fe = fess[rowblk]->GetFE(i);
                        ElementTransformation * eltrans = fess[rowblk]->GetElementTransformation(i);
                        DenseMatrix elmat;
                        blfis(rowblk,colblk)->AssembleElementMatrix(*fe, *eltrans, elmat);

                        Array<int> eldofs;
                        fess[rowblk]->GetElementDofs(i, eldofs);
                        Vector localv;
                        grfuns[rowblk]->GetSubVector(eldofs, localv);

                        Vector localAv(localv.Size());
                        elmat.Mult(localv, localAv);

                        err += localAv * localv;
                    }
                }
                else
                // only using one of the off-diagonal integrators at symmetric places,
                // since a FOSLS functional must be symmetric
                {
                    if (blfis(rowblk,colblk) || blfis(colblk,rowblk))
                    {
                        int trial, test;
                        if (blfis(rowblk,colblk))
                        {
                            trial = colblk;
                            test = rowblk;
                        }
                        else // using an integrator for (colblk, rowblk) instead
                        {
                            trial = rowblk;
                            test = colblk;
                        }


                        FiniteElementSpace * fes1 = fess[trial];
                        FiniteElementSpace * fes2 = fess[test];
                        const FiniteElement * fe1 = fes1->GetFE(i);
                        const FiniteElement * fe2 = fes2->GetFE(i);
                        ElementTransformation * eltrans = fes2->GetElementTransformation(i);
                        DenseMatrix elmat;
                        blfis(test,trial)->AssembleElementMatrix2(*fe1, *fe2, *eltrans, elmat);

                        Vector localv1;
                        Array<int> eldofs1;
                        fes1->GetElementDofs(i, eldofs1);
                        grfuns[trial]->GetSubVector(eldofs1, localv1);

                        Vector localv2;
                        Array<int> eldofs2;
                        fes2->GetElementDofs(i, eldofs2);
                        grfuns[test]->GetSubVector(eldofs2, localv2);

                        Vector localAv1(localv2.Size());
                        elmat.Mult(localv1, localAv1);

                        //std::cout << "sigma linf norm = " << sigma.Normlinf() << "\n";
                        //sigma.Print();
                        //eldofs.Print();
                        //localv.Print();
                        //localAv.Print();

                        // factor 2.0 comes from the fact that we look only on one of the symmetrically placed
                        // bilinear forms in the functional
                        err += 2.0 * (localAv1 * localv2);
                    }
                } // end of else for off-diagonal blocks
            }
        } // end of loop over blocks in the functional

        error_estimates(i) = std::sqrt(err);
        total_error += err;
    }

    //std::cout << "error estimates linf norm = " << error_estimates.Normlinf() << "\n";

    return std::sqrt(total_error);
}

FOSLSEstimator::FOSLSEstimator(MPI_Comm Comm, Array<ParGridFunction *> &solutions,
                               Array2D<BilinearFormIntegrator *> &integrators, bool verbose_)
    : comm(Comm), numblocks(solutions.Size()), current_sequence(-1),
      global_total_error(0.0), verbose(verbose_)
{
    grfuns.SetSize(numblocks);
    for (int i = 0; i < numblocks; ++i)
        grfuns[i] = solutions[i];
    integs.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            integs(i,j) = integrators(i,j);
}

// grfuns_desciptor:
// vector of length = number of blocks in the estimator
// each entry is a pair <a,b>, where:
// 'a' must be 1, if the corresponding grid function
// can be found inside the grfuns array of the problem
// and -1, if it must be found in the extra_grfuns.

FOSLSEstimator::FOSLSEstimator(FOSLSProblem& problem, std::vector<std::pair<int,int> > & grfuns_descriptor,
               Array<ParGridFunction*>* extra_grfuns, Array2D<BilinearFormIntegrator*>& integrators, bool verbose_)
    : comm (problem.GetComm()), numblocks(integrators.NumRows()), current_sequence(-1),
      global_total_error(0.0), verbose(verbose_)
{
    grfuns.SetSize(numblocks);

    MFEM_ASSERT(numblocks == (int)(grfuns_descriptor.size()), "Numblocks mismatch the length of the grfuns_descriptor");

    for (int i = 0; i < numblocks; ++i)
    {
        MFEM_ASSERT(grfuns_descriptor[i].first == 1 || grfuns_descriptor[i].first == -1,
                    "Values of grfuns_descriptor must be either 1 or -1");
        if (grfuns_descriptor[i].first == 1)
            grfuns[i] = problem.GetGrFuns()[grfuns_descriptor[i].second];
        else
        {
            MFEM_ASSERT(extra_grfuns, "Trying to use extra_grfuns which is NULL \n");
            grfuns[i] = (*extra_grfuns)[grfuns_descriptor[i].second];
        }
    }

    integs.SetSize(numblocks, numblocks);
    for (int i = 0; i < numblocks; ++i)
        for (int j = 0; j < numblocks; ++j)
            integs(i,j) = integrators(i,j);
}


void FOSLSEstimator::Update()
{
    for (int i = 0; i < grfuns.Size(); ++i)
        grfuns[i]->Update();
}


bool FOSLSEstimator::MeshIsModified()
{
   long mesh_sequence = grfuns[0]->FESpace()->GetMesh()->GetSequence();
   MFEM_ASSERT(mesh_sequence >= current_sequence, "");
   return (mesh_sequence > current_sequence);
}

const Vector & FOSLSEstimator::GetLocalErrors()
{
    if (MeshIsModified())
    {
        ComputeEstimates();
    }
    return error_estimates;
}

void FOSLSEstimator::ComputeEstimates()
{
    double local_total_error = FOSLSErrorEstimator(integs, grfuns, error_estimates);
    local_total_error *= local_total_error;

    global_total_error = 0.0;
    MPI_Allreduce(&local_total_error, &global_total_error, 1, MPI_DOUBLE, MPI_SUM, comm);
    global_total_error = std::sqrt(global_total_error);

    if (verbose)
        std::cout << "global_total_error = " << global_total_error << "\n";

    //error_estimates.Print();

    current_sequence = grfuns[0]->FESpace()->GetMesh()->GetSequence();
}

void FOSLSEstimator::Reset()
{
    current_sequence = -1;
}

} // for namespace mfem
