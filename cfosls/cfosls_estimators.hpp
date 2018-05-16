#include <iostream>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_ESTIMATORS
#define MFEM_CFOSLS_ESTIMATORS

using namespace std;
using namespace mfem;

namespace mfem
{

// old implementation, now is a simplified form of the blocked case
// here FOSLS functional is given as a bilinear form(sigma, sigma)
double FOSLSErrorEstimator(BilinearFormIntegrator &blfi,
                           GridFunction &sigma, Vector &error_estimates);


// here FOSLS functional is given as a symmetric block matrix with bilinear forms for
// different grid functions (each for all solution and rhs components)
double FOSLSErrorEstimator(Array2D<BilinearFormIntegrator*> &blfis,
                           Array<ParGridFunction*> & grfuns, Vector &error_estimates);

class FOSLSEstimator : public ErrorEstimator
{
protected:
    MPI_Comm comm;
    const int numblocks;
    long current_sequence;
    Array<ParGridFunction*> grfuns;
    Array2D<BilinearFormIntegrator*> integs;
    Vector error_estimates;
    double global_total_error;
    bool verbose;

    /// Check if the mesh of the solution was modified (copied from L2ZienkkiewiczZhuEstimator).
    bool MeshIsModified();

    /// Compute the element error estimates (copied from L2ZienkkiewiczZhuEstimator).
    void ComputeEstimates();
public:
    //~FOSLSEstimator() {}
    //FOSLSEstimator(MPI_Comm& Comm, ParGridFunction &solution,
                   //BilinearFormIntegrator &integrator, bool verbose_ = false);

    FOSLSEstimator(MPI_Comm Comm, Array<ParGridFunction*>& solutions,
                   Array2D<BilinearFormIntegrator*>& integrators, bool verbose_ = false);

    FOSLSEstimator(FOSLSProblem& problem, std::vector<std::pair<int,int> > & grfuns_descriptor,
                   Array<ParGridFunction *> *extra_grfuns, Array2D<BilinearFormIntegrator*>& integrators,
                   bool Verbose = false);

    virtual const Vector & GetLocalErrors () override;
    double GetEstimate() {ComputeEstimates(); return global_total_error;}
    virtual void Reset () override;
    void Update();
};

} // for namespace mfem


#endif
