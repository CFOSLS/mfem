
#ifndef MFEM_LS_TEMP
#define MFEM_LS_TEMP

#include "../config/config.hpp"
#include "../general/array.hpp"
#include "../mesh/mesh.hpp"
#include "../fem/estimators.hpp"

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>


namespace mfem
{


class NDLSRefiner: public MeshOperator
    {
    protected:
        ErrorEstimator &estimator;
        AnisotropicErrorEstimator *aniso_estimator;
        std::vector< std::pair <double, int> > eta;
        
        int refinementstrategy;
        
        double total_error_estimate;
        double total_norm_p;
        double total_err_goal;
        double total_fraction;
        long max_elements;
        
        long num_marked_faces;
        
        long prev_num_elements;
        long prev_num_vertices;
        
        Array<int> child_faces;
        Array<int> marked_faces;
        Array<Refinement> marked_elements;
        
        long num_marked_elements;
        
        long current_sequence;
        
        int non_conforming;
        int nc_limit;
        
        double BetaA;
        int BetaMode;


        class RefinerImpl;

        RefinerImpl * impl;

        
        
        double GetNorm(const Vector &local_err, Mesh &mesh) const;
        
        /** @brief Apply the operator to the mesh.
         @return STOP if a stopping criterion is satisfied or no elements were
         marked for refinement; REFINED + CONTINUE otherwise. */
        virtual int ApplyImpl(Mesh &mesh);

        double BetaCalc(int index, const Mesh &mesh);
        //doesn't actually modify mesh, just didn't want to go down the rabbit hole of making a function in mfem::mesh constant that should've been constant. 
        void EtaCalc( Mesh &mesh, const Vector &local_errs);
        long MinimalSubset_Sum(const std::vector< std::pair <double,int> > &set, double weight, unsigned int tolerance);

    public:
        
        bool version_difference; 
        
        /// Construct a NDLSRefiner using the given ErrorEstimator.
        NDLSRefiner(ErrorEstimator &est);
        ~NDLSRefiner();
        
        // default destructor (virtual)
        
        /** @brief Set the exponent, p, of the discrete p-norm used to compute the
         total error from the local element errors. */
        void SetTotalErrorNormP(double norm_p = std::numeric_limits<double>::infinity())
        { total_norm_p = norm_p; }
        
        double GetTotalErrorEstimate(Mesh &mesh);
    
        
        /** @brief Set the total error stopping criterion: stop when
         total_err <= total_err_goal. The default value is zero. */
        void SetTotalErrorGoal(double err_goal) { total_err_goal = err_goal; }
        
        void SetRefinementStrategy(int strat){refinementstrategy = strat;}
        
        /** Set the version Beta Calc to be used in the LS ReFinement Algorithm. This is the function that determines the weight that the estimated error difference across a face is given. This is an absolute, not relative weight.
         
         Current modes are
         
         0 - BetaCalc (face i) =  1/BetaA; [Constant Fixed Beta Value]
         1 - BetaCalc(face i) = Length of Face [In 2-D], Area of Face [In 3-D];  (DEFAULT)
         */
        void SetBetaCalc(int mode){
            switch(mode)
            {
        case 0 :
            BetaMode = 0;
            break;
        case 1 :
            BetaMode = 1;
            break;
        default :
            mfem_error("SetBetaCalc: Not a Valid Input");
            break;
            }
            
        }
        void SetBetaConstants(double a) {
            if (a >0){
            BetaA = a;
            } else {
                mfem_error("SetBetaConstants: BetaA is required to be greater than or equal to 0");
            }
        }
        /** @brief Set the total fraction used in the computation of the threshold.
         The default value is 1/2.
         @note If fraction == 0, total_err is essentially ignored in the threshold
         computation, i.e. threshold = local error goal. */
        void SetTotalErrorFraction(double fraction) { total_fraction = fraction; }
        
        /** @brief Set the maximum number of elements stopping criterion: stop when
         the input mesh has num_elements >= max_elem. The default value is
         LONG_MAX. */
        void SetMaxElements(long max_elem) { max_elements = max_elem; }
        
        /// Use nonconforming refinement, if possible (triangles, quads, hexes).
        void PreferNonconformingRefinement() { non_conforming = 1; }
        
        /** @brief Use conforming refinement, if possible (triangles, tetrahedra)
         -- this is the default. */
        void PreferConformingRefinement() { non_conforming = -1; }
        
        /** @brief Set the maximum ratio of refinement levels of adjacent elements
         (0 = unlimited). */
        void SetNCLimit(int nc_limit)
        {
            MFEM_ASSERT(nc_limit >= 0, "Invalid NC limit");
            this->nc_limit = nc_limit;
        }
        
        /// Get the number of marked elements in the last Apply() call.
        long GetNumMarkedElements() const { return num_marked_faces; }
        
        /// Reset the associated estimator.
        virtual void Reset();



    };

}

#endif


