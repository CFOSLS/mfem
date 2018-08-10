#include <iostream>
#include "testhead.hpp"

#ifndef MFEM_CFOSLS_TIMESTEPPING
#define MFEM_CFOSLS_TIMESTEPPING

using namespace std;
using namespace mfem;

namespace mfem
{

/// Base class for a FOSLS problem in a time cylinder
/// The main difference between this class and FOSLSProblem
/// is that we have a bottom and top bases of the cylinder
/// and we can specify initial condition (in space-time it's a
/// boundary condition at the bottom base) in it.
/// This class provides a number of utilities which might be useful
/// for handling such a problem.
/// Specific terms used around this class:
/// 1) base = one of the two bases of the cylinder (bottom or top)
/// 2) bot = bottom ~ bottom base of the cylinder
/// 3) top ~ top base of the cylinder
/// 4) tdofs link ~ link ~ see ConstructTdofLink()
/// Remark: Be careful about the difference between the boundary
/// where initial condition is defined and the essential boundary.
/// Initial condition is specified at the bottom base but it might not coincide
/// with the essential boundary (e.g, for parabolic problem).
/// The code was not actually tested against that case.
class FOSLSCylProblem : virtual public FOSLSProblem
{
public:
    using FOSLSProblem::Solve;
protected:
    ParMeshCyl &pmeshcyl;

    // doesn't own this, it's just a ptr to address the underlying cylinder
    // hierarchy (if any) without dynamic casts
    GeneralCylHierarchy * cyl_hierarchy;

    // These two define what is the space for the variable with initial condition
    // (used to construct a corrseponding tdofs link) and what is its index
    // as an unknown in the system
    SpaceName init_cond_space;
    int init_cond_block;

    // Link between tdofs for the f.e. space which corresponds to
    // the variable with specific initial condition
    // See ConstructTdofLink()
    std::vector<std::pair<int,int> > tdofs_link;

    // Restriction operators, see ConstructRestrictions()
    // FIXME: Not used?
    HypreParMatrix* Restrict_bot;
    HypreParMatrix* Restrict_top;

    // temporary storage vectors, used in CorrectFromInitCond()
    Vector* temp_vec1;
    Vector* temp_vec2;

protected:
    // One of the key functions of the class.
    // Constructs a tdof link (as a vector of pairs) between tdofs at the top
    // and bottom bases of the cylinder
    // Depending on the values of init_cond_space, it's either a link for RT0 or
    // linear H1 f.e. tdofs
    void ConstructTdofLink();

    // Creates a HypreParMatrix which restricts a vector of tdofs in the entire domain into
    // the vector of tdofs at the boundary base only (top or bottom)
    // FIXME: Not used?
    void ConstructRestrictions();

    // extracts from vector x its values (tdofs) at the top boundary of the cylinder
    void ExtractTopTdofs(const Vector& x, Vector& bnd_tdofs_top) const;

    void ExtractBotTdofs(const Vector& x, Vector& bnd_tdofs_bot) const;

public:
    virtual ~FOSLSCylProblem()
    {
        //delete Restrict_bot;
        //delete Restrict_top;
        delete temp_vec1;
        delete temp_vec2;
    }

    FOSLSCylProblem (ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          pmeshcyl(Pmeshcyl), cyl_hierarchy(NULL),
          init_cond_block(fe_formul.GetFormulation()->GetUnknownWithInitCnd())
    {
        const Array<SpaceName>* spacenames = fe_formul.GetFormulation()->GetSpacesDescriptor();
        init_cond_space = (*spacenames)[init_cond_block];
        ConstructTdofLink();

        temp_vec1 = new Vector(GlobalTrueProblemSize());
        temp_vec2 = new Vector(GlobalTrueProblemSize());
    }

    FOSLSCylProblem(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          pmeshcyl(*Hierarchy.GetPmeshcyl(level)), cyl_hierarchy(&Hierarchy),
          init_cond_block(fe_formul.GetFormulation()->GetUnknownWithInitCnd())
    {
        const Array<SpaceName>* spacenames = fe_formul.GetFormulation()->GetSpacesDescriptor();
        init_cond_space = (*spacenames)[init_cond_block];
        tdofs_link = *cyl_hierarchy->GetTdofsLink(level, init_cond_space);

        temp_vec1 = new Vector(GlobalTrueProblemSize());
        temp_vec2 = new Vector(GlobalTrueProblemSize());
    }

    // Solves the problem in the cylinder with trueRhs as rhs, and absorbing
    // given initial condition at the bottom (bnd_tdofs_bot).
    // The output values are extracted into the output vector at the top (bnd_tdofs_top)
    void Solve(const Vector& bnd_tdofs_bot, Vector &bnd_tdofs_top, bool compute_error) const;

    // The same as the previous one except it uses the given input rhs instead of internal trueRhs
    void Solve(const Vector& rhs, const Vector& bnd_tdofs_bot, Vector& bnd_tdofs_top, bool compute_error) const;

    // The same as the previous one except it stores the solution in the entire domain in the
    // given output parameter sol
    void Solve(const Vector& rhs, const Vector& bnd_tdofs_bot, Vector& sol, Vector& bnd_tdofs_top, bool compute_error) const;

    // Takes a vector defined at the bottom base, and then
    // modifes the non-boundary entries of vec as
    // vec -= op * Extend_from_BotBase * bnd_tdofs_bot
    // and sets the essential boundary entries to
    // vec = bnd_tdofs_bot
    void CorrectFromInitCnd(const Operator& op, const Vector& bnd_tdofs_bot, Vector& vec) const;

    // The same as the previous except it uses internal CFOSLSop_nobnd as op
    void CorrectFromInitCnd(const Vector& bnd_tdofs_bot, Vector& vec) const
    { CorrectFromInitCnd(*CFOSLSop_nobnd, bnd_tdofs_bot, vec);}

    // Takes a vector of size of initial condition
    // and computes
    // vec_out += coeff * Restrict_to_essbdr * CFOSLSop_nobnd * Extend_from_BotBase * init_cond
    void CorrectFromInitCond(const Vector& init_cond, Vector& vec_out, double coeff);

    // vec_in is considered as a vector defined only at the bottom base,
    // vec_out is a full vector which coincides with vec_in at initial boundary and
    // has 0's for all the rest entries
    void ConvertInitCndToFullVector(const Vector& vec_in, Vector& vec_out);

    // Takes a vector of full problem size and computes the corresponding change
    // to the rhs side as
    // vec_out = Restrict_to_essbdr * CFOSLSop_nobnd * vec_in
    void ConvertBdrCndIntoRhs(const Vector& vec_in, Vector& vec_out);

    // Checks the error for the given vector of values (only at the boundary tdofs)
    void ComputeErrorAtBase(const char * top_or_bot, const Vector& base_vec);

    // Extracts from the given input Vector x its values at the base (top or bottom)
    void ExtractAtBase(const char * top_or_bot, const Vector &x, Vector& base_tdofs) const;

    // The same as above, but with a different interface
    Vector& ExtractAtBase(const char * top_or_bot, const Vector &x) const;

    // modifies the given Vector vec by setting its values on the top or bottom boundary
    // (depending on the flag top_or_bot) from the given values in base_tdofs
    void SetAtBase(const char * top_or_bot, const Vector &base_tdofs, Vector& vec) const;

    // Getters
    std::vector<std::pair<int,int> > * GetTdofsLink() {return &tdofs_link;}

    ParMeshCyl * GetParMeshCyl() {return &pmeshcyl;}

    // Returns a vector at the tom or bottom boundary of the cylinder with the values
    // of the projection of the exact solution (provided by FOSLStest) onto the mesh
    Vector* GetExactBase(const char * top_or_bot);

    // Returns number of tdofs at the base (which is the same for bottom and top bases)
    int GetInitCondSize() const { return tdofs_link.size();}
};

// Specific children of FOSLSCylProblem
// These classes were created as a solution to a "diamond" inheritance problem
// when it was required to create objects which are both FOSLSCylProblem and a particular
// FOSLSProblem's child at the same time
class FOSLSCylProblem_HdivL2L2hyp : public FOSLSCylProblem, public FOSLSProblem_HdivL2L2hyp
{
public:
    FOSLSCylProblem_HdivL2L2hyp(ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivL2L2hyp(Pmeshcyl, bdr_conditions, fe_formulation, precond_option, verbose_)
    {}

    FOSLSCylProblem_HdivL2L2hyp(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivL2L2hyp(Hierarchy, level, bdr_conditions, fe_formulation, precond_option, verbose_)
    {}
};

class FOSLSCylProblem_HdivH1L2hyp : public FOSLSCylProblem, public FOSLSProblem_HdivH1L2hyp
{
public:
    FOSLSCylProblem_HdivH1L2hyp(ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivH1L2hyp(Pmeshcyl, bdr_conditions, fe_formulation, precond_option, verbose_)
    {}
    FOSLSCylProblem_HdivH1L2hyp(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivH1L2hyp(Hierarchy, level, bdr_conditions, fe_formulation, precond_option, verbose_)
    {}

};

/// Generic class-wrapper for doing time-stepping (~ time-slabbing) in CFOSLS
/// Problem is assumed to be at least of type FOSLSProblem (or it's children)
template <class Problem> class TimeStepping
{
protected:
    Array<int> global_offsets;
    Array<Problem*> timeslabs_problems;
    Array<Vector*>  base_inputs;
    Array<Vector*>  base_outputs;
    bool verbose;
    bool problems_initialized;
    int nslabs;

    Vector * vecbase_temp;

protected:
    void SetProblems(Array<Problem*>& timeslabs_problems_);

public:
    ~TimeStepping()
    {
        for (int i = 0; i < base_inputs.Size(); ++i)
            delete base_inputs[i];
        for (int i = 0; i < base_outputs.Size(); ++i)
            delete base_outputs[i];

        delete vecbase_temp;
    }

    TimeStepping(Array<Problem*>& timeslabs_problems_, bool verbose_)
        : timeslabs_problems(0), base_inputs(0), base_outputs(0),
          verbose(verbose_), problems_initialized(false)
    {
        SetProblems(timeslabs_problems_);
        global_offsets.SetSize(nslabs + 1);

        global_offsets[0] = 0;
        for (int tslab = 0; tslab < nslabs; ++tslab)
            global_offsets[tslab + 1] = timeslabs_problems[tslab]->GlobalTrueProblemSize();
        global_offsets.PartialSum();

        vecbase_temp = new Vector(timeslabs_problems[0]->GetInitCondSize());
    }


    void SequentialSolve(const Vector &init_vector, bool compute_error);
    void SequentialSolve(const Vector& rhs, const Vector &init_vector, bool compute_error);
    void SequentialSolve(const Vector& rhs, const Vector &init_vector, Vector& sol, bool compute_error);

    void ParallelSolve(const Array<Vector*> &init_vectors, bool compute_error) const;
    void ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, bool compute_error) const;
    void ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, Vector& sol, bool compute_error) const;

    Array<Vector*>& GetSolutions();

    bool NeedSignSwitch(SpaceName space_name) const
    {
        switch (space_name)
        {
        case HDIV:
            return true;
        case H1:
            return false;
        default:
        {
            MFEM_ABORT("Unsupported space name argument in NeedSignSwitch()");
            break;
        }
        }
        return false;
    }

    const Array<int>& GetGlobalOffsets() const {return global_offsets;}

    Problem * GetProblem(int i) {return timeslabs_problems[i];}

    Array<Vector*> & ConvertFullvecIntoArray(const Vector& x);
    void ConvertArrayIntoFullvec(const Array<Vector*>& vec_inputs, Vector& out);

    int GetGlobalProblemSize() const { return global_offsets[nslabs]; }

    int GetInitCondSize() const
    { return timeslabs_problems[0]->GetInitCondSize();}

    void SeqOp(const Vector& x, Vector& y) const { SeqOp(x, NULL, y);}
    void SeqOp(const Vector& x, const Vector* init_bot, Vector& y) const;

    int Nslabs() const {return nslabs;}

    void ComputeGlobalRhs(Vector& rhs);

    void ZeroBndValues(Vector& vec);

    void ComputeError(const Vector& vec) const;

    void ComputeBndError(const Vector& vec) const;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const;

    void UpdateInterfaceFromPrev(Vector& vec) const;
};

template <class Problem>
Array<Vector*>& TimeStepping<Problem>::ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
{
    BlockVector fullvec_viewer(fullvec.GetData(), global_offsets);

    Array<Vector*> * res = new Array<Vector*>(nslabs);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * problem = timeslabs_problems[tslab];
        //(*res)[tslab] = new Vector(problem->GetInitCondSize());
        //problem->ExtractAtBase(top_or_bot, fullvec_viewer.GetBlock(tslab), *res[tslab]);
        (*res)[tslab] = &(problem->ExtractAtBase(top_or_bot, fullvec_viewer.GetBlock(tslab)));
    }

    return *res;
}

template <class Problem>
void TimeStepping<Problem>::UpdateInterfaceFromPrev(Vector& vec) const
{
    BlockVector vec_viewer(vec.GetData(), global_offsets);

    for (int tslab = 1; tslab < nslabs; ++tslab)
    {
        Problem * prev_problem = timeslabs_problems[tslab - 1];
        Problem * problem = timeslabs_problems[tslab];

        *vecbase_temp = prev_problem->ExtractAtBase("top", vec_viewer.GetBlock(tslab - 1));

        problem->SetAtBase("bot", *vecbase_temp, vec_viewer.GetBlock(tslab));
    }

}


template <class Problem>
void TimeStepping<Problem>::ComputeError(const Vector& vec) const
{
    const BlockVector vec_viewer(vec.GetData(), GetGlobalOffsets());
    bool checkbnd;

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * prob = timeslabs_problems[tslab];
        if (tslab == 0)
            checkbnd = true;
        else
            checkbnd = false;
        prob->ComputeError(vec_viewer.GetBlock(tslab), verbose, checkbnd);
    }
}

template <class Problem>
void TimeStepping<Problem>::ComputeBndError(const Vector& vec) const
{
    const BlockVector vec_viewer(vec.GetData(), GetGlobalOffsets());
    timeslabs_problems[0]->ComputeBndError(vec_viewer.GetBlock(0));
    /*
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * prob = timeslabs_problems[tslab];

        prob->ComputeBndError(vec_viewer.GetBlock(tslab));
    }
    */
}



template <class Problem>
void TimeStepping<Problem>::SetProblems(Array<Problem*> &timeslabs_problems_)
{
    nslabs = timeslabs_problems_.Size();
    timeslabs_problems.SetSize(nslabs);
    base_inputs.SetSize(nslabs);
    base_outputs.SetSize(nslabs);
    for (int i = 0; i < nslabs; ++i)
    {
        timeslabs_problems[i] = timeslabs_problems_[i];
        int init_cond_size = timeslabs_problems[i]->GetInitCondSize();
        base_inputs[i] = new Vector(init_cond_size);
        base_outputs[i] = new Vector(init_cond_size);
    }

    problems_initialized = true;
}

template <class Problem>
void TimeStepping<Problem>::ComputeGlobalRhs(Vector& rhs)
{
    BlockVector rhs_viewer(rhs.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];
        tslab_problem->ComputeAnalyticalRhs(rhs_viewer.GetBlock(tslab));
    }
}

template <class Problem>
void TimeStepping<Problem>::ZeroBndValues(Vector& vec)
{
    BlockVector vec_viewer(vec.GetData(), GetGlobalOffsets());
    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];
        tslab_problem->ZeroBndValues(vec_viewer.GetBlock(tslab));
    }
}


template <class Problem>
void TimeStepping<Problem>::SequentialSolve(const Vector& init_vector, bool compute_error)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(init_vector.Size() == base_inputs[0]->Size(), "Input vector length mismatch the length of the base_input");

    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];
        FOSLSFEFormulation& fe_formul = tslab_problem->GetFEformulation();
        int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();
        SpaceName space_name = fe_formul.GetFormulation()->GetSpaceName(index);

        if (tslab == 0)
            tslab_problem->Solve(init_vector, *base_outputs[tslab]);
        else
            tslab_problem->Solve(*base_inputs[tslab], *base_outputs[tslab]);

        if (tslab < nslabs - 1)
        {
            *base_inputs[tslab + 1] = *base_outputs[tslab];
            if (NeedSignSwitch(space_name))
                *base_inputs[tslab + 1] *= -1;
        }

        if (compute_error)
            tslab_problem->ComputeErrorAtBase("top", *base_outputs[tslab]);

    } // end of loop over all time slab problems
}

// rhs is a Vector of size of full time-stepping problem
template <class Problem>
void TimeStepping<Problem>::SequentialSolve(const Vector& rhs, const Vector& init_vector, bool compute_error)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(init_vector.Size() == base_inputs[0]->Size(), "Input vector length mismatch the length of the base_input");

    Problem * tslab_startproblem = timeslabs_problems[0];
    FOSLSFEFormulation& fe_formul = tslab_startproblem->GetFEformulation();
    int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();
    SpaceName space_name = fe_formul.GetFormulation()->GetSpaceName(index);

    const BlockVector rhs_viewer(rhs.GetData(), GetGlobalOffsets());

    /*
    std::cout << "rhs, block 0 in the call to SequentialSolve \n";
    rhs_viewer.GetBlock(0).Print();

    std::cout << "rhs, block 1 in the call to SequentialSolve \n";
    rhs_viewer.GetBlock(1).Print();

    Array<Vector*> & debug_botbases2 = ExtractAtBases("bot", rhs);
    std::cout << "botbases of rhs in the call to SequentialSolve \n";
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "tslab = " << tslab << "\n";
        debug_botbases2[tslab]->Print();
    }
    */


    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        /*
        if (tslab == 0)
            init_vector.Print();
        else
            base_inputs[tslab]->Print();
        */

        if (tslab == 0)
        {
            tslab_problem->Solve(rhs_viewer.GetBlock(tslab), init_vector, *base_outputs[tslab], compute_error);

            //base_outputs[tslab]->Print();
            //tslab_problem->ComputeBndError(tslab_problem->GetSol());
            //tslab_problem->GetSol().Print();
        }
        else
            tslab_problem->Solve(rhs_viewer.GetBlock(tslab), *base_inputs[tslab], *base_outputs[tslab], compute_error);


        if (tslab < nslabs - 1)
        {
            *base_inputs[tslab + 1] = *base_outputs[tslab];
            if (NeedSignSwitch(space_name))
                *base_inputs[tslab + 1] *= -1;
        }

        if (compute_error)
            tslab_problem->ComputeErrorAtBase("top", *base_outputs[tslab]);

    } // end of loop over all time slab problems
}

// rhs is a Vector of size of full time-stepping problem
template <class Problem>
void TimeStepping<Problem>::SequentialSolve(const Vector& rhs, const Vector& init_vector, Vector& sol, bool compute_error)
{
    SequentialSolve(rhs, init_vector, compute_error);

    BlockVector sol_viewer(sol.GetData(), GetGlobalOffsets());
    for (int tslab = 0; tslab < nslabs; ++tslab)
        sol_viewer.GetBlock(tslab) = timeslabs_problems[tslab]->GetSol();
}

template <class Problem>
void TimeStepping<Problem>::ParallelSolve(const Array<Vector*> &init_vectors, bool compute_error) const
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");

    MFEM_ASSERT(init_vectors.Size() == nslabs, "Number of input vectors must equal number of time slabs");

    // renaming init_vectors into internal base_inputs
    for (int tslab = 0; tslab < nslabs; ++tslab )
        *base_inputs[tslab] = *init_vectors[tslab];

    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        tslab_problem->Solve(*init_vectors[tslab], *base_outputs[tslab]);

        if (compute_error)
            tslab_problem->ComputeErrorAtBase("top", *base_outputs[tslab]);
    } // end of loop over all time slab problems
}

// rhs is a Vector of size of full time-stepping problem
template <class Problem>
void TimeStepping<Problem>::ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, bool compute_error) const
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(init_vectors.Size() == nslabs, "Number of input vectors (for initial "
                                               "conditions) must equal the number of time slabs");

    const BlockVector rhs_viewer(rhs.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        MFEM_ASSERT(init_vectors[tslab]->Size() == GetInitCondSize(),
                    "For the given timeslab initcond vector size mismatch the problem");
        Problem * tslab_problem = timeslabs_problems[tslab];

        tslab_problem->Solve(rhs_viewer.GetBlock(tslab), *init_vectors[tslab], *base_outputs[tslab], compute_error);

        if (compute_error)
            tslab_problem->ComputeErrorAtBase("top", *base_outputs[tslab]);

    } // end of loop over all time slab problems
}

// rhs is a Vector of size of full time-stepping problem
template <class Problem>
void TimeStepping<Problem>::ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, Vector& sol, bool compute_error) const
{
    ParallelSolve(rhs, init_vectors, compute_error);

    BlockVector sol_viewer(sol.GetData(), GetGlobalOffsets());
    for (int tslab = 0; tslab < nslabs; ++tslab)
        sol_viewer.GetBlock(tslab) = timeslabs_problems[tslab]->GetSol();
}


template <class Problem>
Array<Vector*>& TimeStepping<Problem>::GetSolutions()
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");

    Array<Vector*> * res = new Array<Vector*>(nslabs);

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];
        res[tslab] = &(tslab_problem->GetSol());
    }

    return *res;
}

/*
template <class Problem>
Array<int>& TimeStepping<Problem>::GetGlobalOffsets() const
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");

    Array<int> * res = new Array<int>(nslabs + 1);
    (*res)[0] = 0;
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];
        (*res)[tslab + 1] = (*res)[tslab] + tslab_problem->GlobalTrueProblemSize();
    }

    return *res;
}
*/

/*
// FIXME: Why not converting into a BlockVector?
template <class Problem>
Array<Vector*> & TimeStepping<Problem>::ConvertFullVecIntoInitConds(const Vector& x)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(x.Size() == GetGlobalProblemSize(), "Input vector size mismatch the global problem size!");

    Array<Vector*> * res = new Array<Vector*>(nslabs);

    BlockVector x_viewer(x.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        res[tslab] = tslab_problem->ExtractAtBase("bot")

        res[tslab] = new Vector(tslab_problem->GlobalTrueProblemSize());

        *(*res)[tslab] = x_viewer.GetBlock(tslab);
    }

    return *res;
}
*/

/*
template <class Problem>
void TimeStepping<Problem>::ConvertFullVecIntoInitConds(const Vector& x)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(x.Size() == GetGlobalProblemSize(), "Input vector size mismatch the global problem size!");

    BlockVector x_viewer(x.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        *base_inputs[tslab] =


        res[tslab] = tslab_problem->ExtractAtBase("bot")

        res[tslab] = new Vector(tslab_problem->GlobalTrueProblemSize());

        *(*res)[tslab] = x_viewer.GetBlock(tslab);
    }

}
*/


// FIXME: Do we really need that?
template <class Problem>
void TimeStepping<Problem>::ConvertArrayIntoFullvec(const Array<Vector*>& vec_inputs, Vector& out)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(out.Size() == GetGlobalProblemSize(), "Output vector size mismatch the global problem size!");

    BlockVector out_viewer(out.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        out_viewer.GetBlock(tslab) = *vec_inputs[tslab];
    }
}

// FIXME: Why not converting into a BlockVector?
template <class Problem>
Array<Vector*> & TimeStepping<Problem>::ConvertFullvecIntoArray(const Vector& x)
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(x.Size() == GetGlobalProblemSize(), "Input vector size mismatch the global problem size!");

    Array<Vector*> * res = new Array<Vector*>(nslabs);

    BlockVector x_viewer(x.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        res[tslab] = new Vector(tslab_problem->GlobalTrueProblemSize());

        *(*res)[tslab] = x_viewer.GetBlock(tslab);
    }

    return *res;
}

template <class Problem>
void TimeStepping<Problem>::SeqOp(const Vector& x, const Vector *init_bot, Vector& y) const
{
    MFEM_ASSERT(problems_initialized, "Cannot solve if the problems are not set");
    MFEM_ASSERT(x.Size() == GetGlobalProblemSize(), "Input vector size mismatch the global problem size!");
    MFEM_ASSERT(y.Size() == GetGlobalProblemSize(), "Output vector size mismatch the global problem size!");

    BlockVector x_viewer(x.GetData(), GetGlobalOffsets());
    BlockVector y_viewer(y.GetData(), GetGlobalOffsets());

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        // 1. y_block = CFOSLSop * x_block
        tslab_problem->GetOp()->Mult(x_viewer.GetBlock(tslab), y_viewer.GetBlock(tslab));
        // zeroing bnd values is required here since the correct bnd values will be added
        // afterwards from the "correction from the initial condition"
        tslab_problem->ZeroBndValues(y_viewer.GetBlock(tslab));

        // 2. yblock := yblock + InitCondOp_prevblock * x_prevblock
        if (tslab > 0)
        {
            Problem * prevtslab_problem = timeslabs_problems[tslab - 1];
            //prevtslab_problem->ZeroBndValues(y_viewer.GetBlock(tslab));

            // FIXME: Memory allocation every time
            Vector * prev_initcond = new Vector(prevtslab_problem->GetInitCondSize());
            prevtslab_problem->ExtractAtBase("top",x_viewer.GetBlock(tslab - 1), *prev_initcond);

            //std::cout << "prev_initcond norm = " <<
                         //prev_initcond->Norml2() / sqrt(prev_initcond->Size()) << "\n";

            tslab_problem->CorrectFromInitCond(*prev_initcond, y_viewer.GetBlock(tslab), 1.0);
            delete prev_initcond;
        }
        else
            if (init_bot)
                tslab_problem->CorrectFromInitCond(*init_bot, y_viewer.GetBlock(tslab), 1.0);

        //std::cout << "after \n";
        //y_viewer.GetBlock(tslab).Print();
        //std::cout << "do you see? \n";
    }

}

// classes for time-stepping related operators used as components for constructing GeneralMultigrid instance

template <class Problem> class TwoGridTimeStepping
{
protected:
    int nslabs;
    Array<FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* >& cyl_probhierarchies;

    // fine problems are owned by cyl_probhierarchies
    Array<Problem*> fine_problems;
    Array<int> fine_global_offsets;
    TimeStepping<Problem> * fine_timestepping;

    // coarse problems are owned by cyl_probhierarchies
    Array<Problem*> coarse_problems;
    Array<int> coarse_global_offsets;
    TimeStepping<Problem> * coarse_timestepping;

    BlockOperator * interpolation_op;
    BlockOperator * interpolation_op_withbnd;
    bool verbose;

protected:
    void ConstructFineTimeStp();
    void ConstructCoarseTimeStp();
    void ConstructGlobalInterpolation();
    void ConstructGlobalInterpolationWithBnd();

public:
    virtual ~TwoGridTimeStepping()
    {
        //for (int i = 0; i < fine_problems.Size(); ++i)
            //delete fine_problems[i];
        delete fine_timestepping;

        delete coarse_timestepping;
        //for (int i = 0; i < coarse_problems.Size(); ++i)
            //delete coarse_problems[i];

        delete interpolation_op;
        delete interpolation_op_withbnd;
    }

    TwoGridTimeStepping(Array<FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* >& cyl_probhierarchies_, bool verbose_)
        : nslabs(cyl_probhierarchies_.Size()), cyl_probhierarchies(cyl_probhierarchies_),
          verbose(verbose_)
    {
        ConstructFineTimeStp();
        fine_global_offsets.SetSize(nslabs + 1);
        fine_global_offsets[0] = 0;
        for (int tslab = 0; tslab < nslabs; ++tslab)
            fine_global_offsets[tslab + 1] = fine_global_offsets[tslab] + fine_problems[tslab]->GlobalTrueProblemSize();

        ConstructCoarseTimeStp();
        coarse_global_offsets.SetSize(nslabs + 1);
        coarse_global_offsets[0] = 0;
        for (int tslab = 0; tslab < nslabs; ++tslab)
            coarse_global_offsets[tslab + 1] = coarse_global_offsets[tslab] + coarse_problems[tslab]->GlobalTrueProblemSize();

        ConstructGlobalInterpolation();
        ConstructGlobalInterpolationWithBnd();
    }

    TimeStepping<Problem> * GetFineTimeStp() { return fine_timestepping;}
    TimeStepping<Problem> * GetCoarseTimeStp() { return coarse_timestepping;}
    BlockOperator * GetGlobalInterpolationOp() { return interpolation_op;}
    BlockOperator * GetGlobalInterpolationOpWithBnd() { return interpolation_op_withbnd;}
    Array<int>& GetFineOffsets() {return fine_global_offsets;}
    Array<int>& GetCoarseOffsets() {return coarse_global_offsets;}
};

template <class Problem>
void TwoGridTimeStepping<Problem>::ConstructFineTimeStp()
{
    int fine_level = 0;
    fine_problems.SetSize(nslabs);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* cyl_probhierarchy = cyl_probhierarchies[tslab];

        fine_problems[tslab] = cyl_probhierarchy->GetProblem(fine_level);

        //BdrConditions& bdr_conds = cyl_probhierarchy->GetProblem(0)->GetBdrConditions();
        //FOSLSFEFormulation & fe_formulation = cyl_probhierarchy->GetProblem(0)->GetFEformulation();
        //fine_problems[tslab] = new Problem(cyl_hierarchy, 0, bdr_conds, fe_formulation, verbose);

        //(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                           //FOSLSFEFormulation& fe_formulation, bool verbose_)
    }
    fine_timestepping = new TimeStepping<Problem>(fine_problems, verbose);
}

template <class Problem>
void TwoGridTimeStepping<Problem>::ConstructCoarseTimeStp()
{
    int coarse_level = 1;
    coarse_problems.SetSize(nslabs);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* cyl_probhierarchy = cyl_probhierarchies[tslab];

        coarse_problems[tslab] = cyl_probhierarchy->GetProblem(coarse_level);

        // replacing the native coarse problem operators by the coarsened versions
        BlockOperator * coarsened_solveop = cyl_probhierarchy->GetCoarsenedOp(coarse_level);
        coarse_problems[tslab]->ResetOp(*coarsened_solveop);
        BlockOperator * coarsened_solveop_nobnd = cyl_probhierarchy->GetCoarsenedOp_nobnd(coarse_level);
        coarse_problems[tslab]->ResetOp_nobnd(*coarsened_solveop_nobnd);

    }
    coarse_timestepping = new TimeStepping<Problem>(coarse_problems, verbose);
}


template <class Problem>
void TwoGridTimeStepping<Problem>::ConstructGlobalInterpolation()
{
    int fine_level = 0;
    interpolation_op = new BlockOperator(fine_global_offsets, coarse_global_offsets);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* cyl_probhierarchy = cyl_probhierarchies[tslab];
        interpolation_op->SetDiagonalBlock(tslab, cyl_probhierarchy->GetTrueP(fine_level));
    }
}

template <class Problem>
void TwoGridTimeStepping<Problem>::ConstructGlobalInterpolationWithBnd()
{
    int fine_level = 0;
    int coarse_level = 1;
    //fine_global_offsets.Print();
    //coarse_global_offsets.Print();
    interpolation_op_withbnd = new BlockOperator(fine_global_offsets, coarse_global_offsets);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* cyl_probhierarchy = cyl_probhierarchies[tslab];
        Array<int>* coarser_bnd_indices = cyl_probhierarchy->ConstructBndIndices(coarse_level);
        Operator * InterpolationOpWithBnd = new InterpolationWithBNDforTranspose(
                    *cyl_probhierarchy->GetTrueP(fine_level), coarser_bnd_indices);

        interpolation_op_withbnd->SetDiagonalBlock(tslab, InterpolationOpWithBnd);
    }
}


template <class Problem> class TimeSteppingSmoother : public Operator
{
    int nslabs;
    TimeStepping<Problem> &time_stepping;
    Array<Vector*> initvec_inputs; // always 0's
    bool verbose;
public:
    TimeSteppingSmoother(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : Operator(time_stepping_.GetGlobalProblemSize()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_), verbose(verbose_)
    {
        initvec_inputs.SetSize(nslabs);
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            initvec_inputs[tslab] = new Vector(time_stepping.GetInitCondSize());
            *initvec_inputs[tslab] = 0.0;
        }
    }

    void Mult(const Vector &x, Vector &y) const override;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }
};

template <class Problem>
void TimeSteppingSmoother<Problem>::Mult(const Vector &x, Vector &y) const
{
    bool compute_error = false;
    // initvec_inputs should be 0 in this call
#ifdef MFEM_DEBUG
    for (int i = 0; i < initvec_inputs.Size(); ++i)
    {
        MFEM_ASSERT(initvec_inputs[i]->Normlinf() < MYZEROTOL,
                    "Initvec_inputs must be 0 here but they are not!");
    }
#endif
    time_stepping.ParallelSolve(x, initvec_inputs, y, compute_error);
}

template <class Problem> class TimeSteppingSolveOp : public BlockOperator
{
protected:
    int nslabs;
    TimeStepping<Problem> &time_stepping;
    const Array<int>& global_offsets;
    bool verbose;
    Vector init_vec;
public:
    TimeSteppingSolveOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()), verbose(verbose_)
    {
        init_vec.SetSize(time_stepping.GetInitCondSize());
        init_vec = 0.0;
    }

    void Mult(const Vector &x, Vector &y) const override;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }
};

// it is implicitly assumed that the problem is solved with zero initial condition for the first time slab
template <class Problem>
void TimeSteppingSolveOp<Problem>::Mult(const Vector &x, Vector &y) const
{
    //Array<Vector*>& vec_inputs = time_stepping.ConvertFullvecIntoArray(x);
    //Array<Vector*>& initvec_inputs = time_stepping.ConvertIntoInitialConditions(vec_inputs);

    // init_vec is actually always 0 in this call
    //init_vec.Print();
    MFEM_ASSERT(init_vec.Normlinf() < MYZEROTOL, "Initvec must be 0 here but it is not!");

    /*
    Array<Vector*> & debug_botbases = time_stepping.ExtractAtBases("bot", x);
    std::cout << "botbases of input x in SolveOp Mult \n";
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "tslab = " << tslab << "\n";
        debug_botbases[tslab]->Print();
    }
    */

    bool compute_error = false;

    /*
    const BlockVector x_viewer(x.GetData(), global_offsets);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "input x in SolveOp, tslab = " << tslab << ": norm =  " <<
                     x_viewer.GetBlock(tslab).Norml2() / sqrt (x_viewer.GetBlock(tslab).Size()) << "\n";
    }
    */

    time_stepping.SequentialSolve(x, init_vec, y, compute_error);

    /*
    BlockVector y_viewer(y.GetData(), global_offsets);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "output y in SolveOp, tslab = " << tslab << ": norm =  " <<
                     y_viewer.GetBlock(tslab).Norml2() / sqrt (y_viewer.GetBlock(tslab).Size()) << "\n";
    }
    */

    /*
    Array<Vector*> & debug_botbases2 = time_stepping.ExtractAtBases("bot", y);
    std::cout << "botbases of output y in SolveOp Mult \n";
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "tslab = " << tslab << "\n";
        debug_botbases2[tslab]->Print();
    }
    */
}

template <class Problem> class TimeSteppingSeqOp : public BlockOperator
{
    int nslabs;
    TimeStepping<Problem> &time_stepping;
    const Array<int>& global_offsets;
    bool verbose;
public:
    TimeSteppingSeqOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()),
          verbose(verbose_) {}

    void Mult(const Vector &x, Vector &y) const override
    {
        /*
        Array<Vector*> & debug_botbases = time_stepping.ExtractAtBases("bot", x);
        std::cout << "botbases of input x in SeqOp Mult \n";
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            std::cout << "tslab = " << tslab << "\n";
            debug_botbases[tslab]->Print();
        }
        */

        /*
        const BlockVector x_viewer(x.GetData(), global_offsets);
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            std::cout << "input x in SeqOp, tslab = " << tslab << ": norm = " <<
                         x_viewer.GetBlock(tslab).Norml2() / sqrt (x_viewer.GetBlock(tslab).Size()) << "\n";
        }
        */

        time_stepping.SeqOp(x,y);

        /*
        BlockVector y_viewer(y.GetData(), global_offsets);
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            std::cout << "output x in SeqOp, tslab = " << tslab << ": norm = " <<
                         y_viewer.GetBlock(tslab).Norml2() / sqrt (y_viewer.GetBlock(tslab).Size()) << "\n";
        }
        */

        /*
        Array<Vector*> & debug_botbases2 = time_stepping.ExtractAtBases("bot", y);
        std::cout << "botbases of output y in SeqOp Mult \n";
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            std::cout << "tslab = " << tslab << "\n";
            debug_botbases2[tslab]->Print();
        }
        */
    }

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }

};

template <class Problem> class TSTSpecialSolveOp : public BlockOperator
{
protected:
    int nslabs;
    TimeStepping<Problem> &time_stepping;
    const Array<int>& global_offsets;
    bool verbose;
    Vector * init_vec;
    Vector * init_vec2;
public:
    TSTSpecialSolveOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()), verbose(verbose_)
    {
        init_vec = new Vector(time_stepping.GetInitCondSize());
        *init_vec = 0.0;

        init_vec2 = new Vector(time_stepping.GetInitCondSize());
        *init_vec2 = 0.0;
    }

    void Mult(const Vector &x, Vector &y) const override;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }
};

// it is implicitly assumed that the problem is solved with zero initial condition for the first time slab
template <class Problem>
void TSTSpecialSolveOp<Problem>::Mult(const Vector &x, Vector &y) const
{
    // init_vec is actually always 0 in this call
    MFEM_ASSERT(init_vec->Normlinf() < MYZEROTOL, "Initvec must be 0 here but it is not!");

    bool compute_error = false;

    const BlockVector x_viewer(x.GetData(), global_offsets);
    BlockVector y_viewer(y.GetData(), global_offsets);

    /*
    Array<Vector*> & debug_botbases = ExtractAtBases("bot", x);
    std::cout << "botbases of x in the call to TSTSpecialSolveOp Mult \n";
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "tslab = " << tslab << "\n";
        debug_botbases[tslab]->Print();
    }
    */


    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * problem = time_stepping.GetProblem(tslab);

        FOSLSFEFormulation& fe_formul = problem->GetFEformulation();
        int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();
        SpaceName space_name = fe_formul.GetFormulation()->GetSpaceName(index);

        if (tslab > 0)
        {
            problem->ExtractAtBase("bot", x_viewer.GetBlock(tslab), *init_vec);
        }

        //std::cout << "init_vec for tslab = " << tslab << "\n";
        //init_vec->Print();

        problem->Solve(x_viewer.GetBlock(tslab), *init_vec,
                       y_viewer.GetBlock(tslab), *init_vec2, compute_error);

        if (time_stepping.NeedSignSwitch(space_name))
            *init_vec2 *= -1;

        if (tslab > 0)
        {
            problem->ExtractAtBase("bot", x_viewer.GetBlock(tslab), *init_vec);

            *init_vec += *init_vec2;
        }
    }

    /*

    Array<Vector*> & debug_botbases2 = ExtractAtBases("bot", y);
    std::cout << "botbases of y in the call to TSTSpecialSolveOp Mult \n";
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        std::cout << "tslab = " << tslab << "\n";
        debug_botbases2[tslab]->Print();
    }
    */



    *init_vec = 0.0;
}



/*
template <class Problem> class TwoGridTimeST
{
protected:
    int nslabs;
    Array<FOSLSCylProblHierarchy<Problem> *>& cylproblems_hierarchies;
    Array<Problem*> timeslabs_fineproblems;
    TimeStepping<Problem> * fine_timestepping;
    std::vector<Vector*> timeslabs_fineresiduals;
    Array<Problem*> timeslabs_coarseproblems;
    TimeStepping<Problem> * coarse_timestepping;
    std::vector<Vector*> timeslabs_coarseresiduals;
    bool verbose;
public:
    TwoGridTimeST(Array<FOSLSCylProblHierarchy<Problem> *>& tslabs_probl_hierarchy, bool verbose_)
        : nslabs(tslabs_probl_hierarchy.Size()), cylproblems_hierarchies(tslabs_probl_hierarchy), verbose(verbose_)
    {
        timeslabs_fineproblems.SetSize(nslabs);
        timeslabs_coarseproblems.SetSize(nslabs);
        timeslabs_fineresiduals.resize(nslabs);
        timeslabs_coarseresiduals.resize(nslabs);
        for (int tslab = 0; tslab < nslabs; ++tslab)
        {
            MFEM_ASSERT(cylproblems_hierarchies[tslab]->Nlevels() >= 2, "Each problem hierarchy must have at least two levels");
            timeslabs_fineproblems[tslab] = cylproblems_hierarchies[tslab]->GetProblem(0);
            timeslabs_coarseproblems[tslab] = cylproblems_hierarchies[tslab]->GetProblem(1);

            timeslabs_fineresiduals[tslab] = new Vector(timeslabs_fineproblems[tslab]->GlobalTrueProblemSize());
            timeslabs_coarseresiduals[tslab] = new Vector(timeslabs_coarseproblems[tslab]->GlobalTrueProblemSize());
        }

        fine_timestepping = new TimeStepping<Problem>(timeslabs_fineproblems, verbose);
        coarse_timestepping = new TimeStepping<Problem>(timeslabs_coarseproblems, verbose);
    }

    void Iterate();
    void SmootherAction();
    void Restrict();
    void Interpolate();
    void CoarseSolve(const Vector &vec_in, std::vector<Vector *> &corrections);
};

template <class Problem> void TwoGridTimeST<Problem>::Restrict()
{

}

template <class Problem> void TwoGridTimeST<Problem>::Interpolate()
{

}

template <class Problem> void TwoGridTimeST<Problem>::CoarseSolve(const Vector& vec_in, std::vector<Vector*> & corrections)
{
    MFEM_ASSERT((int)(corrections.size()) == nslabs, "Incorrect number of Vectors for the output argument!");

    coarse_timestepping->SequentialSolve(vec_in, verbose);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        *correction[tslab] = timeslabs_coarseproblems[tslab]->GetSol();
    }
}

template <class Problem> void TwoGridTimeST<Problem>::Iterate()
{

}

*/



//#####################################################################################################

// abstract base class for a problem in a time cylinder
class TimeCyl
{
protected:
    ParMeshCyl * pmeshtsl;
    double t_init;
    double tau;
    int nt;
    bool own_pmeshtsl;
public:
    virtual ~TimeCyl();
    TimeCyl (ParMesh& Pmeshbase, double T_init, double Tau, int Nt);
    TimeCyl (ParMeshCyl& Pmeshtsl) : pmeshtsl(&Pmeshtsl), own_pmeshtsl(false) {}

    // interpretation of the input and output vectors depend on the implementation of Solve
    // but they are related to the boundary conditions at the input and at he output
    virtual void Solve(const Vector& vec_in, Vector& vec_out) const = 0;
};

// specific class for time-slabbing in hyperbolic problems
class TimeCylHyper : public TimeCyl
{
private:
    MPI_Comm comm;

protected:
    int ref_lvls;
    const char *formulation;
    const char *space_for_S;
    const char *space_for_sigma;
    int feorder;
    int dim;
    int numsol;

    std::vector<int> ess_bdrat_S;
    std::vector<int> ess_bdrat_sigma;

    GeneralCylHierarchy * hierarchy;

    std::vector<ParFiniteElementSpace* > Sigma_space_lvls; // shortcut (may be useful if consider vector H1 for sigma at some moment
    std::vector<ParFiniteElementSpace* > S_space_lvls;     // shortcut

    std::vector<Array<int>*> block_trueOffsets_lvls;
    std::vector<BlockOperator*> CFOSLSop_lvls;
    std::vector<BlkHypreOperator*> CFOSLSop_coarsened_lvls;
    std::vector<BlockOperator*> CFOSLSop_nobnd_lvls;
    std::vector<BlockDiagonalPreconditioner*> prec_lvls;
    std::vector<MINRESSolver*> solver_lvls;
    std::vector<BlockVector*> trueRhs_nobnd_lvls;
    std::vector<BlockVector*> trueX_lvls;

    std::vector<BlockOperator*> TrueP_lvls;

    bool visualization;
public:
    bool verbose;

protected:
    void InitProblem(int numsol);

public:
    ~TimeCylHyper();
    TimeCylHyper (ParMesh& Pmeshbase, double T_init, double Tau, int Nt, int Ref_lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma, int Numsol);
    TimeCylHyper (ParMeshCyl& Pmeshtsl, int Ref_Lvls,
                   const char *Formulation, const char *Space_for_S, const char *Space_for_sigma, int Numsol);

    virtual void Solve(const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const override
    { Solve(0, bnd_tdofs_bot, bnd_tdofs_top); }

    void Solve(int lvl, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;
    void Solve(const char * mode, int lvl, const Vector& rhs, Vector& sol, const Vector &bnd_tdofs_bot, Vector &bnd_tdofs_top) const;

    void ComputeAnalyticalRhs(int lvl);

    GeneralCylHierarchy * GetHierarchy() {return hierarchy;}

    int GetInitCondSize(int lvl) const
    {
        if (strcmp(space_for_S,"H1") == 0)
            return hierarchy->GetTdofs_H1_link(lvl)->size();
        else
            return hierarchy->GetTdofs_Hdiv_link(lvl)->size();
    }

    int GetNLevels() {return ref_lvls;}

    std::vector<std::pair<int,int> > * GetTdofsLink(int lvl)
    {
        if (strcmp(space_for_S,"H1") == 0)
            return hierarchy->GetTdofs_H1_link(lvl);
        else
            return hierarchy->GetTdofs_Hdiv_link(lvl);
    }

    ParFiniteElementSpace * Get_S_space(int lvl = 0) {return S_space_lvls[lvl];}
    ParFiniteElementSpace * Get_Sigma_space(int lvl = 0) {return Sigma_space_lvls[lvl];}

    /*
    HypreParMatrix * Get_TrueP_H1(int lvl)
    {
        if (lvl >= 0 && lvl < ref_lvls)
            if (TrueP_H1_lvls[lvl])
                return TrueP_H1_lvls[lvl];
        return NULL;
    }
    HypreParMatrix * Get_TrueP_Hdiv(int lvl)
    {
        if (lvl >= 0 && lvl < ref_lvls)
            if (TrueP_Hdiv_lvls[lvl])
                return TrueP_Hdiv_lvls[lvl];
        return NULL;
    }
    */

    ParMeshCyl * GetParMeshCyl(int lvl) { return hierarchy->GetPmeshcyl(lvl);}

    Vector* GetSol(int lvl)
    { return trueX_lvls[lvl]; }

    Vector *GetExactBase(const char * top_or_bot, int level);

    Array<int>* GetBlockTrueOffsets(int lvl)
    { return block_trueOffsets_lvls[lvl]; }

    int ProblemSize(int lvl)
    {return CFOSLSop_lvls[lvl]->Height();}

    bool NeedSignSwitch() {return (strcmp(space_for_S, "L2") == 0);}

    void InterpolateAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out);

    void Interpolate(int lvl, const Vector& vec_in, Vector& vec_out);

    // FIXME: Does one need to scale the restriction?
    // Yes, in general it should be a canonical interpolator transpose, not just transpose of the standard
    void RestrictAtBase(const char * top_or_bot, int lvl, const Vector& vec_in, Vector& vec_out);

    void Restrict(int lvl, const Vector& vec_in, Vector& vec_out);

    // Takes a vector of values corresponding to the bdr condition
    // and computes the corresponding change to the rhs side
    void ConvertBdrCndIntoRhs(int lvl, const Vector& vec_in, Vector& vec_out);

    // vec_in is considered as a vector of strictly values at the bottom boundary,
    // vec_out is a full vector which coincides with vec_in at initial boundary and
    // has 0's for all the rest entries
    void ConvertInitCndToFullVector(int lvl, const Vector& vec_in, Vector& vec_out);

    void ComputeResidual(int lvl, const Vector& initcond_in, const Vector& sol, Vector& residual);

    void ComputeError(int lvl, Vector& sol) const;

};

class TimeSteppingScheme
{
protected:
    std::vector<TimeCylHyper*> timeslab_problems;
    int nslabs;
    int nlevels;
    bool verbose;
    std::vector<std::vector<Vector*> > vec_ins_lvls;
    std::vector<std::vector<Vector*> > vec_outs_lvls;

    std::vector<std::vector<Vector*> > residuals_lvls;
    std::vector<std::vector<Vector*> > sols_lvls;

public:
    TimeSteppingScheme (std::vector<TimeCylHyper*> & timeslab_problems);

    void Solve(char const * mode, const char *level_mode, std::vector<Vector *> rhss, int level, bool compute_accuracy);
    void Solve(char const * mode, const char *level_mode, int level, bool compute_accuracy);

    void ComputeResiduals(int level);

    void RestrictToCoarser(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs);
    void InterpolateToFiner(int level, std::vector<Vector*> vec_ins, std::vector<Vector*> vec_outs);

    void ComputeAnalyticalRhs(int level);

    std::vector<Vector*> * Get_vec_ins(int level){return &vec_ins_lvls[level];}
    std::vector<Vector*> * Get_vec_outs(int level){return &vec_outs_lvls[level];}
    std::vector<Vector*> * Get_sols(int level){return &sols_lvls[level];}
    std::vector<Vector*> * Get_residuals(int level){return &residuals_lvls[level];}

    TimeCylHyper * GetTimeSlab(int tslab) {return timeslab_problems[tslab];}

    void SetInitialCondition(const Vector& x_init, int level)
    { *vec_ins_lvls[level][0] = x_init;}

    void SetInitialConditions(std::vector<Vector*> x_inits, int level)
    {
        MFEM_ASSERT( (int) (x_inits.size()) >= nslabs, "Number of initial vectors is less than number of time slabs! \n");
        for (int tslab = 0; tslab < nslabs; ++tslab)
            *vec_ins_lvls[level][tslab] = *x_inits[tslab];
    }

    int GetNSlabs() {return nslabs;}
    int GetNLevels() {return nlevels;}
};


class SpaceTimeTwoGrid
{
protected:
    TimeSteppingScheme& timestepping;
    int nslabs;
    int max_iter;
    double tol;

    const int num_lvls = 2;

    std::vector<std::vector<Vector*> > res_lvls;
    std::vector<std::vector<Vector*> > corr_lvls;

public:
    SpaceTimeTwoGrid(TimeSteppingScheme& TimeStepping, int Max_Iter, double Tol )
        : timestepping(TimeStepping),
          nslabs(TimeStepping.GetNSlabs()),
          max_iter(Max_Iter),
          tol(Tol)
    {
        MFEM_ASSERT(timestepping.GetNLevels() > 1, "For a two-grid method at least two levels must exist! \n");

        res_lvls.resize(num_lvls);
        for (unsigned int l = 0; l < res_lvls.size(); ++l)
        {
            res_lvls[l].resize(nslabs);
            for (int slab = 0; slab < nslabs; ++slab)
                res_lvls[l][slab] = new Vector(timestepping.GetTimeSlab(slab)->ProblemSize(l));
        }

        corr_lvls.resize(num_lvls);
        for (unsigned int l = 0; l < corr_lvls.size(); ++l)
        {
            corr_lvls[l].resize(nslabs);
            for (int slab = 0; slab < nslabs; ++slab)
                corr_lvls[l][slab] = new Vector(timestepping.GetTimeSlab(slab)->ProblemSize(l));
        }

    }
    void Solve(std::vector<Vector *> rhss, std::vector<Vector *> sols);

protected:
    void Iterate(std::vector<Vector*> ress, std::vector<Vector*> corrs);
    void ComputeResidual(std::vector<Vector*> rhss, std::vector<Vector*> sols);
    void UpdateResidual(std::vector<Vector*> corrs);
    void UpdateSolution(std::vector<Vector*> sols, std::vector<Vector*> corrs);

};

} // for namespace mfem


#endif
