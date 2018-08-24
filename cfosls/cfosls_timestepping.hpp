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

    // temporary storage vectors, used in CorrectFromInitCond()
    mutable Vector temp_vec1;
    mutable Vector temp_vec2;

protected:
    // One of the key functions of the class.
    // Constructs a tdof link (as a vector of pairs) between tdofs at the top
    // and bottom bases of the cylinder
    // Depending on the values of init_cond_space, it's either a link for RT0 or
    // linear H1 f.e. tdofs
    void ConstructTdofLink();

    // extracts from vector x its values (tdofs) at the top boundary of the cylinder
    void ExtractTopTdofs(const Vector& x, Vector& bnd_tdofs_top) const;

    void ExtractBotTdofs(const Vector& x, Vector& bnd_tdofs_bot) const;

public:
    virtual ~FOSLSCylProblem() {}

    FOSLSCylProblem (ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          pmeshcyl(Pmeshcyl), cyl_hierarchy(NULL),
          init_cond_block(fe_formul.GetFormulation()->GetUnknownWithInitCnd())
    {
        const Array<SpaceName>* spacenames = fe_formul.GetFormulation()->GetSpacesDescriptor();
        init_cond_space = (*spacenames)[init_cond_block];
        ConstructTdofLink();

        temp_vec1.SetSize(TrueProblemSize());
        temp_vec2.SetSize(TrueProblemSize());
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

        temp_vec1.SetSize(TrueProblemSize());
        temp_vec2.SetSize(TrueProblemSize());
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

    // construct on demand a restriction to either top or bottom base for
    // the pfespace tdofs corresponding to the initial condition, i.e.
    // for which the initial condition is being set for the problem
    HypreParMatrix * ConstructRestriction(const char * top_or_bot) const;

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
class FOSLSCylProblem_HdivL2hyp : public FOSLSCylProblem, public FOSLSProblem_HdivL2hyp
{
public:
    FOSLSCylProblem_HdivL2hyp(ParMeshCyl& Pmeshcyl, BdrConditions& bdr_conditions,
                    FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Pmeshcyl, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivL2hyp(Pmeshcyl, bdr_conditions, fe_formulation, precond_option, verbose_)
    {}

    FOSLSCylProblem_HdivL2hyp(GeneralCylHierarchy& Hierarchy, int level, BdrConditions& bdr_conditions,
                   FOSLSFEFormulation& fe_formulation, int precond_option, bool verbose_)
        : FOSLSProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSCylProblem(Hierarchy, level, bdr_conditions, fe_formulation, verbose_),
          FOSLSProblem_HdivL2hyp(Hierarchy, level, bdr_conditions, fe_formulation, precond_option, verbose_)
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
/// Contained problems is assumed to be at least of type FOSLSProblem (or it's children)
/// The general setup is that we are solving a problem in a time cylinder which
/// is split into multiple time slabs (each being a cylinder itself)
/// Terms:
/// 1) entire domain = entire time cylinder domain (to distinguish from a time slab)
/// 2) slab ~ time slab, one of the time cylinders which is a part of the entire domain
/// 3) global vector ~ vector defined in the entire domain as a set of time slabs.
/// (!) This means that this vector actually have two values at each interface between time slabs,
/// one from the first time slab (as top base values) and one from the second (as bottom base
/// values)
template <class Problem> class TimeStepping
{
protected:
    // stores the offsets w.r.t to the time slabs,
    // e.g. global_offsets[1] - global_offsets[0] = size of the problem
    // related to the first time slab
    Array<int> global_offsets;

    // object doesn't own the problems, it merely copies them via a call to
    // SetProblems()
    Array<Problem*> timeslabs_problems;

    // to each time-like boundary t = const, i.e. bottom and top bases
    // of the entire domain and interfaces between time slabs
    // two vectors are assigned, called base_inputs[i] and
    // base outputs[i], i = 0, .. N(time slabs)
    // These are used to transfer information between time slabs
    Array<Vector*>  base_inputs;

    // See above
    Array<Vector*>  base_outputs;

    bool verbose;

    bool problems_initialized;

    // number of time slabs
    int nslabs;

    // temporary vector used in UpdateInterfaceFromPrev()
    mutable Vector vecbase_temp;

protected:
    void SetProblems(Array<Problem*>& timeslabs_problems_);

public:
    ~TimeStepping()
    {
        for (int i = 0; i < base_inputs.Size(); ++i)
            delete base_inputs[i];
        for (int i = 0; i < base_outputs.Size(); ++i)
            delete base_outputs[i];
    }

    TimeStepping(Array<Problem*>& timeslabs_problems_, bool verbose_)
        : timeslabs_problems(0), base_inputs(0), base_outputs(0),
          verbose(verbose_), problems_initialized(false)
    {
        SetProblems(timeslabs_problems_);
        global_offsets.SetSize(nslabs + 1);

        global_offsets[0] = 0;
        for (int tslab = 0; tslab < nslabs; ++tslab)
            global_offsets[tslab + 1] = timeslabs_problems[tslab]->TrueProblemSize();
        global_offsets.PartialSum();

        vecbase_temp.SetSize(timeslabs_problems[0]->GetInitCondSize());
    }

    // Performs a sequential solve (sequential-in-time) with a given init_vector
    // at the bottom base of the entire domain (and the bottom base of the first time slab)
    void SequentialSolve(const Vector &init_vector, bool compute_error);

    // The same as previous but also takes in the given global rhs vector
    void SequentialSolve(const Vector& rhs, const Vector &init_vector, bool compute_error);

    // The same as previous but also copies the computed solutions within time slabs into
    // the output Vector sol
    void SequentialSolve(const Vector& rhs, const Vector &init_vector, Vector& sol,
                         bool compute_error);

    // Takes a bunch of initial vectors at bottom bases for the time slabs
    // and performs a solve within each time slab
    // It's called parallel because it is parallel-in-time,
    // each time slab problem depends only on the corresponding init_vector
    // and doesn't depend on other time slabs
    void ParallelSolve(const Array<Vector*> &init_vectors, bool compute_error) const;

    // The same as previous, but also takes the global rhs vector as an input
    void ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, bool compute_error) const;

    // The same as previous, but computes the solution into the global output Vector sol
    // (actually, it copies the internal solutions within each time slab into the output)
    void ParallelSolve(const Vector& rhs, const Array<Vector*> &init_vectors, Vector& sol,
                       bool compute_error) const;

    // decides whether it is required to change the sign when transferring
    // interface values between time slabs
    // e.g., if we transfer values of Hdiv function, then since the normal vector
    // flips its sign after corssing the interface between slabs, we should change the sign
    // if we want to take values from a grid function from Hdiv at the top base of the
    // previous time slab and use them as initial data for the next one.
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

    // Takes a vector of the entire time cylinder's problem size and
    // converts it into an array of Vectors, which of those corresponds
    // to a time slab
    // FIXME: Is there a memory leak here?
    Array<Vector*> & ConvertFullvecIntoArray(const Vector& x);

    // Opposite to ConvertFullvecIntoArray()
    // output vector must be allocated in advance
    // FIXME: Unused, probably not needed at all
    void ConvertArrayIntoFullvec(const Array<Vector*>& vec_inputs, Vector& out);

    // Applies a sequential (in a time-stepping fashion) solution of the entire problem
    // x is treated as the global righthand side
    // init_bot are the initial values, at the bottom base of the very first time slab
    void SeqOp(const Vector& x, const Vector* init_bot, Vector& y) const;

    // See above, this one takes zero initial values
    // This is useful for certain multigrid-in-time algorithms
    // e.g, through the class TimeSteppingSeqOp
    void SeqOp(const Vector& x, Vector& y) const { SeqOp(x, NULL, y);}

    // Computes a global analytical (from the FOSLStest) rhs vector,
    // for the entire problem, by computing analytical righthand side
    // in each time slab
    void ComputeGlobalRhs(Vector& rhs);

    // takes a vector, and in each time slab sets the boundary values to zero
    // (at the essential part of the boundary)
    void ZeroBndValues(Vector& vec);

    // Takes a global vector (for the entire problem) and extracts within each time slab
    // it's values on top or bottom base.
    // FIXME: A memory leak here?
    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const;

    // Takes a global vector, and for each time slab except the first one,
    // sets it's values on the bottom base from the previous time slab's top values
    // (*) doesn't do any sign switch
    void UpdateInterfaceFromPrev(Vector& vec) const;

    // various getters

    // getting solutions from all time slabs
    Array<Vector*>& GetSolutions();

    const Array<int>& GetGlobalOffsets() const {return global_offsets;}

    Problem * GetProblem(int i) {return timeslabs_problems[i];}

    int GetGlobalProblemSize() const { return global_offsets[nslabs]; }

    int GetInitCondSize() const
    { return timeslabs_problems[0]->GetInitCondSize();}

    int Nslabs() const {return nslabs;}

    // routines for computing errors for a global vector
    void ComputeError(const Vector& vec) const;

    void ComputeBndError(const Vector& vec) const;
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

        prev_problem->ExtractAtBase("top", vec_viewer.GetBlock(tslab - 1), vecbase_temp);

        problem->SetAtBase("bot", vecbase_temp, vec_viewer.GetBlock(tslab));
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

    for (int tslab = 0; tslab < nslabs; ++tslab )
    {
        Problem * tslab_problem = timeslabs_problems[tslab];

        if (tslab == 0)
            tslab_problem->Solve(rhs_viewer.GetBlock(tslab), init_vector, *base_outputs[tslab], compute_error);
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
// FIXME: It's a memory leak?
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

        res[tslab] = new Vector(tslab_problem->TrueProblemSize());

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

            tslab_problem->CorrectFromInitCond(*prev_initcond, y_viewer.GetBlock(tslab), 1.0);
            delete prev_initcond;
        }
        else
            if (init_bot)
                tslab_problem->CorrectFromInitCond(*init_bot, y_viewer.GetBlock(tslab), 1.0);
    }

}

// classes for time-stepping related operators used as components for
// constructing GeneralMultigrid instance to be used for parallel-in-time algorothms

/// Generic class for two-grid time-stepping schemes
/// aka parallel-in-time
/// This class is merely a wrapper around two time-stepping objects
/// which correspond to the fine and coarse meshes, plus interpolation between them
/// Terms:
/// See terms for TimeStepping class
template <class Problem> class TwoGridTimeStepping
{
protected:
    int nslabs;

    // not owned by the object
    Array<FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* >& cyl_probhierarchies;

    // fine problems are owned by cyl_probhierarchies
    Array<Problem*> fine_problems;
    Array<int> fine_global_offsets;

    // fine-grid time-stepping, built from finest levels in cyl_probhierarchies
    TimeStepping<Problem> * fine_timestepping;

    // coarse problems are owned by cyl_probhierarchies
    Array<Problem*> coarse_problems;
    Array<int> coarse_global_offsets;

    // coarse-grid time-stepping, built from coarsest (second) levels in cyl_probhierarchies
    TimeStepping<Problem> * coarse_timestepping;

    // a "global" operator which interpolates global vectors from the coarse mesh
    // (actually, coarse meshES in time slabs) to fine mesh
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
        delete fine_timestepping;
        delete coarse_timestepping;

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
            fine_global_offsets[tslab + 1] = fine_global_offsets[tslab] +
                    fine_problems[tslab]->TrueProblemSize();

        ConstructCoarseTimeStp();
        coarse_global_offsets.SetSize(nslabs + 1);
        coarse_global_offsets[0] = 0;
        for (int tslab = 0; tslab < nslabs; ++tslab)
            coarse_global_offsets[tslab + 1] = coarse_global_offsets[tslab] +
                    coarse_problems[tslab]->TrueProblemSize();

        ConstructGlobalInterpolation();
        ConstructGlobalInterpolationWithBnd();
    }

    // getters
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
        coarse_problems[tslab]->ResetOp(*coarsened_solveop, false);
        coarse_problems[tslab]->UpdateSolverMat(*coarse_problems[tslab]->GetOp());
        BlockOperator * coarsened_solveop_nobnd = cyl_probhierarchy->GetCoarsenedOp_nobnd(coarse_level);
        coarse_problems[tslab]->ResetOp_nobnd(*coarsened_solveop_nobnd, false);

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

    interpolation_op_withbnd = new BlockOperator(fine_global_offsets, coarse_global_offsets);
    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        FOSLSCylProblHierarchy<Problem, GeneralCylHierarchy>* cyl_probhierarchy = cyl_probhierarchies[tslab];
        Array<int>* coarser_bnd_indices = cyl_probhierarchy->ConstructBndIndices(coarse_level);
        Operator * InterpolationOpWithBnd = new InterpolationWithBNDforTranspose(
                    *cyl_probhierarchy->GetTrueP(fine_level), coarser_bnd_indices);

        // the constructor of InterpolationWithBNDforTranspose above actually copies the bnd indices
        // inside itself so we can delete memory here
        delete coarser_bnd_indices;

        interpolation_op_withbnd->SetDiagonalBlock(tslab, InterpolationOpWithBnd);
        interpolation_op_withbnd->owns_blocks = true;
    }
}

/// Generic class for a smoother built on a parallel-in-time algorithm
/// Used as component for parallel-in-time GeneralMultigrid objects
/// Terms: See terms of TimeStepping class
template <class Problem> class TimeSteppingSmoother : public Operator
{
    int nslabs;
    // doesn't own this
    TimeStepping<Problem> &time_stepping;

    // Temporary zero vectors used in Mult()
    // One might get rid of them, these seem to be redundant
    Array<Vector*> initvec_inputs; // always 0's
    bool verbose;
public:
    virtual ~TimeSteppingSmoother()
    {
        for (int i = 0; i < initvec_inputs.Size(); ++i)
            delete initvec_inputs[i];
    }

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

    // Performs a parallel solve, using x as global rhs vector,
    // and initvec_inputs (which are always zero) as initial values for each time slab
    void Mult(const Vector &x, Vector &y) const override;

    // FIXME: Is anyone using it?
    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }

    // Getter (unused)
    TimeStepping<Problem> & GetTimeStepping() { return time_stepping;}
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

/// Generic class for a solve operator built on a sequential time-stepping soving algorithm
/// Used as component for parallel-in-time GeneralMultigrid objects
/// Actually, it's an Operator wrapper around sequential time-stepping algorithm
/// with zero starting guess
/// Terms: See terms of TimeStepping class
template <class Problem> class TimeSteppingSolveOp : public BlockOperator
{
protected:
    int nslabs;
    // doesn't own this
    TimeStepping<Problem> &time_stepping;

    // doesn't own this
    const Array<int>& global_offsets;

    bool verbose;

    // Temporary zero vector (used as zero starting guess in Mult())
    // One might get rid of this, it's redundant
    Vector init_vec;

public:
    virtual ~TimeSteppingSolveOp() {}
    TimeSteppingSolveOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()), verbose(verbose_)
    {
        init_vec.SetSize(time_stepping.GetInitCondSize());
        init_vec = 0.0;
    }

    // Peforms a sequential solve with x as a global rhs vector, and init_vec (always zero)
    // as starting values at the botom base of the entire domain
    void Mult(const Vector &x, Vector &y) const override;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }
};

// it is implicitly assumed that the problem is solved with zero initial condition for the first time slab
template <class Problem>
void TimeSteppingSolveOp<Problem>::Mult(const Vector &x, Vector &y) const
{
    // init_vec is actually always 0 in this call
    MFEM_ASSERT(init_vec.Normlinf() < MYZEROTOL, "Initvec must be 0 here but it is not!");

    bool compute_error = false;

    time_stepping.SequentialSolve(x, init_vec, y, compute_error);
}

/// Generic class for an operator action built on a sequential time-stepping algorithm
/// operator. Unlike TimeSteppingSolveOp, this one corresponds to an action of the
/// sequential time-stepping operator rather than corresponding solution operator
/// Used as component for parallel-in-time GeneralMultigrid objects
/// Actually, it's an Operator wrapper around sequential time-stepping operator action
/// SeqOp()
/// Terms: See terms of TimeStepping class
template <class Problem> class TimeSteppingSeqOp : public BlockOperator
{
    int nslabs;
    // doesn't own this
    TimeStepping<Problem> &time_stepping;

    // doesn't own this
    const Array<int>& global_offsets;

    bool verbose;
public:
    virtual ~TimeSteppingSeqOp() {}

    TimeSteppingSeqOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()),
          verbose(verbose_)
    {}

    // Simply calls SeqOp() for the underlying time-stepping instance
    void Mult(const Vector &x, Vector &y) const override
    { time_stepping.SeqOp(x,y); }

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }

};

/// Generic class for ...
/// Terms: See terms of TimeStepping class
template <class Problem> class TSTSpecialSolveOp : public BlockOperator
{
protected:
    int nslabs;

    // doesn't own this
    TimeStepping<Problem> &time_stepping;

    // doesn't own this
    const Array<int>& global_offsets;

    bool verbose;

    // Temporary vectors used in Mult()
    mutable Vector init_vec;
    mutable Vector init_vec2;
public:
    virtual ~TSTSpecialSolveOp() {}

    TSTSpecialSolveOp(TimeStepping<Problem> &time_stepping_, bool verbose_)
        : BlockOperator(time_stepping_.GetGlobalOffsets()),
          nslabs(time_stepping_.Nslabs()), time_stepping(time_stepping_),
          global_offsets(time_stepping_.GetGlobalOffsets()), verbose(verbose_)
    {
        init_vec.SetSize(time_stepping.GetInitCondSize());
        init_vec = 0.0;

        init_vec2.SetSize(time_stepping.GetInitCondSize());
        init_vec2 = 0.0;
    }

    void Mult(const Vector &x, Vector &y) const override;

    Array<Vector*>& ExtractAtBases(const char * top_or_bot, const Vector& fullvec) const
    { return time_stepping.ExtractAtBases(top_or_bot, fullvec); }
};

// It's assumed implicitly that the problem is solved with zero initial condition for the first time slab
template <class Problem>
void TSTSpecialSolveOp<Problem>::Mult(const Vector &x, Vector &y) const
{
    // init_vec is actually always 0 in this call
    MFEM_ASSERT(init_vec.Normlinf() < MYZEROTOL, "Initvec must be 0 here but it is not!");

    bool compute_error = false;

    const BlockVector x_viewer(x.GetData(), global_offsets);
    BlockVector y_viewer(y.GetData(), global_offsets);

    for (int tslab = 0; tslab < nslabs; ++tslab)
    {
        Problem * problem = time_stepping.GetProblem(tslab);

        FOSLSFEFormulation& fe_formul = problem->GetFEformulation();
        int index = fe_formul.GetFormulation()->GetUnknownWithInitCnd();
        SpaceName space_name = fe_formul.GetFormulation()->GetSpaceName(index);

        if (tslab > 0)
            problem->ExtractAtBase("bot", x_viewer.GetBlock(tslab), init_vec);

        problem->Solve(x_viewer.GetBlock(tslab), init_vec,
                       y_viewer.GetBlock(tslab), init_vec2, compute_error);

        if (time_stepping.NeedSignSwitch(space_name))
            init_vec2 *= -1;

        if (tslab > 0)
        {
            problem->ExtractAtBase("bot", x_viewer.GetBlock(tslab), init_vec);

            init_vec += init_vec2;
        }
    }

    init_vec = 0.0;
}

} // for namespace mfem


#endif
