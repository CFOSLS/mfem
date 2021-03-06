// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_LINEARFORM
#define MFEM_LINEARFORM

#include "../config/config.hpp"
#include "lininteg.hpp"
#include "gridfunc.hpp"

namespace mfem
{

/// Class for linear form - Vector with associated FE space and LFIntegrators.
class LinearForm : public Vector
{
private:
   /// FE space on which LF lives.
   FiniteElementSpace * fes;

   /// Indicates the LinerFormIntegrators are owned by another LinearForm
   int extern_lfs;

   /// Set of Domain Integrators to be applied.
   Array<LinearFormIntegrator*> dlfi;
   Array<bool> dlfi_owned;

   /// Set of Boundary Integrators to be applied.
   Array<LinearFormIntegrator*> blfi;
   Array<bool> blfi_owned;

   /// Set of Boundary Face Integrators to be applied.
   Array<LinearFormIntegrator*> flfi;
   Array<Array<int>*>           flfi_marker;
   Array<bool> flfi_owned;

public:
   /// Creates linear form associated with FE space *f.
   LinearForm (FiniteElementSpace * f) : Vector (f -> GetVSize())
   { fes = f; extern_lfs = 0; }

   LinearForm (FiniteElementSpace * f, LinearForm *lf);

   LinearForm() { fes = NULL; }

   FiniteElementSpace * GetFES() { return fes; }

   /// Adds new Domain Integrator.
   void AddDomainIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Domain Integrator, but doesn't take the ownership.
   void BorrowDomainIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Integrator.
   void AddBoundaryIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Integrator, but doesn't take the ownership.
   void BorrowBoundaryIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Face Integrator.
   void AddBdrFaceIntegrator (LinearFormIntegrator * lfi);

   /// Adds new Boundary Face Integrator, but doesn't take the ownership.
   void BorrowBdrFaceIntegrator (LinearFormIntegrator * lfi);

   /** @brief Add new Boundary Face Integrator, restricted to the given boundary
       attributes. */
   void AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                             Array<int> &bdr_attr_marker);

   Array<LinearFormIntegrator*> *GetDLFI() { return &dlfi; }

   Array<LinearFormIntegrator*> *GetBLFI() { return &blfi; }

   Array<LinearFormIntegrator*> *GetFLFI() { return &flfi; }
   Array<Array<int>*> *GetFLFI_Marker() { return &flfi_marker; }

   /// Assembles the linear form i.e. sums over all domain/bdr integrators.
   void Assemble();

   void Update() { SetSize(fes->GetVSize()); }

   void Update(FiniteElementSpace *f) { fes = f; SetSize(f->GetVSize()); }

   void Update(FiniteElementSpace *f, Vector &v, int v_offset);

   /// Return the action of the LinearForm as a linear mapping.
   /** Linear forms are linear functionals which map GridFunctions to
       the real numbers.  This method performs this mapping which in
       this case is equivalent as an inner product of the LinearForm
       and GridFunction. */
   double operator()(const GridFunction &gf) const { return (*this)*gf; }

   /// Destroys linear form.
   ~LinearForm();
};

}

#endif
