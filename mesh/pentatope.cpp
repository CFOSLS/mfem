// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

// Implementation of class Pentatope


#include "mesh_headers.hpp"

namespace mfem
{


Pentatope::Pentatope(const int *ind, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute = attr;
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }

   transform = 0;
}

Pentatope::Pentatope(int ind1, int ind2, int ind3, int ind4, int ind5, int attr)
   : Element(Geometry::PENTATOPE)
{
   attribute  = attr;
   indices[0] = ind1;
   indices[1] = ind2;
   indices[2] = ind3;
   indices[3] = ind4;
   indices[4] = ind5;

   transform = 0;
}

void Pentatope::GetVertices(Array<int> &v) const
{
   v.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      v[i] = indices[i];
   }
}

void Pentatope::SetVertices(const int *ind)
{
   for (int i = 0; i < 5; i++)
   {
      indices[i] = ind[i];
   }
}

// static method, doesn't take into account the "swapped" parameter for now
void Pentatope::GetPointMatrix(unsigned transform, DenseMatrix &pm)
{
   double *a = &pm(0,0), *b = &pm(0,1), *c = &pm(0,2), *d = &pm(0,3), *e = &pm(0,4);

   // initialize to identity
   a[0] = 0.0, a[1] = 0.0, a[2] = 0.0; a[3] = 0.0;
   b[0] = 1.0, b[1] = 0.0, b[2] = 0.0; b[3] = 0.0;
   c[0] = 0.0, c[1] = 1.0, c[2] = 0.0; c[3] = 0.0;
   d[0] = 0.0, d[1] = 0.0, d[2] = 1.0; d[3] = 0.0;
   e[0] = 0.0, e[1] = 0.0, e[2] = 0.0; e[3] = 1.0;

   int chain[12], n = 0;
   bool swapped[12];
   while (transform)
   {
      chain[n++] = (transform & 31) - 1;
      swapped[n-1] = (transform / 32 == 1);
      transform >>= 6;
   }

   // The transformations and orientations here should match
   // Mesh::RedRefinementPentatope and Mesh::Bisection for pentatopes:

#define ASGN(a, b) (a[0] = b[0], a[1] = b[1], a[2] = b[2], a[3] = b[3])
#define SWAP(a, b) for (int i = 0; i < 4; i++) { std::swap(a[i], b[i]); }
#define AVG(a, b, c) for (int i = 0; i < 4; i++) { a[i] = (b[i]+c[i])*0.5; }

   double mid1[4], mid2[4], mid3[4], mid4[4], mid5[4];

   while (n)
   {
      switch (chain[--n])
      {
         case 0:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, d); AVG(mid2, a, e); AVG(mid3, b, e); AVG(mid4, c, e); AVG(mid5,d,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 1:
            if (swapped[n]) SWAP(a,b);
            AVG(b, a, b); AVG(c, a, c); AVG(d, a, d); AVG(e, a, e);
            if (swapped[n]) SWAP(a,b);
            break;
         case 2:
            if (swapped[n]) SWAP(a,b);
            AVG(a, b, a); AVG(c, b, c); AVG(d, b, d); AVG(e, b, e);
            if (swapped[n]) SWAP(a,b);
            break;
         case 3:
            if (swapped[n]) SWAP(a,b);
            AVG(a, c, a); AVG(b, c, b); AVG(d, c, d); AVG(e, c, e);
            if (swapped[n]) SWAP(a,b);
            break;
         case 4:
            if (swapped[n]) SWAP(a,b);
            AVG(a, d, a); AVG(b, d, b); AVG(c, d, c); AVG(e, d, e);
            if (swapped[n]) SWAP(a,b);
            break;
         case 5:
            if (swapped[n]) SWAP(a,b);
            AVG(a, e, a); AVG(b, e, b); AVG(c, e, c); AVG(d, e, d);
            if (swapped[n]) SWAP(a,b);
            break;
         case 6:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, b); AVG(mid2, a, c); AVG(mid3, b, c); AVG(mid4, b, d); AVG(mid5,b,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 7:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, b); AVG(mid2, a, c); AVG(mid3, a, d); AVG(mid4, b, d); AVG(mid5,b,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (swapped[n]) SWAP(a,b);
            break;
         case 8:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, b); AVG(mid2, a, c); AVG(mid3, a, d); AVG(mid4, a, e); AVG(mid5,b,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 9:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, c); AVG(mid2, b, c); AVG(mid3, b, d); AVG(mid4, c, d); AVG(mid5,c,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 10:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, c); AVG(mid2, b, c); AVG(mid3, b, d); AVG(mid4, b, e); AVG(mid5,c,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (swapped[n]) SWAP(a,b);
            break;
         case 11:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, c); AVG(mid2, a, d); AVG(mid3, b, d); AVG(mid4, c, d); AVG(mid5,c,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (swapped[n]) SWAP(a,b);
            break;
         case 12:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, c); AVG(mid2, a, d); AVG(mid3, b, d); AVG(mid4, b, e); AVG(mid5,c,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 13:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, c); AVG(mid2, a, d); AVG(mid3, a, e); AVG(mid4, b, e); AVG(mid5,c,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (swapped[n]) SWAP(a,b);
            break;
         case 14:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, d); AVG(mid2, b, d); AVG(mid3, c, d); AVG(mid4, c, e); AVG(mid5,d,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (!swapped[n]) SWAP(a,b);
            break;
         case 15:
            if (swapped[n]) SWAP(a,b);
            AVG(mid1, a, d); AVG(mid2, b, d); AVG(mid3, b, e); AVG(mid4, c, e); AVG(mid5,d,e);
            ASGN(a, mid1); ASGN(b,mid2); ASGN(c,mid3); ASGN(d,mid4); ASGN(e,mid5);
            if (swapped[n]) SWAP(a,b);
            break;
         default:
            MFEM_ABORT("Invalid transform.");
      }
   }

   /*
   DenseMatrix Volume(4,4);
   for (int i = 0; i < 4; i++)
   {
       for (int j = 0; j < 4; ++j)
       {
           Volume(i,j) = pm(i,j+1) - pm(i,j);
       }
   }
   if (Volume.Det() < 0)
   {
       std::cout << "negative \n";
   }
   else
   {
       //SWAP(a,b); // doesn't help
       std::cout << "positive \n";
   }
   */

   return;
}

Element *Pentatope::Duplicate(Mesh *m) const
{
   Pentatope *pent = new Pentatope;
   pent->SetVertices(indices);
   pent->SetAttribute(attribute);
   return pent;
}


void Pentatope::MarkEdge(const DSTable &v_to_v, const int *length)
{
   int ind[5], i, j, l, L; // (unused) type;

   // determine the longest edge
   L = length[v_to_v(indices[0], indices[1])]; j = 0;
   if ((l = length[v_to_v(indices[1], indices[2])]) > L) { L = l; j = 1; }
   if ((l = length[v_to_v(indices[2], indices[0])]) > L) { L = l; j = 2; }
   if ((l = length[v_to_v(indices[0], indices[3])]) > L) { L = l; j = 3; }
   if ((l = length[v_to_v(indices[1], indices[3])]) > L) { L = l; j = 4; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; j = 5; }
   if ((l = length[v_to_v(indices[0], indices[4])]) > L) { L = l; j = 6; }
   if ((l = length[v_to_v(indices[1], indices[4])]) > L) { L = l; j = 7; }
   if ((l = length[v_to_v(indices[2], indices[4])]) > L) { L = l; j = 8; }
   if ((l = length[v_to_v(indices[3], indices[4])]) > L) { L = l; j = 9; }

   for (i = 0; i < 5; i++)
   {
      ind[i] = indices[i];
   }

   // reordering vertices so that the longest edge will be from vertex 0 to vertex 1
   // preserving the orientation (even number of transpositions)
   switch (j)
   {
      case 1:
         indices[0] = ind[1]; indices[1] = ind[2];
         indices[2] = ind[0]; indices[3] = ind[3]; indices[4] = ind[4];
         break;
      case 2:
         indices[0] = ind[2]; indices[1] = ind[0];
         indices[2] = ind[1]; indices[3] = ind[3]; indices[4] = ind[4];
         break;
      case 3:
         indices[0] = ind[3]; indices[1] = ind[0];
         indices[2] = ind[2]; indices[3] = ind[1]; indices[4] = ind[4];
         break;
      case 4:
         indices[0] = ind[1]; indices[1] = ind[3];
         indices[2] = ind[2]; indices[3] = ind[0]; indices[4] = ind[4];
         break;
      case 5:
         indices[0] = ind[2]; indices[1] = ind[3];
         indices[2] = ind[0]; indices[3] = ind[1]; indices[4] = ind[4];
         break;
      case 6:
        indices[0] = ind[4]; indices[1] = ind[0];
        indices[2] = ind[2]; indices[3] = ind[3]; indices[4] = ind[1];
        break;
      case 7:
        indices[0] = ind[1]; indices[1] = ind[4];
        indices[2] = ind[2]; indices[3] = ind[3]; indices[4] = ind[0];
        break;
      case 8:
        indices[0] = ind[4]; indices[1] = ind[2];
        indices[2] = ind[1]; indices[3] = ind[3]; indices[4] = ind[0];
        break;
      case 9:
        indices[0] = ind[3]; indices[1] = ind[4];
        indices[2] = ind[1]; indices[3] = ind[2]; indices[4] = ind[0];
        break;
   }

   // There are three faces (tetrahedrons) which contain the longest edge.
   // Determine the two longest edges for the other two faces and
   // store them in ind[0] and ind[1]
   ind[0] = 2; ind[1] = 1;
   L = length[v_to_v(indices[0], indices[2])];
   if ((l = length[v_to_v(indices[0], indices[3])]) > L) { L = l; ind[0] = 3; }
   if ((l = length[v_to_v(indices[0], indices[4])]) > L) { L = l; ind[0] = 6; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; ind[0] = 5; }
   if ((l = length[v_to_v(indices[2], indices[4])]) > L) { L = l; ind[0] = 8; }
   if ((l = length[v_to_v(indices[3], indices[4])]) > L) { L = l; ind[0] = 9; }

   L = length[v_to_v(indices[1], indices[2])];
   if ((l = length[v_to_v(indices[1], indices[3])]) > L) { L = l; ind[1] = 4; }
   if ((l = length[v_to_v(indices[1], indices[4])]) > L) { L = l; ind[1] = 7; }
   if ((l = length[v_to_v(indices[2], indices[3])]) > L) { L = l; ind[0] = 5; }
   if ((l = length[v_to_v(indices[2], indices[4])]) > L) { L = l; ind[0] = 8; }
   if ((l = length[v_to_v(indices[3], indices[4])]) > L) { L = l; ind[1] = 9; }

   MFEM_ABORT("PENTATOPE:: MarkEdge not finished for Pentatope \n");
   /*
   j = 0;
   switch (ind[0])
   {
      case 2:
         switch (ind[1])
         {
            case 1:  type = Tetrahedron::TYPE_PU; break;
            case 4:  type = Tetrahedron::TYPE_PU; break;
            case 7:  type = Tetrahedron::TYPE_A;  break;
            case 5:  type = Tetrahedron::TYPE_M;  break;
            case 8:  type = Tetrahedron::TYPE_M;break;
            case 9:  type = Tetrahedron::TYPE_M;break;
            default:
                std::cout << "Error: cannot be here \n";
                break;
         }
         break;
      case 3:
         switch (ind[1])
         {
            case 1:  type = Tetrahedron::TYPE_A;  break;
            case 4:  type = Tetrahedron::TYPE_PU;
               j = 1; ind[0] = 2; ind[1] = 1; break;
            case 5:
            default: type = Tetrahedron::TYPE_M;
               j = 1; ind[0] = 5; ind[1] = 1;
         }
         break;
      case 6:
         switch (ind[1])
         {
            case 1:  type = Tetrahedron::TYPE_M;  break;
            case 4:  type = Tetrahedron::TYPE_M;
               j = 1; ind[0] = 2; ind[1] = 5; break;
            case 5:
            default: type = Tetrahedron::TYPE_O;
         }
   }

   if (j)
   {
      j = indices[0]; indices[0] = indices[1]; indices[1] = j;
      j = indices[2]; indices[2] = indices[3]; indices[3] = j;
   }

   CreateRefinementFlag(ind, type);
   */
}


Linear4DFiniteElement PentatopeFE;

}
