Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
r[n_] := Module[{l = Floor[Sqrt[n]], m = n - Floor[Sqrt[n]]^2 - 
        Floor[Sqrt[n]], \[Mu], \[Nu]}, \[Mu] = l - m; \[Nu] = l + m; 
      Piecewise[{{(Gamma[\[Mu]/4 + 1/2]*Gamma[\[Nu]/4 + 1/2])/
          Gamma[(\[Mu] + \[Nu])/4 + 2], Mod[\[Mu]/2, 2] == Mod[\[Nu]/2, 2] == 
          0}, {(Sqrt[Pi]/2)*((Gamma[\[Mu]/4 + 1/4]*Gamma[\[Nu]/4 + 1/4])/
           Gamma[(\[Mu] + \[Nu])/4 + 2]), Mod[(\[Mu] - 1)/2, 2] == 
          Mod[(\[Nu] - 1)/2, 2] == 0}, {0, True}}]]
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {z -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
A1[lmax_] := Transpose[Flatten[Table[p[l, m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
p[l_, m_, lmax_] := Module[{Ylm}, Ylm = Y[l, m, x, y]; 
      Join[{Evaluate[Ylm /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[Ylm, bp[n, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {n, 1, (lmax + 1)^2 - 1}]]]
 
p[n_, lmax_] := Module[{g}, g = bg[n, x, y]; 
      Join[{Evaluate[g /. {Sqrt[1 - x^2 - y^2] -> 0, x -> 0, y -> 0}]}, 
       Table[Coefficient[g, bp[j, x, y]] /. {Sqrt[1 - x^2 - y^2] -> 0, 
          x -> 0, y -> 0}, {j, 1, (lmax + 1)^2 - 1}]]]
 
Y[l_, m_, x_, y_] := Expand[FullSimplify[Yxyz[l, m, x, y, 
       Sqrt[1 - x^2 - y^2]]]]
 
Yxyz[l_, m_, x_, y_, z_] := Piecewise[
     {{Sum[Sum[(-1)^(j/2)*A[l, m]*B[l, m, j, k]*x^(m - j)*y^j*z^k, 
         {k, 0, l - m}], {j, 0, m, 2}], m >= 0}, 
      {Sum[Sum[(-1)^((j - 1)/2)*A[l, Abs[m]]*B[l, Abs[m], j, k]*
          x^(Abs[m] - j)*y^j*z^k, {k, 0, l - Abs[m]}], {j, 1, Abs[m], 2}], 
       m < 0}}]
 
A[l_, m_] := Sqrt[((2 - KroneckerDelta[m, 0])*(2*l + 1)*(l - m)!)/
      (4*Pi*(l + m)!)]
 
A[lmax_] := A1[lmax] . A2[lmax]
 
A2[lmax_] := Inverse[A2Inv[lmax]]
 
A2Inv[lmax_] := Transpose[Flatten[Table[p[l^2 + l + m, lmax], {l, 0, lmax}, 
       {m, -l, l}], 1]]
 
B[l_, m_, j_, k_] := (2^l*m!*((l + m + k - 1)/2)!)/
     (j!*k!*(m - j)!*(l - m - k)!*((-l + m + k - 1)/2)!)
 
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
 
bg[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      Which[EvenQ[\[Nu]], ((\[Mu] + 2)/2)*x^(\[Mu]/2)*y^(\[Nu]/2), 
       \[Nu] == 1 && \[Mu] == 1, Sqrt[1 - x^2 - y^2], \[Mu] > 1, 
       Sqrt[1 - x^2 - y^2]*(((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] - 1)/2) - ((\[Mu] - 3)/2)*x^((\[Mu] - 5)/2)*
          y^((\[Nu] + 3)/2) - ((\[Mu] + 3)/2)*x^((\[Mu] - 1)/2)*
          y^((\[Nu] - 1)/2)), OddQ[l], Sqrt[1 - x^2 - y^2]*
        (-x^(l - 3) + x^(l - 1) + 4*x^(l - 3)*y^2), True, 
       3*x^(l - 2)*y*Sqrt[1 - x^2 - y^2]]]
bp[n_, x_, y_] := Module[{l, m, \[Mu], \[Nu]}, l = Floor[Sqrt[n]]; 
      m = n - l^2 - l; \[Mu] = l - m; \[Nu] = l + m; 
      If[EvenQ[\[Nu]], x^(\[Mu]/2)*y^(\[Nu]/2), x^((\[Mu] - 1)/2)*
        y^((\[Nu] - 1)/2)*Sqrt[1 - x^2 - y^2]]]
