# Theory and Mathematical Background

##  Overview

This document provides a comprehensive mathematical foundation for the **Surface Semi-Geostrophic (SSG)** equations implemented in SSG.jl. The package solves sophisticated geophysical fluid dynamics problems with spectral accuracy and advanced numerical methods.

##  Table of Contents

1. [Surface Semi-Geostrophic Equations](#surface-semi-geostrophic-equations)
2. [3D Semi-Geostrophic Theory](#3d-semi-geostrophic-theory)
3. [Spectral Methods](#spectral-methods)
4. [Boundary Conditions](#boundary-conditions)
5. [Numerical Methods](#numerical-methods)
6. [References](#references)

---

## Surface Semi-Geostrophic Equations

### Governing Equations

The surface semi-geostrophic model describes the evolution of **surface buoyancy anomalies** under geostrophic constraint. The fundamental equation is:

```math
\frac{\partial b}{\partial t} + J(\psi, b) = 0
```

where:
- **b(x,y,t)**: surface buoyancy anomaly [m/s²]
- **ψ(x,y,t)**: geostrophic streamfunction [m²/s]  
- **J(ψ,b)**: Jacobian operator (advection term)

### Jacobian Operator

The Jacobian represents **advection of buoyancy** by the geostrophic flow:

```math
J(\psi, b) = \frac{\partial \psi}{\partial x} \frac{\partial b}{\partial y} - \frac{\partial \psi}{\partial y} \frac{\partial b}{\partial x}
```

This can also be written in **conservative form**:

```math
J(\psi, b) = \nabla \cdot (b \mathbf{u}^g) = \frac{\partial}{\partial x}(b u) + \frac{\partial}{\partial y}(b v)
```

### Geostrophic Velocities

The velocity field is **diagnostic** (determined by the streamfunction):

```math
\mathbf{u}^g = (u, v) = \left(-\frac{\partial \psi}{\partial y}, \frac{\partial \psi}{\partial x}\right)
```

This ensures the flow is **non-divergent**: ∇ · **u**^g = 0.

### Conservation Properties

The surface SSG equations conserve several important quantities:

1. **Buoyancy Conservation**: ∂b/∂t + J(ψ,b) = 0 implies material conservation of b
2. **Energy**: E = ½∫(|∇ψ|² + b²) dx dy 
3. **Enstrophy**: Ω = ½∫(∇²ψ)² dx dy
4. **Potential Vorticity**: q = ∇²ψ + b

---

## 3D Semi-Geostrophic Theory

### Core 3D SSG Equation

For the full three-dimensional problem, SSG.jl solves the **Semi-Geostrophic equation**:

```math
\nabla^2 \Phi = \varepsilon D\Phi \quad \text{(SSG Equation)}
```

where:
- **∇²**: 3D Laplacian operator = ∂²/∂X² + ∂²/∂Y² + ∂²/∂Z²
- **DΦ**: Nonlinear differential operator (Monge-Ampère type)
- **ε**: External parameter (measure of global Rossby number)
- **Φ(X,Y,Z)**: 3D geostrophic streamfunction in transformed coordinates

### Nonlinear Operator

The nonlinear operator **DΦ** is defined as:

```math
D\Phi = \frac{\partial^2\Phi}{\partial X^2} \frac{\partial^2\Phi}{\partial Y^2} - \left(\frac{\partial^2\Phi}{\partial X \partial Y}\right)^2
```

This is the **determinant of the horizontal Hessian**:

```math
D\Phi = \det\begin{pmatrix}
\frac{\partial^2\Phi}{\partial X^2} & \frac{\partial^2\Phi}{\partial X \partial Y} \\
\frac{\partial^2\Phi}{\partial X \partial Y} & \frac{\partial^2\Phi}{\partial Y^2}
\end{pmatrix}
```

### Coordinate Transformation

The SSG equations are solved in **geostrophic coordinates** (X,Y,Z) related to physical coordinates (x,y,z) through:

```math
\frac{DX}{Dt} = u^g, \quad \frac{DY}{Dt} = v^g, \quad \frac{DZ}{Dt} = 0
```

This transformation ensures the **geostrophic momentum approximation** is satisfied.

### Rossby Number Scaling

The parameter **ε** represents the ratio of nonlinear to linear terms:

```math
\varepsilon \sim \frac{U}{fL} = \text{Rossby Number}
```

where:
- **U**: Characteristic velocity scale
- **f**: Coriolis parameter  
- **L**: Characteristic length scale

### Surface Monge-Ampère Connection

For surface dynamics, the streamfunction and buoyancy are related through the **Monge-Ampère equation**:

```math
\det(D^2\phi) = b
```

where **D²φ** is the Hessian matrix of the surface streamfunction:

```math
D^2\phi = \begin{pmatrix}
\frac{\partial^2\phi}{\partial x^2} & \frac{\partial^2\phi}{\partial x \partial y} \\
\frac{\partial^2\phi}{\partial x \partial y} & \frac{\partial^2\phi}{\partial y^2}
\end{pmatrix}
```

---

## Spectral Methods

### Fourier Transform Approach

SSG.jl uses **spectral methods** for horizontal derivatives with exceptional accuracy. For a periodic function f(x,y), the Fourier representation is:

```math
f(x,y) = \sum_{k_x} \sum_{k_y} \hat{f}(k_x, k_y) e^{i(k_x x + k_y y)}
```

### Spectral Derivatives

Derivatives become **multiplications in spectral space**:

```math
\frac{\partial f}{\partial x} \leftrightarrow ik_x \hat{f}(k_x, k_y)
```

```math
\frac{\partial f}{\partial y} \leftrightarrow ik_y \hat{f}(k_x, k_y)
```

```math
\nabla^2_h f \leftrightarrow -(k_x^2 + k_y^2) \hat{f}(k_x, k_y)
```

### Real FFTs

For real fields, SSG.jl uses **real-to-complex FFTs** which exploit Hermitian symmetry:

```math
\hat{f}(-k_x, -k_y) = \hat{f}^*(k_x, k_y)
```

This reduces storage by ~50% and computational cost.

### Dealiasing

**Two-thirds rule** prevents aliasing errors in nonlinear terms:

- Compute: Nₓ × Nᵧ grid points
- Transform: (3Nₓ/2) × (3Nᵧ/2) modes  
- Keep: Nₓ × Nᵧ modes after dealiasing

Wavenumbers satisfying |kₓ| ≤ Nₓ/3 and |kᵧ| ≤ Nᵧ/3 are retained.

### Parseval's Theorem

Energy computations use **Parseval's theorem**:

```math
\int f^2 \, dx \, dy = \sum_{k_x, k_y} |\hat{f}(k_x, k_y)|^2
```

This enables **spectrally exact** energy and enstrophy calculations.

---

## Boundary Conditions

### Vertical Boundary Conditions

The 3D SSG equation uses **sophisticated boundary conditions** at top and bottom:

#### Surface (Z = 0)
```math
\frac{\partial \Phi}{\partial Z} = \tilde{b}_s(X,Y)
```

where **b̃ₛ** is the **surface buoyancy anomaly** in geostrophic coordinates.

#### Bottom (Z = -H)  
```math
\frac{\partial \Phi}{\partial Z} = 0
```

This represents a **rigid bottom boundary** with no vertical flow.

### Numerical Implementation

Boundary conditions are implemented using **higher-order finite differences**:

#### Surface Boundary (Second-Order Backward)
For non-uniform vertical grid with spacings h₁, h₂:

```math
\frac{\partial \Phi}{\partial Z}\bigg|_{\text{surface}} = \frac{a \Phi_k + b \Phi_{k-1} + c \Phi_{k-2}}{h}
```

where coefficients are:
```math
a = \frac{2h_1 + h_2}{h_1(h_1 + h_2)}, \quad 
b = -\frac{h_1 + h_2}{h_1 h_2}, \quad 
c = \frac{h_1}{h_2(h_1 + h_2)}
```

#### Bottom Boundary (Second-Order Forward)
```math
\frac{\partial \Phi}{\partial Z}\bigg|_{\text{bottom}} = \frac{a' \Phi_1 + b' \Phi_2 + c' \Phi_3}{h'} = 0
```

### Horizontal Periodicity

Horizontal directions are **periodic**:
- **u(0,y,z) = u(Lₓ,y,z)**
- **v(x,0,z) = v(x,Lᵧ,z)**

This is naturally satisfied by the Fourier basis functions.

---

## Numerical Methods

### Time Integration Schemes

#### Adams-Bashforth 2 (Low Storage)
```math
b^{n+1} = b^n + \Delta t \left[\frac{3}{2}f^n - \frac{1}{2}f^{n-1}\right]
```

where **f = -J(ψ,b)** is the tendency term.

#### Runge-Kutta 3 (Classical)
```math
\begin{align}
k_1 &= f(t_n, b_n) \\
k_2 &= f(t_n + \Delta t/3, b_n + \Delta t k_1/3) \\
k_3 &= f(t_n + 2\Delta t/3, b_n + 2\Delta t k_2/3) \\
b^{n+1} &= b_n + \frac{\Delta t}{4}(k_1 + 3k_3)
\end{align}
```

### Multigrid Methods

For the 3D SSG equation, SSG.jl implements **geometric multigrid**:

#### V-Cycle Algorithm
1. **Pre-smoothing**: Apply relaxation on fine grid
2. **Restriction**: Transfer residual to coarse grid  
3. **Coarse solve**: Recursive V-cycle or direct solve
4. **Prolongation**: Interpolate correction to fine grid
5. **Post-smoothing**: Final relaxation on fine grid

#### Smoothing Operators

**Spectral Smoother**: Uses spectral preconditioning
```math
\hat{\Phi}^{new} = \hat{\Phi}^{old} + \frac{\omega \hat{r}}{k^2 + \varepsilon k^4 + \delta}
```

**SOR Smoother**: Successive over-relaxation with linearization
```math
\Phi_{i,j,k}^{new} = \Phi_{i,j,k}^{old} + \omega \frac{r_{i,j,k}}{-2(\Delta x^{-2} + \Delta y^{-2}) + \beta_k}
```

### Adaptive Time Stepping

CFL-based time step control:

```math
\Delta t^{new} = \min\left(\Delta t_{\max}, \Delta t^{old} \frac{\text{CFL}_{\text{target}}}{\text{CFL}_{\text{current}}}\right)
```

where:
```math
\text{CFL} = \max\left(\frac{|u|\Delta t}{\Delta x}, \frac{|v|\Delta t}{\Delta y}\right)
```

### Numerical Methods

7. **Boyd, J. P.** (2001). *Chebyshev and Fourier spectral methods*. Dover Publications.

8. **Briggs, W. L., Henson, V. E., & McCormick, S. F.** (2000). *A multigrid tutorial*. SIAM.

9. **Canuto, C., Hussaini, M. Y., Quarteroni, A., & Zang, T. A.** (2007). *Spectral methods: fundamentals in single domains*. Springer.

### Geophysical Fluid Dynamics

10. **Vallis, G. K.** (2017). *Atmospheric and Oceanic Fluid Dynamics: Fundamentals and Large-scale Circulation*. Cambridge University Press.

11. **Pedlosky, J.** (1987). *Geophysical fluid dynamics*. Springer-Verlag.

12. **Salmon, R.** (1998). *Lectures on geophysical fluid dynamics*. Oxford University Press.

---
