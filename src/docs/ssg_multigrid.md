## SSG EQUATION (1) IMPLEMENTATION SUMMARY:

### Core Equation Solved:
∇²Φ = εDΦ                                 (1)

where:
- ∇² = ∂²/∂X² + ∂²/∂Y² + ∂²/∂Z²            (3D Laplacian)
- DΦ = ∂²Φ/∂X²∂Y² - (∂²Φ/∂X∂Y)²            (2, nonlinear operator)
- ε: external parameter (Rossby number measure)

### Boundary Conditions (A4):
- ∂Φ/∂Z = b̃s  at Z = 0 (surface)
- ∂Φ/∂Z = 0   at Z = -1 (bottom)

### Key Features:
1. **Spectral accuracy** in horizontal directions (X,Y)
2. **Finite differences** in vertical direction (Z) with non-uniform grid support
3. **Multigrid solver** for fast convergence
4. **Compatible interface** with existing Fields structure
5. **Boundary condition handling** per equation (A4)
6. **Linearized smoothers** for nonlinear operator DΦ

### Main Functions:
- `solve_ssg_equation()`: Direct 3D SSG solver
- `solve_monge_ampere_fields!()`: Interface compatible with existing code
- `compute_ma_residual_fields!()`: Residual computation for time stepping
- `demo_ssg_solver()`: Comprehensive testing and validation

### Integration:
- Drop-in replacement for existing Monge-Ampère solver
- Maintains all existing interfaces in nonlinear.jl and timestep.jl
- Supports adaptive time stepping and spectral filtering
- Compatible with MPI parallel execution via PencilArrays

### Technical Implementation:
- **Equation (1)**: Implemented in `compute_ssg_residual!()`
- **Operator DΦ (2)**: Implemented in `compute_d_operator!()`
- **Boundary conditions (A4)**: Implemented in `apply_ssg_boundary_conditions!()`
- **Multigrid**: V-cycles with spectral and SOR smoothers
- **Compatibility**: 2D ↔ 3D field conversion for existing codebase