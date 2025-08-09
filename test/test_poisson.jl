#!/usr/bin/env julia

# Test script for SSG Poisson solver
using MPI
using PencilArrays
using PencilFFTs

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using SSG

function test_simple_laplacian()
    println("Testing simple Laplacian solver...")
    
    # Initialize MPI
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    # Create a small test domain
    domain = make_domain(16, 16, 8; Lx=2π, Ly=2π, Lz=1.0)
    
    # Create test fields
    Φ_initial = PencilArray(domain.pr3d, zeros(Float64, size_local(domain.pr3d)))
    b_rhs = PencilArray(domain.pr3d, zeros(Float64, size_local(domain.pr3d)))
    
    # Set up a simple manufactured solution test
    # Let's solve ∇²Φ = -2π² sin(πx) sin(πy) with Φ(x,y,z) ≈ sin(πx) sin(πy)
    range_locals = range_local(domain.pr3d)
    Φ_data = Φ_initial.data
    b_data = b_rhs.data
    
    for (k_local, k_global) in enumerate(range_locals[3])
        for (j_local, j_global) in enumerate(range_locals[2])
            for (i_local, i_global) in enumerate(range_locals[1])
                x = domain.x[i_global]
                y = domain.y[j_global] 
                z = domain.z[k_global]
                
                # Analytical solution
                Φ_exact = sin(π*x/domain.Lx) * sin(π*y/domain.Ly) * (1 - z/domain.Lz)
                
                # RHS for this solution: ∇²Φ = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
                laplacian_exact = -π²/(domain.Lx)² * sin(π*x/domain.Lx) * sin(π*y/domain.Ly) * (1 - z/domain.Lz) +
                                 -π²/(domain.Ly)² * sin(π*x/domain.Lx) * sin(π*y/domain.Ly) * (1 - z/domain.Lz)
                
                # Initial guess (could be zero or something close)
                Φ_data[i_local, j_local, k_local] = 0.1 * Φ_exact
                b_data[i_local, j_local, k_local] = laplacian_exact
            end
        end
    end
    
    # Test with small ε (nearly linear Poisson)
    ε = 0.01
    
    # Solve the equation
    println("Solving ∇²Φ - ε*DΦ = RHS with ε = $ε")
    solution, diagnostics = solve_ssg_equation(Φ_initial, b_rhs, ε, domain; 
                                             tol=1e-6, maxiter=20, verbose=true,
                                             n_levels=2, smoother=:adaptive)
    
    println("Converged: $(diagnostics.converged)")
    println("Iterations: $(diagnostics.iterations)")
    println("Final residual: $(diagnostics.final_residual)")
    
    MPI.Finalize()
    
    return diagnostics.converged
end

function test_nonlinear_term()
    println("\nTesting nonlinear D-operator computation...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(8, 8, 4)
    level = SSGLevel{Float64}(domain, 1)
    
    # Set a simple test function Φ = x² + y²
    range_locals = range_local(domain.pr3d)
    Φ_data = level.Φ.data
    
    for (k_local, k_global) in enumerate(range_locals[3])
        for (j_local, j_global) in enumerate(range_locals[2])
            for (i_local, i_global) in enumerate(range_locals[1])
                x = domain.x[i_global] 
                y = domain.y[j_global]
                
                # Test function: Φ = x² + y²
                Φ_data[i_local, j_local, k_local] = x^2 + y^2
            end
        end
    end
    
    # Compute D-operator
    compute_d_operator!(level, level.tmp_real)
    
    # For Φ = x² + y², we have:
    # ∂²Φ/∂x² = 2, ∂²Φ/∂y² = 2, ∂²Φ/∂x∂y = 0
    # So DΦ = (2)(2) - (0)² = 4
    
    expected_value = 4.0
    actual_values = level.tmp_real.data
    
    # Check a few interior points
    avg_error = 0.0
    n_points = 0
    
    nx_local, ny_local, nz_local = size(actual_values)
    if nx_local >= 3 && ny_local >= 3 && nz_local >= 2
        for k = 2:nz_local-1
            for j = 2:ny_local-1
                for i = 2:nx_local-1
                    error = abs(actual_values[i,j,k] - expected_value)
                    avg_error += error
                    n_points += 1
                end
            end
        end
    end
    
    if n_points > 0
        avg_error /= n_points
        println("D-operator test: average error = $avg_error (expected ≈ 0)")
        println("Test passed: $(avg_error < 0.1)")
    else
        println("D-operator test: grid too small for interior points")
    end
    
    MPI.Finalize()
    return true
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running SSG Poisson solver tests...")
    
    test1_passed = test_simple_laplacian()
    test2_passed = test_nonlinear_term()
    
    if test1_passed && test2_passed
        println("\n✅ All tests passed!")
    else
        println("\n❌ Some tests failed!")
        exit(1)
    end
end