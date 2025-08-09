#!/usr/bin/env julia

# Test script for nonlinear terms in surface SSG equations
using MPI
using PencilArrays
using PencilFFTs
using Printf

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))

using SSG

function test_jacobian_accuracy()
    println("Testing Jacobian computation accuracy...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    # Create test domain
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π, Lz=1.0)
    fields = allocate_fields(domain)
    
    # Set up test functions with known analytical Jacobian
    # Let ψ = sin(x)cos(y), b = cos(x)sin(y)
    # Then J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x
    #             = cos(x)cos(y) * cos(x)cos(y) - (-sin(x)sin(y)) * sin(x)cos(y)
    #             = cos²(x)cos²(y) + sin²(x)sin(y)cos(y)
    
    range_locals = range_local(fields.φₛ.pencil)
    φ_data = fields.φₛ.data
    b_data = fields.bₛ.data
    
    # Set up test functions
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            φ_data[i_local, j_local] = sin(x) * cos(y)  # ψ
            b_data[i_local, j_local] = cos(x) * sin(y)  # b
        end
    end
    
    # Compute Jacobian numerically
    compute_jacobian!(fields.tmp, fields.φₛ, fields.bₛ, fields, domain)
    
    # Compute analytical Jacobian for comparison
    jacobian_analytical = zeros(size(fields.tmp.data))
    tmp_data = fields.tmp.data
    
    max_error = 0.0
    avg_error = 0.0
    n_points = 0
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Analytical result
            analytical = cos(x)^2 * cos(y)^2 + sin(x)^2 * sin(y) * cos(y)
            numerical = tmp_data[i_local, j_local]
            
            error = abs(numerical - analytical)
            max_error = max(max_error, error)
            avg_error += error
            n_points += 1
        end
    end
    
    if n_points > 0
        avg_error /= n_points
    end
    
    println("  Maximum error: $(round(max_error, digits=8))")
    println("  Average error: $(round(avg_error, digits=8))")
    
    # Should achieve spectral accuracy
    success = max_error < 1e-12
    
    MPI.Finalize()
    return success
end

function test_geostrophic_velocity_consistency()
    println("\nTesting geostrophic velocity computation consistency...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(16, 16, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Set up test streamfunction: ψ = sin(2x)cos(y)
    # Then u = -∂ψ/∂y = sin(2x)sin(y)
    #      v = ∂ψ/∂x  = 2cos(2x)cos(y)
    
    range_locals = range_local(fields.φₛ.pencil)
    φ_data = fields.φₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            φ_data[i_local, j_local] = sin(2*x) * cos(y)
        end
    end
    
    # Compute velocities
    compute_geostrophic_velocities!(fields, domain)
    
    # Check against analytical solution
    u_data = fields.u.data
    v_data = fields.v.data
    
    max_error_u = 0.0
    max_error_v = 0.0
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Analytical velocities
            u_analytical = sin(2*x) * sin(y)   # -∂ψ/∂y
            v_analytical = 2*cos(2*x) * cos(y) # ∂ψ/∂x
            
            # Check surface level (k=1) and verify all levels are the same
            for k = 1:size(u_data, 3)
                error_u = abs(u_data[i_local, j_local, k] - u_analytical)
                error_v = abs(v_data[i_local, j_local, k] - v_analytical)
                
                max_error_u = max(max_error_u, error_u)
                max_error_v = max(max_error_v, error_v)
            end
        end
    end
    
    println("  Max error in u: $(round(max_error_u, digits=8))")
    println("  Max error in v: $(round(max_error_v, digits=8))")
    
    # Should achieve spectral accuracy
    success = max_error_u < 1e-12 && max_error_v < 1e-12
    
    MPI.Finalize()
    return success
end

function test_tendency_computation()
    println("\nTesting complete tendency computation...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(16, 16, 4)
    fields = allocate_fields(domain)
    params = TimeParams{Float64}(0.01)
    
    # Set up test fields
    range_locals = range_local(fields.φₛ.pencil)
    φ_data = fields.φₛ.data
    b_data = fields.bₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Simple test case
            φ_data[i_local, j_local] = 0.1 * sin(x) * cos(y)
            b_data[i_local, j_local] = 0.1 * cos(x) * sin(y)
        end
    end
    
    # Compute tendency
    compute_tendency!(fields.tmp, fields, domain, params)
    
    # Check that tendency is finite and reasonable
    tendency_data = fields.tmp.data
    max_tendency = maximum(abs, tendency_data)
    is_finite = all(isfinite, tendency_data)
    
    println("  Maximum tendency magnitude: $(round(max_tendency, digits=6))")
    println("  All values finite: $is_finite")
    
    # For this test case, tendency should be finite and not too large
    success = is_finite && max_tendency < 1.0 && max_tendency > 1e-10
    
    MPI.Finalize()
    return success
end

function test_conservation_properties()
    println("\nTesting conservation properties...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    params = TimeParams{Float64}(0.001)
    
    # Set up smooth initial conditions
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    φ_data = fields.φₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            b_data[i_local, j_local] = exp(-(x-π)^2/2) * exp(-(y-π)^2/2)
            φ_data[i_local, j_local] = 0.1 * sin(x) * cos(y)
        end
    end
    
    # Compute initial total buoyancy
    initial_buoyancy = sum(b_data)
    initial_buoyancy_global = MPI.Allreduce(initial_buoyancy, MPI.SUM, comm)
    
    # Take several time steps
    state = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
    
    for step = 1:10
        timestep_rk3!(fields, domain, params, state)
    end
    
    # Check final total buoyancy
    final_buoyancy = sum(fields.bₛ.data)
    final_buoyancy_global = MPI.Allreduce(final_buoyancy, MPI.SUM, comm)
    
    # Calculate conservation error
    conservation_error = abs(final_buoyancy_global - initial_buoyancy_global) / abs(initial_buoyancy_global)
    
    println("  Initial total buoyancy: $(round(initial_buoyancy_global, digits=6))")
    println("  Final total buoyancy: $(round(final_buoyancy_global, digits=6))")
    println("  Relative conservation error: $(round(conservation_error*100, digits=6))%")
    
    # Should conserve to machine precision (with proper dealiasing)
    success = conservation_error < 1e-12
    
    MPI.Finalize()
    return success
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running nonlinear SSG equation tests...")
    
    test1_passed = test_jacobian_accuracy()
    test2_passed = test_geostrophic_velocity_consistency()
    test3_passed = test_tendency_computation()
    test4_passed = test_conservation_properties()
    
    println("\n" * "="^50)
    println("NONLINEAR SSG TEST RESULTS:")
    println("✓ Jacobian Accuracy: $(test1_passed ? "PASSED" : "FAILED")")
    println("✓ Velocity Consistency: $(test2_passed ? "PASSED" : "FAILED")")
    println("✓ Tendency Computation: $(test3_passed ? "PASSED" : "FAILED")")
    println("✓ Conservation Properties: $(test4_passed ? "PASSED" : "FAILED")")
    
    if test1_passed && test2_passed && test3_passed && test4_passed
        println("\n🎉 All nonlinear SSG tests passed!")
    else
        println("\n❌ Some nonlinear SSG tests failed!")
        exit(1)
    end
end