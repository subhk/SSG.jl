#!/usr/bin/env julia

# Test script for time integration in SSG
using MPI
using PencilArrays
using PencilFFTs

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using SSG

function test_rk3_coefficients()
    println("Testing RK3 time integration coefficients...")
    
    # Test against known analytical solution
    # dy/dt = -k*y, y(0) = 1 => y(t) = exp(-k*t)
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    # Create simple test domain
    domain = make_domain(8, 8, 4; Lx=2π, Ly=2π, Lz=1.0)
    fields = allocate_fields(domain)
    
    # Initialize with simple exponential decay test
    k_decay = 1.0  # decay constant
    dt = 0.01
    t_final = 0.1
    
    # Set initial condition: b = exp(0) = 1 everywhere
    fields.bₛ.data .= 1.0
    
    # Create time integration parameters
    params = TimeParams{Float64}(dt; scheme=RK3, adaptive_dt=false)
    state = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
    
    # Mock tendency function for exponential decay: db/dt = -k*b
    function mock_tendency!(db_dt, fields, domain, params)
        db_dt.data .= -k_decay .* fields.bₛ.data
        return db_dt
    end
    
    # Replace the tendency computation temporarily
    original_compute_tendency! = SSG.compute_tendency!
    SSG.compute_tendency! = mock_tendency!
    
    try
        # Integrate forward in time
        n_steps = Int(t_final / dt)
        errors = Float64[]
        
        for step = 1:n_steps
            timestep_rk3!(fields, domain, params, state)
            
            # Compare with analytical solution
            t_current = state.t
            analytical = exp(-k_decay * t_current)
            numerical = fields.bₛ.data[1,1]  # Check first point
            
            error = abs(numerical - analytical) / analytical
            push!(errors, error)
            
            if step % 5 == 0
                println("  Step $step: t=$(round(t_current, digits=3)), analytical=$(round(analytical, digits=6)), numerical=$(round(numerical, digits=6)), error=$(round(error*100, digits=4))%")
            end
        end
        
        final_error = errors[end]
        println("Final relative error: $(round(final_error*100, digits=4))%")
        
        # Check if RK3 is achieving expected order of accuracy
        success = final_error < 1e-4  # Should be very accurate for this simple problem
        
    finally
        # Restore original function
        SSG.compute_tendency! = original_compute_tendency!
    end
    
    MPI.Finalize()
    return success
end

function test_adams_bashforth_stability()
    println("\nTesting Adams-Bashforth stability...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(16, 16, 4)
    fields = allocate_fields(domain)
    
    # Initialize with small random perturbations
    fields.bₛ.data .= 0.01 .* randn(size(fields.bₛ.data))
    
    params = TimeParams{Float64}(0.001; scheme=AB2_LowStorage, adaptive_dt=false)
    state = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
    
    # Mock tendency for damping: db/dt = -0.1*b
    function damping_tendency!(db_dt, fields, domain, params)
        db_dt.data .= -0.1 .* fields.bₛ.data
        return db_dt
    end
    
    original_compute_tendency! = SSG.compute_tendency!
    SSG.compute_tendency! = damping_tendency!
    
    try
        initial_norm = sqrt(sum(abs2, fields.bₛ.data))
        
        # Run for many time steps
        for step = 1:500
            timestep_ab2_ls!(fields, domain, params, state)
        end
        
        final_norm = sqrt(sum(abs2, fields.bₛ.data))
        
        println("  Initial norm: $(round(initial_norm, digits=6))")
        println("  Final norm: $(round(final_norm, digits=6))")
        println("  Decay ratio: $(round(final_norm/initial_norm, digits=6))")
        
        # Should decay exponentially - check if stable
        success = final_norm < initial_norm && final_norm > 0 && isfinite(final_norm)
        
    finally
        SSG.compute_tendency! = original_compute_tendency!
    end
    
    MPI.Finalize()
    return success
end

function test_cfl_calculation()
    println("\nTesting CFL number calculation...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Set up simple velocity field: u = 1, v = 0
    fields.u.data .= 1.0
    fields.v.data .= 0.0
    
    dt = 0.01
    
    # Calculate CFL
    cfl = compute_cfl_number(fields, domain, dt)
    
    # Expected CFL = u_max * dt / dx = 1.0 * 0.01 / (2π/32) ≈ 0.051
    dx = domain.Lx / domain.Nx
    expected_cfl = 1.0 * dt / dx
    
    println("  Grid spacing dx: $(round(dx, digits=4))")
    println("  Time step dt: $dt")
    println("  Max velocity: 1.0")
    println("  Expected CFL: $(round(expected_cfl, digits=4))")
    println("  Computed CFL: $(round(cfl, digits=4))")
    println("  Relative error: $(round(abs(cfl - expected_cfl)/expected_cfl * 100, digits=2))%")
    
    success = abs(cfl - expected_cfl) / expected_cfl < 0.01
    
    MPI.Finalize()
    return success
end

function test_spectral_filtering()
    println("\nTesting spectral filtering...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Create high-frequency noise
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Mix low and high frequency components
            b_data[i_local, j_local] = sin(x) + 0.5 * sin(15*x) * cos(15*y)  # High freq noise
        end
    end
    
    # Calculate initial energy spectrum (approximate)
    initial_max = maximum(abs, b_data)
    initial_mean = sum(abs, b_data) / length(b_data)
    
    # Apply filter
    filter_strength = 1.0
    apply_spectral_filter!(fields, domain, filter_strength)
    
    # Check result
    final_max = maximum(abs, b_data)
    final_mean = sum(abs, b_data) / length(b_data)
    
    println("  Before filtering - Max: $(round(initial_max, digits=4)), Mean: $(round(initial_mean, digits=4))")
    println("  After filtering - Max: $(round(final_max, digits=4)), Mean: $(round(final_mean, digits=4))")
    println("  Reduction ratio: $(round(final_mean/initial_mean, digits=3))")
    
    # Filter should reduce high-frequency content
    success = final_mean < initial_mean && final_max <= initial_max
    
    MPI.Finalize()
    return success
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running time integration tests...")
    
    test1_passed = test_rk3_coefficients()
    test2_passed = test_adams_bashforth_stability()
    test3_passed = test_cfl_calculation()
    test4_passed = test_spectral_filtering()
    
    println("\n" * "="^50)
    println("TEST RESULTS:")
    println("✓ RK3 Integration: $(test1_passed ? "PASSED" : "FAILED")")
    println("✓ AB2 Stability: $(test2_passed ? "PASSED" : "FAILED")")
    println("✓ CFL Calculation: $(test3_passed ? "PASSED" : "FAILED")")
    println("✓ Spectral Filtering: $(test4_passed ? "PASSED" : "FAILED")")
    
    if test1_passed && test2_passed && test3_passed && test4_passed
        println("\n🎉 All time integration tests passed!")
    else
        println("\n❌ Some tests failed!")
        exit(1)
    end
end