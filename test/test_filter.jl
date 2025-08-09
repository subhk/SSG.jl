#!/usr/bin/env julia

# Test script for spectral filtering in SSG
using MPI
using PencilArrays
using PencilFFTs
using Printf

# Add the source directory to the load path
push!(LOAD_PATH, joinpath(dirname(@__DIR__), "src"))

using SSG

function test_filter_types()
    println("Testing filter type construction...")
    
    T = Float64
    
    # Test ExponentialFilter
    exp_filter = ExponentialFilter{T}(1.0, 4, 0.65)
    @assert exp_filter.strength == 1.0
    @assert exp_filter.order == 4
    @assert exp_filter.cutoff == 0.65
    
    # Test HyperviscosityFilter
    hyper_filter = HyperviscosityFilter{T}(1e-4, 2)
    @assert hyper_filter.coefficient == 1e-4
    @assert hyper_filter.order == 2
    
    # Test CutoffFilter
    cutoff_filter = CutoffFilter{T}(2/3)
    @assert cutoff_filter.cutoff ≈ 2/3
    
    # Test CesaroFilter
    cesaro_filter = CesaroFilter{T}(0.5, 2)
    @assert cesaro_filter.cutoff == 0.5
    @assert cesaro_filter.order == 2
    
    println("  ✓ All filter types constructed correctly")
    return true
end

function test_transfer_functions()
    println("\nTesting transfer function implementations...")
    
    T = Float64
    
    # Test exponential transfer
    exp_filter = ExponentialFilter{T}(1.0, 2, 0.5)
    
    # Below cutoff should be 1.0
    @assert exponential_transfer(0.3, exp_filter) ≈ 1.0
    
    # Above cutoff should decay
    transfer_high = exponential_transfer(0.8, exp_filter)
    @assert 0.0 < transfer_high < 1.0
    
    # Test hyperviscosity transfer
    hyper_filter = HyperviscosityFilter{T}(1e-3, 1)
    dt = 0.01
    k_mag = 2.0
    
    transfer_hyper = hyperviscosity_transfer(k_mag, dt, hyper_filter)
    expected = exp(-1e-3 * k_mag^2 * dt)
    @assert abs(transfer_hyper - expected) < 1e-14
    
    # Test cutoff transfer
    cutoff_filter = CutoffFilter{T}(0.6)
    @assert cutoff_transfer(0.4, cutoff_filter) ≈ 1.0
    @assert cutoff_transfer(0.8, cutoff_filter) ≈ 0.0
    
    println("  ✓ All transfer functions working correctly")
    return true
end

function test_2d_filtering()
    println("\nTesting 2D field filtering...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    # Create test domain
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Create test field with high-frequency components
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Low frequency + high frequency noise
            b_data[i_local, j_local] = sin(x) * cos(y) + 0.5 * sin(15*x) * cos(15*y)
        end
    end
    
    # Store initial state
    b_initial = copy(b_data)
    initial_max = maximum(abs, b_data)
    
    # Test exponential filter
    exp_filter = ExponentialFilter{Float64}(1.0, 4, 0.65)
    apply_filter_to_field!(fields.bₛ, fields.bhat, domain, exp_filter)
    
    final_max = maximum(abs, b_data)
    reduction_ratio = final_max / initial_max
    
    println("  Initial max amplitude: $(round(initial_max, digits=4))")
    println("  Final max amplitude: $(round(final_max, digits=4))")
    println("  Reduction ratio: $(round(reduction_ratio, digits=3))")
    
    # Should reduce high-frequency content
    success = reduction_ratio < 0.9 && reduction_ratio > 0.3
    
    if success
        println("  ✓ 2D exponential filter working correctly")
    end
    
    MPI.Finalize()
    return success
end

function test_cutoff_filter_accuracy()
    println("\nTesting cutoff filter accuracy...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(64, 64, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Create field with specific wavenumber content
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    
    # Low wavenumber mode: k = (1, 1)
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            b_data[i_local, j_local] = cos(x) * sin(y)  # Should pass 2/3 filter
        end
    end
    
    # Store initial energy
    initial_energy = sum(abs2, b_data)
    initial_energy_global = MPI.Allreduce(initial_energy, MPI.SUM, comm)
    
    # Apply 2/3 cutoff filter
    cutoff_filter = CutoffFilter{Float64}(2/3)
    apply_filter_to_field!(fields.bₛ, fields.bhat, domain, cutoff_filter)
    
    # Check final energy
    final_energy = sum(abs2, b_data)
    final_energy_global = MPI.Allreduce(final_energy, MPI.SUM, comm)
    
    energy_ratio = final_energy_global / initial_energy_global
    
    println("  Energy ratio after 2/3 filter: $(round(energy_ratio, digits=6))")
    
    # Low wavenumber should pass through almost unchanged
    success = abs(energy_ratio - 1.0) < 0.01
    
    if success
        println("  ✓ Cutoff filter preserves low wavenumbers correctly")
    end
    
    MPI.Finalize()
    return success
end

function test_hyperviscosity_damping()
    println("\nTesting hyperviscosity filter damping...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Create high-wavenumber field
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            # High wavenumber mode
            b_data[i_local, j_local] = sin(10*x) * cos(10*y)
        end
    end
    
    initial_max = maximum(abs, b_data)
    
    # Apply hyperviscosity filter
    dt = 0.01
    hyper_filter = HyperviscosityFilter{Float64}(1e-2, 2)  # Strong damping
    apply_filter_to_field!(fields.bₛ, fields.bhat, domain, hyper_filter; dt=dt)
    
    final_max = maximum(abs, b_data)
    damping_ratio = final_max / initial_max
    
    println("  Initial amplitude: $(round(initial_max, digits=4))")
    println("  Final amplitude: $(round(final_max, digits=4))")
    println("  Damping ratio: $(round(damping_ratio, digits=4))")
    
    # Should provide significant damping for high wavenumbers
    success = damping_ratio < 0.5
    
    if success
        println("  ✓ Hyperviscosity filter provides proper damping")
    end
    
    MPI.Finalize()
    return success
end

function test_filter_conservation()
    println("\nTesting filter conservation properties...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4; Lx=2π, Ly=2π)
    fields = allocate_fields(domain)
    
    # Create constant field (should be conserved by linear filters)
    fields.bₛ.data .= 1.0
    
    initial_sum = sum(fields.bₛ.data)
    initial_sum_global = MPI.Allreduce(initial_sum, MPI.SUM, comm)
    
    # Apply exponential filter
    exp_filter = ExponentialFilter{Float64}(0.5, 4, 0.6)
    apply_filter_to_field!(fields.bₛ, fields.bhat, domain, exp_filter)
    
    final_sum = sum(fields.bₛ.data)
    final_sum_global = MPI.Allreduce(final_sum, MPI.SUM, comm)
    
    conservation_error = abs(final_sum_global - initial_sum_global) / abs(initial_sum_global)
    
    println("  Initial sum: $(round(initial_sum_global, digits=6))")
    println("  Final sum: $(round(final_sum_global, digits=6))")
    println("  Conservation error: $(round(conservation_error*100, digits=8))%")
    
    # DC component should be perfectly conserved
    success = conservation_error < 1e-14
    
    if success
        println("  ✓ Filter conserves DC component to machine precision")
    end
    
    MPI.Finalize()
    return success
end

function test_filter_stability()
    println("\nTesting filter stability with time integration...")
    
    MPI.Init()
    comm = MPI.COMM_WORLD
    
    domain = make_domain(32, 32, 4)
    fields = allocate_fields(domain)
    
    # Initialize with smooth field
    range_locals = range_local(fields.bₛ.pencil)
    b_data = fields.bₛ.data
    φ_data = fields.φₛ.data
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            b_data[i_local, j_local] = exp(-(x-π)^2) * exp(-(y-π)^2)
            φ_data[i_local, j_local] = 0.1 * sin(x) * cos(y)
        end
    end
    
    # Set up time integration with filtering
    params = TimeParams{Float64}(0.001; filter_freq=5, filter_strength=0.1)
    state = TimeState{Float64, typeof(fields.bₛ)}(0.0, fields)
    
    initial_energy = sum(abs2, b_data)
    
    # Run integration with periodic filtering
    for step = 1:20
        timestep_rk3!(fields, domain, params, state)
        
        # Manual filtering every few steps (simulating what happens in timestepper)
        if step % 5 == 0
            filter = ExponentialFilter{Float64}(0.1, 4, 0.7)
            apply_filter_to_field!(fields.bₛ, fields.bhat, domain, filter)
        end
    end
    
    final_energy = sum(abs2, b_data)
    
    println("  Initial energy: $(round(initial_energy, digits=6))")
    println("  Final energy: $(round(final_energy, digits=6))")
    
    # Should remain stable and finite
    is_stable = isfinite(final_energy) && final_energy > 0 && final_energy < 10*initial_energy
    
    if is_stable
        println("  ✓ Filtering maintains stability during time integration")
    end
    
    MPI.Finalize()
    return is_stable
end

# Run tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running spectral filtering tests...")
    
    test1_passed = test_filter_types()
    test2_passed = test_transfer_functions()
    test3_passed = test_2d_filtering()
    test4_passed = test_cutoff_filter_accuracy()
    test5_passed = test_hyperviscosity_damping()
    test6_passed = test_filter_conservation()
    test7_passed = test_filter_stability()
    
    println("\n" * "="^60)
    println("SPECTRAL FILTERING TEST RESULTS:")
    println("✓ Filter Types: $(test1_passed ? "PASSED" : "FAILED")")
    println("✓ Transfer Functions: $(test2_passed ? "PASSED" : "FAILED")")
    println("✓ 2D Field Filtering: $(test3_passed ? "PASSED" : "FAILED")")
    println("✓ Cutoff Filter Accuracy: $(test4_passed ? "PASSED" : "FAILED")")
    println("✓ Hyperviscosity Damping: $(test5_passed ? "PASSED" : "FAILED")")
    println("✓ Filter Conservation: $(test6_passed ? "PASSED" : "FAILED")")
    println("✓ Filter Stability: $(test7_passed ? "PASSED" : "FAILED")")
    
    all_passed = test1_passed && test2_passed && test3_passed && test4_passed && 
                 test5_passed && test6_passed && test7_passed
    
    if all_passed
        println("\n🎉 All spectral filtering tests passed!")
    else
        println("\n❌ Some spectral filtering tests failed!")
        exit(1)
    end
end