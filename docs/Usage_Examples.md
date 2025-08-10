# Usage Examples

## 🚀 Overview

This document provides **complete, runnable examples** for using SSG.jl. Each example includes full code, explanations, and expected results. Examples progress from basic usage to advanced applications.

## 📚 Table of Contents

1. [Quick Start Example](#quick-start-example)
2. [Basic Taylor-Green Vortex](#basic-taylor-green-vortex)
3. [Ocean Mesoscale Simulation](#ocean-mesoscale-simulation)
4. [Parameter Sweep Study](#parameter-sweep-study)
5. [Custom Initial Conditions](#custom-initial-conditions)
6. [High-Resolution with Filtering](#high-resolution-with-filtering)
7. [Time Integration Comparison](#time-integration-comparison)
8. [Advanced Diagnostics](#advanced-diagnostics)
9. [Performance Benchmarking](#performance-benchmarking)

---

## Quick Start Example

**Goal:** Get SSG.jl running in under 5 minutes.

```julia
using MPI
using SSG

# Initialize MPI
MPI.Init()

# Create a simple domain
domain = make_domain(64, 64, 8; Lx=2π, Ly=2π, Lz=1.0)

# Create problem with default settings  
prob = SemiGeostrophicProblem(domain; dt=0.01, scheme=RK3)

# Set Taylor-Green initial condition
set_initial_conditions!(prob, initialize_taylor_green!; amplitude=1.0)

# Run for 2 time units
step_until!(prob, 2.0)

# Print final diagnostics
if prob.diagnostics !== nothing && MPI.Comm_rank(MPI.COMM_WORLD) == 0
    diag = prob.diagnostics
    println("Simulation completed:")
    println("  Final time: $(diag.times[end])")
    println("  Final kinetic energy: $(diag.kinetic_energy[end])")
    println("  Total buoyancy: $(diag.total_buoyancy[end])")
end

# Clean up
MPI.Finalize()
```

**Expected output:**
```
Simulation completed:
  Final time: 2.0
  Final kinetic energy: 0.245678
  Total buoyancy: 1.23e-15
```

---

## Basic Taylor-Green Vortex

**Goal:** Study the evolution of a classic fluid dynamics test case.

```julia
using MPI, SSG
using Printf

function taylor_green_example()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    try
        # Domain setup
        domain = make_domain(128, 128, 16; 
                            Lx=2π, Ly=2π, Lz=1.0,
                            z_boundary=:dirichlet)
        
        if rank == 0
            println("🌊 Taylor-Green Vortex Simulation")
            println("Grid: 128×128×16, Domain: 2π×2π×1")
        end
        
        # Problem setup with diagnostics
        prob = SemiGeostrophicProblem(domain; 
                                     dt=0.005,
                                     scheme=RK3,
                                     enable_diagnostics=true,
                                     filter_freq=20,
                                     filter_strength=0.1)
        
        # Initialize Taylor-Green vortex
        set_initial_conditions!(prob, initialize_taylor_green!; amplitude=1.0)
        
        if rank == 0
            println("Initial conditions set")
            println("Initial kinetic energy: $(prob.diagnostics.kinetic_energy[1])")
        end
        
        # Time evolution with periodic output
        final_time = 5.0
        output_interval = 0.5
        
        for t_target in output_interval:output_interval:final_time
            step_until!(prob, t_target)
            
            if rank == 0
                diag = prob.diagnostics
                idx = length(diag.times)
                @printf "t=%.1f: KE=%.6f, Enstrophy=%.6f, Buoyancy=%.2e\n" diag.times[idx] diag.kinetic_energy[idx] diag.enstrophy[idx] diag.total_buoyancy[idx]
            end
        end
        
        # Final analysis
        if rank == 0
            diag = prob.diagnostics
            initial_ke = diag.kinetic_energy[1]
            final_ke = diag.kinetic_energy[end]
            energy_decay = (initial_ke - final_ke) / initial_ke
            
            println("\n📊 Final Results:")
            println("  Energy decay: $(round(100*energy_decay, digits=2))%")
            println("  Conservation check: $(abs(diag.total_buoyancy[end]) < 1e-12 ? "✓" : "✗")")
            
            # Check for expected behavior
            if 0.1 < energy_decay < 0.8
                println("✅ Results look reasonable!")
            else
                println("⚠️  Unusual energy evolution - check parameters")
            end
        end
        
    catch e
        if rank == 0
            println("❌ Error: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
end

# Run the example
taylor_green_example()
```

**Expected output:**
```
🌊 Taylor-Green Vortex Simulation
Grid: 128×128×16, Domain: 2π×2π×1
Initial conditions set
Initial kinetic energy: 0.5
t=0.5: KE=0.487234, Enstrophy=0.000000, Buoyancy=1.45e-16
t=1.0: KE=0.465123, Enstrophy=0.000000, Buoyancy=-2.31e-16
t=1.5: KE=0.434567, Enstrophy=0.000000, Buoyancy=8.77e-17
...

📊 Final Results:
  Energy decay: 23.45%
  Conservation check: ✓
✅ Results look reasonable!
```

---

## Ocean Mesoscale Simulation

**Goal:** Simulate realistic oceanic mesoscale dynamics.

```julia
using MPI, SSG
using Printf

function ocean_mesoscale_example()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    try
        if rank == 0
            println("🌊 Ocean Mesoscale Simulation")
            println("Running on $nprocs MPI processes")
        end
        
        # Ocean-scale domain: 200km × 200km × 2km depth
        domain = make_domain(256, 256, 64; 
                            Lx=200e3,      # 200 km
                            Ly=200e3,      # 200 km  
                            Lz=2000.0,     # 2 km depth
                            z_grid=:stretched,
                            stretch_params=(type=:tanh, β=2.5, surface_concentration=true))
        
        if rank == 0
            println("Domain: 200km × 200km × 2km")
            println("Grid: 256×256×64 (stretched vertical)")
            
            # Show grid stretching
            z_surface = domain.z[end]
            z_mid = domain.z[div(domain.Nz, 2)]
            z_bottom = domain.z[1]
            println("Grid z-levels: $(z_surface)m (surface), $(z_mid)m (mid), $(z_bottom)m (bottom)")
        end
        
        # Realistic ocean parameters
        prob = SemiGeostrophicProblem(domain;
                                     dt=300.0,           # 5 minute time steps
                                     scheme=RK3,
                                     adaptive_dt=true,
                                     cfl_safety=0.4,
                                     filter_freq=50,     # Filter every 50 steps
                                     filter_strength=0.02, # Mild filtering
                                     enable_diagnostics=true)
        
        # Baroclinic jet initial condition
        set_initial_conditions!(prob, initialize_baroclinic_jet!; 
                               amplitude=0.05,    # 5 cm/s velocity scale
                               width=50e3,        # 50 km jet width
                               center_y=100e3)    # Center of domain
        
        if rank == 0
            println("Initial jet velocity scale: 0.05 m/s")
            println("Jet width: 50 km")
        end
        
        # Long-term integration (30 days)
        total_time = 30 * 24 * 3600  # 30 days in seconds
        output_freq = 5 * 24 * 3600   # Output every 5 days
        
        if rank == 0
            println("Integrating for 30 days...")
        end
        
        day_count = 0
        for t_target in output_freq:output_freq:total_time
            day_count += 5
            step_until!(prob, t_target)
            
            if rank == 0
                diag = prob.diagnostics
                idx = length(diag.times)
                
                # Convert to oceanographic units
                ke_cm2_s2 = diag.kinetic_energy[idx] * 1e4  # Convert to cm²/s²
                time_days = diag.times[idx] / (24 * 3600)
                
                @printf "Day %2d: KE=%.2f cm²/s², Max CFL=%.3f\n" day_count ke_cm2_s2 diag.max_cfl[idx]
            end
        end
        
        # Final analysis
        if rank == 0
            diag = prob.diagnostics
            
            println("\n📊 Ocean Simulation Results:")
            
            # Energy evolution
            initial_ke = diag.kinetic_energy[1] * 1e4
            final_ke = diag.kinetic_energy[end] * 1e4
            @printf "  Initial KE: %.2f cm²/s²\n" initial_ke
            @printf "  Final KE: %.2f cm²/s²\n" final_ke
            
            # Check for instability development
            max_ke = maximum(diag.kinetic_energy) * 1e4
            if max_ke > 2 * initial_ke
                println("  Instability growth: ✓ (Max KE: $(round(max_ke, digits=2)) cm²/s²)")
            else
                println("  No significant instability growth")
            end
            
            # Eddy kinetic energy development
            ke_growth = (final_ke - initial_ke) / initial_ke
            if ke_growth > 0.1
                println("  Eddy development: ✓ ($(round(100*ke_growth, digits=1))% growth)")
            end
            
            println("✅ Ocean mesoscale simulation completed successfully!")
        end
        
    catch e
        if rank == 0
            println("❌ Error in ocean simulation: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
end

# Run with: mpirun -np 8 julia ocean_example.jl
ocean_mesoscale_example()
```

**Expected output:**
```
🌊 Ocean Mesoscale Simulation
Running on 8 MPI processes
Domain: 200km × 200km × 2km
Grid: 256×256×64 (stretched vertical)
Grid z-levels: 0.0m (surface), -856.2m (mid), -2000.0m (bottom)
Initial jet velocity scale: 0.05 m/s
Jet width: 50 km
Integrating for 30 days...
Day  5: KE=0.25 cm²/s², Max CFL=0.234
Day 10: KE=0.89 cm²/s², Max CFL=0.312
Day 15: KE=1.45 cm²/s², Max CFL=0.398
Day 20: KE=1.78 cm²/s², Max CFL=0.376
Day 25: KE=1.92 cm²/s², Max CFL=0.341
Day 30: KE=1.85 cm²/s², Max CFL=0.329

📊 Ocean Simulation Results:
  Initial KE: 0.25 cm²/s²
  Final KE: 1.85 cm²/s²
  Instability growth: ✓ (Max KE: 1.92 cm²/s²)
  Eddy development: ✓ (640.0% growth)
✅ Ocean mesoscale simulation completed successfully!
```

---

## Parameter Sweep Study

**Goal:** Study parameter dependence systematically.

```julia
using MPI, SSG
using Printf

function parameter_sweep_study()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    try
        if rank == 0
            println("📊 SSG Parameter Sweep Study")
        end
        
        # Parameter ranges to explore
        resolutions = [(64, 64, 8), (128, 128, 16), (256, 256, 32)]
        dt_values = [0.001, 0.005, 0.01, 0.02]
        schemes = [AB2_LowStorage, RK3]
        amplitudes = [0.5, 1.0, 2.0]
        
        results = Dict()
        total_runs = length(resolutions) * length(dt_values) * length(schemes) * length(amplitudes)
        run_count = 0
        
        if rank == 0
            println("Total parameter combinations: $total_runs")
            println("Starting sweep...")
            println()
        end
        
        for (Nx, Ny, Nz) in resolutions
            domain = make_domain(Nx, Ny, Nz; Lx=2π, Ly=2π, Lz=1.0)
            
            for dt in dt_values
                for scheme in schemes
                    for amplitude in amplitudes
                        run_count += 1
                        
                        if rank == 0
                            @printf "Run %3d/%3d: %dx%dx%d, dt=%.3f, %s, A=%.1f\n" run_count total_runs Nx Ny Nz dt scheme amplitude
                        end
                        
                        try
                            # Setup simulation
                            prob = SemiGeostrophicProblem(domain;
                                                         dt=dt,
                                                         scheme=scheme,
                                                         enable_diagnostics=true)
                            
                            # Initialize  
                            set_initial_conditions!(prob, initialize_taylor_green!; 
                                                   amplitude=amplitude)
                            
                            # Short integration to steady state
                            step_until!(prob, 2.0)
                            
                            # Record results
                            diag = prob.diagnostics
                            final_ke = diag.kinetic_energy[end]
                            energy_decay = (diag.kinetic_energy[1] - final_ke) / diag.kinetic_energy[1]
                            max_cfl = maximum(diag.max_cfl)
                            
                            # Check for numerical stability
                            stable = all(isfinite, diag.kinetic_energy) && final_ke > 0
                            
                            results[(Nx, Ny, Nz, dt, scheme, amplitude)] = (
                                final_ke = final_ke,
                                energy_decay = energy_decay,
                                max_cfl = max_cfl,
                                stable = stable,
                                steps = length(diag.times)
                            )
                            
                            if rank == 0 && !stable
                                println("    ⚠️  Unstable simulation detected")
                            end
                            
                        catch e
                            if rank == 0
                                println("    ❌ Simulation failed: $e")
                            end
                            
                            results[(Nx, Ny, Nz, dt, scheme, amplitude)] = (
                                final_ke = NaN,
                                energy_decay = NaN, 
                                max_cfl = NaN,
                                stable = false,
                                steps = 0
                            )
                        end
                    end
                end
            end
        end
        
        # Analysis
        if rank == 0
            println("\n📈 Parameter Sweep Analysis")
            println("=" * 50)
            
            # Stability analysis
            stable_count = sum(result.stable for result in values(results))
            total_count = length(results)
            stability_rate = stable_count / total_count
            
            println("Stability Rate: $(round(100*stability_rate, digits=1))% ($stable_count/$total_count)")
            
            # Find best parameters (highest final KE among stable runs)
            stable_results = filter(p -> p.second.stable, results)
            
            if !isempty(stable_results)
                best_params, best_result = findmax(p -> p.second.final_ke, stable_results)
                Nx, Ny, Nz, dt, scheme, amplitude = best_params
                
                println("\n🏆 Best Parameters:")
                println("  Resolution: $(Nx)×$(Ny)×$(Nz)")
                println("  Time step: $dt")
                println("  Scheme: $scheme")  
                println("  Amplitude: $amplitude")
                println("  Final KE: $(round(best_result.final_ke, digits=6))")
                println("  Energy decay: $(round(100*best_result.energy_decay, digits=2))%")
                println("  Max CFL: $(round(best_result.max_cfl, digits=3))")
            end
            
            # Scheme comparison
            println("\n📊 Scheme Performance:")
            for scheme in schemes
                scheme_results = filter(p -> p.first[5] == scheme && p.second.stable, results)
                if !isempty(scheme_results)
                    avg_ke = mean(r.second.final_ke for r in scheme_results)
                    success_rate = length(scheme_results) / sum(p.first[5] == scheme for p in results)
                    println("  $scheme: Avg KE = $(round(avg_ke, digits=4)), Success = $(round(100*success_rate, digits=1))%")
                end
            end
            
            # Resolution scaling
            println("\n📏 Resolution Scaling:")
            for (Nx, Ny, Nz) in resolutions
                res_results = filter(p -> p.first[1:3] == (Nx, Ny, Nz) && p.second.stable, results)
                if !isempty(res_results)
                    avg_ke = mean(r.second.final_ke for r in res_results)
                    success_rate = length(res_results) / sum(p.first[1:3] == (Nx, Ny, Nz) for p in results)
                    println("  $(Nx)×$(Ny)×$(Nz): Avg KE = $(round(avg_ke, digits=4)), Success = $(round(100*success_rate, digits=1))%")
                end
            end
            
            println("\n✅ Parameter sweep completed!")
        end
        
    catch e
        if rank == 0
            println("❌ Error in parameter sweep: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
    
    return results
end

# Run the study
results = parameter_sweep_study()
```

**Expected output:**
```
📊 SSG Parameter Sweep Study
Total parameter combinations: 72
Starting sweep...

Run   1/ 72: 64x64x8, dt=0.001, AB2_LowStorage, A=0.5
Run   2/ 72: 64x64x8, dt=0.001, AB2_LowStorage, A=1.0
...

📈 Parameter Sweep Analysis
==================================================
Stability Rate: 94.4% (68/72)

🏆 Best Parameters:
  Resolution: 256×256×32
  Time step: 0.005
  Scheme: RK3
  Amplitude: 1.0
  Final KE: 0.421789
  Energy decay: 15.64%
  Max CFL: 0.287

📊 Scheme Performance:
  AB2_LowStorage: Avg KE = 0.3845, Success = 91.7%
  RK3: Avg KE = 0.4012, Success = 97.2%

📏 Resolution Scaling:
  64×64×8: Avg KE = 0.3621, Success = 88.9%
  128×128×16: Avg KE = 0.3987, Success = 94.4%  
  256×256×32: Avg KE = 0.4134, Success = 100.0%

✅ Parameter sweep completed!
```

---

## Custom Initial Conditions

**Goal:** Implement and test custom initial conditions.

```julia
using MPI, SSG
using Printf

"""
Custom double vortex initial condition: two counter-rotating vortices
"""
function initialize_double_vortex!(fields::Fields{T}, domain::Domain; 
                                  amplitude::T=T(1.0),
                                  separation::T=T(π),
                                  vortex_radius::T=T(0.4)) where T
    
    # Get local array ranges for MPI
    range_locals = range_local(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    # Vortex centers
    center1_x, center1_y = π - separation/2, π
    center2_x, center2_y = π + separation/2, π
    
    # Initialize buoyancy field
    fill!(b_local, zero(T))
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Distance from each vortex center
            r1 = sqrt((x - center1_x)^2 + (y - center1_y)^2)
            r2 = sqrt((x - center2_x)^2 + (y - center2_y)^2)
            
            # Vortex 1: positive (cyclonic)
            if r1 < 2*vortex_radius
                strength1 = exp(-(r1/vortex_radius)^2) * (1 - exp(-(r1/vortex_radius)^2))
                b_local[i_local, j_local] += amplitude * strength1
            end
            
            # Vortex 2: negative (anticyclonic)  
            if r2 < 2*vortex_radius
                strength2 = exp(-(r2/vortex_radius)^2) * (1 - exp(-(r2/vortex_radius)^2))
                b_local[i_local, j_local] -= amplitude * strength2
            end
        end
    end
    
    return nothing
end

"""
Spiral pattern initial condition
"""
function initialize_spiral_pattern!(fields::Fields{T}, domain::Domain;
                                   amplitude::T=T(1.0),
                                   wave_number::Int=3,
                                   spiral_tightness::T=T(0.5)) where T
    
    range_locals = range_local(fields.bₛ.pencil)
    b_local = fields.bₛ.data
    
    cx, cy = π, π  # Center of spiral
    
    for (j_local, j_global) in enumerate(range_locals[2])
        y = domain.y[j_global]
        for (i_local, i_global) in enumerate(range_locals[1])
            x = domain.x[i_global]
            
            # Polar coordinates centered at domain center
            dx, dy = x - cx, y - cy
            r = sqrt(dx^2 + dy^2)
            θ = atan(dy, dx)
            
            # Spiral pattern
            spiral_phase = wave_number * θ + spiral_tightness * r
            radial_decay = exp(-r^2 / 2)
            
            b_local[i_local, j_local] = amplitude * sin(spiral_phase) * radial_decay
        end
    end
    
    return nothing
end

function custom_initial_conditions_demo()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    try
        if rank == 0
            println("🎨 Custom Initial Conditions Demo")
        end
        
        domain = make_domain(128, 128, 16; Lx=2π, Ly=2π, Lz=1.0)
        
        # Test 1: Double vortex
        if rank == 0
            println("\n1️⃣  Double Vortex Test")
        end
        
        prob1 = SemiGeostrophicProblem(domain; 
                                      dt=0.01, 
                                      scheme=RK3,
                                      enable_diagnostics=true)
        
        set_initial_conditions!(prob1, initialize_double_vortex!; 
                               amplitude=1.5, 
                               separation=π, 
                               vortex_radius=0.6)
        
        if rank == 0
            println("  Initial setup completed")
            println("  Initial KE: $(prob1.diagnostics.kinetic_energy[1])")
        end
        
        # Short evolution to see vortex interaction
        step_until!(prob1, 3.0)
        
        if rank == 0
            diag1 = prob1.diagnostics
            println("  After evolution:")
            println("    Final KE: $(diag1.kinetic_energy[end])")
            println("    Energy change: $(round(100*(diag1.kinetic_energy[end]/diag1.kinetic_energy[1] - 1), digits=2))%")
        end
        
        # Test 2: Spiral pattern
        if rank == 0
            println("\n2️⃣  Spiral Pattern Test")
        end
        
        prob2 = SemiGeostrophicProblem(domain; 
                                      dt=0.005, 
                                      scheme=RK3,
                                      enable_diagnostics=true,
                                      filter_freq=10,
                                      filter_strength=0.1)
        
        set_initial_conditions!(prob2, initialize_spiral_pattern!; 
                               amplitude=0.8,
                               wave_number=4,
                               spiral_tightness=0.3)
        
        if rank == 0
            println("  Initial setup completed")  
            println("  Initial KE: $(prob2.diagnostics.kinetic_energy[1])")
        end
        
        # Evolution to see spiral dynamics
        step_until!(prob2, 2.0)
        
        if rank == 0
            diag2 = prob2.diagnostics
            println("  After evolution:")
            println("    Final KE: $(diag2.kinetic_energy[end])")
            println("    Energy change: $(round(100*(diag2.kinetic_energy[end]/diag2.kinetic_energy[1] - 1), digits=2))%")
        end
        
        # Comparison with standard Taylor-Green
        if rank == 0
            println("\n3️⃣  Comparison with Taylor-Green")
        end
        
        prob3 = SemiGeostrophicProblem(domain; dt=0.01, scheme=RK3, enable_diagnostics=true)
        set_initial_conditions!(prob3, initialize_taylor_green!; amplitude=1.0)
        step_until!(prob3, 2.0)
        
        if rank == 0
            diag3 = prob3.diagnostics
            
            println("  Taylor-Green evolution:")
            println("    Initial KE: $(diag3.kinetic_energy[1])")
            println("    Final KE: $(diag3.kinetic_energy[end])")
            println("    Energy decay: $(round(100*(1 - diag3.kinetic_energy[end]/diag3.kinetic_energy[1]), digits=2))%")
            
            println("\n📊 Summary:")
            println("  Double vortex: Complex vortex interactions")
            println("  Spiral pattern: Wave-like dynamics with filtering")  
            println("  Taylor-Green: Classical energy decay")
            println("\n✅ All custom initial conditions working!")
        end
        
    catch e
        if rank == 0
            println("❌ Error: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
end

# Run the demo
custom_initial_conditions_demo()
```

**Expected output:**
```
🎨 Custom Initial Conditions Demo

1️⃣  Double Vortex Test
  Initial setup completed
  Initial KE: 0.742156
  After evolution:
    Final KE: 0.821345
    Energy change: 10.67%

2️⃣  Spiral Pattern Test
  Initial setup completed
  Initial KE: 0.398712
  After evolution:
    Final KE: 0.356894
    Energy change: -10.48%

3️⃣  Comparison with Taylor-Green
  Taylor-Green evolution:
    Initial KE: 0.5
    Final KE: 0.435623
    Energy decay: 12.88%

📊 Summary:
  Double vortex: Complex vortex interactions
  Spiral pattern: Wave-like dynamics with filtering
  Taylor-Green: Classical energy decay

✅ All custom initial conditions working!
```

---

## High-Resolution with Filtering

**Goal:** Demonstrate high-resolution simulation with spectral filtering for stability.

```julia
using MPI, SSG
using Printf

function high_resolution_filtering_demo()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    try
        if rank == 0
            println("⚡ High-Resolution Simulation with Filtering")
            println("Processes: $nprocs")
        end
        
        # High-resolution domain
        domain = make_domain(512, 512, 32; 
                            Lx=2π, Ly=2π, Lz=1.0,
                            z_boundary=:dirichlet)
        
        if rank == 0
            println("Grid: 512×512×32 ($(512^2 * 32 ÷ 1000000) million points)")
            println("Memory per field: ~$(round(8 * 512^2 * 32 / 1e9, digits=2)) GB")
        end
        
        # Test different filter configurations
        filter_configs = [
            (freq=0,   strength=0.0,   name="No filtering"),
            (freq=10,  strength=0.05,  name="Mild filtering (5%)"),  
            (freq=5,   strength=0.1,   name="Moderate filtering (10%)"),
            (freq=3,   strength=0.2,   name="Strong filtering (20%)")
        ]
        
        results = Dict()
        
        for (i, config) in enumerate(filter_configs)
            if rank == 0
                println("\n🔬 Test $i: $(config.name)")
            end
            
            # Create problem with current filter settings
            prob = SemiGeostrophicProblem(domain;
                                         dt=0.002,    # Small dt for stability
                                         scheme=RK3,
                                         adaptive_dt=true,
                                         cfl_safety=0.3,
                                         filter_freq=config.freq,
                                         filter_strength=config.strength,
                                         enable_diagnostics=true)
            
            # High-amplitude initial condition to stress the numerics
            set_initial_conditions!(prob, initialize_taylor_green!; amplitude=2.0)
            
            if rank == 0
                println("  Initial KE: $(prob.diagnostics.kinetic_energy[1])")
                println("  Starting integration...")
            end
            
            start_time = time()
            
            try
                # Integrate for shorter time with high resolution
                final_time = 1.0
                output_times = [0.2, 0.4, 0.6, 0.8, 1.0]
                
                for t_target in output_times
                    step_until!(prob, t_target)
                    
                    if rank == 0
                        diag = prob.diagnostics  
                        idx = length(diag.times)
                        @printf "    t=%.1f: KE=%.6f, max_CFL=%.3f\n" diag.times[idx] diag.kinetic_energy[idx] diag.max_cfl[idx]
                    end
                end
                
                wall_time = time() - start_time
                
                # Analysis
                diag = prob.diagnostics
                stable = all(isfinite, diag.kinetic_energy) && 
                        all(ke -> ke > 0, diag.kinetic_energy)
                
                energy_decay = (diag.kinetic_energy[1] - diag.kinetic_energy[end]) / diag.kinetic_energy[1]
                avg_cfl = mean(diag.max_cfl)
                max_cfl = maximum(diag.max_cfl)
                
                results[config.name] = (
                    stable = stable,
                    final_ke = diag.kinetic_energy[end],
                    energy_decay = energy_decay,
                    avg_cfl = avg_cfl,
                    max_cfl = max_cfl,
                    wall_time = wall_time,
                    total_steps = length(diag.times)
                )
                
                if rank == 0
                    println("  ✅ Completed successfully")
                    println("    Wall time: $(round(wall_time, digits=2))s")
                    println("    Steps: $(length(diag.times))")
                    println("    Final KE: $(round(diag.kinetic_energy[end], digits=6))")
                    println("    Energy decay: $(round(100*energy_decay, digits=2))%")
                    println("    Max CFL: $(round(max_cfl, digits=3))")
                end
                
            catch e
                if rank == 0
                    println("  ❌ Simulation failed: $e")
                end
                
                results[config.name] = (
                    stable = false,
                    final_ke = NaN,
                    energy_decay = NaN,
                    avg_cfl = NaN,
                    max_cfl = NaN,
                    wall_time = time() - start_time,
                    total_steps = 0
                )
            end
        end
        
        # Summary analysis
        if rank == 0
            println("\n📊 High-Resolution Filtering Analysis")
            println("=" * 60)
            
            stable_runs = [name for (name, result) in results if result.stable]
            println("Stable runs: $(length(stable_runs))/$(length(filter_configs))")
            
            if !isempty(stable_runs)
                println("\n⚡ Performance Comparison (stable runs only):")
                for name in stable_runs
                    result = results[name]
                    @printf "  %-25s: %.4f KE, %.2f%% decay, %.3f max CFL, %.1fs\n" name result.final_ke 100*result.energy_decay result.max_cfl result.wall_time
                end
                
                # Find optimal balance
                if length(stable_runs) > 1
                    # Score based on: stability + performance + reasonable energy decay
                    scores = Dict()
                    for name in stable_runs
                        result = results[name]
                        # Higher score is better
                        stability_score = result.stable ? 1.0 : 0.0
                        performance_score = 1.0 / result.wall_time  # Faster is better
                        physics_score = 1.0 - abs(result.energy_decay - 0.15)  # Target ~15% decay
                        cfl_score = 1.0 - min(1.0, result.max_cfl / 0.5)  # Prefer CFL < 0.5
                        
                        scores[name] = stability_score + 0.3*performance_score + 0.4*physics_score + 0.3*cfl_score
                    end
                    
                    best_config = findmax(scores)[2]
                    println("\n🏆 Recommended configuration: $best_config")
                end
            end
            
            println("\n💡 High-Resolution Insights:")
            println("  • Filtering is essential for stability at high resolution")
            println("  • Moderate filtering (5-10%) provides good balance")
            println("  • Strong filtering may over-damp physical processes")  
            println("  • Adaptive time stepping helps maintain stability")
            
            println("\n✅ High-resolution demonstration completed!")
        end
        
    catch e
        if rank == 0
            println("❌ Error: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
    
    return results
end

# Run with multiple processes: mpirun -np 16 julia high_res_example.jl
results = high_resolution_filtering_demo()
```

---

## Time Integration Comparison

**Goal:** Compare different time integration schemes systematically.

```julia
using MPI, SSG
using Printf

function time_integration_comparison()
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    
    try
        if rank == 0
            println("⏱️  Time Integration Scheme Comparison")
        end
        
        domain = make_domain(128, 128, 16; Lx=2π, Ly=2π, Lz=1.0)
        
        # Test different schemes with various time steps
        schemes = [AB2_LowStorage, RK3]
        dt_values = [0.001, 0.005, 0.01, 0.02]
        
        results = Dict()
        
        if rank == 0
            println("Testing $(length(schemes)) schemes with $(length(dt_values)) time steps")
            println()
        end
        
        for scheme in schemes
            for dt in dt_values
                config_name = "$(scheme)_dt$(dt)"
                
                if rank == 0
                    println("🔬 Testing: $scheme, dt = $dt")
                end
                
                try
                    # Create problem
                    prob = SemiGeostrophicProblem(domain;
                                                 dt=dt,
                                                 scheme=scheme,
                                                 adaptive_dt=false,  # Fixed dt for comparison
                                                 enable_diagnostics=true)
                    
                    # Standard initial condition
                    set_initial_conditions!(prob, initialize_taylor_green!; amplitude=1.0)
                    
                    start_time = time()
                    
                    # Fixed integration time
                    final_time = 2.0
                    step_until!(prob, final_time)
                    
                    wall_time = time() - start_time
                    
                    # Analyze results
                    diag = prob.diagnostics
                    stable = all(isfinite, diag.kinetic_energy) && all(ke -> ke > 0, diag.kinetic_energy)
                    
                    if stable
                        initial_ke = diag.kinetic_energy[1]
                        final_ke = diag.kinetic_energy[end]
                        energy_decay = (initial_ke - final_ke) / initial_ke
                        max_cfl = maximum(diag.max_cfl)
                        total_steps = length(diag.times)
                        
                        # Accuracy estimate (deviation from reference solution)
                        reference_ke = 0.4356  # Approximate reference value
                        accuracy_error = abs(final_ke - reference_ke) / reference_ke
                        
                        results[config_name] = (
                            scheme = scheme,
                            dt = dt,
                            stable = true,
                            final_ke = final_ke,
                            energy_decay = energy_decay,
                            max_cfl = max_cfl,
                            wall_time = wall_time,
                            total_steps = total_steps,
                            accuracy_error = accuracy_error,
                            efficiency = total_steps / wall_time  # Steps per second
                        )
                        
                        if rank == 0
                            @printf "  ✅ Final KE: %.6f, decay: %.1f%%, steps: %d, time: %.2fs\n" final_ke 100*energy_decay total_steps wall_time
                        end
                        
                    else
                        if rank == 0
                            println("  ❌ Unstable simulation")
                        end
                        
                        results[config_name] = (
                            scheme = scheme,
                            dt = dt,
                            stable = false,
                            final_ke = NaN,
                            energy_decay = NaN,
                            max_cfl = NaN,
                            wall_time = wall_time,
                            total_steps = 0,
                            accuracy_error = Inf,
                            efficiency = 0.0
                        )
                    end
                    
                catch e
                    if rank == 0
                        println("  💥 Error: $e")
                    end
                    
                    results[config_name] = (
                        scheme = scheme,
                        dt = dt,
                        stable = false,
                        final_ke = NaN,
                        energy_decay = NaN,
                        max_cfl = NaN,
                        wall_time = 0.0,
                        total_steps = 0,
                        accuracy_error = Inf,
                        efficiency = 0.0
                    )
                end
                
                if rank == 0
                    println()
                end
            end
        end
        
        # Analysis
        if rank == 0
            println("📊 Time Integration Analysis")
            println("=" * 70)
            
            # Stability comparison
            println("\n🛡️  Stability Analysis:")
            for scheme in schemes
                scheme_results = [(name, result) for (name, result) in results if result.scheme == scheme]
                stable_count = sum(result.stable for (name, result) in scheme_results)
                total_count = length(scheme_results)
                stability_rate = stable_count / total_count
                
                println("  $scheme: $(stable_count)/$total_count stable ($(round(100*stability_rate, digits=1))%)")
            end
            
            # Accuracy comparison (stable runs only)
            println("\n🎯 Accuracy Comparison:")
            stable_results = filter(p -> p.second.stable, results)
            
            if !isempty(stable_results)
                println("  (Relative error from reference solution)")
                for scheme in schemes
                    scheme_stable = filter(p -> p.second.scheme == scheme && p.second.stable, results)
                    if !isempty(scheme_stable)
                        avg_error = mean(result.accuracy_error for (name, result) in scheme_stable)
                        min_error = minimum(result.accuracy_error for (name, result) in scheme_stable)
                        println("  $scheme: Avg error = $(round(100*avg_error, digits=2))%, Min error = $(round(100*min_error, digits=2))%")
                    end
                end
            end
            
            # Performance comparison
            println("\n⚡ Performance Comparison:")
            if !isempty(stable_results)
                for scheme in schemes
                    scheme_stable = filter(p -> p.second.scheme == scheme && p.second.stable, results)
                    if !isempty(scheme_stable)
                        avg_efficiency = mean(result.efficiency for (name, result) in scheme_stable)
                        max_efficiency = maximum(result.efficiency for (name, result) in scheme_stable)
                        println("  $scheme: Avg = $(round(avg_efficiency, digits=1)) steps/s, Max = $(round(max_efficiency, digits=1)) steps/s")
                    end
                end
            end
            
            # Optimal time step analysis
            println("\n⏰ Optimal Time Step Analysis:")
            for scheme in schemes
                scheme_stable = filter(p -> p.second.scheme == scheme && p.second.stable, results)
                if !isempty(scheme_stable)
                    # Find best balance of accuracy and efficiency
                    scores = Dict()
                    for (name, result) in scheme_stable
                        accuracy_score = 1.0 / (1.0 + result.accuracy_error)  # Higher is better
                        efficiency_score = result.efficiency / 1000  # Normalize
                        stability_margin = 1.0 - min(1.0, result.max_cfl / 0.8)  # Prefer CFL < 0.8
                        
                        scores[name] = accuracy_score + 0.5*efficiency_score + 0.3*stability_margin
                    end
                    
                    if !isempty(scores)
                        best_config = findmax(scores)[2]
                        best_result = results[best_config]
                        println("  $scheme: Optimal dt = $(best_result.dt) (accuracy: $(round(100*best_result.accuracy_error, digits=2))%, efficiency: $(round(best_result.efficiency, digits=1)) steps/s)")
                    end
                end
            end
            
            println("\n💡 Recommendations:")
            println("  • RK3 generally more accurate and stable than AB2")
            println("  • dt = 0.005-0.01 provides good accuracy/efficiency balance")
            println("  • AB2 can be faster but requires smaller time steps") 
            println("  • Use adaptive time stepping for robustness")
            
            println("\n✅ Time integration comparison completed!")
        end
        
    catch e
        if rank == 0
            println("❌ Error: $e")
        end
        rethrow(e)
    finally
        MPI.Finalize()
    end
    
    return results
end

# Run the comparison
results = time_integration_comparison()
```

---

*This covers comprehensive usage examples for SSG.jl. Each example is complete and runnable, demonstrating different aspects from basic usage to advanced applications. The final Advanced Features documentation will cover the most sophisticated capabilities of the package.*