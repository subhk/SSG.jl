"""
Create coarser domain for multigrid hierarchy by reducing grid resolution

This function takes a fine domain and creates a coarser domain with reduced
resolution by the given coarsening factor. Essential for multigrid methods.

Parameters:
- domain: The actual Domain object
- factor: Coarsening factor (typically 2, meaning half resolution in each direction)

Returns:
- coarse_domain: New Domain object with coarser resolution
"""
function create_coarse_domain(domain::Domain, factor::Int=2)
    # Extract fine domain parameters
    fine_Nx = domain.Nx
    fine_Ny = domain.Ny  
    fine_Nz = domain.Nz
    
    # Physical domain size remains the same
    Lx = domain.Lx
    Ly = domain.Ly
    Lz = domain.Lz
    
    # Coarsen grid resolution
    coarse_Nx = max(fine_Nx ÷ factor, 5)  # Ensure minimum size
    coarse_Ny = max(fine_Ny ÷ factor, 5)
    coarse_Nz = fine_Nz  #  don't coarsen in the z-direction
    
    # For FFT compatibility, ensure even numbers
    coarse_Nx = coarse_Nx % 2 == 0 ? coarse_Nx : coarse_Nx + 1
    coarse_Ny = coarse_Ny % 2 == 0 ? coarse_Ny : coarse_Ny + 1
    
    # Get communicator and other parameters from fine domain
    comm = domain.pc.comm
    
    # Create new domain with coarser resolution
    # This depends on your Domain constructor - here are common patterns:
    
    # OPTION 1: If your Domain constructor takes (Nx, Ny, Nz, Lx, Ly, Lz, comm)
    coarse_domain = Domain(coarse_Nx, coarse_Ny, coarse_Nz, Lx, Ly, Lz, comm)
    
    # OPTION 2: If you need to preserve more parameters from fine domain
    # coarse_domain = Domain(
    #     Nx = coarse_Nx,
    #     Ny = coarse_Ny, 
    #     Nz = coarse_Nz,
    #     Lx = Lx,
    #     Ly = Ly,
    #     Lz = Lz,
    #     pc = create_pencil_config(coarse_Nx, coarse_Ny, coarse_Nz, comm),
    #     boundary_conditions = domain.boundary_conditions,
    #     # ... other parameters from domain
    # )
    
    return coarse_domain
end

# ============================================================================
# ALTERNATIVE IMPLEMENTATIONS BASED ON YOUR DOMAIN STRUCTURE
# ============================================================================

"""
Implementation if your Domain uses a configuration struct/NamedTuple
"""
function create_coarse_domain_config(domain::Domain, factor::Int=2)
    # Coarsen grid dimensions
    coarse_Nx = max(domain.Nx ÷ factor, 5)
    coarse_Ny = max(domain.Ny ÷ factor, 5) 
    coarse_Nz = domain.Nz  # Keep Z resolution
    
    # Ensure even numbers for FFT
    coarse_Nx = coarse_Nx % 2 == 0 ? coarse_Nx : coarse_Nx + 1
    coarse_Ny = coarse_Ny % 2 == 0 ? coarse_Ny : coarse_Ny + 1
    
    # Create new pencil configuration for coarser grid
    coarse_pencil_config = (
        comm = domain.pc.comm,
        # Add other pencil parameters as needed
    )
    
    # Create coarser domain preserving all other properties
    coarse_domain = Domain(
        Nx = coarse_Nx,
        Ny = coarse_Ny,
        Nz = coarse_Nz,
        Lx = domain.Lx,  # Physical size unchanged
        Ly = domain.Ly,
        Lz = domain.Lz,
        pc = coarse_pencil_config,
        #boundary_conditions = domain.boundary_conditions,
        # Copy other fields from domain as needed
    )
    
    return coarse_domain
end

"""
Implementation for transforms.jl Domain with FFT plans
"""
function create_coarse_domain_fft(domain::Domain, factor::Int=2)
    # Coarsen dimensions
    coarse_Nx = max(domain.Nx ÷ factor, 8)  # Minimum 8 for FFT
    coarse_Ny = max(domain.Ny ÷ factor, 8)
    coarse_Nz = domain.Nz
    
    # Ensure power of 2 or highly composite for efficient FFT
    coarse_Nx = next_fft_size(coarse_Nx)
    coarse_Ny = next_fft_size(coarse_Ny)
    
    # Create new domain with updated FFT plans
    coarse_domain = Domain(
        Nx = coarse_Nx,
        Ny = coarse_Ny, 
        Nz = coarse_Nz,
        Lx = domain.Lx,
        Ly = domain.Ly,
        Lz = domain.Lz,
        pc = domain.pc  # May need to update pencil configuration
    )
    
    # Initialize FFT plans for coarser grid
    setup_fft_plans!(coarse_domain)
    
    return coarse_domain
end

"""
Helper function to find next efficient FFT size
"""
function next_fft_size(n::Int)
    # Find next number that's efficient for FFT (powers of 2, 3, 5, 7)
    while true
        temp = n
        for p in [2, 3, 5, 7]
            while temp % p == 0
                temp ÷= p
            end
        end
        if temp == 1
            return n
        end
        n += 1
    end
end

"""
Semi-coarsening version (coarsen only in one direction)
Useful for anisotropic problems
"""
function create_semicoarse_domain(domain::Domain, direction::Symbol, factor::Int=2)
    coarse_Nx = domain.Nx
    coarse_Ny = domain.Ny
    coarse_Nz = domain.Nz
    
    if direction == :x
        coarse_Nx = max(domain.Nx ÷ factor, 5)
    elseif direction == :y  
        coarse_Ny = max(domain.Ny ÷ factor, 5)
    elseif direction == :z
        coarse_Nz = max(domain.Nz ÷ factor, 3)
    else
        error("Unknown coarsening direction: $direction")
    end
    
    # Create domain with semi-coarsened resolution
    coarse_domain = Domain(coarse_Nx, coarse_Ny, coarse_Nz, 
                          domain.Lx, domain.Ly, domain.Lz,
                          domain.pc.comm)
    
    return coarse_domain
end

# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

"""
Example of how to integrate with your specific Domain constructor
"""
function integrate_with_your_domain()
    println("Integration examples for create_coarse_domain:")
    println("")
    
    println("1. If your Domain constructor is:")
    println("   Domain(Nx, Ny, Nz, Lx, Ly, Lz, comm)")
    println("   → Use: create_coarse_domain(domain, factor)")
    println("")
    
    println("2. If your Domain has complex configuration:")
    println("   Domain(config_dict)")
    println("   → Use: create_coarse_domain_config(domain, factor)")
    println("")
    
    println("3. If your Domain includes FFT plans:")
    println("   Domain(...) with setup_fft_plans!")
    println("   → Use: create_coarse_domain_fft(domain, factor)")
    println("")
    
    println("4. For anisotropic problems:")
    println("   → Use: create_semicoarse_domain(domain, :x, factor)")
    println("")
    
    # Example usage in multigrid hierarchy creation
    println("Example multigrid hierarchy creation:")
    println("""
    function create_mg_hierarchy(base_domain::Domain, n_levels::Int=4)
        levels = SSGLevel{Float64}[]
        current_domain = base_domain
        
        for level = 1:n_levels
            push!(levels, SSGLevel{Float64}(current_domain, level))
            
            if level < n_levels
                # Coarsen domain for next level
                current_domain = create_coarse_domain(current_domain, 2)
            end
        end
        
        return levels
    end
    """)
end

# ============================================================================
# DOMAIN VALIDATION AND UTILITIES
# ============================================================================

"""
Validate that coarsening is appropriate for multigrid
"""
function validate_coarsening(domain::Domain, coarse_domain::Domain)
    checks_passed = true
    
    # Check that coarse grid is actually coarser
    if coarse_domain.Nx >= domain.Nx || coarse_domain.Ny >= domain.Ny
        @warn "Coarse domain not actually coarser than fine domain"
        checks_passed = false
    end
    
    # Check minimum size for effective multigrid
    if coarse_domain.Nx < 5 || coarse_domain.Ny < 5
        @warn "Coarse domain too small for effective multigrid"
        checks_passed = false
    end
    
    # Check that physical domain size is preserved
    if abs(coarse_domain.Lx - domain.Lx) > 1e-14 ||
       abs(coarse_domain.Ly - domain.Ly) > 1e-14 ||
       abs(coarse_domain.Lz - domain.Lz) > 1e-14
        @warn "Physical domain size not preserved during coarsening"
        checks_passed = false
    end
    
    # Check FFT compatibility
    if coarse_domain.Nx % 2 != 0 || coarse_domain.Ny % 2 != 0
        @warn "Coarse domain dimensions not even - may cause FFT issues"
    end
    
    return checks_passed
end

"""
Estimate memory usage for multigrid hierarchy
"""
function estimate_mg_memory(base_domain::Domain, n_levels::Int=4)
    total_memory = 0.0
    current_nx, current_ny, current_nz = base_domain.Nx, base_domain.Ny, base_domain.Nz
    
    println("Multigrid memory estimation:")
    for level = 1:n_levels
        # Estimate fields per level (solution, RHS, residual, derivatives, etc.)
        fields_per_level = 10  # Approximate number of 3D fields
        memory_per_level = fields_per_level * current_nx * current_ny * current_nz * 8  # 8 bytes for Float64
        total_memory += memory_per_level
        
        println("  Level $level: $(current_nx)×$(current_ny)×$(current_nz) → $(memory_per_level/1e6:.1f) MB")
        
        # Coarsen for next level
        current_nx = max(current_nx ÷ 2, 5)
        current_ny = max(current_ny ÷ 2, 5)
    end
    
    println("  Total estimated memory: $(total_memory/1e6:.1f) MB")
    return total_memory
end

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
To integrate with your SSG solver:

1. **Identify your Domain constructor pattern:**
   Look at how your Domain is created in your codebase

2. **Choose appropriate implementation:**
   - Simple: create_coarse_domain() 
   - Complex config: create_coarse_domain_config()
   - With FFT plans: create_coarse_domain_fft()
   - Anisotropic: create_semicoarse_domain()

3. **Replace the placeholder in SSG solver:**
   Replace the placeholder function with your chosen implementation

4. **Test the coarsening:**
   ```julia
   domain = Domain(...)  # Your fine domain
   coarse_domain = create_coarse_domain(domain, 2)
   validate_coarsening(domain, coarse_domain)
   ```

5. **Integration example:**
   ```julia
   # In your SSG solver, replace:
   current_domain = create_coarse_domain(current_domain, 2)
   
   # With your specific implementation:
   current_domain = create_coarse_domain_fft(current_domain, 2)
   ```
"""
