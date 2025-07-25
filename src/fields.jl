# src/fields.jl
# Field allocation and management

"""
    struct Fields{T, PA, PC}
Container for all simulation fields (prognostic, diagnostic, and scratch).

# Fields
## Prognostic
- `b`: Surface buoyancy anomaly

## Diagnostic  
- `φ`: Geostrophic streamfunction
- `u, v`: Semi-geostrophic velocity components

## Scratch arrays (real space)
- `R`: Residual for Monge-Ampère equation
- `tmp, tmp2`: General scratch arrays

## Spectral arrays
- `bhat, φhat`: Spectral versions of b and φ
- `tmpc`: Spectral scratch array
"""
struct Fields{T, PA, PC}
    # Prognostic fields
    b::PA           # surface buoyancy anomaly
    
    # Diagnostic fields
    φ::PA           # streamfunction
    u::PA           # x-velocity
    v::PA           # y-velocity

    # Multigrid workspace
    φ_mg::PA           # Multigrid solution workspace
    b_mg::PA           # Multigrid RHS workspace
    
    # Scratch arrays (real space)
    R::PA           # residual for MA equation
    tmp::PA         # general scratch
    tmp2::PA        # additional scratch
    
    # Spectral arrays
    bhat::PC        # spectral buoyancy
    φhat::PC        # spectral streamfunction
    tmpc::PC        # spectral scratch
end

"""
    allocate_fields(dom::Domain{T}) -> Fields{T}
Allocate all field arrays for the given domain.

# Arguments
- `dom`: Domain structure

# Returns
- `Fields`: Allocated field structure
"""
function allocate_fields(dom::Domain{T}) where T
    # Real-space fields
    b    = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    φ    = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    u    = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    v    = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    R    = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    tmp  = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    tmp2 = PencilArray(dom.pr, zeros(T, local_size(dom.pr)))
    
    # Spectral fields
    bhat = PencilArray(dom.pc, zeros(Complex{T}, local_size(dom.pc)))
    φhat = PencilArray(dom.pc, zeros(Complex{T}, local_size(dom.pc)))
    tmpc = PencilArray(dom.pc, zeros(Complex{T}, local_size(dom.pc)))
    
    return Fields{T, typeof(b), typeof(bhat)}(
        b, φ, u, v, R, tmp, tmp2, bhat, φhat, tmpc
    )
end

"""
    zero_fields!(fld::Fields)
Set all fields to zero.
"""
function zero_fields!(fld::Fields)
    fld.b .= 0
    fld.φ .= 0
    fld.u .= 0
    fld.v .= 0
    fld.R .= 0
    fld.tmp .= 0
    fld.tmp2 .= 0
    fld.bhat .= 0
    fld.φhat .= 0
    fld.tmpc .= 0
    return fld
end



function ensure_same_grid(dest::PencilArray, src::PencilArray)
    # Check compatible sizes
    if size(dest) != size(src)
        throw(ArgumentError("PencilArrays have incompatible sizes: $(size(dest)) vs $(size(src))"))
    end
    
    # Check MPI communicator compatibility
    if dest.pencil.comm != src.pencil.comm
        throw(ArgumentError("PencilArrays have different MPI communicators"))
    end
    
    return true
end

macro ensuresamegrid(dest, src)
    return quote
        ensure_same_grid($(esc(dest)), $(esc(src)))
    end
end


"""
    copy_field!(dest, src)
Copy one field to another (must be same type and size).
"""
function copy_field!(dest, src)
    @ensuresamegrid(dest, src)
    dest .= src
    return dest
end



"""
    field_stats(field) -> (mean, std, min, max)
Compute basic statistics for a field.
"""
function field_stats(field)
    vals = Array(field)  # Convert to regular array
    return (mean(vals), std(vals), minimum(vals), maximum(vals))
end

"""
    Base.show(io::IO, fld::Fields)
Pretty print field information.
"""
function Base.show(io::IO, fld::Fields)
    T = eltype(fld.b)
    println(io, "Fields{$T}:")
    println(io, "  Prognostic: b (buoyancy)")
    println(io, "  Diagnostic: φ (streamfunction), u, v (velocities)")
    println(io, "  Scratch   : R, tmp, tmp2 (real), tmpc (spectral)")
    println(io, "  Local size: $(size(fld.b))")
end