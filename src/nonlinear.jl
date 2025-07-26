###############################################################################
# SURFACE SEMI-GEOSTROPHIC EQUATIONS
###############################################################################
#
# Governing equation for buoyancy:
# ∂b/∂t + J(ψ, b) = 0                    (buoyancy conservation)
# u = -∂ψ/∂y,  v = ∂ψ/∂x                 (geostrophic velocities)
#
# Where:
# - b(x,y,t): surface buoyancy anomaly
# - ψ(x,y,t): geostrophic streamfunction  
# - J(ψ,b) = ∂ψ/∂x ∂b/∂y - ∂ψ/∂y ∂b/∂x: Jacobian (advection term)
# - D²ψ: Hessian matrix of ψ
#
# TIME INTEGRATION OPTIONS (from timestep.jl):
# 1. Low-storage 2nd order Adams-Bashforth (AB2-LS)
# 2. 3rd order Runge-Kutta (RK3)
# 3. Spectral filtering for numerical stability
#
###############################################################################

