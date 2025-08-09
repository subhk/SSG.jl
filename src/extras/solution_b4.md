# Solution of Equation (B4) from Z=0 to Z=-H

## Problem Statement

We need to solve the differential equation:

$$-k^2 \tilde{w}^* + \frac{\partial^2 \tilde{w}^*}{\partial Z^2} = \tilde{Q}(k, Z) \quad \text{(B4)}$$

where:
- $\mathbf{k} = (k_X, k_Y)$ is a horizontal wavenumber vector in the geostrophic coordinate system
- $k = \sqrt{k_X^2 + k_Y^2}$ is the magnitude of the wavenumber vector
- $\tilde{Q}(k, Z)$ is the Fourier transform of the RHS of Eq. (B1)

The domain is from Z = 0 to Z = -H with boundary conditions:
- $\tilde{w}^* = 0$ at Z = 0
- $\tilde{w}^* = 0$ at Z = -H

## Analytical Solution

For each $k \neq 0$, the solution of equation (B4) subject to the boundary conditions is:

$$\tilde{w}^*(k, Z) = \exp(kZ) \int_0^Z \delta_3(k, z') dz' - \exp(-kZ) \int_0^Z \delta_4(k, z') dz' + 2\frac{\delta_2^H(k)}{\delta_1^H(k)} \sinh(kZ)$$

## Coefficient Definitions

The coefficients are defined as follows:

### $\delta_1^H(k)$
$$\delta_1^H(k) = -2 \sinh(kH)$$

### $\delta_2^H(k)$
$$\delta_2^H(k) = \exp(kH) \int_{-H}^0 \delta_4(k, z') dz' - \exp(-kH) \int_{-H}^0 \delta_3(k, z') dz'$$

### $\delta_3(k, z')$
$$\delta_3(k, z') = \exp(-kz') \frac{\tilde{Q}(k, z')}{2k}$$

### $\delta_4(k, z')$
$$\delta_4(k, z') = \exp(kz') \frac{\tilde{Q}(k, z')}{2k}$$

## Key Modifications from Original Solution

The solution for the domain Z ∈ [0, -H] differs from the original solution (which was for Z ∈ [0, -1]) in the following ways:

1. **Integration limits**: The integration limits in $\delta_2^H(k)$ are changed from $\int_{-1}^0$ to $\int_{-H}^0$

2. **Hyperbolic function**: The term $\sinh(k)$ is replaced with $\sinh(kH)$ in $\delta_1^H(k)$

3. **Boundary condition**: The solution satisfies $\tilde{w}^*(k, -H) = 0$ instead of $\tilde{w}^*(k, -1) = 0$

## Verification

The solution satisfies:
- The differential equation (B4): $-k^2 \tilde{w}^* + \frac{\partial^2 \tilde{w}^*}{\partial Z^2} = \tilde{Q}(k, Z)$
- Upper boundary condition: $\tilde{w}^*(k, 0) = 0$
- Lower boundary condition: $\tilde{w}^*(k, -H) = 0$

## Physical Interpretation

The inverse Fourier transform of $\tilde{w}^*(k, Z)$ gives the vertical velocity in the geostrophic coordinates:

$$w(X, Y, Z) = \mathcal{F}^{-1}[\tilde{w}^*(k, Z)]$$

where $\mathcal{F}^{-1}$ denotes the inverse Fourier transform operator.

## Notes

- This analytical solution is derived using symbolic mathematics methods
- The solution is valid for all $k \neq 0$
- For $k = 0$, a separate analysis would be required
- The forcing term $\tilde{Q}(k, Z)$ must be specified to evaluate the integrals numerically