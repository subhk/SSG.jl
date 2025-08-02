using Plots

"""
Make a non-periodic function periodic by extending it with a given period.

Parameters:
- f: the original non-periodic function
- period: the desired period for the periodic extension
- method: extension method (:repeat, :reflect, or :zero_pad)
"""
function make_periodic(f, period; method=:repeat)
    function periodic_f(x)
        # Normalize x to be within [0, period)
        x_normalized = mod(x, period)
        
        if method == :repeat
            # Simple periodic repetition
            return f(x_normalized)
        elseif method == :reflect
            # Reflect the function at the boundaries for smoother transitions
            half_period = period / 2
            if x_normalized <= half_period
                return f(x_normalized)
            else
                # Reflect around the half-period point
                return f(period - x_normalized)
            end
        elseif method == :zero_pad
            # Return zero outside the original domain, then repeat
            if 0 <= x_normalized <= period
                return f(x_normalized)
            else
                return 0.0
            end
        else
            error("Unknown method: $method. Use :repeat, :reflect, or :zero_pad")
        end
    end
    
    return periodic_f
end

"""
Alternative approach: Make periodic using Fourier series approximation
"""
function fourier_periodic(f, period, n_terms=10; domain_start=0, domain_end=nothing)
    if domain_end === nothing
        domain_end = period
    end
    
    # Calculate Fourier coefficients
    function integrate_simpson(func, a, b, n=1000)
        h = (b - a) / n
        x = a:h:b
        y = func.(x)
        return h/3 * (y[1] + 4*sum(y[2:2:end-1]) + 2*sum(y[3:2:end-2]) + y[end])
    end
    
    # Calculate a0 (DC component)
    a0 = (2/period) * integrate_simpson(x -> f(x), domain_start, domain_end)
    
    # Calculate an and bn coefficients
    an = zeros(n_terms)
    bn = zeros(n_terms)
    
    for n in 1:n_terms
        an[n] = (2/period) * integrate_simpson(x -> f(x) * cos(2π*n*x/period), domain_start, domain_end)
        bn[n] = (2/period) * integrate_simpson(x -> f(x) * sin(2π*n*x/period), domain_start, domain_end)
    end
    
    # Return the Fourier series approximation
    function fourier_series(x)
        result = a0/2
        for n in 1:n_terms
            result += an[n] * cos(2π*n*x/period) + bn[n] * sin(2π*n*x/period)
        end
        return result
    end
    
    return fourier_series
end

# Example usage and demonstration
function demo_periodic_functions()
    # Define a non-periodic function (e.g., a Gaussian pulse)
    original_func(x) = exp(-(x-2)^2) * sin(x)
    
    period = 4π
    
    # Create periodic versions using different methods
    periodic_repeat = make_periodic(original_func, period, method=:repeat)
    periodic_reflect = make_periodic(original_func, period, method=:reflect)
    periodic_fourier = fourier_periodic(original_func, period, 20, domain_end=period)
    
    # Plot comparison
    x_range = -2*period:0.1:2*period
    
    p1 = plot(x_range, original_func.(mod.(x_range, period)), 
              label="Original (repeated)", linewidth=2, title="Periodic Extensions")
    plot!(p1, x_range, periodic_repeat.(x_range), 
          label="Simple Repeat", linewidth=2, linestyle=:dash)
    plot!(p1, x_range, periodic_reflect.(x_range), 
          label="Reflected", linewidth=2, linestyle=:dot)
    plot!(p1, x_range, periodic_fourier.(x_range), 
          label="Fourier Series", linewidth=2, linestyle=:dashdot)
    
    # Add vertical lines to show period boundaries
    for i in -2:2
        vline!(p1, [i*period], color=:gray, alpha=0.3, label="")
    end
    
    xlabel!(p1, "x")
    ylabel!(p1, "f(x)")
    
    return p1
end

# Example with a step function
function demo_step_function()
    # Define a step function
    step_func(x) = x < 2π ? 1.0 : 0.0
    
    period = 4π
    
    # Create periodic versions
    periodic_step = make_periodic(step_func, period)
    fourier_step = fourier_periodic(step_func, period, 50, domain_end=period)
    
    x_range = -period:0.05:2*period
    
    p2 = plot(x_range, periodic_step.(x_range), 
              label="Periodic Step", linewidth=2, title="Step Function - Periodic Extension")
    plot!(p2, x_range, fourier_step.(x_range), 
          label="Fourier Approximation", linewidth=2, linestyle=:dash)
    
    # Add vertical lines to show period boundaries
    for i in -1:2
        vline!(p2, [i*period], color=:gray, alpha=0.3, label="")
    end
    
    xlabel!(p2, "x")
    ylabel!(p2, "f(x)")
    
    return p2
end

# Run demonstrations
println("Creating periodic function demonstrations...")

# Generate plots
plot1 = demo_periodic_functions()
plot2 = demo_step_function()

# Display plots
display(plot1)
display(plot2)

# Example of using the functions directly
println("\nExample: Evaluating periodic functions at specific points")
original(x) = x^2
periodic_func = make_periodic(original, 2π)

test_points = [0, π, 2π, 3π, 4π]
for x in test_points
    println("x = $x: original(mod(x,2π)) = $(original(mod(x,2π))), periodic(x) = $(periodic_func(x))")
end