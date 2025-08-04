using Plots, FFTW

"""
Make a 2D non-periodic function periodic by extending it with given periods in x and y directions.

Parameters:
- f: the original 2D non-periodic function f(x,y)
- period_x: the desired period in x direction
- period_y: the desired period in y direction  
- method: extension method (:repeat, :reflect, :symmetric, :zero_pad)
"""
function make_periodic_2d(f, period_x, period_y; method=:repeat)
    function periodic_f(x, y)
        # Normalize coordinates to be within their respective periods
        x_norm = mod(x, period_x)
        y_norm = mod(y, period_y)
        
        if method == :repeat
            # Simple periodic repetition in both dimensions
            return f(x_norm, y_norm)
            
        elseif method == :reflect
            # Reflect in both dimensions independently
            half_px, half_py = period_x/2, period_y/2
            
            x_eval = x_norm <= half_px ? x_norm : period_x - x_norm
            y_eval = y_norm <= half_py ? y_norm : period_y - y_norm
            
            return f(x_eval, y_eval)
            
        elseif method == :symmetric
            # 4-fold symmetry: reflect across both axes at period boundaries
            half_px, half_py = period_x/2, period_y/2
            
            # Determine which quadrant we're in and apply appropriate symmetry
            x_quad = x_norm <= half_px ? 1 : 2
            y_quad = y_norm <= half_py ? 1 : 2
            
            x_eval = x_quad == 1 ? x_norm : period_x - x_norm
            y_eval = y_quad == 1 ? y_norm : period_y - y_norm
            
            return f(x_eval, y_eval)
            
        elseif method == :zero_pad
            # Return original function in first period, zero elsewhere
            if 0 <= x_norm <= period_x && 0 <= y_norm <= period_y
                return f(x_norm, y_norm)
            else
                return 0.0
            end
            
        else
            error("Unknown method: $method. Use :repeat, :reflect, :symmetric, or :zero_pad")
        end
    end
    
    return periodic_f
end

"""
2D Fourier series approximation for periodic extension
"""
function fourier_periodic_2d(f, period_x, period_y, n_terms_x=10, n_terms_y=10; 
                             domain_x=(0, nothing), domain_y=(0, nothing))
    
    # Set default domain bounds
    x_start, x_end = domain_x[1], domain_x[2] === nothing ? period_x : domain_x[2]
    y_start, y_end = domain_y[1], domain_y[2] === nothing ? period_y : domain_y[2]
    
    # 2D numerical integration using Simpson's rule
    function integrate_2d(func, x_bounds, y_bounds, nx=50, ny=50)
        x_start, x_end = x_bounds
        y_start, y_end = y_bounds
        
        hx, hy = (x_end - x_start)/nx, (y_end - y_start)/ny
        
        total = 0.0
        for i in 0:nx, j in 0:ny
            x, y = x_start + i*hx, y_start + j*hy
            
            # Simpson's rule weights
            wx = i == 0 || i == nx ? 1 : (i % 2 == 0 ? 2 : 4)
            wy = j == 0 || j == ny ? 1 : (j % 2 == 0 ? 2 : 4)
            
            total += wx * wy * func(x, y)
        end
        
        return (hx * hy / 9) * total
    end
    
    # Calculate 2D Fourier coefficients
    A = zeros(2*n_terms_x+1, 2*n_terms_y+1)
    
    for m in -n_terms_x:n_terms_x, n in -n_terms_y:n_terms_y
        coeff_func(x, y) = f(x, y) * exp(-2π*im*(m*x/period_x + n*y/period_y))
        A[m+n_terms_x+1, n+n_terms_y+1] = integrate_2d(coeff_func, (x_start, x_end), (y_start, y_end)) / (period_x * period_y)
    end
    
    # Return the 2D Fourier series
    function fourier_2d(x, y)
        result = 0.0 + 0.0im
        for m in -n_terms_x:n_terms_x, n in -n_terms_y:n_terms_y
            result += A[m+n_terms_x+1, n+n_terms_y+1] * exp(2π*im*(m*x/period_x + n*y/period_y))
        end
        return real(result)
    end
    
    return fourier_2d
end

"""
Create a 2D periodic function from discrete data (like an image)
"""
function make_periodic_2d_discrete(data::Matrix; method=:repeat)
    rows, cols = size(data)
    
    function periodic_discrete(x, y)
        # Convert continuous coordinates to discrete indices
        i = mod(round(Int, x), rows) + 1
        j = mod(round(Int, y), cols) + 1
        
        # Ensure indices are within bounds
        i = clamp(i, 1, rows)
        j = clamp(j, 1, cols)
        
        if method == :repeat
            return data[i, j]
        elseif method == :reflect
            # Reflect indices if needed
            i_refl = i <= rows÷2 ? i : rows - i + 1
            j_refl = j <= cols÷2 ? j : cols - j + 1
            return data[clamp(i_refl, 1, rows), clamp(j_refl, 1, cols)]
        else
            return data[i, j]
        end
    end
    
    return periodic_discrete
end

"""
Visualize 2D periodic function
"""
function plot_2d_periodic(f, period_x, period_y; 
                          x_range=(-period_x, 2*period_x), 
                          y_range=(-period_y, 2*period_y),
                          resolution=100,
                          title="2D Periodic Function")
    
    x_vals = range(x_range[1], x_range[2], length=resolution)
    y_vals = range(y_range[1], y_range[2], length=resolution)
    
    z_vals = [f(x, y) for y in y_vals, x in x_vals]
    
    p = heatmap(x_vals, y_vals, z_vals, 
                aspect_ratio=:equal, 
                title=title,
                xlabel="x", ylabel="y",
                colorbar_title="f(x,y)")
    
    # Add grid lines to show period boundaries
    for i in -2:2
        vline!(p, [i * period_x], color=:white, alpha=0.7, linewidth=1, label="")
        hline!(p, [i * period_y], color=:white, alpha=0.7, linewidth=1, label="")
    end
    
    return p
end

# Demonstration functions
function demo_2d_gaussian()
    println("Demo: 2D Gaussian function made periodic")
    
    # Original 2D Gaussian
    gaussian_2d(x, y) = exp(-((x-π)^2 + (y-π)^2)/2)
    
    period_x, period_y = 2π, 2π
    
    # Create different periodic versions
    periodic_repeat = make_periodic_2d(gaussian_2d, period_x, period_y, method=:repeat)
    periodic_reflect = make_periodic_2d(gaussian_2d, period_x, period_y, method=:reflect)
    periodic_symmetric = make_periodic_2d(gaussian_2d, period_x, period_y, method=:symmetric)
    
    # Create plots
    p1 = plot_2d_periodic(periodic_repeat, period_x, period_y, title="2D Gaussian - Repeat")
    p2 = plot_2d_periodic(periodic_reflect, period_x, period_y, title="2D Gaussian - Reflect")
    p3 = plot_2d_periodic(periodic_symmetric, period_x, period_y, title="2D Gaussian - Symmetric")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
    display(combined_plot)
    
    return combined_plot
end

function demo_2d_wave_interference()
    println("Demo: 2D wave interference pattern made periodic")
    
    # Wave interference pattern
    wave_2d(x, y) = sin(x) * cos(y) + 0.5 * sin(2*x + y)
    
    period_x, period_y = 2π, 2π
    
    # Create periodic version and Fourier approximation
    periodic_wave = make_periodic_2d(wave_2d, period_x, period_y, method=:repeat)
    fourier_wave = fourier_periodic_2d(wave_2d, period_x, period_y, 15, 15)
    
    # Create plots
    p1 = plot_2d_periodic(periodic_wave, period_x, period_y, title="Wave Pattern - Original Periodic")
    p2 = plot_2d_periodic(fourier_wave, period_x, period_y, title="Wave Pattern - Fourier Series")
    
    combined_plot = plot(p1, p2, layout=(1,2), size=(800, 400))
    display(combined_plot)
    
    return combined_plot
end

function demo_2d_step_function()
    println("Demo: 2D step function (square) made periodic")
    
    # 2D step function (square in center)
    step_2d(x, y) = (1 <= x <= 3 && 1 <= y <= 3) ? 1.0 : 0.0
    
    period_x, period_y = 4, 4
    
    # Create different periodic versions
    periodic_step = make_periodic_2d(step_2d, period_x, period_y, method=:repeat)
    symmetric_step = make_periodic_2d(step_2d, period_x, period_y, method=:symmetric)
    
    # Create plots
    p1 = plot_2d_periodic(periodic_step, period_x, period_y, title="2D Step - Repeat")
    p2 = plot_2d_periodic(symmetric_step, period_x, period_y, title="2D Step - Symmetric")
    
    combined_plot = plot(p1, p2, layout=(1,2), size=(800, 400))
    display(combined_plot)
    
    return combined_plot
end

function demo_discrete_image_periodic()
    println("Demo: Making discrete data (image-like) periodic")
    
    # Create a simple 2D pattern
    pattern = zeros(20, 20)
    for i in 1:20, j in 1:20
        pattern[i, j] = sin(i/3) * cos(j/3) + 0.5 * exp(-((i-10)^2 + (j-10)^2)/50)
    end
    
    # Make it periodic
    periodic_pattern = make_periodic_2d_discrete(pattern, method=:repeat)
    
    # Create extended grid to show periodicity
    extended_x = 1:60
    extended_y = 1:60
    extended_pattern = [periodic_pattern(x, y) for y in extended_y, x in extended_x]
    
    p = heatmap(extended_x, extended_y, extended_pattern,
               aspect_ratio=:equal,
               title="Discrete Pattern Made Periodic",
               xlabel="x", ylabel="y")
    
    # Add grid lines to show original pattern boundaries
    for i in 0:3
        vline!(p, [i*20 + 0.5], color=:white, alpha=0.7, linewidth=2, label="")
        hline!(p, [i*20 + 0.5], color=:white, alpha=0.7, linewidth=2, label="")
    end
    
    display(p)
    return p
end

# Run all demonstrations
println("Creating 2D periodic function demonstrations...")

# Generate all demo plots
demo_2d_gaussian()
demo_2d_wave_interference()
demo_2d_step_function()
demo_discrete_image_periodic()

# Example of direct usage
println("\nExample: Direct evaluation of 2D periodic functions")
original_2d(x, y) = x^2 + y^2
periodic_2d_func = make_periodic_2d(original_2d, 2π, 2π)

test_points = [(0, 0), (π, π), (2π, 0), (3π, 2π), (4π, 4π)]
for (x, y) in test_points
    orig_val = original_2d(mod(x, 2π), mod(y, 2π))
    periodic_val = periodic_2d_func(x, y)
    println("(x,y) = ($x, $y): original = $orig_val, periodic = $periodic_val")
end