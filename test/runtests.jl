#!/usr/bin/env julia

# Main test runner for SSG.jl
println("Running SSG.jl Test Suite")
println("=" ^ 50)

# Run Poisson solver tests
println("\n🧮 POISSON SOLVER TESTS")
println("-" ^ 30)
include("test_poisson.jl")

println("\n⏰ TIME INTEGRATION TESTS") 
println("-" ^ 30)
include("test_timestep.jl")

println("\n" ^ 2 * "=" ^ 50)
println("✅ All SSG.jl tests completed successfully!")