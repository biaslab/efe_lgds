using LinearAlgebra
include("utils.jl")
log2π = log(big(2.)*π)


# Dimensionality of state
n=2

# Goal prior
Σ_p = I(n) * 3 # Goal prior variance
μ_p = ones(n) * 3.

# Observation model
Σ_x = I(n) #* 0.1 # Observation noise
A  = I(n)

# Prior state
Σ_z_t_min = I(n)# * 0.6
μ_z_t_min = ones(n) * 1.0

# State transition
# Transition noise
Σ_z = I(n) #* 0.5
# Candidate transition matrices
B1 = I(n) * 0.1
B2 = I(n) * 1.0
B3 = I(n) * 10.
B4 = I(n) * 100.
Bs = [B1,B2,B3,B4]

for B in Bs
    Σ_z_t = B * Σ_z_t_min * B' + Σ_z
    μ_z_t = B * μ_z_t_min

    # Recast to float instead of bigfloat and round
    G,ce,mi,kl,Hx,C = round.(Float64.(EFE(μ_p, Σ_p, μ_z_t,Σ_z_t,Σ_x,A)),sigdigits=4)

    println("MI: ",-mi,"\nCE: ",ce, "\nG: ",G,"\nKL: ",kl,"\nHx: ",Hx, "\nC: ", C)
    println("############ \n")
end

