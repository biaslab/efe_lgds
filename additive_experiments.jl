using LinearAlgebra,OhMyREPL
include("utils.jl")
log2π = log(big(2.)*π)

# Dimensionality of state
n=2

# Goal prior
Σ_p = I(n) * 1.0 # Goal prior variance
μ_p = ones(n) * 3.0

# Observation model
Σ_x = I(n) * 1.0 # Observation noise
A  = I(n)

# Prior state
Σ_z_t_min = I(n) * 1.0
μ_z_t_min = ones(n) * 0.0

# State transition
# Transition noise
Σ_z = I(n) * 1.0
# Transition matrix
B = I(n)

# Control variances
Σ_u_1 = I(n)
Σ_u_2 = I(n)
Σ_u_3 = I(n) * 2
Σ_u_4 = I(n) * 2
Σ_us = [Σ_u_1,Σ_u_2,Σ_u_3,Σ_u_4]



# Control means
u1 = ones(n) * 0.
u2 = ones(n) * 2.
u3 = ones(n) * 0.
u4 = ones(n) * 2.
us = [u1,u2,u3,u4]

for i in 1:4
    Σ_z_t = B * Σ_z_t_min * B' + Σ_z + Σ_us[i]
    μ_z_t = B * μ_z_t_min + us[i]

    # Recast to float instead of bigfloat and round
    G,ce,mi,kl,Hx,C = round.(Float64.(EFE(μ_p, Σ_p, μ_z_t,Σ_z_t,Σ_x,A)),sigdigits=4)

    println("MI: ",-mi,"\nCE: ",ce, "\nG: ",G,"\nKL: ",kl,"\nHx: ",Hx, "\nC: ", C)
    println("############ \n")
end

