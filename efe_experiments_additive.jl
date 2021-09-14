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
Σ_z_t_min = I(n) * 1.0
μ_z_t_min = ones(n) * 0.0

# State transition
# Transition noise
Σ_z = I(n)
# Transition matrix
B = I(n) * 1.

Θ_1 = [[0.,0.],I(n)]
Θ_2 = [[2.,2.],I(n)]
Θ_3 = [[0.,0.],I(n)*2.]
Θ_4 = [[2.,2.],I(n)*2.]
Θs = [Θ_1,Θ_2,Θ_3,Θ_4]

for i in 1:4
    # Compute prior predictive for next timestep
    Σ_z_t = B * Σ_z_t_min * B' + Θs[i][2] + Σ_z
    μ_z_t = B * μ_z_t_min + Θs[i][1]

    # Recast to float instead of bigfloat and round
    G,ce,mi,kl,Hx,C = round.(Float64.(EFE(μ_p, Σ_p, μ_z_t,Σ_z_t,Σ_x,A)),sigdigits=4)

    println("KL: ",kl, "\nC: ", C, "\nG: ",G ,"\nCE: ",ce, "\nMI: ",-mi,"\nHx: ",Hx)
    println("############ \n")
end

