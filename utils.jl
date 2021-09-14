using LinearAlgebra,Distributions
log2π = log(big(2.)*π)

function MI(Σ_x, Σ_z_t,A)
    n = size(Σ_x)[1]
    0.5 * logdet(I(n) + inv(Σ_x)*A*Σ_z_t*A')
end

function KL(μ_p,μ_q,Σ_p,Σ_q)
    # n is the Distribution Dimension
    n = length(μ_p)
    # Pull out the means, and get the difference
    μdiff = μ_q .- μ_p

    kl = 0.5 * (tr(inv(Σ_p)*Σ_q) + (μdiff' * inv(Σ_p) * μdiff) - n + logdet(Σ_p) - logdet(Σ_q))
    return kl
end

function EFE(μ_p, Σ_p, # Goal prior
	    μ_z_t, Σ_z_t, # State prior predictive
	    Σ_x, A # Observation model
	    )
    # Dimensionality
    n = length(μ_p)

    # Calculate marginal covariance
    Σ_22 = A * Σ_z_t * A' + Σ_x

    # Predicted observation for the next time step
    μ_q = A * μ_z_t

    # Entropy of q(x)
    Hx = 0.5*(n*log2π + logdet(Σ_22) +n)

    # Cross entropy as  KL + H[x]
    kl = KL(μ_p,μ_q,Σ_p,Σ_22)
    ce =  kl + Hx

    # Mutual information
    mi = MI(Σ_x, Σ_z_t, A)

    # G is ce - MI
    G = ce - mi

    # Ambiguity as G - kl
    C = G - kl

    # Return quantities for plotting
    G,ce,mi,kl,Hx,C
end
