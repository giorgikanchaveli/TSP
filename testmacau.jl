include("Macau.jl")
using Plots
using LinearAlgebra


# cost is cost based on our estimated distance matrix
# realcost is cost based on true distance matrix
# sampled_costs are costs observed at each iteration

#c, cost, realcost, sampled_costs, true_d,est = Macau.macau(α = 0.5)


#plot(sampled_costs)

distance = function(d1::Matrix{Float64}, d2::Matrix{Float64}) # distance between estimated distance matrix
                                                              # and true distance matrix
    norm(triu(d1) - triu(d2))
end

least_explored = function(s::Matrix{Int64}) # least explored link and it's value (distance)
    index = argmin(s)
    return index.I, s[index]
end

most_explored = function(s::Matrix{Int64}) # most explored link and it's value (distance)
    index = argmax(s)
    return index.I, s[index]
end

est_distance = function(;N::Int = 10, α::Float64 = 0.0, β_start::Float64 = 1.0, β_final::Float64 = 10.0, n_β::Int = 20, iters::Int = 10)
    d = 0.0
    for i in range(1, iters)
        c, cost, realcost, sampled_costs, true_d,est = Macau.macau(N = N, α = α,β_start = β_start, β_final = β_final, n_β = n_β)
        d += distance(est.estD, true_d)
    end
    print("cost of route is $realcost")
    return d / iters 
end

regret = function(;N::Integer = 10, β_start::Real = 1.0, β_final::Real = 10.0, n_β::Int = 100,
    α::Real = 0.0, iters = 10_000, seed::Int = 716464734)
    c0, cost0, realcost0, sampled_costs0, true_d0, est0 = Macau.macau(N = N,β_start = β_start,β_final = β_final,n_β = n_β,α = α,iters = iters, seed = seed)
    c0, cost1, realcost1, sampled_costs1, true_d1, est1 = Macau.macau(N = N,β_start = β_start,β_final = β_final,n_β = n_β,α = 0.0,iters = iters, seed = seed)
    return (realcost0 - realcost1)
end


start_time = time()
est_distance(N = 50, α= 0.25, β_start = 0.0, β_final = 100.1, n_β = 250, iters = 5)
print("time elapsed = $(time() - start_time)")
