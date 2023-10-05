include("Macau.jl")
include("epsilon_greedy.jl")
using Plots
using LinearAlgebra
using ExtractMacro


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


regret = function(params::Macau.Params,
                ;N::Integer = 10, α::Real = 0.0, seed::Int = 716464734)
    # compares real cost of last routes obtained by algorithm in case of alpha > 0 and alpha = 0 

    c0, cost0, realcost0, sampled_costs0, true_d0, est0 = Macau.macau(N = N, α = α, params = params, seed = seed)
    c0, cost1, realcost1, sampled_costs1, true_d1, est1 = Macau.macau(N = N, α = 0.0, params = params, seed = seed)
    return (realcost0 - realcost1)
end

regret = function(params::epsilon_greedy.Params,
                ;N::Integer = 10, α::Real = 0.0, seed::Int = 716464734)
# compares real cost of last routes obtained by algorithm in case of alpha > 0 and alpha = 0 

    c0, cost0, realcost0, sampled_costs0, true_d0, est0 = epsilon_greedy.eps_greedy(N = N, α = α, params = params, seed = seed)
    c0, cost1, realcost1, sampled_costs1, true_d1, est1 = epsilon_greedy.eps_greedy(N = N, α = 0.0, params = params, seed = seed)
    return (realcost0 - realcost1)
end

params_macau = Macau.Params(1.0, 10.0, 100, 100, false)
params_eps_greedy = epsilon_greedy.Params(0.01, 10000)
regret_macau = regret(params_macau, α=0.5)
regret_eps_greedy = regret(params_eps_greedy) # gives error , ?




seed = 1234
N = 500

param = epsilon_greedy.Params(0.1,10000) # even if epsilon = 0.0 it's better, why?
graph, est_eps, costs_eps = epsilon_greedy.eps_greedy(N = 100, params = param,seed = seed)



param = Macau.Params(1.0, 10.0, 100, 100, false)
c, cost, realcost, costs_macau, d, est_macau = Macau.macau(N = 100, params = param,seed = seed)
print("sum of costs for epsilon_greedy is $(sum(costs_eps))")
print("sum of costs for epsilon_greedy is $(sum(costs_macau))")


print("distance between true distance matrix and estimated distance matrix from eps_greedy $(distance(graph.D, est_eps.estD))")
print("distance between true distance matrix and estimated distance matrix from eps_greedy $(distance(graph.D, est_macau.estD))")