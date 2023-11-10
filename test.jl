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

# params_macau = Macau.Params(1.0, 10.0, 100, 100, false)
# params_eps_greedy = epsilon_greedy.Params(0.01, 10000)
# regret_macau = regret(params_macau, α=0.5)


# regret_eps_greedy = regret(params_eps_greedy) # gives error , ?




test_d_rewards = function(N::Int, param_eps::epsilon_greedy.Params, param_macau::Macau.Params, α::Float64, seed::Int, initreal::Bool)
    graph, est_eps, costs_eps, real_cost_eps = epsilon_greedy.eps_greedy(N=N, params = param_eps, α = α, seed = seed, initreal = true) 
    println("sum of costs for epsilon greedy : $(sum(costs_eps))")
    println("real cost of route given by epsilon greedy : $(real_cost_eps)")
    println("distance between true distance matrix and estimated distance matrix from epsilon greedy : $(distance(graph.D, est_eps.estD))")

    graph, est_macau, costs_macau, real_cost_macau = Macau.macau(N = N, params = param_macau, α = α, seed = seed, initreal = true)
    println("sum of costs for macau : $(sum(costs_macau))")
    println("real cost of route given by macau : $(real_cost_macau)")
    println("distance between true distance matrix and estimated distance matrix from macau : $(distance(graph.D, est_macau.estD))")

    


    
    # fill other epsilon costs with minimum
    x = minimum(costs_eps)
    while length(costs_eps) < length(costs_macau)
        push!(costs_eps, x)
    end
    
    # Plots
    rewards = plot(costs_eps, title = "rewards per routes", lc = "black", label = "epsilon greedy", alpha = 1.0, size = (800,500), linewidth=2)
    plot!(rewards, costs_macau, label =  "macau", lc = "red", linewidth=2)
    
    

    differences = costs_eps .- costs_macau
    first_positive = argmax(differences.>0)
    
    print("first index where macau is better is $(first_positive)")
    scatter!(rewards, [first_positive], [costs_macau[first_positive]], color = "green", label = "first time macau is better", markersize = 5, linewidth=2)
   # display(rewards)

    cum_rewards = plot(cumsum(costs_eps), title = "cumulative rewards", linewidth=2,lc = "red", label = "epsilon greedy", size = (800,500), alpha = 1.0)
    plot!(cum_rewards, cumsum(costs_macau), label = "macau",lc = "blue", alpha = 1.0, linewidth=2)
    #display(cum_rewards)

    plots = plot(rewards, cum_rewards, layout = (1,2))


    #plots = plot(rewards, cum_rewards, layout = (1,2))

    #display(plots)

    return (graph, est_eps.route, est_macau.route, plots)

end


plot_route = function(graph::Macau.Graph, r::Vector{Int})
    coords = graph.coords # tuple of vectors

    cities_x = [city_x for city_x in coords[1]]
    cities_y = [city_y for city_y in coords[2]]
    push!(r, r[1])
    tour_x = [city_x for city_x in coords[1][r]]
    tour_y = [city_y for city_y in coords[2][r]]

    # Plot the cities
    scatter(cities_x, cities_y, label="Cities")

# Plot the TSP tour
    plot!(tour_x, tour_y, label="Route", line=:arrow, legend = false)
    
    annotate!([(tour_x[1], tour_y[1] + 0.04, text(string(1)))])
    annotate!([(tour_x[2], tour_y[2] + 0.04, text(string(2)))])
    annotate!([(tour_x[graph.N], tour_y[graph.N] + 0.02, text(string(graph.N)))])
# Add labels to the cities
#    for (i, (x, y)) in enumerate(coords)
#        annotate!([(x, y + 0.02, text(string(i)))])
#    end

# Set plot attributes
    xlabel!("X")
    ylabel!("Y")
    title!("Traveling Salesman Problem")
    
end








# α = 0.0 and distance is known

seed = 134241
N = 200
α = 0.0

# parameters for eps_greedy
ϵ = 0.1
iters = 1500
initreal = true

param_eps = epsilon_greedy.Params(ϵ,iters) # even if epsilon = 0.0 it's better, why?

# parameters for macau
β_0 = 0.0
β_1 = 600.0
n_β = 300
mcmc_iter = 2000
show = false
initreal = true

param_macau = Macau.Params(β_0, β_1, n_β, mcmc_iter, show)

#@assert mcmc_iter*n_β == param_eps.iters



#test_d_rewards(N, param_eps, param_macau, α, seed, initreal)



# α = 0.2 and distance is unknown

seed = 134241
N = 200
α = 0.2

# parameters for eps_greedy
ϵ = 0.1
iters = 1500
initreal = false

param_eps = epsilon_greedy.Params(ϵ,iters) # even if epsilon = 0.0 it's better, why?

# parameters for macau
β_0 = 0.0
β_1 = 600.0
n_β = 300
mcmc_iter = 2000
show = false
initreal = false

param_macau = Macau.Params(β_0, β_1, n_β, mcmc_iter, show)

#@assert mcmc_iter*n_β == param_eps.iters



graph, r_eps, r_macau, plots_rewards = test_d_rewards(N, param_eps, param_macau, α, seed, initreal)

# convert route of macau so that epsilon and macau start at the same city

# find index of first city in epsilon route in macau's route
first_city = r_eps[1]
index_first_city = collect(1:graph.N)[r_macau .== first_city][1]
r_macau[1:graph.N - index_first_city+1], r_macau[graph.N - index_first_city+2:graph.N] = r_macau[index_first_city:graph.N], r_macau[1:index_first_city-1]




pl_eps = plot_route(graph, r_eps)
title!(pl_eps, "epsilon greedy")

pl_macau = plot_route(graph, r_macau)
title!(pl_macau, "macau")


plot!(pl_eps, size=(800, 600))
plot!(pl_macau, size=(800, 600))



plots_routes = plot(pl_eps, pl_macau, layout = (1,2))
plots = plot(plots_rewards, plots_routes, layout = (2,1))

display(plots)

