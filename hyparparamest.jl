include("Macau.jl")
include("epsilon_greedy.jl")
using Plots
using LinearAlgebra
using ExtractMacro
using Dates
import .Macau: macau
using Statistics


#hyperparameters = Macau.hyperparams(1.0,5.0,50)
#graph, est, sampled_costs, realcost = macau(hyperparams = hyperparameters)

# range for each of the hyperparameters
β_start_min, β_start_max, n_start = 0.00, 200.0, 10 # n_start is number of β_start we consider in that range 
β_end_min, β_end_max, n_final = 0.1, 1000.0, 10  # n_final is number of β_end we consider in that range 
n_β_min, n_β_max, nn = 20, 300, 8 # nn is number of n_β we consider in that range 

# from ranges construct vectors for hyperparameters
β_starts = collect(range(β_start_min, β_start_max, n_start))
β_ends = collect(range(β_end_min, β_end_max, n_final))
n_βs = ceil.(Int, collect(range(n_β_min, n_β_max, nn)))


n_simulate = 10 # number of times we want to simulate data for each hyperparameters





cost_macau = function(;N::Int = N, α::Float64 = α, hyperparams::Macau.hyperparams, seed::Int = 1)
    
    _, _, costs, _ = macau(N = N, α = α, hyperparams = hyperparams, seed = seed)
    return mean(costs)
end





# for each of the pairs of hyperparameters we generate the several graphs ( sim ) and compute average cost. Graphs we consider
# are same for each pair of hyperparameter.
optimize_hypparmac = function(β_starts::Vector{Float64}, β_ends::Vector{Float64}, n_βs::Vector{Int64}; N::Integer = 100, α::Real = 0.0, sim::Int = 10)
    # sim denotes number of times we want to simulate new graph for each pair of hyperparameters
    len1, len2, len3 = length(β_starts), length(β_ends), length(n_βs)
    costs = zeros(Float64, len1, len2, len3) # 3 dimensional matrix that records regrets for each pair of hyperparameters
    min_cost = Inf
    best_params = missing

    seeds = rand(10:10000, sim) # for each of the parameter we want to generate same graphs because
                                # this way best hyperparameter won't be biased towards the seeds that where  
                                # generated for that hyperparameter.
    iter = 0
    println("need to do $(length(β_starts)*length(β_ends)*length(n_βs))")
    for i in 1:len1
        for j in 1:len2
            for k in 1:len3
                t = time()
                β_start = β_starts[i]
                β_end = β_ends[j]
                n_β = n_βs[k]
                if β_start >= β_end
                    costs[i, j, k] = Inf
                    break
                else
                    cost = 0.0

                    for seed in seeds
                        
                        hyperparameters = Macau.hyperparams(β_start,β_end,n_β)
                        cost += cost_macau(N = N, α = α, hyperparams = hyperparameters, seed = seed)
                        
                    end
                    
                    costs[i,j,k] = cost / sim # average regret over several(sim) graphs
                    
                    if min_cost > costs[i,j,k]
                        
                        min_cost = costs[i,j,k]
                        best_params = (i,j,k)
                    end
                end
                #print("time for one pair of hyperparameters is $(time() - t)")
                if iter % 10 ==0
                    println("iter : $(iter)")
                end
                iter += 1

            end
        end
    end
    if !ismissing(best_params)
        #print("Best parameters : $(β_starts[best_params[1]], β_ends[best_params[2]], n_βs[best_params[3]])")
    else
        print("coulnd't find best parameters")
    end
    return best_params, costs
end



# get the argmax of this 3 dimensional matrix
# and it will be our best hyperparameters
println("start")
t = time()
best_params, costs = optimize_hypparmac(β_starts,β_ends,n_βs,N = 100, α = 0.2, sim = n_simulate)
elapsed = time() - t



println("time needed :$(elapsed ÷ 60) minutes and $(elapsed - elapsed ÷ 60) seconds")
println(β_starts[best_params[1]])
println(β_ends[best_params[2]])
println(n_βs[best_params[3]])