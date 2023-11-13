include("Macau.jl")
using Plots
using LinearAlgebra
using ExtractMacro
using Dates
import .Macau: macau2
using Statistics




# range for each of the hyperparameters
β_start_min, β_start_max, n_start = 0.00, 200.0, 100 # n_start is number of β_start we consider in that range 
β_end_min, β_end_max, n_final = 0.1, 1000.0, 100  # n_final is number of β_end we consider in that range 


# from ranges construct vectors for hyperparameters
β_starts = collect(range(β_start_min, β_start_max, n_start))
β_ends = collect(range(β_end_min, β_end_max, n_final))



n_simulate = 10 # number of times we want to simulate data for each hyperparameters




cost_macau = function(;N::Int = N, α::Float64 = α, hyperparams::Macau.hyperparams, seed::Int = 1)
    
    _, _, costs, _, acc = macau2(N = N, α = α, hyperparams = hyperparams, seed = seed)
    return mean(costs), acc
end




# for each of the pairs of hyperparameters we generate the several graphs ( sim ) and compute average cost. Graphs we consider
# are same for each pair of hyperparameter.
optimize_hypparmac = function(β_starts::Vector{Float64}, β_ends::Vector{Float64}; N::Integer = 100, α::Real = 0.0, sim::Int = 10)
    # sim denotes number of times we want to simulate new graph for each pair of hyperparameters
    len1, len2 = length(β_starts), length(β_ends)
    costs = zeros(Float64, len1, len2) # 2 dimensional matrix that records costs for each pair of hyperparameters
    accs = zeros(Float64, len1, len2) # 2 dimensional matrix that records acceptance rate for each pair of hyperparameters
    min_cost = Inf
    best_params = missing
    seeds = rand(10:10000, sim) # for each of the parameter we want to generate same graphs because
                                # this way best hyperparameter won't be biased towards the seeds that where  
                                # generated for that hyperparameter.
    iter = 1
    println("need to do $(length(β_starts)*length(β_ends))")
    for i in 1:len1
        
        for j in 1:len2
           
            t = time()
            β_start = β_starts[i]
            β_end = β_ends[j]
            if β_start >= β_end
                costs[i, j] = Inf
                
            else
                cost = 0.0
                acc = 0.0
                for seed in seeds
                    
                    hyperparameters = Macau.hyperparams(β_start,β_end,100)# 100 here means nothing, step size for beta is fixed in macau file
                    cost_obs, acc_obs = cost_macau(N = N, α = α, hyperparams = hyperparameters, seed = seed)
                   
                    
                    cost += cost_obs
                    acc += acc_obs
                    
                end
               
                accs[i,j] = acc / sim
                costs[i,j] = cost / sim # average regret over several(sim) graphs
                
                if min_cost > costs[i,j]
                    
                    min_cost = costs[i,j]
                    best_params = (i,j)
                end
            end
            #print("time for one pair of hyperparameters is $(time() - t)")
           
            if iter % 10 ==0
                println("iter : $(iter)")
            end
            iter += 1

            
        end
    end
    if !ismissing(best_params)
        #print("Best parameters : $(β_starts[best_params[1]], β_ends[best_params[2]]")
    else
        print("coulnd't find best parameters")
    end
    return best_params, costs,accs
end





# get the argmax of this 2 dimensional matrix
# and it will be our best hyperparameters
println("start")
t = time()
best_params, costs, accs = optimize_hypparmac(β_starts,β_ends,N = 100, α = 0.2, sim = n_simulate)
elapsed = time() - t



heatmap(costs, c=:viridis, xlabel="β_ends", ylabel="β_starts", 
        title="Hyperparameter Cost Matrix", color=:auto, ratio=:equal)