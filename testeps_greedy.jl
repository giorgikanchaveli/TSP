include("epsilon_greedy.jl")


param = epsilon_greedy.Params(0,10)
graph, est, sampled_costs = epsilon_greedy.eps_greedy(params = param)

print(sum(sampled_costs))



