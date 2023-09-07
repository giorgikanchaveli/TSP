include("Macau.jl")
using Plots


# cost is cost based on our estimated distance matrix
# realcost is cost based on true distance matrix
# sampled_costs are costs observed at each iteration
c, cost, realcost, sampled_costs, true_d,est_d = Macau.macau(α = 0.5)


plot(sampled_costs)
