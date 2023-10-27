module Macau

using Random, LinearAlgebra
using ExtractMacro

struct Graph   # TSP problem
    N::Int
    D::Matrix{Float64} 
    coords::Tuple{Vector{Float64}, Vector{Float64}}
    α::Float64   # this parameter is used to generate reward, that is distance + α * unif()
    function Graph(N, α)
        N > 0 || throw(ArgumentError("N must be positive, given: $N"))
        α ≥ 0 || throw(ArgumentError("α must be non-negative, given: $N"))

        x, y = rand(N), rand(N) 
        coords = (x,y)
        D = [√((x[i]-x[j])^2 + (y[i]-y[j])^2) for i = 1:N, j = 1:N]
        return new(N, D, coords, α)
    end
end

mutable struct Est    # encapsulates all useful quantities for inference
    N::Int
    estD::Matrix{Float64}  # 
    route::Vector{Int}
    samples::Matrix{Int}   # matrix that counts for each edge how many times we choose
                           # if we initialize it to 0 then initial estimates of Distance don't matter
            
    Est(N) = new(N, 0.521*(ones(N, N) - I), [i for i = 1:N], zeros(Int, N, N))  # 0.521 is expected distance between two points
                                                               # chosen uniformly in [0,1]^2 
                                                            # zeros MIGHT BE CHANGED
end

mutable struct Params # parameters for the algorithm
    β_start::Float64 # first beta we choose
    β_final::Float64 # last beta we choose
    n_β::Int  # number of betas we want to have
    iters::Int # number of iterations per beta
    show::Bool # if true, you want to print some useful infos
end

function tour_cost(c::Vector{Int}, D::Matrix)
    # c is permutation vector that represents route
    N = length(c)
    @assert size(D) == (N,N)
    cost = sum(D[c[i],c[i+1]] for i = 1:N-1)
    cost += D[c[N],c[1]]
    return cost
end

real_cost(c::Vector{Int}, graph::Graph) = tour_cost(c, graph.D)

est_cost(c::Vector{Int}, est::Est) = tour_cost(c, est.estD)

function sample_cost!(c::Vector{Int}, graph::Graph, est::Est)

    # for route c, we observe cost and update our estimated for distance and other parameters in est object
    
    @extract est : N estD samples
    graph.N == N || throw(ArgumentError("incompatible graphs, graph.N=$(graph.N) est.N=$N"))
    @extract graph : N D coords α
 
    est_cost = 0.0 # estimated cost of route based on our estimated distance after updating it 
    obs_cost = 0.0 # observed cost of route. This will be reward for that route
    for i = 1:N
        c1, c2 = c[i], c[mod1(i+1, N)]
        cost1 = max(D[c1,c2] * (1 + α * randn()), 0.0)
        s = samples[c1,c2]
        estD[c1,c2] = (estD[c1,c2] * s + cost1) / (s + 1)
        estD[c2,c1] = estD[c1,c2]
        samples[c1,c2] += 1
        samples[c2,c1] = samples[c1,c2]
        est_cost += estD[c1,c2]
        obs_cost += cost1
    end
    return obs_cost, est_cost
end

function propose_move(c::Vector{Int}, est::Est)
    
    @extract est : N estD

    local a::Int, b::Int
    while true
        a, b = rand(1:N), rand(1:N)
        a == b && continue
        a > b && ((a,b) = (b,a))
        ((a,b) == (1,N) || (a,b) == (2,N) || (a,b) == (1,N-1)) && continue
        break
    end

    i0 = c[mod1(a-1, N)]
    i1 = c[a]
    j0 = c[b]
    j1 = c[mod1(b+1, N)]

    Δcost = (estD[i0,j0] + estD[i1,j1]) - (estD[i0,i1] + estD[j0,j1])
    return Δcost, (a,b)
end

function accept_move!(c::Vector{Int}, move::Tuple{Int,Int})
    a, b = move

    d = b - a
    @assert d > 0
    for i = 0:(d÷2)
        c[a+i], c[b-i] = c[b-i], c[a+i]
    end
    return c
end


# might change so that macau receives graph object
function macau(;N::Integer = 10, params::Params = Params(1.0,10.0,100, 10000, false),
               α::Real = 0.0, seed::Int = 716464734, initreal::Bool = false)
    Random.seed!(seed)
    graph = Graph(N, α)
    est = Est(N)


    @extract params : β_start β_final n_β iters show
    

    if initreal
        copyto!(est.estD, graph.D)
    end

    c = est.route
    cost = est_cost(c, est) 
    sampled_cost = Inf
    sampled_costs = Array{Float64}(undef, n_β*iters)
    
    acc = 0
    i = 1
    for β in range(β_start, β_final, n_β)
        
        acc = 0 
        for iter = 1:iters
            Δcost, move = propose_move(c, est)
           # println("prob = $(exp(-β * Δcost))")
            if Δcost ≤ 0 || rand() < exp(-β * Δcost)
                accept_move!(c, move)
                # cost += Δcost
                # estcost = est_cost(c, est)
                # @assert cost ≈ estcost
                #sampled_cost, cost = sample_cost!(c, graph, est)
                acc += 1
            end
            sampled_cost, cost = sample_cost!(c, graph, est)
            sampled_costs[i] = sampled_cost # new reward
            i += 1
        end
        if show
            println("β=$β, acc=$(acc/iters),est_cost=$cost, sampled_cost=$sampled_cost")
        end
    end
    
    realcost = real_cost(c, graph)
    
    return graph, est, sampled_costs, realcost
end

end # module


