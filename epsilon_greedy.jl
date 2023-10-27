module epsilon_greedy

using Random, LinearAlgebra
using ExtractMacro

struct Graph # TSP problem
    N::Int
    D::Matrix{Float64}
    coords::Tuple{Vector{Float64}, Vector{Float64}}
    α::Float64   # this parameter is used to generate reward, that is distance + α * unif()

    function Graph(N,α)
        N > 0 || throw(ArgumentError("N must be positive, given: $N"))
        α ≥ 0 || throw(ArgumentError("α must be non-negative, given: $N"))

        x, y = rand(N), rand(N)
        coords = (x,y)
        D = [√((x[i]-x[j])^2 + (y[i]-y[j])^2) for i = 1:N, j = 1:N]
        return new(N,D,coords,α)
    end
end

mutable struct Est # object that encapsulates all the variables of interest for inference
    N::Int
    estD::Matrix{Float64}  # 
    route::Vector{Int}
    samples::Matrix{Int}   # matrix that counts for each edge how many times we choose
                           # if we initialize it to 0 then initial estimates of Distance don't matter
    Est(N) = new(N, 0.521*(ones(N, N) - I),[i for i=1:N], zeros(Int, N, N)) # 0.521 is expected distance between two points
                                                        # chosen uniformly in [0,1]^2 
                                                        # zeros MIGHT BE CHANGED because in this case 
                                                        # initial values for estimated distance don't matter once we observe
                                                        # link.                                                
end



mutable struct Params # parameters for the algorithm
    ϵ::Float64 # probability of choosing random route
    iters::Int # number of iterations 
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


possible_moves = function(N::Int) # Given number of Cities, returns list of acceptible moves which are
                                  # tuple of cities that define links to be swapped.
    N > 2 || throw(ArgumentError("N must be larger than 2, given: $N"))
    moves_1 = [(1,i) for i in range(2,N-2)]
    moves_2 = [(2,i) for i in range(3, N - 1)]
    moves_others = [(i, j) for i = 3:N for j = i+1:N] # other moves from city 3
    append!(moves_1, moves_2)
    append!(moves_1, moves_others)
    return moves_1
end

check_moves = function(N::Int, moves::Vector{Tuple{Int64, Int64}}) # checks that each move is acceptible
    for move in moves
        move[1] < move[2] || throw(ArgumentError("first element must be less than secibd element"))
        (move == (1,N) || move == (2,N) || move == (1,N-1)) && throw(ArgumentError("$move isn't valid"))
    end
    print("moves are acceptible")
end



sample_cost! = function(c::Vector{Int},graph::Graph, est::Est)
    @extract est : N estD samples
    graph.N == N || throw(ArgumentError("incompatible graphs, graph.N=$(graph.N) est.N=$N"))
    @extract graph : N D coords α

    total_cost = 0.0

    for i in 1:N
        city1 = c[i]
        city2 = c[mod1(i+1, N)]
        cost = max(D[city1,city2] * (1 + α * randn()), 0.0)

        # update estimates
        s = samples[city1,city2]
        estD[city1,city2] = (estD[city1,city2] * s + cost) / (s + 1)
        estD[city2,city1] = estD[city1,city2]
        samples[city1,city2] += 1
        samples[city2,city1] = samples[city1,city2]

        total_cost += cost
    end
    return total_cost
end



delta_cost = function(c::Vector{Int}, move, d::Matrix{Float64}) # given route and move, computes delta cost for that move
    N = length(c)
    a,b = move
    i0 = c[mod1(a-1, N)]
    i1 = c[a]
    j0 = c[b]
    j1 = c[mod1(b+1, N)]

    Δcost = (d[i0,j0] + d[i1,j1]) - (d[i0,i1] + d[j0,j1])
    return Δcost
end



delta_costs = function(c::Vector{Int}, moves, d::Matrix{Float64}) # given list of moves, computes delta costs for each of them
    costs = Vector{Float64}(undef, length(moves))

    for i in 1:length(moves)
        costs[i] = delta_cost(c, moves[i], d)
    end
    return costs
end

best_moves = function(c::Vector{Int}, moves, d::Matrix{Float64}) # given current route and distance matrix, returns move
                                                                 # that minimizes cost of each possible route
    costs = delta_costs(c, moves, d)

    return moves[argmin(costs)]
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


eps_greedy = function(;N::Integer = 10, params::Params = Params(0.0, 100),
    α::Float64 = 0.0, seed::Int = 716464734, initreal::Bool = false)
    Random.seed!(seed)
    graph = Graph(N,α)
    moves = possible_moves(N)
    est = Est(N)

    if initreal
        copyto!(est.estD, graph.D)
    end

    @extract params : ϵ iters

    c = est.route

    sampled_costs = Array{Float64}(undef, iters) # observed costs accumulated for each iteration

    for i in 1:iters

        # optimization step
        if rand() < ϵ
            move = moves[rand(1:length(moves))]
        else
            move = best_moves(c, moves, est.estD)
        end
        accept_move!(c, move)

        # update estimates step
        sampled_cost = sample_cost!(c,graph, est)
        sampled_costs[i] = sampled_cost # new reward
    end

    realcost = real_cost(c, graph)

    return graph, est, sampled_costs, realcost
end




end



