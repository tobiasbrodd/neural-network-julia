module Neural

using Statistics

export Network, setup, train!, predict!

mutable struct Network
    a::Array{Any,1}
    W::Array{Any,1}
    b::Array{Any,1}
    epsilon::Float64
    result::Array
end

function sigmoid(x)
    1 ./ (1 .+ exp.(-x))
end

function sigmoid_derivative(x)
    x .* (1 .- x)
end

function tanh(x)
    (exp.(x) - exp.(-x)) / (exp.(x) + exp.(-x))
end

function tanh_derivative(x)
    1 - x.^2
end

function setup(input_size, hidden_sizes, output_size, epsilon=0.01)
    sizes = [input_size hidden_sizes output_size]

    W = []
    for i = 2:length(sizes)
        push!(W, randn((sizes[i-1], sizes[i])))
    end

    b = []
    for i = 2:length(sizes)
        push!(b, zeros((1, sizes[i])))
    end

   return Network([], W, b, epsilon, [])
end

function forward!(network::Network, X)
    network.a = [X]
    for i = 1:length(network.W)
        z = network.a[i] * network.W[i] .+ network.b[i]
        push!(network.a, sigmoid(z))
    end
    network.result = network.a[end]

    return network
end

function backward!(network::Network, X, y)
    error = y - network.result
    delta = error .* sigmoid_derivative(network.result)

    network.W[end] += network.epsilon * transpose(network.a[end-1]) * delta
    network.b[end] .+= network.epsilon * mean(delta)

    W_length = length(network.W)

    for i = 1:W_length-1
        j = W_length - i
        
        error = delta * transpose(network.W[j+1])
        delta = error .* sigmoid_derivative(network.a[j+1])
        network.W[j] += network.epsilon * transpose(network.a[j]) * delta
        network.b[j] .+= network.epsilon * mean(delta)
    end

    return network
end

function train!(network::Network, X, y, n=1000)
    for i = 1:n
        forward!(network, X)
        backward!(network, X, y)
    end
end

function predict!(network::Network, XP)
    forward!(network, XP)
end

end