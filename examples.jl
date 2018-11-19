module Examples

include("./neural.jl")

using .Neural, Gadfly

export fixed_sequence_example, normal_sequence_example, random_sequence_example

function grid(x_min, x_max, y_min, y_max, h)
    n = floor(Int, (y_max - y_min) / h) + 1
    m = floor(Int, (x_max - x_min) / h) + 1
    xx = zeros(n, m)
    yy = zeros(n, m)
    
    for i = 1:n, j = 1:m
        xx[i,j] = x_min + h*(j-1)
    end

    for i = 1:n, j = 1:m
        yy[i,j] = y_min + h*(i-1)
    end

    return (xx, yy)
end

function plot_decision_boundary(network, X, y)
    x_min = minimum(X[:, 1]) - 0.5
    x_max = maximum(X[:, 1]) + 0.5
    y_min = minimum(X[:, 2]) - 0.5
    y_max = maximum(X[:, 2]) + 0.5
    h = 0.01

    xx, yy = grid(x_min, x_max, y_min, y_max, h)
    xy = [reshape(xx, (size(xx, 1)*size(xx, 2), 1)) reshape(yy, (size(yy, 1)*size(yy, 2), 1))]
    Z = reshape(round.(predict!(network, xy).result), size(xx))
    Gadfly.push_theme(:dark)
    boundary = layer(z=collect(transpose(Z)), x=xx[1, :], y=yy[:, 1], Geom.contour)
    scatter = layer(x=X[:,1], y=X[:,2], color=y[:,1], Geom.point)
    return plot(boundary, scatter, Guide.xlabel("x"), Guide.ylabel("y"), Guide.title("Decison Boundary"))
end

function plot_predictions(network, X, y, X_scale, y_scale, title)
    X *= X_scale
    y *= y_scale

    XP = collect(0:X_scale) / X_scale
    yp = predict!(network, XP).result
    XP *= X_scale
    yp *= y_scale
    Gadfly.push_theme(:dark)
    predicted_layer = layer(x=XP, y=yp, Geom.line)
    sequence_layer = layer(x=X, y=y, Theme(default_color="green"), Geom.line)
    return plot(predicted_layer, sequence_layer, Guide.xlabel("X"), Guide.ylabel("y"), Guide.title(title), Guide.manual_color_key("Legend",["Prediction", "Data"], [Gadfly.current_theme().default_color,"green"]))
end

function exponential_sequence_example()
    input_size = 1
    hidden_sizes = [100 200 100]
    output_size = 1

    sequence = collect(0:0.1:9.9)

    X = sequence
    y = exp.(sequence)

    X_scale = 10^(length(digits(floor(Integer, maximum(X)), base=10)))
    y_scale = 10^(length(digits(floor(Integer, maximum(y)), base=10)))
    
    X /= X_scale
    y /= y_scale

    network = setup(input_size, hidden_sizes, output_size)
    train!(network, X, y, 10000)
    nn_plot = plot_predictions(network, X, y, X_scale, y_scale, "Exponential Sequence")
    display(nn_plot)
end

function normal_sequence_example()
    input_size = 1
    hidden_sizes = [25]
    output_size = 1

    sequence = collect(0:99)

    X = sequence
    y = [1.]
    for i = 1:99
        push!(y, y[i] * (1 + abs(randn())) % 1.2)
    end

    X_scale = 10^(length(digits(floor(Integer, maximum(X)), base=10)))
    y_scale = 10^(length(digits(floor(Integer, maximum(y)), base=10)))
    
    X /= X_scale
    y /= y_scale

    network = setup(input_size, hidden_sizes, output_size)
    train!(network, X, y, 10000)
    nn_plot = plot_predictions(network, X, y, X_scale, y_scale, "Normally Distributed Sequence")
    display(nn_plot)
end

function random_sequence_example()
    input_size = 1
    hidden_sizes = [25]
    output_size = 1

    X = sort(rand(0:100, (25, input_size)), dims=1)
    y = sort(rand(0:100, (25, output_size)), dims=1)

    X_scale = 10^(length(digits(floor(Integer, maximum(X)), base=10)))
    y_scale = 10^(length(digits(floor(Integer, maximum(y)), base=10)))
    
    X /= X_scale
    y /= y_scale

    network = setup(input_size, hidden_sizes, output_size)
    train!(network, X, y, 10000)
    nn_plot = plot_predictions(network, X, y, X_scale, y_scale, "Random Sequence")
    display(nn_plot)
end

function decision_boundary_example()
    input_size = 2
    hidden_sizes = [10 20 30 20 10]
    output_size = 1

    X = randn((100, 2))
    y = zeros((100, 1))

    quandrant = (h1=rand(-1:2:1), h2=rand(-1:2:1))

    for i = 1:100
        if quandrant.h1*X[i,1] > 0 && quandrant.h2*X[i,2] > 0
            y[i] = 1
        end 
    end
    
    network = setup(input_size, hidden_sizes, output_size)
    train!(network, X, y, 10000)
    nn_plot = plot_decision_boundary(network, X, y)
    display(nn_plot)
end

end