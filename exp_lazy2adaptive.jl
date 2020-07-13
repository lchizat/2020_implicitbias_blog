# copyright Lénaïc Chizat
# the code shows the training dynamics in parameter and predictor space for a 2-layer relu NN on a two class classification task
# we optimize the exponential loss (the behavior is sensibly the same with the logistic loss)
# we use a specific step-size schedule motivated by the theory in this ref https://arxiv.org/abs/2002.04486 (sections 4 & 5)

# HOW TO RUN THE CODE
# in a prompt first run:
# include("exp_lazy2adaptive.jl")
# Then for fast prototyping, run for instance the following code (takes 1 min)
# illustration(4, 60, 200, 0.4, 100000, 10, 0.05);
# These parameters are explained in the header of the function "illustration"
# A good illustration is obtained as follows (takes 20 min):
# Random.seed!(9); # 9 generates a nice random problem
# illustration(4, 60, 800, 1, 5000000, 400, 0.005);


using LinearAlgebra, Random
using PyPlot, ProgressMeter # these packages need to be installed (via "] add NamePackage" )
using3D()


"""
Gradient descent to train a 2-layers ReLU neural network for the exponential loss
We use the step-size schedule from in https://arxiv.org/abs/2002.04486 (sections 4 & 5)
INPUT: X (training input), Y (training output), m (nb neurons), both: training both layers or just the output layer
OUTPUT: Ws (training trajectory)
"""
function twonet(X, Y, m, stepsize, niter)

    (n,d) = size(X) # n samples in R^d
    # initialize
    W = randn(m, d+1)
    # input weights initialized uniformly on the unit sphere
    beta = 80.0
    W[:,1:d] = beta * W[:,1:d] ./ sqrt.(sum(W[:,1:d].^2, dims=2))
    #W[div(m,2)+1:end,1:d] .= W[1:div(m,2),1:d] # symmetrization to put the initial output to zero (optional)
    # output weights initialized by -1 or 1 with equal proba
    W[1:div(m,2),end] .= beta
    W[div(m,2)+1:end,end] .= -beta

    Ws    = zeros(m, d+1, niter) # store optimization path
    loss  = zeros(niter) # loss is -log of the empirical risk
    margins = zeros(niter)
    betas = zeros(niter)


    @showprogress 1 "Training neural network..." for iter = 1:niter
        Ws[:,:,iter] = W
        act  =  max.( W[:,1:end-1] * X', 0.0) # activations
        out  =  (1/m) * sum( W[:,end] .* act , dims=1) # predictions of the network
        perf = Y .* out[:]
        margin = minimum(perf)
        temp = exp.(margin .- perf) # numerical stabilization of exp
        gradR = temp .* Y ./ sum(temp)' # gradient of the loss
        grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  ))  # gradient for input weights
        grad_w2 = act * gradR  # gradient for output weights

        grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
        betas[iter] = sum(W.^2)/m
        loss[iter] = margin - log(sum(exp.(margin .- perf))/n)
        margins[iter] = margin/betas[iter] # margin in F_1 norm
        W = W + stepsize * grad/beta^2 # notice the more agressive step-size here (but small at init)
    end

    return Ws, loss, margins, betas
end


"Coordinates of the 2d cluster centers, p is k^2 the number of clusters"
function cluster_center(p,k)
    p1 = mod.(p .- 1,k) .+ 1
    p2 = div.(p .- 1,k) .+ 1
    Delta = 1/(3k-1)
    x1 =  Delta*(1 .+ 3*(p1 .- 1)) .- 1/2
    x2 =  Delta*(1 .+ 3*(p2 .- 1)) .- 1/2
    return x1, x2
end


"""
Plot the classifier for a test case, comparing training both  layer
k: number of clusters per dimension in the data set (choose 3, 4 or 5)
n: number of training samples
m: number of neurons
stepsize, niter: parameters for training
nframes, resolution: parameters for the plots
"""
function illustration(k, n, m, stepsize, niter, nframes, resolution)
# data distribution (itself random)
sd = 0 # number of spurious dimensions with random noise
Delta = 1/(3k-1) # interclass distance
A = ones(k^2) # cluster affectation
A[randperm(k^2)[1:div(k^2,2)]] .= -1

# sample from this data distribution
P = rand(1:k^2,n) # cluster label
T = 2π*rand(n)  # shift angle
R = Delta*rand(n) # shift magnitude
X = cat(ones(n), cluster_center(P,k)[1] .+ R .* cos.(T),cluster_center(P,k)[2] + R .* sin.(T), (rand(n,sd) .- 1/2), dims=2)
Y = A[P]

# plot training set
X1 = X[(Y .== 1),:]
X2 = X[(Y .== -1),:]
fig = figure(figsize=[3,3])
plot(X1[:,2],X1[:,3],"+k")
plot(X2[:,2],X2[:,3],"_k")
axis("equal");axis("off");
display(fig)


# train the neural network
Ws, loss, margins, betas = twonet(X, Y, m, stepsize, niter)

# define the sequence of time steps ts to be plotted
a = (niter-1)/(nframes-1)^4
ts = setdiff(Int.(floor.(a*(0:nframes-1).^4)) .+ 1)
Ws = Ws[:,:,ts]

# these are the positions we plot for the particles
Wproj = Ws[:,1:end-1,:] .* abs.(Ws[:,end:end,:])
WN   = sqrt.(sum(Wproj.^2, dims = 2))
Wdir = Wproj ./ WN
Wlog = tanh.(0.5*WN) .* Wdir


@showprogress 1 "Plotting images..." for k = 1:length(ts)
       ioff() # turns off interactive plotting
    fig = figure(figsize=[7,4])
    ax1 = subplot(121, projection="3d")
    ax1.set_position([0,0.1,0.5,0.8])

    if k<11
        indt = 1:k
    else
        indt = (k-10):k
    end

    for i = 1:size(Wlog,1)
        plot3D(Wlog[i,2,indt],Wlog[i,3,indt],Wlog[i,1,indt], color="k", linewidth=0.2) # tail
    end
    plot3D(Wlog[1:div(m,2),2,k],Wlog[1:div(m,2),3,k],Wlog[1:div(m,2),1,k],"o",color="C3", markersize=1)
    plot3D(Wlog[div(m,2)+1:end,2,k],Wlog[div(m,2)+1:end,3,k],Wlog[div(m,2)+1:end,1,k],"o",color="C0", markersize=1)

    ax1.set_xlim3d(-1, 1)
    ax1.set_ylim3d(-1, 1)
    ax1.set_zlim3d(-1, 1)
    ax1.set_xticks([-1/2, 0, 1/2])
    ax1.set_yticks([-1/2, 0, 1/2])
    ax1.set_zticks([-1/2, 0, 1/2])
    ax1.view_init(25-20*sinpi(k/length(ts)),45-45*cospi(k/length(ts)))

    ax2 = subplot(122)
    ax2.set_position([0.45,0.25,0.5,0.5])

    f(x1,x2,k) = (1/m) * sum( Ws[:,end,k] .* max.( Ws[:,1:3,k] * [1;x1;x2], 0.0)) # prediction function

    xs = -0.8:resolution:0.8
    tab = [f(xs[i],xs[j],k) for i=1:length(xs), j=1:length(xs)]
    pcolormesh(xs', xs, tanh.(tab'), cmap="coolwarm", shading="gouraud", vmin=-1.0, vmax=1.0, edgecolor="face")

    xs = -0.8:resolution:0.8
    tab = [f(xs[i],xs[j],k) for i=1:length(xs), j=1:length(xs)]
    contour(xs', xs, tanh.(tab'), levels =0, colors="k", antialiased = true, linewidths=2)

    # plot training set
    X1 = X[(Y .== 1),:]
    X2 = X[(Y .== -1),:]
    plot(X1[:,2],X1[:,3],"+k")
    plot(X2[:,2],X2[:,3],"_k")
    axis("equal");axis("off");
    ax2.set_xticks([-1/2, 0, 1/2])
    ax2.set_yticks([-1/2, 0, 1/2])

    savefig("dynamics_lazy_ns_$(k).png",bbox_inches="tight", dpi=300)
    close(fig)
end


end
