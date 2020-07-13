using PyPlot, Random, ProgressMeter


"""
Train 2-layer relu NN with GD on square loss
INPUT
 m is the number of neurons, 
 alpha1 (resp. alpha2): scale of the initialization of the input (resp. output) weights
 ouly_output: train just the output layer (instead of both)
 sym_init: trick to make the output = 0 exactly at initialization
OUTPUT
 Ws: the whole training path
 loss: evolution of loss during training
"""
function twonet_square(X, Y, m, stepsize, niter;alpha1=1.0, alpha2=1.0, weight_decay=0.0, sym_init=false, only_output = false)
    (n,d) = size(X) # n samples in R^d
    # initialize
    W = randn(m, d+1)
    # input weights initialized uniformly on the unit sphere
    W[:,1:d] = W[:,1:d] .* alpha1 ./ sqrt.(sum(W[:,1:d].^2, dims=2))
    # output weights initialized by -1 or 1 with equal proba
    W[1:div(m,2),end] .= alpha2
    W[div(m,2)+1:end,end] .= -alpha2
    W = W
    if sym_init
        @assert iseven(m)
        W[div(m,2)+1:end,1:d]  .=  W[1:div(m,2),1:d]
    end
    Ws    = zeros(m, d+1, niter) # store optimization path
    loss  = zeros(niter) # loss is -log of the empirical risk

    @showprogress 1 "Training neural network..." for iter = 1:niter
        Ws[:,:,iter] = W
        act  =  max.( W[:,1:end-1] * X', 0.0) # activations
        out  =  (1/m) * sum( W[:,end] .* act , dims=1)[:] # predictions of the network
        gradR   = (out .- Y)/n  # size n
        grad_w1 = (W[:,end] .* float.(act .> 0) * ( X .* gradR  )) + weight_decay * W[:,1:end-1] # gradient for input weights
        grad_w2 = act * gradR + weight_decay * W[:,end]  # gradient for output weights
        if only_output 
            grad = cat(zeros(m,d), grad_w2, dims=2) # size (m × d+1)
        else
            grad = cat(grad_w1, grad_w2, dims=2) # size (m × d+1)
        end
        loss[iter] = (1/2)*sum( ( out - Y).^2 )/n + (weight_decay/2)*sum(W.^2)/m
        W = W - stepsize * grad
    end

    return Ws, loss
end


X = [-1 1; -1/16 1; 1/16 1; 1 1]
Y = [-1; 1/2; -1/2; 1]
alpha1 = 50
alpha2s=[0 5 10 50]
m = 5000
WW = zeros(m,3,length(alpha2s))
for k = 1:length(alpha2s)
    alpha2 = alpha2s[k]
    stepsize = 0.003/(1+alpha2/20)
    niter = 10000
    Ws, loss = twonet_square(X, Y, m, stepsize, niter, alpha1=alpha1, alpha2=alpha2, sym_init=true)
    WW[:,:,k] .= Ws[:,:,end]
end
#loglog(loss)


cm = plt.get_cmap("viridis")
figure(figsize=[8,4])
for k = 1:length(alpha2s)
X_test = cat(range(-1.2,1.2,length=200),ones(200),dims=2 )
    Y_test = (1/m) * sum( WW[:,end,k] .* max.( WW[:,1:end-1,k] * X_test', 0.0) , dims=1)[:]
plot(X_test[:,1], Y_test, color=cm((k-1)/(length(alpha2s)-1)),label="r=$(alpha2s[k]/50)",lw=3)
end
legend()
plot(X[:,1],Y,"ko",ms=10)
xticks([-1,0,1]);xlabel(L"x")
yticks([-1,0,1]);ylabel(L"y")