### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ b405e9cd-cde5-4c2a-b9a0-03f1df01c48f
begin
	using Pkg
	cd(".")
	Pkg.activate(".")
end

# ╔═╡ d05308c8-5949-11ee-0ca3-ab89dcca6ae6
begin
	using Flux, CUDA
	using Plots, LinearAlgebra, Distributions, Makie
	using Random, SymbolicRegression, Latexify
end

# ╔═╡ d2288bce-fc0f-486d-b315-72b90eeb28bc
md"""
# LocalGLMNet
"""

# ╔═╡ afca99c9-3706-4b5e-8a9d-22345829c062
md"""
In this notebook, we discuss neural networks and methodologies to interpret them.

Some of this work is based on _LocalGLMnet: Interpretable deep learning for tabular data_ by Richman & Wüthrich ([2021](https://arxiv.org/pdf/2107.11059.pdf))
"""

# ╔═╡ 8271339c-831c-4a2d-8d6e-28321cb3c67c
md"""
### Setting up our Environment
"""

# ╔═╡ e8041009-be85-4240-bb5d-8ca5b7bdf51a
md"""
We use `Flux.jl` to define the neural network and make use of `CUDA.jl` to move the training over to the GPU.

`SymbolicRegression.jl` and `Latexify.jl` are used to make sense of the neural network's predictions.
"""

# ╔═╡ 40247426-e41d-4092-bd85-8c395a5430a5
md"""
To standardise the process, we use a random seed.
"""

# ╔═╡ 6513ae90-3f8d-40c7-b951-d9b1dbac113c
rng = MersenneTwister(100)

# ╔═╡ 8f3eec65-cb3a-426a-9c97-ab9f747085c6
NN_seed = Flux.glorot_uniform(rng);

# ╔═╡ c263806c-ad06-4172-be72-537ac5ad3950
md"""
### Data
"""

# ╔═╡ 69c81a38-aa6b-467a-8999-955c0da01d4f
md"""
Our synthetic regression problem comes from Richman _et al._ We aim to train a neural network to approximate the following equation:

$$f(x) = \frac{1}{2}x_1 - \frac{1}{4}x_2^2 + \frac{1}{2}|x_3|\sin(2x_3) + \frac{1}{2}x_4 x_5 + \frac{1}{8}x_{5}^2x_{6}$$

Our data input $X$ is an $8×10,000$ matrix of standard Normally distributed variables with a unit variance. We assume a 50% correlation between $x_2$ and $x_8$. Note that variables $x_7$ and $x_8$ are not used in the equation.
"""

# ╔═╡ 621a6026-04a7-4119-aee7-39380f89f810
Σ = [1 0 0 0 0 0 0 0; 
	 0 1 0 0 0 0 0 0.5; 
	 0 0 1 0 0 0 0 0;
	 0 0 0 1 0 0 0 0;
	 0 0 0 0 1 0 0 0;
	 0 0 0 0 0 1 0 0;
	 0 0 0 0 0 0 1 0;
	 0 0.5 0 0 0 0 0 1
]

# ╔═╡ 1a2a0752-75e8-4703-b832-e95eb6e3bc6b
X = rand(MvNormal(zeros(8), Σ), 100_000)' |> f32 |> gpu

# ╔═╡ 1e6a99b9-b1a8-4459-9f70-67961271e4c8
f(x) = 0.5 .* x[:, 1] .- 0.25 .* x[:, 2] .^2 + 0.5 .* abs.(x[:, 3]) .* sin.(2 .* x[:, 3]) .+ 0.5 .* x[:, 4] .* x[:, 5] .+ 0.125 .* x[:, 6] .* x[:, 5] .^ 2

# ╔═╡ 0465bbc4-435d-473f-806c-22710c3d115b
y = f(X)

# ╔═╡ 847a4c6f-dc09-4ae7-a1f6-54447fe1484a
md"""
Below is a plot of $x_2$ against the output:
"""

# ╔═╡ 8a40b065-30ed-43f7-98af-0518cc829d62
Plots.scatter(X[:, 2] |> cpu, y |> cpu, xlabel="x₂", ylabel="f(X)", label="")

# ╔═╡ 9e7557f2-1acb-4893-978b-9e00c1dc30d2
md"""
For training the network, we take 75% of the data as training and the remainder for validation during the training process to identify signs of overfitting.
"""

# ╔═╡ 1061fa8e-c123-41ab-8838-927db6e49485
begin
	X_train = X[1:75_000, :]
	y_train = f(X_train)
	X_valid = X[75_001:end, :]
	y_valid = f(X_valid)
end

# ╔═╡ c649098c-ac0d-49f0-8f23-2300ea5fde99
md"""
### Defining our Neural Network
"""

# ╔═╡ 3a6268d6-10ac-4c17-9987-391d0a3a1b3d
md"""
We set-up a 3-layer feedforward network ($32 → 64 → 8$) to perform regression. Important to note is that we intentionally set the last hidden layer to comprise of 8 neurons (+ bias) -- the size of our paramter space!

Per Richman _et al._, this can lead to a LocalGLMnet by establishing a _skip connection_ between the parameter space and the last hidden layer, setting us up to have a GLM-style neural network. 
"""

# ╔═╡ 9aabe634-0bad-4529-bb8d-760b91d891e9
model = Chain(
	Dense(8 => 32, Flux.tanh; init=NN_seed),
	Dense(32 => 64, Flux.tanh; init=NN_seed),
	Dense(64 => 8, Flux.tanh; init=NN_seed),
	Dense(8 => 1, identity; init=NN_seed)
) |> gpu 

# ╔═╡ 3a941ea0-524a-40f6-a92c-4a8de4918d3f
md"""
Our model uses the default implementation of Adam over 5 000 epochs. The mean squared error (MSE) is used as the loss function.
"""

# ╔═╡ 85381d05-fd50-43a4-a462-e73184c448d5
begin
	opt_params = Flux.setup(Flux.Adam(), model)
	epochs = 7_500
	loss(y_pred, y_actual) = Flux.Losses.mse(y_pred, y_actual)
end

# ╔═╡ 942b49a5-ba5f-460f-81b2-4baa3f76636c
md"""
In our training loop, we retain the losses to plot and validate the training proceedure.
"""

# ╔═╡ 4ffc9795-091d-4d1a-b18b-f2ed9774a747
# ╠═╡ show_logs = false
begin
	losses = zero(rand(epochs))
	losses_valid = zero(rand(epochs))
	for epoch ∈ 1:epochs
		Flux.train!(model, [(X_train', y_train')], opt_params) do m, x, y
	    	loss(m(x), y)
		end
		losses[epoch] = loss(model(X_train'), y_train')
		losses_valid[epoch] = loss(model(X_valid'), y_valid')
	end
end

# ╔═╡ 5b54ecde-4ef7-472c-9295-bc71493d78cd
md"""
Based on the plot of the losses below, the neural network has performed very well on both training and validation splits, having reached stability from about 3 000 epochs, suggesting early stopping may have been used.

The training time however is not too significant, just under 1 minute on this Pop!_OS MSI laptop (11th Gen i5, Nvidia GTX 1650, 16GB RAM).
"""

# ╔═╡ e6b56c4c-5197-40ba-97e7-75a0b52dda6d
begin
	Plots.plot(title="Loss (MSE)", xlabel="Epochs", ylabel="Loss (MSE)", legend=true)
	Plots.scatter!(1:epochs, losses, color="purple", markeralpha=0.25, label="Training")
	Plots.scatter!(1:epochs, losses_valid, color="orange", markeralpha=0.25, label="Validation")
end

# ╔═╡ 42223eb0-9b7a-47d2-b6fc-ab6786de6775
md"""
### Testing the Model
"""

# ╔═╡ 95f8974a-f865-487a-9632-f755a94e1a4a
md"""
Per Richman _et al._, we create a new sample of 100 000 data points and investigate how the model performs on entirely unseen data.
"""

# ╔═╡ 0a9e7b6a-0f16-4ac8-bb99-bd729a719e13
X_test = rand(MvNormal(zeros(8), Σ), 100_000)' |> f32 |> gpu

# ╔═╡ 08fd28d4-8376-4031-80be-84f88055f11c
y_test = f(X_test)

# ╔═╡ 2ccb1f57-9b22-4923-aa38-e593ff5d4d98
y_pred = model(X_test')

# ╔═╡ 6e0a0873-da7f-459b-9228-429c65afe8c1
md"""
The Actual v. Predicted plot below shows the majority of the data points lie along the diagonla, indicating a good fit overall on unseen data.
"""

# ╔═╡ 4114157c-cc23-4a1c-9aaf-bb6abab4bd1a
begin
	Plots.scatter(y_test |> cpu, y_pred' |> cpu, xlim = (-5, 5), ylim = (-5, 5), yaxis = "Predicted", xaxis = "Actual", label="")
	Plots.plot!(-5:5, -5:5, width=2, color="black", label="")
end

# ╔═╡ 9c99aa6f-d85f-4781-861c-5eae78e8011c
md"""
For comparison, we can retrieve the MSE scores for the training, validation, and testing splits:
"""

# ╔═╡ b0665c1e-c7a3-4f67-83a6-54f5df6f3789
mse_scores = [loss(y_train, model(X_train')'), loss(y_valid, model(X_valid')'), loss(y_test, y_pred')]

# ╔═╡ 44e76113-4fd1-4772-9d82-503deb33c1f1
md"""
### Model Explaination
"""

# ╔═╡ fbd109eb-2810-4c8f-9cee-e826de665f42
md"""
To assist with explaining our model, we will be constructing naïve partial dependence plots by generating outputs with one feature containing the data and the remainder as zeros. This provides a rough approximation for the impact a feature has on its output in isolation.

The function below returns a matrix with $n$ features isolated.
"""

# ╔═╡ 7ab66a7f-d593-4706-b0bf-dc8be4f30503
function n_feature(X_::Union{CuArray, Matrix}, i::Union{Vector{Int64}, UnitRange{Int64}})::Matrix{Float32}
	ret::Matrix{Float32} = deepcopy(X_)
	N::Int64, m::Int64 = size(ret)
	for k ∈ 1:m
		if k ∉ i
			ret[:, k] = zeros(N)
		end
	end
	return ret
end

# ╔═╡ 697279ec-e144-44da-87c6-2cffca56acc1
md"""
We can then pass our data through the model to generate output, subtracting any bias term (below, we assume the bais comes from our dataset, $X$, for convenience).
"""

# ╔═╡ e4a90c03-be04-453b-b317-ae41d59a6612
function naive_pdp(xᵢ)
	weights::Vector, _ = Flux.destructure(model) |> cpu
	β₀ = repeat([weights[end]], size(xᵢ)[1]) |> cpu
	return (model(xᵢ' |> gpu) |> cpu)
end

# ╔═╡ b430a6a5-369b-4b8d-9994-19121cbedd51
md"""
Below is the output if feature $x_1$ is isolated:
"""

# ╔═╡ 06f69c1e-046d-4e2c-a0e0-159fd28f6af4
begin
	Plots.plot(xlabel = "x₁", ylabel = "f(X)")
	Plots.scatter!(n_feature(X, [1])[:, 1], naive_pdp(n_feature(X, [1]))' |> cpu, label="")
end

# ╔═╡ 3ddc7775-77f3-47b1-a8d3-0dba2dd5e1d1
md"""
The above suggests the model has identified a fully linear relation between $x_1$ and the output.

Recalling our original function used to synthesise the output data, this is exactly what we had:

$$f(x) = \frac{1}{2}x_1 - \frac{1}{4}x_2^2 + \frac{1}{2}|x_3|\sin(2x_3) + \frac{1}{2}x_4 x_5 + \frac{1}{8}x_{5}^2x_{6}$$

 $f(x)$ was generated with $x_1$ having a linear relation with the output. Judging by the plot, the slope is approximately $\frac{1}{2}$.

We can also investigate the bias term -- if any.
"""

# ╔═╡ b58aa8a7-3053-4c1f-920f-5c1f72de4e2a
naive_pdp(n_feature(X, [0]))'

# ╔═╡ d664f697-0f94-4aa0-8595-36055b9381a0
#Flux.jacobian(model |> cpu, n_feature(X_valid, [0])')

# ╔═╡ 67d3165d-ac55-43ff-ae08-afcd6949318b
md"""
Even though the synthetic function did not have a bias term, the model assumed a small term of 0.00116.

Looking at a feature like $x_7$ and $x_8$, we see that there is a small impact on outputs through their inclusion.
"""

# ╔═╡ b263ddaa-1442-446c-b275-58daebd051bf
begin
	Plots.plot(xlabel = "x", ylabel = "f(X)")
	Plots.scatter!(n_feature(X, [7])[:, 7], naive_pdp(n_feature(X, [7]))' |> cpu, label="x₇")
	Plots.scatter!(n_feature(X, [8])[:, 8], naive_pdp(n_feature(X, [8]))'  |> cpu, label="x₈")
end

# ╔═╡ 7812ba51-d7eb-42b7-be5b-453859a2604c
md"""
We can also use this approach to construct plots to capture interdependency between features.
"""

# ╔═╡ 633d4f7e-1786-42b3-8f70-4f2e827d8023
begin
	l = @layout [a d; b c]
	p1 = scatter3d(n_feature(X, [4])[:, 4], n_feature(X, [5])[:, 5], vec(naive_pdp(n_feature(X, [4,5]))' |> cpu), label="", xlabel="x₄", ylabel="x₅", zlabel="f(x)")
	p2 = Plots.scatter(n_feature(X, [4])[:, 4], naive_pdp(n_feature(X, [4]))' |> cpu, xlabel="x₄", ylabel = "f(x)", label="")
	p3 = Plots.scatter(n_feature(X, [5])[:, 5], naive_pdp(n_feature(X, [5]))' |> cpu, xlabel="x₅", ylabel = "f(x)", label="")
	p4 = Plots.scatter(n_feature(X, [4])[:, 4], n_feature(X, [5])[:, 5], xlabel="x₄", ylabel = "x₅", label="")
	Plots.plot(p1, p2, p3, p4, layout = l)
end

# ╔═╡ 9fb27561-c74f-40aa-9a0f-29c651c6cb3f
begin
	l2 = @layout [a d; b c]
	p5 = scatter3d(n_feature(X, [5])[:, 5], n_feature(X, [6])[:, 6], vec(naive_pdp(n_feature(X, [5,6]))' |> cpu), label="", xlabel="x₅", ylabel="x₆", zlabel="f(x)")
	p6 = Plots.scatter(n_feature(X, [5])[:, 5], naive_pdp(n_feature(X, [5]))' |> cpu, xlabel="x₅", ylabel = "f(x)", label="")
	p7 = Plots.scatter(n_feature(X, [6])[:, 6], naive_pdp(n_feature(X, [6]))' |> cpu, xlabel="x₆", ylabel = "f(x)", label="")
	p8 = Plots.scatter(n_feature(X, [5])[:, 5], n_feature(X, [6])[:, 6], xlabel="x₅", ylabel = "x₆", label="")
	Plots.plot(p5, p6, p7, p8, layout = l2)
end

# ╔═╡ bc36d6f7-4021-4137-9d80-eac0ba05fa8c
md"""
_I am not too sure what the significance of the interpretation below is, but if we take the output from the final hidden layer and plot it, we can see which 'channels' have more variety than others. For instance, channel 7 appears to mainly produce a constant output, whereas 1-4 produce a greater variety based on inputs._
"""

# ╔═╡ c86733fe-2f68-44b0-a7ad-6e34424d5177
Plots.heatmap(model[1:(end-1)](X_train')' |> cpu, title="Heatmap over the last hidden layer of the trained NN", xticks=1:8)

# ╔═╡ fc0d6dfe-8136-4fcb-8802-d0d32b309cdb
md"""
### LocalGLMnet
"""

# ╔═╡ 8712bce1-a627-40b5-9f75-6b5ccdfc847d
md"""
A LocalGLMnet model is defined as a skip-connection model, where an additive decomposition of the model is created as follows:

$$g(𝒙) = \beta_0 + \sum_{i}\beta(x_i)x_i$$


"""

# ╔═╡ 49267399-03c4-42b0-b9a2-f49abf1c0d69


# ╔═╡ 492726a6-5108-4754-be7a-7bb146e911f9
function localglmnet(m, X_::Union{CuArray, Matrix})::Matrix
	ret::Matrix = deepcopy(X_) |> cpu
	weights::Vector, _ = Flux.destructure(m) |> cpu
	N::Int64, M::Int64 = size(ret)
	output = zeros(N)
	for i ∈ 0:M
		output = output + (m(n_feature(ret, [i])' |> gpu) |> cpu)'
	end
	return output
end

# ╔═╡ a1308b55-46a4-41ee-90c4-b8ad91310058
y_lgn = localglmnet(model, X_train)

# ╔═╡ 7670f058-702d-4d6c-9d08-018f14b29f1c
md"""
We compute the MSE for comparitive purposes, and produce an Actual v. Predicted plot.
"""

# ╔═╡ c0e10a7f-44f5-4f95-a3b3-d04d34dee199
loss(y_train |> cpu, y_lgn)

# ╔═╡ 377551b4-0033-40ed-abb1-b9c8d86aaed9
begin
	Plots.scatter(y_train |> cpu, y_lgn, xlim = (-5, 5), ylim = (-5, 5), yaxis = "LocalGLMNet Prediction", xaxis = "Actual", label="")
	Plots.plot!(-5:5, -5:5, width=2, color="black", label="")
end

# ╔═╡ 59c4a287-753d-43d7-8995-6a4def819a89
md"""
The results, while not as accurate as the full NN, are at least an interpetable approximation of the model.

We 
"""

# ╔═╡ 73f28ea1-11da-4020-b60b-d2ccb7e88b36
options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, exp],
    populations=20
)

# ╔═╡ 139b2d3e-92e4-453a-bc90-1bb2a113016c
#hall_of_fame = equation_search(
    (X_train' |> cpu), vec(model(X_train')' |> cpu), niterations=40, options=options,
    parallelism=:multithreading
)

# ╔═╡ 80b89d86-6058-48c5-95f4-0d6b1a495616
dominating = calculate_pareto_frontier(hall_of_fame)

# ╔═╡ c94b572d-123f-45f2-8fd3-00db8df3e2ee
trees = [member.tree for member in dominating]

# ╔═╡ ee4b9e54-bd46-41ff-821b-5cd66deddb9d
y_sr, _ = eval_tree_array(trees[end], X_valid' |> cpu, options)

# ╔═╡ 01114b16-0885-422b-9272-1363b42f64e4
latexify(string(trees[end]))

# ╔═╡ e0c6917f-9f27-4cbd-8514-f42211d2ce2a
begin
	scatter(y_valid |> cpu, y_sr, xlim = (-5, 5), ylim = (-5, 5), yaxis = "Symbolic Prediction", xaxis = "Actual", label="SR", markeralpha=.4)
	scatter!(y_valid |> cpu, (model(X_valid')' |> cpu), xlim = (-5, 5), ylim = (-5, 5), label="NN", markeralpha=.4)
	scatter!(y_valid |> cpu, localglmnet(model, X_valid), xlim = (-5, 5), ylim = (-5, 5), label="LGN", markeralpha=.4)
	plot!(-5:5, -5:5, width=2, color="black", label="")
end

# ╔═╡ cdb815d9-e76b-40d1-b95e-0b934e7688f5
mean(((y_valid |> cpu) .- y_sr).^2)

# ╔═╡ b9282db5-92e5-403d-85ad-650cd5f95126
mean(((y_valid |> cpu) .- (model(X_valid')' |> cpu)).^2)

# ╔═╡ d57c7b5a-f795-43bd-84b3-9d48a96f905c
mean(((y_valid |> cpu) .- localglmnet(model, X_valid)).^2)

# ╔═╡ 222665da-c1f3-4f9f-b281-7d0bcd6ce0c4


# ╔═╡ Cell order:
# ╟─d2288bce-fc0f-486d-b315-72b90eeb28bc
# ╟─afca99c9-3706-4b5e-8a9d-22345829c062
# ╟─8271339c-831c-4a2d-8d6e-28321cb3c67c
# ╠═b405e9cd-cde5-4c2a-b9a0-03f1df01c48f
# ╟─e8041009-be85-4240-bb5d-8ca5b7bdf51a
# ╠═d05308c8-5949-11ee-0ca3-ab89dcca6ae6
# ╟─40247426-e41d-4092-bd85-8c395a5430a5
# ╠═6513ae90-3f8d-40c7-b951-d9b1dbac113c
# ╠═8f3eec65-cb3a-426a-9c97-ab9f747085c6
# ╟─c263806c-ad06-4172-be72-537ac5ad3950
# ╟─69c81a38-aa6b-467a-8999-955c0da01d4f
# ╠═621a6026-04a7-4119-aee7-39380f89f810
# ╠═1a2a0752-75e8-4703-b832-e95eb6e3bc6b
# ╠═1e6a99b9-b1a8-4459-9f70-67961271e4c8
# ╠═0465bbc4-435d-473f-806c-22710c3d115b
# ╟─847a4c6f-dc09-4ae7-a1f6-54447fe1484a
# ╠═8a40b065-30ed-43f7-98af-0518cc829d62
# ╟─9e7557f2-1acb-4893-978b-9e00c1dc30d2
# ╠═1061fa8e-c123-41ab-8838-927db6e49485
# ╟─c649098c-ac0d-49f0-8f23-2300ea5fde99
# ╟─3a6268d6-10ac-4c17-9987-391d0a3a1b3d
# ╠═9aabe634-0bad-4529-bb8d-760b91d891e9
# ╟─3a941ea0-524a-40f6-a92c-4a8de4918d3f
# ╠═85381d05-fd50-43a4-a462-e73184c448d5
# ╟─942b49a5-ba5f-460f-81b2-4baa3f76636c
# ╠═4ffc9795-091d-4d1a-b18b-f2ed9774a747
# ╟─5b54ecde-4ef7-472c-9295-bc71493d78cd
# ╠═e6b56c4c-5197-40ba-97e7-75a0b52dda6d
# ╟─42223eb0-9b7a-47d2-b6fc-ab6786de6775
# ╟─95f8974a-f865-487a-9632-f755a94e1a4a
# ╠═0a9e7b6a-0f16-4ac8-bb99-bd729a719e13
# ╠═08fd28d4-8376-4031-80be-84f88055f11c
# ╠═2ccb1f57-9b22-4923-aa38-e593ff5d4d98
# ╟─6e0a0873-da7f-459b-9228-429c65afe8c1
# ╠═4114157c-cc23-4a1c-9aaf-bb6abab4bd1a
# ╟─9c99aa6f-d85f-4781-861c-5eae78e8011c
# ╠═b0665c1e-c7a3-4f67-83a6-54f5df6f3789
# ╟─44e76113-4fd1-4772-9d82-503deb33c1f1
# ╟─fbd109eb-2810-4c8f-9cee-e826de665f42
# ╠═7ab66a7f-d593-4706-b0bf-dc8be4f30503
# ╟─697279ec-e144-44da-87c6-2cffca56acc1
# ╠═e4a90c03-be04-453b-b317-ae41d59a6612
# ╟─b430a6a5-369b-4b8d-9994-19121cbedd51
# ╠═06f69c1e-046d-4e2c-a0e0-159fd28f6af4
# ╟─3ddc7775-77f3-47b1-a8d3-0dba2dd5e1d1
# ╠═b58aa8a7-3053-4c1f-920f-5c1f72de4e2a
# ╠═d664f697-0f94-4aa0-8595-36055b9381a0
# ╠═67d3165d-ac55-43ff-ae08-afcd6949318b
# ╠═b263ddaa-1442-446c-b275-58daebd051bf
# ╟─7812ba51-d7eb-42b7-be5b-453859a2604c
# ╠═633d4f7e-1786-42b3-8f70-4f2e827d8023
# ╠═9fb27561-c74f-40aa-9a0f-29c651c6cb3f
# ╟─bc36d6f7-4021-4137-9d80-eac0ba05fa8c
# ╠═c86733fe-2f68-44b0-a7ad-6e34424d5177
# ╟─fc0d6dfe-8136-4fcb-8802-d0d32b309cdb
# ╠═8712bce1-a627-40b5-9f75-6b5ccdfc847d
# ╠═49267399-03c4-42b0-b9a2-f49abf1c0d69
# ╠═492726a6-5108-4754-be7a-7bb146e911f9
# ╠═a1308b55-46a4-41ee-90c4-b8ad91310058
# ╟─7670f058-702d-4d6c-9d08-018f14b29f1c
# ╠═c0e10a7f-44f5-4f95-a3b3-d04d34dee199
# ╠═377551b4-0033-40ed-abb1-b9c8d86aaed9
# ╠═59c4a287-753d-43d7-8995-6a4def819a89
# ╠═73f28ea1-11da-4020-b60b-d2ccb7e88b36
# ╠═139b2d3e-92e4-453a-bc90-1bb2a113016c
# ╠═80b89d86-6058-48c5-95f4-0d6b1a495616
# ╠═c94b572d-123f-45f2-8fd3-00db8df3e2ee
# ╠═ee4b9e54-bd46-41ff-821b-5cd66deddb9d
# ╠═01114b16-0885-422b-9272-1363b42f64e4
# ╠═e0c6917f-9f27-4cbd-8514-f42211d2ce2a
# ╠═cdb815d9-e76b-40d1-b95e-0b934e7688f5
# ╠═b9282db5-92e5-403d-85ad-650cd5f95126
# ╠═d57c7b5a-f795-43bd-84b3-9d48a96f905c
# ╠═222665da-c1f3-4f9f-b281-7d0bcd6ce0c4
