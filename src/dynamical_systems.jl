

abstract type Integrator end   
struct forwardEuler <: Integrator
    h::Float64
end
struct RungeKutta4thOrder <: Integrator
    h::Float64
end

struct DynamicalSystems <: Algorithm 
    maxiter::Int
    ϵ::Float64
    integrator::Integrator
    which::Symbol
    normalize::Bool
end

tol(args::DynamicalSystems) = args.ϵ



function dynamical_systems(A_tensor, Λ, integration, maxiter, ε, normalize_vector, xₖ)
    normalize!(xₖ)
    xₖ₊₁ = similar(xₖ)
    return xₖ₊₁, dynamical_systems!(A_tensor, Λ, integration, maxiter, ε, normalize_vector, xₖ, xₖ₊₁)...
end 

dynamical_systems(A_ten, Λ, max_iter, ε) = dynamical_systems(A_ten, Λ, forwardEuler(1e-2), max_iter, ε, true, randn(A_ten.n))


function dynamical_systems!(A_tensor, Λ::Symbol, integration::Integrator, 
                            max_iter::Int, ε::Float64, normalize::Bool,
                            xₖ::AbstractArray{T},xₖ₊₁::AbstractArray{T}) where T 
    #using forward Euler for itegrator 

    binom_factor = binomial(A_tensor.order,A_tensor.order-2)
    is = Array{Int}(undef, binom_factor*size(A_tensor.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_tensor.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_tensor.indices, 2))

    #y = similar(xₖ)
    xₖ₊₁ .= 0.0
    λₖ = Inf
    R = Inf
    iter = 0

    while true 

        iter += 1
        Axₖᵐ⁻² = contract_to_mat!(A_tensor,xₖ,is,js,vs)
        mul!(xₖ₊₁,Axₖᵐ⁻²,xₖ)
        λₖ₊₁ = dot(xₖ₊₁,xₖ)

        R = norm(xₖ₊₁ - xₖ) / norm(xₖ₊₁)

        if (sqrt((λₖ - λₖ₊₁)^2/λₖ^2) < ε) || iter >= max_iter
            break
        else
            xₖ .= xₖ₊₁
            λₖ = copy(λₖ₊₁)

            #=
            output = symeigs(Axₖᵐ⁻²,1,which = Λ,maxiter=1000,failonmaxiter=false)
            if output.state.aupd_mxiter[] > 1000
                return Axₖᵐ⁻²
            end=# 

            integrate!(integration,Λ,Axₖᵐ⁻²,xₖ)
            if normalize; normalize!(xₖ); end
            
        end 
        
    end 
    if normalize; normalize!(xₖ₊₁); end
    return λₖ, R, iter
end


function dynamical_systems_profiled!(A_tensor, Λ, integration, max_iter, 
                                     ε, normalize, xₖ, xₖ₊₁)

    x_history = Matrix{Float64}(undef,n(A_tensor), max_iter)
    iter_history = -ones(Float64,max_iter,5)
    return dynamical_systems_profiled!(A_tensor, Λ, integration, max_iter, ε, normalize,xₖ,xₖ₊₁,x_history, iter_history)..., x_history, iter_history
end 

dynamical_systems_profiled(A_tensor, Λ, integration, max_iter, ε, normalize, xₖ) = dynamical_systems_profiled!(A_tensor, Λ, integration, max_iter, ε, normalize, xₖ, rand(length(xₖ)))

function dynamical_systems_profiled!(A_tensor, Λ::Symbol, integration::Integrator, 
                                    max_iter::Int, ε::Float64, normalize::Bool,
                                    xₖ::AbstractArray{T},xₖ₊₁::AbstractArray{T},
                                    x_history::AbstractArray{T,2},
                                    iter_history::AbstractArray{F,2}) where {T, F <: AbstractFloat} 

    binom_factor = binomial(A_tensor.order,A_tensor.order-2)
    is = Array{Int}(undef, binom_factor*size(A_tensor.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_tensor.indices, 2))

    vs = Array{T}(undef, binom_factor*size(A_tensor.indices, 2))

    #y = similar(xₖ)
    #xₖ₋₁ = similar(xₖ)
    λₖ₋₁ = Inf
    λₖ = Inf
    R = Inf
    iter = 1

    iterate_diffs = @view(iter_history[:,1])
    λs = @view(iter_history[:,2])
    λ_checks = @view(iter_history[:,3]) 
    mat_contract_ts = @view(iter_history[:,4])
    solver_ts = @view(iter_history[:,5])

    while true 
        Axₖᵐ⁻², t = @timed contract_to_mat!(A_tensor,xₖ,is,js,vs)
        mat_contract_ts[iter] = t 
        mul!(xₖ₊₁,Axₖᵐ⁻²,xₖ)

        λₖ₊₁ = dot(xₖ₊₁,xₖ)
        λs[iter] = λₖ
        λ_checks[iter] = sqrt((λₖ₊₁ - λₖ)^2/λₖ^2)
        
        if normalize; normalize!(xₖ₊₁); end
        iterate_diffs[iter] = norm(xₖ₊₁ - xₖ) / norm(xₖ₊₁) 
        
        if (λ_checks[iter] < ε && iterate_diffs[iter] < ε) || iter >= max_iter
            
            x_history[:,iter] = xₖ₊₁
            break
        else
            xₖ .= xₖ₊₁
            λₖ = copy(λₖ₊₁)

            #=
            output,t = @timed symeigs(Axₖᵐ⁻²,1,which=Λ,maxiter=1000,failonmaxiter=false)
            solver_ts[iter] = t

            if output.state.aupd_mxiter[] > 1000
                return Axₖᵐ⁻²
            end 
            =#
            integrate!(integration,Λ,Axₖᵐ⁻²,xₖ)
            if normalize; normalize!(xₖ); end
            x_history[:,iter] = xₖ
            iter += 1
        end 
        
    end 
    return λₖ, iter
end


function integrate!(integration::forwardEuler,Λ::Symbol,A,x)
    x .+= integration.h .*(eigenmap(Λ,A) .- x)
end 

function integrate!(integration::RungeKutta4thOrder,Λ::Symbol,A,x)
    throw(error("unimplemented"))
    #x .+= Integration.h .*(eigenmap(Λ,A) .- x)
end 


function eigenmap(which::Symbol,A)
    output = symeigs(A,1,which=which,maxiter=1000000,failonmaxiter=true)
    V = output.vectors
    return V .*= sign(V[1])
end 
