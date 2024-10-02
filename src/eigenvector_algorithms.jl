abstract type Algorithm end 
abstract type InverseSolver end
struct GMRES <: InverseSolver end
struct MINRES <: InverseSolver end  
struct Backslash <: InverseSolver end 


include("dynamical_systems.jl")
include("newton_correction_methods.jl")

#
#   Sampling Procedure
#
function sample_eigenvectors(A_tensor,samples::Int,seed::Union{Nothing,UInt}=nothing,
                             zeig_method::Algorithm=NCM(1000,1e-5)) #TODO:edit it to how david does seeding

    if seed !== nothing 
        seed!(seed)
    end 

    Λ = Vector{Float64}(undef,samples)
    R = Vector{Float64}(undef,samples)
    iterations = Vector{Int}(undef,samples)

    X = randn(n(A_tensor),samples)
    V = similar(X)

    for s = 1:samples

        Λ[s], R[s], iterations[s] = tensor_eigenvector_method!(zeig_method, A_tensor, X[:,s], @view V[:,s])
        if s % 10 == 0
            println("finished sample $s")
        end
    end

    return (;vecs=V, vals=Λ, norm_diffs=R, iterations=iterations, 
            unique_eigs=unique_eigenvalues(((i,λ) for (i,λ) in enumerate(Λ) if iterations[i] < zeig_method.maxiter),tol(zeig_method)))
end 

function tensor_eigenvector_method!(args::NCM, A_tensor, x, v)
    return NCM!(A_tensor, args.maxiter, args.δ, x, v, args.solver)
end

function tensor_eigenvector_method!(args::ONCM, A_tensor, x, v)
    return ONCM!(A_tensor, args.maxiter, args.δ, x, v, args.solver)
end

function tensor_eigenvector_method!(args::DynamicalSystems, A_tensor, x, v)
    return dynamical_systems!(A_tensor, args.which, args.integrator, args.maxiter, args.ϵ, args.normalize, x, v)
end


#
#   Sampling Procedure
#
function sample_eigenvectors_profiled(A_tensor,samples::Int,seed::Union{Nothing,UInt}=nothing,
                             zeig_method::Algorithm=NCM(1000,1e-5)) #TODO:edit it to how david does seeding

    if seed !== nothing 
        seed!(seed)
    end 

    Λ = Vector{Float64}(undef,samples)
    R = Vector{Float64}(undef,samples)
    iterations = Vector{Int}(undef,samples)

    x_history = Array{Float64,3}(undef, n(A_tensor), zeig_method.maxiter, samples)
    iter_history = Array{Float64,3}(undef, zeig_method.maxiter, 6, samples)


    X = randn(n(A_tensor),samples)
    V = similar(X)
    Λ = Vector{Float64}(undef,samples)

    for s = 1:samples

        Λ[s], iterations[s] = @time tensor_eigenvector_method_profiled!(zeig_method, A_tensor,  
                                                                             @view(X[:,s]), @view(V[:,s]),
                                                                             @view(x_history[:,:,s]),
                                                                             @view(iter_history[:,:,s]))
        if s % 10 == 0  
            println("finished sample $s")
        end
    end

    converged_eigenvalues = unique_eigenvalues(((i,λ) for (i,λ) in enumerate(Λ) if iterations[i] < zeig_method.maxiter),tol(zeig_method))
    return (;vals=Λ, norm_diff=R, iter=iterations, xs=x_history, tracked_stats=iter_history, vecs=V,
           unique_vals =converged_eigenvalues)
end 

function tensor_eigenvector_method_profiled!(args::NCM, A_tensor, x, v, x_history, iter_history)
    return NCM_profiled!(A_tensor, args.maxiter, args.δ, x, v, x_history, iter_history, args.solver)
end

function tensor_eigenvector_method_profiled!(args::ONCM, A_tensor, x, v, x_history, iter_history)
    return ONCM_profiled!(A_tensor, args.maxiter, args.δ, x, v, x_history, iter_history, args.solver)
end

function tensor_eigenvector_method_profiled!(args::DynamicalSystems, A_tensor, x, v, x_history, iter_history)
    return dynamical_systems_profiled!(A_tensor, args.which, args.integrator, args.maxiter, 
                                       args.ϵ, args.normalize, x, v, x_history, iter_history)
end


"""
    unique_eigenvalues()
"""
function unique_eigenvalues(Λ,tolerance_computed)
    eigenvalues = Dict{Float64,Vector{Int}}() 
    for (i,λ) in Λ

        λ = abs(round(λ,digits=-Int(ceil(log10(tolerance_computed)))))
                            # Z-eigs computed to δ tolerance
        if haskey(eigenvalues,λ)
            push!(eigenvalues[λ],i)
        else
            eigenvalues[λ] = Float64[i]
        end 
    end

    return eigenvalues
end

function unique_eigenvalues(Λ::AbstractVector{T},tolerance_computed) where {T <: Real}
    return unique_eigenvalues(enumerate(Λ), tolerance_computed)
end



function eigenspace_residual(A,x;verbose=false)
    
    Axᵏ⁻¹ = contraction(A,x)
    λ =  Axᵏ⁻¹' * x 

    verbose && println("λ:$(λ)")
    return norm(Axᵏ⁻¹ - λ * x)
end 

function eigenspace_residual(A,λ,x;verbose=false)
    
    Axᵏ⁻¹ = contraction(A,x)
    computed_λ =  Axᵏ⁻¹' * x 

    verbose && println("λ:$(λ)")
    return abs(computed_λ - λ), norm(Axᵏ⁻¹ - λ * x)
end 


# all methods should profile the same things 
# TODO: may need to re-evaluate this setup 
function setup_profiling(n,max_iter)

    x_history = Matrix{Float64}(undef,n, max_iter)
    iter_history = -ones(Float64,max_iter,6)

    #  -- views made help document column to measurement 
    

    return x_history, iter_history, # views may be ignored if unneeded
           (iterate_diffs,λs,method_residuals,solver_residuals,
            mat_contract_ts,solver_ts)   
           

end 