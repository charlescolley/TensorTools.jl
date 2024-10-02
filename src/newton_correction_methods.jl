
# TODO: Make Algorithm take maxiter and δ
struct NCM <: Algorithm 
    maxiter::Int
    δ::Float64
    solver::InverseSolver
end 

struct ONCM <: Algorithm 
    maxiter::Int
    δ::Float64
    solver::InverseSolver
end

tol(args::Union{NCM,ONCM}) = args.δ 


function NCM(A_ten,max_iter,δ,x;kwargs...)
    y = similar(x)
    return y, NCM!(A_ten,max_iter,δ,x,y;kwargs...)...
end 
NCM(A_ten,max_iter::Int,δ::Float64;kwargs...) = NCM(A_ten,max_iter,δ,randn(A_ten.n);kwargs...)

function ONCM(A_ten,max_iter,δ,x;kwargs...)
    y = similar(x)
    return y, ONCM!(A_ten,max_iter,δ,x,y;kwargs...)...
end 
ONCM(A_ten,max_iter::Int,δ::Float64) = ONCM(A_ten,max_iter,δ,randn(A_ten.n);kwargs...)

#
#    NCM
#

function ncm_linop(order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ)
    
    #=
    return function ncm_linop_curry(z,x)

        z .= zero(eltype(x))

        mul!(z,Axₖᵐ⁻²,x)
        z .*= (order-1)
        z .-= λ .* x

        z .-= ((order * (dot(Axₖᵐ⁻¹,x))) .* xₖ)
        #z .-= ((dot(Axₖᵐ⁻¹,x)) .* xₖ)
        
    end=#

    return function ncm_linop_curry(x)

        z = zeros(eltype(x),length(x))
        # -- A(x) = H(x) - m ⋅ x(Axᵐ⁻¹)ᵀ
        #    H(x) = (m−1) ⋅ Axᵐ⁻² − μ(x) ⋅I.
        #    μ(x) = Axᵐ = λ
        # -- A(x) = (m−1) ⋅ Axᵐ⁻² − Axᵐ⋅I. - m ⋅ x(Axᵐ⁻¹)ᵀ

        # -- (m−1) ⋅ Axᵐ⁻²
        mul!(z,Axₖᵐ⁻²,x)
        z .*= (order-1)

        #  -- -Axᵐ⋅I
        z .-= λ .* x
    
        #  -- - m ⋅ x(Axᵐ⁻¹)ᵀ
        z .-= ((order * (dot(Axₖᵐ⁻¹,x))) .* xₖ)
        #z .-= ((dot(Axₖᵐ⁻¹,x)) .* xₖ)
        return z 
    end 
end

function ncm_linop_adjoint(order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ)
    
    #=return function ncm_linop_curry(z,x)

        z .= zero(eltype(x))
        
       
        mul!(z,Axₖᵐ⁻²,x)
        z .*= (order-1)
        z .-= λ .* x

        z .-= ((order * (dot(xₖ,x))) .* Axₖᵐ⁻¹)
        #z .-= ((dot(xₖ,x)) .* Axₖᵐ⁻¹)
        
    end =#

    return function ncm_linop_curry(x)

        z = zeros(eltype(x),length(x))
        
       
        mul!(z,Axₖᵐ⁻²,x)
        z .*= (order-1)
        z .-= λ .* x

        z .-= ((order * (dot(xₖ,x))) .* Axₖᵐ⁻¹)
        #z .-= ((dot(xₖ,x)) .* Axₖᵐ⁻¹)
        return z 
    end 
end

"""
   solves (A + uvᵀ)x = b through the sherman morrison formula
   (A + uvᵀ)⁻¹ = A⁻¹ + (A⁻¹uvᵀA⁻¹)/(1 + vᵀA⁻¹u) 
"""
function sherman_morrison_inverse!(A,u,v,b,x,solver::InverseSolver;verbose=false)

    A⁻¹b = similar(b)
    A⁻¹u = similar(u)
    solve!(A,b,A⁻¹b, solver)
    solve!(A,u,A⁻¹u, solver)
    #A⁻¹b = gmres(A,b) 
    #A⁻¹u = gmres(A,u)
        
    verbose && @printf("res(A⁻¹u)=%.18f -- res(A⁻¹b)=%.18f\n",
            norm(A*(A⁻¹u) - u),norm(A*(A⁻¹b) - b))
    #@printf("vᵀA⁻¹u=%.18f\n",dot(v,A⁻¹u))

    #  -- x = A⁻¹b - ((A⁻¹u * (v' * A⁻¹b)) /  (1 + (v'* A⁻¹u)))

    x .= A⁻¹u .* (v' * A⁻¹b)
    x ./= -(1 + (v'* A⁻¹u))
    x .+= A⁻¹b 
    
    verbose && @printf("res((A+uvᵀ)⁻¹b)=%.18f\n",
                       norm(A*x + u*dot(v,x) - b))

end 

function solve!(A,b,y,::GMRES)
    y .= gmres(A,b) 
end

function solve!(A,b,y,::MINRES)
    minres!(y,A,b) 
end

function solve!(A,b,y,::Backslash)
    y .= A \ b
end

function NCM!(A_ten, max_iter::Integer, δ::Float64,
              xₖ::AbstractVector{T}, xₖ₊₁::AbstractVector{T},
              solver::InverseSolver;verbose=false) where T
                                                #   k=0

    xₖ ./= norm(xₖ)

    #X = zeros(A_ten.n,max_iter)
    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    Axₖᵐ⁻¹ = similar(xₖ)    
    g = similar(xₖ)

    h::Float64 = 1.0 
    iter = 1 

    residual = (A,u,v,x,b) -> norm(A*x + u*(v'*x) - b)/norm(b)

    while true
   
        Axₖᵐ⁻² = contract_to_mat!(A_ten,xₖ,is,js,vs)
        mul!(Axₖᵐ⁻¹,Axₖᵐ⁻²,xₖ)
        λ = Axₖᵐ⁻¹'*xₖ

        #  -- g = Axₖᵐ⁻¹ - λxₖ
        g .= xₖ
        g .*= -λ
        g .+= Axₖᵐ⁻¹

        #  -- H(xₖ) = (m-1)Axₖᵐ⁻² - λI 
        Axₖᵐ⁻².data.nzval .*= (A_ten.order-1)
        Axₖᵐ⁻²[diagind(Axₖᵐ⁻²)] .-= λ # add in a shift
        
        #  -- y = (H(xₖ) - mxₖ)(Axₖᵐ⁻¹)ᵀ)⁻¹(-g)
        #                                  -1 included for xₖ₊₁ .+= xₖ - y
        #g .*= -1 # 
        sherman_morrison_inverse!(Axₖᵐ⁻²,-A_ten.order*xₖ,Axₖᵐ⁻¹,g,xₖ₊₁,solver)
        
        # =
        xₖ₊₁ .*= -h#^(Int(ceil(log10(iter))))
        xₖ₊₁ .+= xₖ
        xₖ₊₁ ./= norm(xₖ₊₁)
        
        step_residual = residual(Axₖᵐ⁻²,-A_ten.order*xₖ,Axₖᵐ⁻¹,xₖ₊₁,g)#norm(Axₖᵐ⁻²*xₖ₊₁ - A_ten.order*xₖ*(Axₖᵐ⁻¹'*xₖ₊₁) - g)/norm(g)
        R = norm(xₖ₊₁ - xₖ);

        verbose && @printf("%5d -- ||xₖ₊₁ - xₖ||=%.18f  λ=%.18f ||Axₖᵐ⁻¹||=%.12f  ||Lh - g||/||g||=%.18f  <m*xₖ,Axₖᵐ⁻¹>=%.12f\n",
                            iter,R,λ,norm(Axₖᵐ⁻¹),step_residual,A_ten.order*dot(xₖ,Axₖᵐ⁻¹))

        if (R < δ || iter >= max_iter)
            return (contraction(A_ten,xₖ₊₁)'*xₖ, R, iter)
        else
            xₖ .= xₖ₊₁ # TODO: check if I need to explicitly copy.
            #X[:,iter] = xₖ₊₁
            iter += 1
            
        end 
    end
end

function NCM_profiled!(A_ten, max_iter, δ, xₖ, xₖ₊₁)

    x_history = Matrix{Float64}(undef,A_ten.n, max_iter)
    iter_history = -ones(Float64,max_iter,6)

    return NCM_profiled!(A_ten, max_iter, δ, xₖ, xₖ₊₁, x_history, iter_history)    
end

function NCM_profiled!(A_ten, max_iter::Integer, δ::Float64, xₖ::AbstractVector{T}, xₖ₊₁::AbstractVector{T},
                       x_history::AbstractArray{T,2}, iter_history::AbstractArray{F,2},solver::InverseSolver) where {T, F <: AbstractFloat}
                                                #   k=0

    xₖ ./= norm(xₖ)

    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    Axₖᵐ⁻¹ = similar(xₖ)    
    g = similar(xₖ)

    h::Float64 = 1.0 # TODO: consider making this input 
    iter = 1
    
    residual = (A,u,v,x,b) -> norm(A*x + u*(v'*x) - b)/norm(b)


    #  -- Profiling setup --  #
    
    #x_history, iter_history, 
    #(iterate_diffs,λs,method_residuals,solver_residuals,mat_contract_ts,solver_ts) = setup_profiling(A_ten.n,max_iter)


    iterate_diffs = @view(iter_history[:,1])
    λs = @view(iter_history[:,2])
    method_residuals = @view(iter_history[:,3]) # NCM Method residuals
    solver_residuals = @view(iter_history[:,4]) # iterative solver residual
    mat_contract_ts = @view(iter_history[:,5])
    solver_ts = @view(iter_history[:,6])

    while true
        
        
        Axₖᵐ⁻²,t = @timed contract_to_mat!(A_ten,xₖ,is,js,vs)
        mat_contract_ts[iter] = t
        mul!(Axₖᵐ⁻¹,Axₖᵐ⁻²,xₖ)
        λ = Axₖᵐ⁻¹'*xₖ

        #  -- g = Axₖᵐ⁻¹ - λxₖ
        g .= xₖ
        g .*= -λ
        g .+= Axₖᵐ⁻¹


        #  -- H(xₖ) = (m-1)Axₖᵐ⁻² - λI 
        Axₖᵐ⁻².data.nzval .*= (A_ten.order-1)
        Axₖᵐ⁻²[diagind(Axₖᵐ⁻²)] .-= λ # add in a shift
        

        #L = LinearMap(ncm_linop(A_ten.order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ),A_ten.n,
        #              issymmetric=true,ismutating=true)
        

        #  -- y = (H(xₖ) - (mxₖ)(Axₖᵐ⁻¹)ᵀ)⁻¹(-g)
        #g,t = @timed gmres(L,g)        
        
        t = @timed sherman_morrison_inverse!(Axₖᵐ⁻²,-A_ten.order*xₖ,Axₖᵐ⁻¹,g,xₖ₊₁,solver)
        solver_ts[iter] = t.time

        xₖ₊₁ .*= -h#^(Int(ceil(log10(iter))))
        xₖ₊₁ .+= xₖ
        xₖ₊₁ ./= norm(xₖ₊₁)
        
        #xₖ₊₁ .= (xₖ - g)
        #xₖ₊₁ ./= norm(xₖ₊₁)
    
        method_residuals[iter] = residual(Axₖᵐ⁻²,-A_ten.order*xₖ,Axₖᵐ⁻¹,xₖ₊₁,g)
        iterate_diffs[iter] = norm(xₖ₊₁ - xₖ); 
        x_history[:,iter] = xₖ₊₁ # TODO: maybe only store last k=100 iterations?
        λs[iter] = λ


        if (iterate_diffs[iter] < δ || iter >= max_iter)
            return contraction(A_ten,xₖ₊₁)'*xₖ, iter
        else
            xₖ .= xₖ₊₁ # TODO: check if I need to explicitly copy.
            iter += 1
        end 
    end

    
end


#
#    ONCM
#

function oncm_linop(order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ)
    return function oncm_linop_curry(z,y)
        # PᵀAP = (I - xxᵀ)((m-1)Axᵐ⁻² - λI + k ̇x(Axᵏ⁻¹)ᵀ)(I - xxᵀ)

        #z .= zero(eltype(y))
        #  -- (I - xxᵀ)
        y .-= (dot(xₖ,y) .* xₖ)

        #  -- ((m-1)Axₖᵐ⁻² - λI)
        mul!(z,Axₖᵐ⁻²,y)   
        z .*= (order-1)
        z .-= λ .* y

        z .-= ((order * (dot(Axₖᵐ⁻¹,y))) .* xₖ)

        #  -- (I - xxᵀ)
        z .-= dot(xₖ,z) .* xₖ

    end 
end

function ONCM!(A_ten, max_iter::Integer, δ::Float64, 
               xₖ::AbstractVector{T}, xₖ₊₁::AbstractVector{T},solver::InverseSolver;
               #k=0
               verbose=false) where T


    xₖ ./= norm(xₖ)

    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    Axₖᵐ⁻¹ = similar(xₖ)
    g = similar(xₖ)
    h = similar(xₖ)
    xₖ₊₁ .= rand(Float64)    

    iter = 1

    while true
   
        Axₖᵐ⁻² = contract_to_mat!(A_ten,xₖ,is,js,vs)
        mul!(Axₖᵐ⁻¹,Axₖᵐ⁻²,xₖ)
        λ = Axₖᵐ⁻¹'*xₖ

        #g = -λ*xₖ + Axₖᵐ⁻¹
        g .= xₖ
        g .*= -λ
        g .+= Axₖᵐ⁻¹
       

        L = LinearMap(oncm_linop(A_ten.order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ),A_ten.n,
                      issymmetric=true,ismutating=true)
        g .-= (dot(g,xₖ) .* xₖ) # project out xₖ

        solve!(L,g,h,solver)
    
        xₖ₊₁ .= (xₖ .- h)
        xₖ₊₁ ./= norm(xₖ₊₁)

        #println()
        #println("residual:$( norm( L * h - (g - dot(g,xₖ) .* xₖ) ))")
        #println("dot(h,xₖ):$( dot(h, xₖ) )")
        
        
        R = norm(xₖ₊₁ - xₖ);
        verbose && @printf("%5d -- ||xₖ₊₁ - xₖ||=%.10f  λ=%.10f\n",iter,R,λ)
        #verbose && @printf("%5d -- ||xₖ₊₁ - xₖ||=%.18f  λ=%.18f ||Axₖᵐ⁻¹||=%.12f  ||Lh - g||/||g||=%.18f  <m*xₖ,Axₖᵐ⁻¹>=%.12f\n",
        #                    iter,R,λ,norm(Axₖᵐ⁻¹),step_residual,A_ten.n*dot(xₖ,Axₖᵐ⁻¹))
        #=
        verbose && @printf("%5d -- ||xₖ₊₁ - xₖ||=%.10f  λ=%.10f  max(|λ(L)|)=%.10f  min(|λ(L)|)=%.10f  κ(L)=%.10f  |λ(L) > 0|=%d  |λ(L) <= 0|=%d\n",
        iter,R,λ,maximum(abs_eigs),minimum(abs_eigs),
        maximum(abs_eigs)/minimum(abs_eigs),
        sum(eig_vals .> 0),sum(eig_vals .<= 0))
        =#
        
        if (R < δ || iter >= max_iter)
            return (contraction(A_ten,xₖ₊₁)'*xₖ, R, iter)
        else
            xₖ .= xₖ₊₁ # TODO: check if I need to explicitly copy.
            iter += 1
        end 
        
    end

    
end

function ONCM_profiled!(A_ten, max_iter, δ, xₖ, xₖ₊₁)

    x_history = Matrix{Float64}(undef,n, max_iter)
    iter_history = -ones(Float64,max_iter,6)

    return ONCM_profiled!(A_ten, max_iter, δ, xₖ, xₖ₊₁, x_history, iter_history), x_history, iter_history
end 

function ONCM_profiled!(A_ten, max_iter::Integer, δ::Float64, xₖ::AbstractVector{T}, xₖ₊₁::AbstractVector{T},
                                                            #  k=0
                        x_history::AbstractArray{T,2}, iter_history::AbstractArray{F,2},solver::InverseSolver) where {T, F <: AbstractFloat}
                                                

    xₖ ./= norm(xₖ)

    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))
    

    Axₖᵐ⁻¹ = similar(xₖ)
    g = similar(xₖ)
    h = similar(xₖ)
    xₖ₊₁ .= rand(Float64)

    iter = 1

    #  -- Profiling Setup --  #

    iterate_diffs = @view(iter_history[:,1])
    λs = @view(iter_history[:,2])
    method_residuals = @view(iter_history[:,3]) # NCM Method residuals
    solver_residuals = @view(iter_history[:,4]) # iterative solver residual
    mat_contract_ts = @view(iter_history[:,5])
    solver_ts = @view(iter_history[:,6])

    #x_history, iter_history, 
    #(iterate_diffs,λs,method_residuals,solver_residuals,mat_contract_ts,solver_ts) = setup_profiling(A_ten.n,max_iter)


    while true
   
        Axₖᵐ⁻²,t = @timed contract_to_mat!(A_ten,xₖ,is,js,vs)
        mat_contract_ts[iter] = t
        mul!(Axₖᵐ⁻¹,Axₖᵐ⁻²,xₖ)
        λ = Axₖᵐ⁻¹'*xₖ

        g .= xₖ
        g .*= -λ
        g .+= Axₖᵐ⁻¹
        #g = -λ*xₖ + Axₖᵐ⁻¹

        L = LinearMap(oncm_linop(A_ten.order,Axₖᵐ⁻²,Axₖᵐ⁻¹,λ,xₖ),A_ten.n,
                      issymmetric=true,ismutating=true)
        g .-= (dot(g,xₖ) .* xₖ) # project out xₖ

        t = @timed solve!(L,-g,xₖ₊₁,solver)
        solver_ts[iter] = t.time
        solver_residuals[iter] = norm(L*xₖ₊₁ + g)/norm(g)
                                            # use `+` bc 
                                            # solve uses -g
        #xₖ₊₁ .*= -1
        xₖ₊₁ .+= xₖ
        #xₖ₊₁ .= (xₖ .- h)
        xₖ₊₁ ./= norm(xₖ₊₁)

        R = norm(xₖ₊₁ - xₖ);
        iterate_diffs[iter] = R;
        x_history[:,iter] = xₖ₊₁ # TODO: maybe only store last k=100 iterations?
        λs[iter] = λ

        if (R < δ || iter >= max_iter)
            return contraction(A_ten,xₖ₊₁)'*xₖ, iter      
        else
            xₖ .= xₖ₊₁ # TODO: check if I need to explicitly copy.
            iter += 1
        end 
    end

    
end

