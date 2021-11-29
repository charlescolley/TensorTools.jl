#
#    Clique based contraction routines
#

#TODO: test
function contraction!(A::SymTensorUnweighted{Clique}, x::Array{T,1},y::Array{T,1}) where T 

    order, edges = size(A.indices)

    scaling_factor = factorial(order-1)

    for i=1:edges
        for j=1:order

            val = scaling_factor

            for k=1:j-1
                val *= x[A.indices[k,i]]
            end

            for k=j+1:order
                val *= x[A.indices[k,i]]
            end
            y[A.indices[j,i]] += val
        end
    end
end

function contraction!(A::SymTensor{Clique}, x::Array{T,1},y::Array{T,1}) where T 

    order, edges = size(A.indices)

    scaling_factor = factorial(order-1)

    for i=1:edges
        for j=1:order

            val = scaling_factor*A.weights[i]

            for k=1:j-1
                val *= x[A.indices[k,i]]
            end

            for k=j+1:order
                val *= x[A.indices[k,i]]
            end
            y[A.indices[j,i]] += val
        end
    end
end



#TODO: test
function contraction_divide_out!(A::SymTensorUnweighted{Clique}, x::Array{T,1},y::Array{T,1}) where T 

    order, edges = size(A.indices)
    scaling_factor = factorial(order-1)

    for i=1:edges
        val = scaling_factor
        for j=1:order
            val *= x[A.indices[j,i]]
        end

        for j=1:order
            y[A.indices[j,i]] += (val/x[A.indices[j,i]])
        end
    end
end

function contraction_divide_out!(A::SymTensor{Clique}, x::Array{T,1},y::Array{T,1}) where T 

    order, edges = size(A.indices)
    scaling_factor = factorial(order-1)

    for i=1:edges
        val = scaling_factor*A.weights[i]
        for j=1:order
            val *= x[A.indices[j,i]]
        end

        for j=1:order
            y[A.indices[j,i]] += (val/x[A.indices[j,i]])
        end
    end
end

function contraction_divide_out!(simplicial_complexes::Union{Vector{SymTensorUnweighted{Clique}},Vector{SymTensor{Clique}}},x, y)
    #assuming simplicial_complexes are sorted by their motif order

    for i=1:length(simplicial_complexes)
        contraction_divide_out!(simplicial_complexes[i],x,y)
    end
end

function contract_to_mat(A_ten::SymTensorUnweighted{Clique},x::Array{T}) where T

    scaling_factor = factorial(A_ten.order-2)
    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    new_edge_idx = 1 

    for edge_idx = 1:size(A_ten.indices,2)

        for i in 1:size(A_ten.indices,1)
            for j = i + 1: size(A_ten.indices,1)
                val = scaling_factor;
                for k = 1:size(A_ten.indices,1)
                    if k != i && k != j 
                        val *= x[A_ten.indices[k,edge_idx]]
                    end
                end

                is[new_edge_idx] = A_ten.indices[i,edge_idx]    
                js[new_edge_idx] = A_ten.indices[j,edge_idx]
                vs[new_edge_idx] = val
                new_edge_idx  += 1
            end

        end
    
    end 
    A = sparse(is,js,vs,A_ten.n,A_ten.n)
    return A + A' 
        # expecting diag(A) = \vec{0}
end

function contract_to_mat(A_ten::SymTensor{Clique},x::Array{T}) where T

    scaling_factor = factorial(A_ten.order-2)
    binom_factor = binomial(A_ten.order,A_ten.order-2)
    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    new_edge_idx = 1 

    for edge_idx = 1:size(A_ten.indices,2)

        for i in 1:size(A_ten.indices,1)
            for j = i + 1: size(A_ten.indices,1)
                val = scaling_factor*A_ten.weights[edge_idx];
                for k = 1:size(A_ten.indices,1)
                    if k != i && k != j 
                        val *= x[A_ten.indices[k,edge_idx]]
                    end
                end

                is[new_edge_idx] = A_ten.indices[i,edge_idx]    
                js[new_edge_idx] = A_ten.indices[j,edge_idx]
                vs[new_edge_idx] = val
                new_edge_idx  += 1
            end

        end
    
    end 
    A = sparse(is,js,vs,A_ten.n,A_ten.n)
    return A + A' 
        # expecting diag(A) = \vec{0}
end

function contract_to_mat_divide_out(A_ten::SymTensorUnweighted{Clique},x::Array{T}) where T

    scaling_factor = factorial(A_ten.order-2)
    binom_factor = binomial(A_ten.order,A_ten.order-2)

    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    new_edge_idx = 1 

    for edge_idx = 1:size(A_ten.indices,2)

        val = scaling_factor;
        for i = 1:size(A_ten.indices,1)
            val *= x[A_ten.indices[i,edge_idx]]
        end
        #val = A->vals[i]*scaling_factor;

        for i in 1:size(A_ten.indices,1)
            val /= x[A_ten.indices[i,edge_idx]]
            for j = i + 1: size(A_ten.indices,1)
                val /= x[A_ten.indices[j,edge_idx]]

                is[new_edge_idx] = A_ten.indices[i,edge_idx]    
                js[new_edge_idx] = A_ten.indices[j,edge_idx]
                vs[new_edge_idx] = val
                new_edge_idx  += 1

                val *= x[A_ten.indices[j,edge_idx]]

            end
            val *= x[A_ten.indices[i,edge_idx]]
        end
    end 
    A = sparse(is,js,vs,A_ten.n,A_ten.n)
    return A + A' 
        # expecting diag(A) = \vec{0}
end

function contract_to_mat_divide_out(A_ten::SymTensor{Clique},x::Array{T}) where T

    scaling_factor = factorial(A_ten.order-2)
    binom_factor = binomial(A_ten.order,A_ten.order-2)

    is = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    js = Array{Int}(undef, binom_factor*size(A_ten.indices, 2))
    vs = Array{T}(undef, binom_factor*size(A_ten.indices, 2))

    new_edge_idx = 1 

    for edge_idx = 1:size(A_ten.indices,2)

        val = scaling_factor*A_ten.weights[edge_idx];
        for i = 1:size(A_ten.indices,1)
            val *= x[A_ten.indices[i,edge_idx]]
        end
        #val = A->vals[i]*scaling_factor;

        for i in 1:size(A_ten.indices,1)
            val /= x[A_ten.indices[i,edge_idx]]
            for j = i + 1: size(A_ten.indices,1)
                val /= x[A_ten.indices[j,edge_idx]]

                is[new_edge_idx] = A_ten.indices[i,edge_idx]    
                js[new_edge_idx] = A_ten.indices[j,edge_idx]
                vs[new_edge_idx] = val
                new_edge_idx  += 1

                val *= x[A_ten.indices[j,edge_idx]]

            end
            val *= x[A_ten.indices[i,edge_idx]]
        end
    end 
    A = sparse(is,js,vs,A_ten.n,A_ten.n)
    return A + A' 
        # expecting diag(A) = \vec{0}
end


function embedded_contraction!(A::SymTensorUnweighted{Clique}, x::Array{T,1},y::Array{T,1},embedded_mode::Int) where T

    order,edges = size(A.indices)
    #@assert order == 3

    # compute contraction template
    partition::Vector{Vector{Int}} = [ x for x in integer_partitions(embedded_mode - order) if length(x) <= order]
    #println(partition)
    partition = [length(y) < order ? vcat(y,zeros(Int,order - length(y))) : y  for y in partition]
                     #pad the entries to all have the same length
    partition = [x .+= 1 for x in partition]
    
    multiplicities::Array{Array{Int,1},1} = unique(hcat([collect(permutations(y)) for y in partition]...))

    scaling_factors = Array{Int,2}(undef,order,length(multiplicities))
    for i =1:length(multiplicities)
        for j=1:order
            row = copy(multiplicities[i])
            row[j] -= 1
            scaling_factors[j,i] = multinomial(row...)
        end
        #push!(scaling_factors,row_factors)
    end

    #return partition, scaling_factors

    for idx =1:edges

        for idx2 =1:length(multiplicities)
            for idx3 =1:order
                val::Float64 = scaling_factors[idx3,idx2]

                for idx4=1:idx3-1
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end
                val *= x[A.indices[idx3,idx]]^(multiplicities[idx2][idx3] -1)
                for idx4=idx3+1:order
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end

                y[A.indices[idx3,idx]] += val
            end
        end
    
    end
end

function embedded_contraction!(simplicial_complexes::Array{SymTensorUnweighted{S},1},x, y) where {S <: Motif}
    #assuming simplicial_complexes are sorted by their motif order

    max_order = size(simplicial_complexes[end].indices,1)

    for i=1:length(simplicial_complexes)
        if i == length(simplicial_complexes)
            #contraction_divide_out!(simplicial_complexes[i],x,y)
            contraction!(simplicial_complexes[i],x,y)
        else
            embedded_contraction!(simplicial_complexes[i],x,y,max_order)
        end
    end
end



#
#    Cycle based contraction routines
#

function contraction!(A::SymTensorUnweighted{Cycle}, x::Array{T,1},y::Array{T,1}) where T

    order, edges = size(A.indices)

    for i=1:edges
        for j=1:order

            val = 2 #for each orientation

            for k=1:j-1
                val *= x[A.indices[k,i]]
            end

            for k=j+1:order
                val *= x[A.indices[k,i]]
            end
            y[A.indices[j,i]] += val
        end
    end
end


function embedded_contraction!(A::SymTensorUnweighted{Cycle}, x::Array{T,1},y::Array{T,1},embedded_mode::Int) where T

    order,edges = size(A.indices)
    #@assert order == 3

    # compute contraction template
    partition = [ x for x in integer_partitions(embedded_mode - order) if length(x) <= order]
    #println(partition)
    partition = [length(y) < order ? vcat(y,zeros(Int,order - length(y))) : y  for y in partition]
                     #pad the entries to all have the same length
    partition = [x .+= 1 for x in partition]
    
    multiplicities::Array{Array{Integer,1},1} = unique(hcat([collect(permutations(y)) for y in partition]...))

    #=
    scaling_factors = Array{Integer,2}(undef,order,length(multiplicities))
    for i =1:length(multiplicities)
        for j=1:order
            row = copy(multiplicities[i])
            row[j] -= 1
            scaling_factors[j,i] = multinomial(row...)
        end
        #push!(scaling_factors,row_factors)
    end
    =#

    for idx =1:edges

        for idx2 =1:length(multiplicities)
            for idx3 =1:order
                val::Float64 = 2.0#scaling_factors[idx3,idx2]

                for idx4=1:idx3-1
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end
                val *= x[A.indices[idx3,idx]]^(multiplicities[idx2][idx3] -1)
                for idx4=idx3+1:order
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end

                y[A.indices[idx3,idx]] += val
            end
        end
    
    end
end



#
#  Duck Typed Allocators
#

function contraction(A, x::Array{T,1}) where T 

    # using length of x rather than A.n to make more robust against different types of A 
    y = zeros(T,length(x))
    contraction!(A, x, y)
    return y 
end

function embedded_contraction(A, x::Array{T,1}) where T 
    # using length of x rather than A.n to make more robust against different types of A 
    y = zeros(T,length(x))
    embedded_contraction!(A, x, y)
    return y 
end

function contraction_divide_out(A, x::Array{T,1}) where T 
    # using length of x rather than A.n to make more robust against different types of A 
    y = zeros(T,length(x))
    contraction_divide_out!(A, x, y)
    return y 
end