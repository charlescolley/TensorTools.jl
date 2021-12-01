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

function single_mode_ttv(A::SymTensorUnweighted{Clique},x::Vector{T}) where T
    #=
        Only computes a contraction for a single mode. Helpful for low rank 
        contraction. This version uses vector version of subedges to make it 
        easier to sort values.
    =#
    @assert A.n == length(x)

    edge_count = size(A.indices,2)
    #new_subedges = Array{Int,2}(undef,A.order-1,edge_count*A.order)
    new_subedges = Array{Tuple{Vector{Int},T},1}(undef,edge_count*A.order)

    
    #new_subedge_weights = Array{Int,1}(undef,edge_count*A.order)
    sub_edges_idx = 1

    free_mode_pattern = collect(combinations(1:A.order, A.order-1))
    contracted_mode_pattern = collect(A.order:-1:1)

    for edge_idx in 1:size(A.indices,2)
        subedge_offset = (edge_idx -1)*A.order
        for j in 1:A.order

            nodes_to_touch = free_mode_pattern[j]
            new_edge = Array{Int,1}(undef,A.order-1)
            for k in 1:A.order-1
                new_edge[k] = A.indices[nodes_to_touch[k],edge_idx]
            end
            new_subedges[subedge_offset + j] = (new_edge,x[A.indices[contracted_mode_pattern[j],edge_idx]])
        end
    end

    indices, weights = reduce_to_unique_edges(new_subedges)
    return SymTensor{Clique}(A.n,A.order-1,indices,weights)
end

function single_mode_ttv(A::SymTensor{Clique},x::Vector{T}) where T
    #=
        Only computes a contraction for a single mode. Helpful for low rank 
        contraction. This version uses vector version of subedges to make it 
        easier to sort values.
    =#
    @assert A.n == length(x)

    edge_count = size(A.indices,2)
    new_subedges = Array{Tuple{Vector{Int},T},1}(undef,edge_count*A.order)

    free_mode_pattern = collect(combinations(1:A.order, A.order-1))
    contracted_mode_pattern = collect(A.order:-1:1)

    for edge_idx in 1:size(A.indices,2)
        subedge_offset = (edge_idx -1)*A.order
        for j in 1:A.order

            nodes_to_touch = free_mode_pattern[j]
            new_edge = Array{Int,1}(undef,A.order-1)
            for k in 1:A.order-1
                new_edge[k] = A.indices[nodes_to_touch[k],edge_idx]
            end
            new_weight = x[A.indices[contracted_mode_pattern[j],edge_idx]]*A.weights[edge_idx]
            new_subedges[subedge_offset + j] = (new_edge,new_weight)
        end
    end

    indices, weights = reduce_to_unique_edges(new_subedges)
    return SymTensor{Clique}(A.n,A.order-1,indices,weights)
end

function reduce_to_unique_edges(new_subedges::Vector{Tuple{Vector{Int},T}}) where T 
    #assuming that new_subedges has at least 1 entry
    sort!(new_subedges,by=i->i[1])


    start_ptr = 1 
    consolidated_indices = Array{Int,2}(undef,length(new_subedges[1][1]),length(new_subedges))
    consolidated_weights = Array{T,1}(undef,length(new_subedges))
    idx = 1
    while start_ptr < length(new_subedges)

        (edge,weight) = new_subedges[start_ptr]
        consolidated_indices[:,idx] = edge
        consolidated_weights[idx] = weight

        # aggregate the edge weights together 
        edge_check_ptr = start_ptr + 1

        while new_subedges[edge_check_ptr][1] == edge
            consolidated_weights[idx] += new_subedges[edge_check_ptr][2]
            edge_check_ptr += 1          
            if edge_check_ptr > length(new_subedges)
                break
            end
        end

        start_ptr = edge_check_ptr
        idx += 1
    end

    #copy in the last edge 
    if start_ptr == length(new_subedges)
        (edge,weight) = new_subedges[start_ptr]
        consolidated_indices[:,idx] = edge
        consolidated_weights[idx] = weight
        return consolidated_indices[:,1:idx], consolidated_weights[1:idx]
    else
        return consolidated_indices[:,1:idx-1], consolidated_weights[1:idx-1]
    end

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


function compute_multinomial(indices::Vector{Int})

    label_freqency = Dict{Int,Int}()

    for i in indices
        if haskey(label_freqency,i)
            label_freqency[i] += 1
        else
            label_freqency[i] = 1
        end
    end

    return multinomial(values(label_freqency)...)
end

function contract_all_unique_permutations(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},U::Matrix{T}) where {M <: Motif,T}

    m,d = size(U)
    #@assert A.n == m 
    
    contraction_components = Array{T,2}(undef,m,binomial(d + A.order-2, A.order-1))
                                                # n choose k w/ replacement
    #println("contraction comps:$(size(contraction_components))")
    contract_all_unique_permutations!(A,U,contraction_components,0,size(U,2),Array{Int}(undef,0))
    
    return contraction_components
end

function contract_all_unique_permutations!(A::Union{SymTensor{M,T},SymTensorUnweighted{M}},U::Matrix{T},
                            contraction_components::Matrix{T},offset::Int,
                            end_idx::Int,prefix_indices::Vector{Int}) where {M <: Motif,T}

    d = size(U,2)
    if A.order == 3 
        indices = Array{Int}(undef,length(prefix_indices)+2)
        indices[1:end-2] = prefix_indices

        for i = 1:end_idx
            indices[end-1] = i  
            sub_A_mat = contract_to_mat(A, U[:,i])
            #for j = i:size(U,2)
            for j = 1:i
                indices[end] = j

                factor = compute_multinomial(indices)
                #println("edge:$indices  factor:$factor  ")

                idx = offset + i*(i-1)÷2 + j
            
                contraction_components[:,idx] = factor*(sub_A_mat*U[:,j])
            end
        end
    else

        running_offset = copy(offset)
        for i = 1:end_idx
            sub_A = single_mode_ttv(A,U[:,i])
            prefix = copy(prefix_indices)
            push!(prefix,i)
     
            contract_all_unique_permutations!(sub_A,U, contraction_components,running_offset,i,prefix) 

            running_offset += binomial(i + A.order-3, A.order-2)
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