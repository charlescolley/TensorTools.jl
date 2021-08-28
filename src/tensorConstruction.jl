

#
#    Clique Motif Routines
#

function tensors_from_graph(A, orders::Array{Int,1}, sample_size::Int, motif::Clique)

    tensors = Array{SymTensorUnweighted{Clique},1}(undef,length(orders))

    for (i,order) in enumerate(orders)
        tensor = tensor_from_graph(A, order, sample_size,motif)
        tensors[i] = tensor
    end

    return tensors

end

function tensors_from_graph(A, orders::Array{Int,1}, sample_sizes::Array{Int,1}, motif::Clique)

    @assert length(orders) == length(sample_sizes)
    tensors = Array{SymTensorUnweighted{Clique},1}(undef,length(orders))

    for (i,(order,sample)) in enumerate(zip(orders,sample_sizes))
        if order == 2
            tensors[i] = matrix_to_SymTensorUnweighted(A, motif)
        else
            tensors[i] = tensor_from_graph(A, order, sample, motif)
        end
    end

    return tensors
end

function tensor_from_graph(A, order, t, motif::Clique)

    _, cliques::Array{Array{Int64,1},1} = TuranShadow(A,order,t)

    reduce_to_unique_cliques!(cliques)

    indices = zeros(order,length(cliques))
    idx = 1
    #n = -1
    
    for clique in cliques #is there a better way to do this? 
        indices[:,idx] = clique
        #n_c = maximum(clique)
        #if n < n_c
        #    n = n_c
        #end
        idx += 1;
    end

    
    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices))

end


#
#    Cycle Motif Routines
#

function tensor_from_graph(A, order::Int, sample_size::Int, motif::Cycle)

    cycle_tensors = tensors_from_graph(A,order,sample_size,motif) 
    return cycle_tensors[end]
        # Johnson's algorithm returns many smaller cycles too, only return the requested ones

end

function tensors_from_graph(A, orders::Array{Int,1}, sample_size::Int, motif::Cycle)

    cycle_tensors = tensors_from_graph(A, maximum(orders), sample_size, motif) 
    return [cycle_tensors[order-1] for order in orders]
        # Johnson's algorithm returns many smaller cycles too, only return the requested ones

end

"""
function tensors_from_graph(A::SparseMatrixCSC{T,Int64}, max_length::Int,samples::Int,motif::Cycle) where T

	LG_A = LightGraphs.SimpleGraph(A)

	cycles = simplecycles_limited_length(LG_A,max_length,samples)

	cycleLengthCount = zeros(Int, max_length)
	for cycle in cycles
		cycleLengthCount[length(cycle)] += 1
	end


	#Preallocate needed memory
	IncidenceTensors = Array{Array{Integer,2},1}(undef,max_length)
	for l =1:max_length
		IncidenceTensors[l] = Array{Integer,2}(undef,l,cycleLengthCount[l]) 
	end
	IncidenceHeadPters = ones(Integer,max_length)

	for cycle in cycles
		l = length(cycle)
        IncidenceTensors[l][:,IncidenceHeadPters[l]] = cycle
        IncidenceHeadPters[l] += 1
        ""
        if issorted(cycle)
			IncidenceTensors[l][:,IncidenceHeadPters[l]] = cycle
			IncidenceHeadPters[l] += 1
		end
        ""
	end

	#trim the 2D arrays for the incidence matrix
	for l = 2:max_length
		IncidenceTensors[l] = IncidenceTensors[l][:,1:(IncidenceHeadPters[l]-1)]
	end 	


	#return IncidenceTensors
	return [DistributedTensorConstruction.SymTensorUnweighted{Cycle}(size(A,1),l,IncidenceTensors[l]) for l in 2:max_length]
                                                                                                             #ignore the singleton node returned 
end
"""

function tensors_from_graph(A::SparseMatrixCSC{T,Int64}, max_length::Int,samples::Int,motif::Cycle) where T

	LG_A = LightGraphs.SimpleGraph(A)

	cycles = simplecycles_limited_length(LG_A,max_length,samples)
    sort!(cycles,by=c->length(c))

    cycle_length_ptrs = ones(Int,max_length-1) #LightGraphs doesn't return anything len(c)==1
    l_prev = length(cycles[1])
    cl_ptr = 2 # we know cycles of len 2 start at 1
    for i = 2:length(cycles)
        l = length(cycles[i])

        if l > l_prev
            cycle_length_ptrs[cl_ptr] = i
            cl_ptr += 1
            l_prev = l
        end
    end

    IncidenceTensors = Array{Array{Integer,2},1}(undef,max_length-1)

    for l = 1:(max_length-1)
        
        if l == (max_length-1)
            sub_cycles = @view cycles[cycle_length_ptrs[l]:end]
        else
            sub_cycles = @view cycles[cycle_length_ptrs[l]:(cycle_length_ptrs[l+1]-1)]
        end
        
        IncidenceTensors[l] = unique_cycles(sub_cycles)
    end

    return [DistributedTensorConstruction.SymTensorUnweighted{Cycle}(size(A,1),size(IncidenceTensors[l],1),IncidenceTensors[l]) for l in 1:max_length-1]


end

function unique_cycles(cycles::AbstractVector{Vector{T}}) where T

    cycles_to_add = Array{Int,1}(undef,0)
    hashes = Set()
    #precision = Int(ceil(log(length(cycles))))
    for (i,cycle) in enumerate(cycles)
        hash = cycle_hash(cycle) 

        if hash in hashes
            continue
        else
            push!(cycles_to_add,i)
            push!(hashes,hash)
        end
    end

    Incidence = Array{T,2}(undef,length(cycles[1]),length(cycles_to_add))
    head = 1
    for cycle_idx in cycles_to_add
        Incidence[:,head] = cycles[cycle_idx]
        head += 1 
    end
    return Incidence
end

#TODO: make more robust to numerical inprecision
function cycle_hash2(cycle::Vector{T}) where T

    forward_val = 0.0
    for i =1:(length(cycle) - 1)
        
        forward_val += log10(cycle[i])*cycle[i+1]
    end
    forward_val += log10(cycle[end])*cycle[1]

    backward_val = 0.0
    for i =1:(length(cycle) - 1)
        backward_val += cycle[i]log10(cycle[i+1])
    end
    backward_val += cycle[end]log10(cycle[1])

    return backward_val*forward_val
end


function cycle_hash(cycle::Vector{T}) where T

    forward_val = 0.0
    for i =1:(length(cycle) - 1)
        
        forward_val += cycle[i]^2*cycle[i+1]
    end
    forward_val += cycle[end]^2*cycle[1]

    backward_val = 0.0
    for i =1:(length(cycle) - 1)
        backward_val += cycle[i]*cycle[i+1]^2
    end
    backward_val += cycle[end]*cycle[1]^2

    return log10(backward_val) + log10(forward_val)
end



#
#    Matrix to Tensor Constructor
#


function matrix_to_SymTensorUnweighted(A::SparseMatrixCSC{T,Int},motif::S) where {T, S<: Motif}

    n,m = size(A)
    @assert n == m

    is,js,_ = findnz(A)
    edge_list = [[i,j] for (i,j) in zip(is,js)]
    reduce_to_unique_cliques!(edge_list)

    edges = Array{Int,2}(undef,2,length(edge_list))

    for idx = 1:length(edge_list)
        edges[:,idx] = edge_list[idx]
    end

    return SymTensorUnweighted{S}(n,2,edges)
    

end


