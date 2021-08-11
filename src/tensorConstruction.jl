

#
#    Clique Motif Routines
#

function tensors_from_graph(A, orders::Array{Int,1}, sample_size::Int, motif::Clique)

    tensors = Array{SymTensorUnweighted,1}(undef,length(orders))

    for (i,order) in enumerate(orders)
        tensor = tensor_from_graph(A, order, sample_size,motif)
        tensors[i] = tensor
    end

    return tensors

end

function tensors_from_graph(A, orders::Array{Int,1}, sample_sizes::Array{Int,1}, motif::Clique)

    @assert length(orders) == length(sample_sizes)
    tensors = Array{SymTensorUnweighted,1}(undef,length(orders))

    for (i,(order,sample)) in enumerate(zip(orders,sample_sizes))
        if order == 2
            tensors[i] = matrix_to_SymTensorUnweighted(A)
        else
            tensors[i] = tensor_from_graph(A, order, sample,motif)
        end
    end

    return tensors
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

    cycle_tensors = tensors_from_graph(A,maximum(orders),sample_size,motif) 
    return [cycle_tensors[order-1] for order in orders]
        # Johnson's algorithm returns many smaller cycles too, only return the requested ones

end


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
															#only need one orientation
	end
	IncidenceHeadPters = ones(Integer,max_length)


	for cycle in cycles
		l = length(cycle)

		if issorted(cycle)
			IncidenceTensors[l][:,IncidenceHeadPters[l]] = cycle
			IncidenceHeadPters[l] += 1
		end
	end

	#trim the 2D arrays for the incidence matrix
	for l = 2:max_length
		IncidenceTensors[l] = IncidenceTensors[l][:,1:(IncidenceHeadPters[l]-1)]
	end 	


	#return IncidenceTensors
	return [DistributedTensorConstruction.SymTensorUnweighted(size(A,1),l,IncidenceTensors[l]) for l in 2:max_length]
            #ignore the singleton node returned 
end

#
#    Matrix to Tensor Constructor
#


function matrix_to_SymTensorUnweighted(A::SparseMatrixCSC{T,Int}) where T

    n,m = size(A)
    @assert n == m

    is,js,_ = findnz(A)
    edge_list = [[i,j] for (i,j) in zip(is,js)]
    reduce_to_unique_cliques!(edge_list)

    edges = Array{Int,2}(undef,2,length(edge_list))

    for idx = 1:length(edge_list)
        edges[:,idx] = edge_list[idx]
    end

    return SymTensorUnweighted(n,2,edges)
    

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

    
    return SymTensorUnweighted(size(A,1),order,round.(Int,indices))

end

