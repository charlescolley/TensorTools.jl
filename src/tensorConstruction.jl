
#
#    Clique Motif Routines
#

#  -- Multiple Tensor Generation

function tensors_from_graph(A, orders::Array{Int,1}, ε::Float64, δ::Float64, motif::Clique)
    
    estimated_total_cliques = Array{Float64,1}(undef,length(orders))
    samples_used = Array{Float64,1}(undef,length(orders))
    tensors = Array{SymTensorUnweighted{Clique},1}(undef,length(orders))

    for (i,order) in enumerate(orders)
        tensors[i], estimated_total_cliques[i], samples_used[i] = tensor_from_graph(A, order,  ε, δ, motif)
    end

    return tensors, estimated_total_cliques, samples_used

end

function tensors_from_graph(A, orders::Array{Int,1}, samples::Int, motif::Clique)
    
    estimated_total_cliques = Array{Float64,1}(undef,length(orders))
    tensors = Array{SymTensorUnweighted{Clique},1}(undef,length(orders))

    for (i,order) in enumerate(orders)
        tensors[i], estimated_total_cliques[i] = tensor_from_graph(A, order, samples, motif)
    end

    return tensors, estimated_total_cliques
end

function tensors_from_graph(A, orders::Array{Int,1}, sample_sizes::Array{Int,1}, motif::Clique)

    @assert length(orders) == length(sample_sizes)
    tensors = Array{SymTensorUnweighted{Clique},1}(undef,length(orders))
    estimated_total_cliques = Array{Int,1}(undef,length(orders))


    for (i,(order,sample)) in enumerate(zip(orders,sample_sizes))
        if order == 2
            tensors[i] = matrix_to_SymTensorUnweighted(A, motif)
            estimated_total_cliques[i] = -1
        else
            tensors[i],estimated_total_cliques[i] = tensor_from_graph(A, order, sample, motif)
        end
    end

    return tensors, estimated_total_cliques
end

#  -- Single Tensor Generation

function tensor_from_graph(A, order, ε::Float64, δ::Float64, motif::Clique)
    samples_used, estimated_total_cliques::Float64, all_cliques::Array{Array{Int64,1},1} = TuranShadow(A,order,ε,δ)
    reduce_to_unique_cliques!(all_cliques)

    indices = Array{Int,2}(undef,order,length(all_cliques))
    idx = 1
    for clique in all_cliques #is there a better way to do this? 
        indices[:,idx] = clique
        idx += 1;
    end

    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices)), estimated_total_cliques, samples_used
end

function tensor_from_graph(A, order, t, motif::Clique)

    estimated_total_cliques::Float64, cliques::Array{Array{Int64,1},1} = TuranShadow(A,order,t)

    reduce_to_unique_cliques!(cliques)

    indices = zeros(order,length(cliques))
    idx = 1
    
    for clique in cliques #is there a better way to do this? 
        indices[:,idx] = clique
        idx += 1;
    end

    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices)), estimated_total_cliques

end

function tensor_from_graph(A, order::Int, motif::Clique;max_samples=10^9,verbose=false)

    t = 10000

    all_cliques = Array{Array{Int64,1},1}(undef,0)
    prev_motif_count = 0
    used_samples = 0
    last_estimated_total_cliques = 0
    while t < max_samples

        last_estimated_total_cliques::Float64, cliques::Array{Array{Int64,1},1} = TuranShadow(A,order,t)
        used_samples += t

        append!(all_cliques,cliques)
        reduce_to_unique_cliques!(all_cliques)

        if length(all_cliques) == prev_motif_count
            break 
        else
            prev_motif_count =  length(all_cliques)
            t *= 2 
        end
    end


    indices = zeros(order,length(all_cliques))
    idx = 1
    
    for clique in all_cliques #is there a better way to do this? 
        indices[:,idx] = clique
        idx += 1;
    end

    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices)), last_estimated_total_cliques

end

#  -- Profiled Versions

function tensor_from_graph_profiled(A, order::Int,ε::Float64, δ::Float64,motif::Clique;verbose=false)
    profiling = Dict()
   
    (bound_samples, approx_number_of_cliques::Float64, cliques::Array{Array{Int64,1},1}),rt = @timed TuranShadow(A, order, ε, δ)
    profiling["samples"] = bound_samples
    profiling["TuranShadow_rt"] = rt
    profiling["TuranShadow_approx_clique_count"] = approx_number_of_cliques

    _,rt = @timed reduce_to_unique_cliques!(cliques)
    profiling["reduction_rt"] = rt 
    profiling["motif_count"]= length(cliques)

    indices = Array{Int,2}(undef,order,length(cliques))
    idx = 1
    for clique in cliques #is there a better way to do this? 
        indices[:,idx] = clique
        idx += 1;
    end

    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices)), profiling

end


# This version runs multiple phases and doubles the number of samples used until no new cliques are
# found. 
function tensor_from_graph_profiled(A, order::Int, motif::Clique;max_samples=10^9,verbose=false)

    t = 10000

    profiling = Dict([
        ("motif_count",[]),
        ("samples",[]),
        ("reduction_rt",[]),
        ("TuranShadow_approx_clique_count",[]),
        ("TuranShadow_rt",[]),
        ("append!_rt",[]),
    ])

    all_cliques = Array{Array{Int64,1},1}(undef,0)
    prev_motif_count = 0
    used_samples = 0
    while t < max_samples

        push!(profiling["samples"],t)
        (approx_number_of_cliques::Float64, cliques::Array{Array{Int64,1},1}),rt = @timed TuranShadow(A,order,t)
        push!(profiling["TuranShadow_rt"],rt)
        push!(profiling["TuranShadow_approx_clique_count"],approx_number_of_cliques)

        used_samples += t
        _,rt = @timed append!(all_cliques,cliques)
        push!(profiling["append!_rt"],rt)

        _,rt = @timed reduce_to_unique_cliques!(all_cliques)
        push!(profiling["reduction_rt"],rt)

        push!(profiling["motif_count"],length(all_cliques))


        iter_rt = profiling["reduction_rt"][end] + profiling["append!_rt"][end] + profiling["TuranShadow_rt"][end]
        if length(all_cliques) == prev_motif_count
            println("found $(prev_motif_count) motifs using $(t) samples in $iter_rt (s)")
            break 
        else
            prev_motif_count =  length(all_cliques)
            println("found $(prev_motif_count) motifs using $(t) samples in $iter_rt (s)")
            t *= 2 
        end
    end


    indices = zeros(order,length(all_cliques))
    idx = 1
    
    for clique in all_cliques #is there a better way to do this? 
        indices[:,idx] = clique
        idx += 1;
    end

    return SymTensorUnweighted{Clique}(size(A,1),order,round.(Int,indices)), profiling

end

#
#    Cycle Motif Routines
#

function tensor_from_graph(A, order::Int, sample_size::Int, motif::Cycle)

    cycle_tensors = tensors_from_graph(A,order,sample_size,motif) 
    return filter!(c-> c.order == order, cycle_tensors)
        # Johnson's algorithm returns many smaller cycles too, only return the requested ones
        filter!(c-> c.order in orders, cycle_tensors)
end

function tensors_from_graph(A, orders::Array{Int,1}, sample_size::Int, motif::Cycle)

    cycle_tensors = tensors_from_graph(A, maximum(orders), sample_size, motif) 
    filter!(c-> c.order in orders, cycle_tensors)
        # Johnson's algorithm returns many smaller cycles too, only return the requested ones
        # Assuming length(orders) is small. 

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
	return [SymTensorUnweighted{Cycle}(size(A,1),l,IncidenceTensors[l]) for l in 2:max_length]
                                                                                                             #ignore the singleton node returned 
end
"""

function tensors_from_graph(A::SparseMatrixCSC{T,Int64}, maxCycleLength::Int,samples::Int,motif::Cycle) where T

	LG_A = LightGraphs.SimpleGraph(A)

	cycles = simplecycles_limited_length(LG_A,maxCycleLength,samples)
    if length(cycles) == 0 
        return [SymTensorUnweighted{Cycle}(size(A,1),maxCycleLength,Array{T,2}(undef,maxCycleLength,0))]
    end 

    sort!(cycles,by=c->length(c))

    cycle_length_ptrs = ones(Int,maxCycleLength-1) #LightGraphs doesn't return anything len(c)==1
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

    IncidenceTensors = Array{Array{Integer,2},1}(undef,maxCycleLength-1)

    for l = 1:(maxCycleLength-1)
        
        if l == (maxCycleLength-1)
            sub_cycles = @view cycles[cycle_length_ptrs[l]:end]
        else
            sub_cycles = @view cycles[cycle_length_ptrs[l]:(cycle_length_ptrs[l+1]-1)]
        end
        
        IncidenceTensors[l] = unique_cycles(sub_cycles)
    end

    return [SymTensorUnweighted{Cycle}(size(A,1),size(IncidenceTensors[l],1),IncidenceTensors[l]) for l in 1:maxCycleLength-1]

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
    
    if length(cycles) == 0 
        Incidence = Array{T,2}(undef,0,length(cycles_to_add))
    else
        Incidence = Array{T,2}(undef,length(cycles[1]),length(cycles_to_add))
    end

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


