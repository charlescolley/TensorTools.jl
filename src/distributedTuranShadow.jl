function compute_splitting_idx(n,p)
    #  computes the uniform splitting indices for 
    #  an array of length n into p pieces
    splitting_idx = zeros(Int,p-1)
    idx = 1
    count = 1
    while count <= n - Int(ceil(n/p))
        splitting_idx[idx] += 1 
        idx = (idx % (p-1)) + 1
        count += 1
    end 
    for i = 2:(p-1)
        splitting_idx[i] += splitting_idx[i-1]
    end
    return splitting_idx
end

function parallel_clique_sampling_proc(smat_file,order,samples,pids, ssten_filename,
                                        gather_sending_to, gather_receiving_from,
                                        bcast_sending_to,  bcast_receiving_from,
                                        p_a2a_sending_to,  p_a2a_my_channel,
                                        serialized_write_channel)

    proc_count = length(pids)
    pid_idx = Dict([(pid,i) for (i,pid) in enumerate(pids)])
    A = MatrixNetworks.readSMAT(smat_file)
    _, cliques::Vector{Vector{Int64}} = TuranShadow(A,order,samples)
    reduce_to_unique_cliques!(cliques)

    splitting_indices = compute_splitting_idx(length(cliques),proc_count)

    #
    #  --  collect clique splitting samples  --  #
    #
    #all_data = Vector{Vector{Vector{Int}}}(undef,1)
    #all_data[1] = cliques[splitting_indices]

    all_data = cliques[splitting_indices]

    for channel in reverse(gather_receiving_from)
        their_data = take!(channel)
        append!(all_data,their_data)
    end

    if gather_sending_to === nothing
        #process the correct splittings and broadcast them 
        sort!(all_data)

        splitting_indices = compute_splitting_idx(length(all_data),proc_count)

        final_splittings = all_data[splitting_indices]
    else 
        put!(gather_sending_to,all_data)
    end

    #
    #  --  send final clique splittings  --  #  
    #

    if bcast_receiving_from !== nothing 
        final_splittings = take!(bcast_receiving_from)
    end
    for channel in bcast_sending_to
        put!(channel,final_splittings)
    end
        
    #  --  determine which cliques should be sent to which processors --  # 

    start_idx_for_processors = Vector{Int}(undef,proc_count)
    end_idx_for_processors = Vector{Int}(undef,proc_count)
    start_idx_for_processors[1] = 1 
    end_idx_for_processors[end] = length(cliques)


    split_idx = 1

    idx = 1 
    for edge_split in final_splittings
        
        while cliques[idx] < edge_split
            idx +=1 
        end
        start_idx_for_processors[split_idx+1] = idx
        end_idx_for_processors[split_idx] = idx -1  
        split_idx += 1
    end


    #
    #  --  send all cliques to the correct processors  --  #
    #

    for channel in p_a2a_sending_to
        i = pid_idx[channel.where]
        put!(channel,cliques[start_idx_for_processors[i]:end_idx_for_processors[i]])
    end 

    data_taken = 1 
    cliques_for_me = Vector{Vector{Int}}(undef,0)
    while data_taken <= length(p_a2a_sending_to)
        their_data = take!(p_a2a_my_channel)

        append!(cliques_for_me,their_data)
        data_taken += 1 
    end 

    reduce_to_unique_cliques!(cliques_for_me)
    
    #
    #   Compute the total number of cliques found and reuse the gather channels to reduce 
    #
    clique_count = [[length(cliques_for_me)]]
    for channel in reverse(gather_receiving_from)
        their_counts = take!(channel)
        clique_count[1][1] += their_counts[1][1]
    end

    if gather_sending_to !== nothing
        put!(gather_sending_to,clique_count)
    end


    #
    #  Create the ssten file serially 
    #
    if gather_sending_to === nothing

        write_ssten(cliques_for_me, size(A,1), ssten_filename,clique_count[1][1])
        put!(serialized_write_channel,[[1]])

    else #wait for a message from the p_a2a channel before appending 

        take!(p_a2a_my_channel)
        append_to_ssten(cliques_for_me,ssten_filename)
        if serialized_write_channel !== nothing 
            put!(serialized_write_channel,[[1]])
        end 

    end


end

function parallel_clique_sampling_proc_profiled(smat_file,order,samples,pids,ssten_filename,
                                                   gather_sending_to, gather_receiving_from,
                                                   bcast_sending_to,  bcast_receiving_from,
                                                   p_a2a_sending_to,  p_a2a_my_channel,
                                                   serialized_write_channel)

    profiling = Dict()
    (_,profiling["internal_runtime"]) = @timed begin 
        proc_count = length(pids)
        pid_idx = Dict([(pid,i) for (i,pid) in enumerate(pids)])
        A = MatrixNetworks.readSMAT(smat_file)
        # =  @timed begin 
        (_, cliques::Vector{Vector{Int64}}),profiling["TuranShadow"] = @timed TuranShadow(A,order,samples)
        (_,profiling["clique_reduction1"]) = @timed reduce_to_unique_cliques!(cliques)
        
        splitting_indices = compute_splitting_idx(length(cliques),proc_count)

        #
        #  --  collect clique splitting samples  --  #
        #
        #all_data = Vector{Vector{Vector{Int}}}(undef,1)
        #all_data[1] = cliques[splitting_indices]

        all_data = cliques[splitting_indices]
        #println(gather_receiving_from)

        (_,profiling["gather_runtime"]) = @timed begin 
            for channel in reverse(gather_receiving_from)
                their_data = take!(channel)
                append!(all_data,their_data)
            end

            #println(all_data)
            if gather_sending_to !== nothing
                put!(gather_sending_to,all_data)
            end
        end

        if gather_sending_to === nothing
            #process the correct splittings and broadcast them 
            sort!(all_data)
            splitting_indices = compute_splitting_idx(length(all_data),proc_count)
            final_splittings = all_data[splitting_indices]
        end
        #
        #  --  send final clique splittings  --  #  
        #
        (_,profiling["bcast_runtime"]) = @timed begin 
            if bcast_receiving_from !== nothing 
                final_splittings = take!(bcast_receiving_from)
            end
            for channel in bcast_sending_to
                put!(channel,final_splittings)
            end
        end
            
        #println(final_splittings)
        #  --  determine which cliques should be sent to which processors --  # 

        start_idx_for_processors = Vector{Int}(undef,proc_count)
        end_idx_for_processors = Vector{Int}(undef,proc_count)
        start_idx_for_processors[1] = 1 
        end_idx_for_processors[end] = length(cliques)


        split_idx = 1

        idx = 1 
        for edge_split in final_splittings
            
            while cliques[idx] < edge_split
                idx +=1 
            end
            start_idx_for_processors[split_idx+1] = idx
            end_idx_for_processors[split_idx] = idx -1  
            split_idx += 1
        end

        #println(start_idx_for_processors)
        #println(end_idx_for_processors)

        #
        #  --  send all cliques to the correct processors  --  #
        #
        (_,profiling["pa2a_runtime"]) = @timed begin 
            for channel in p_a2a_sending_to
                i = pid_idx[channel.where]
                put!(channel,cliques[start_idx_for_processors[i]:end_idx_for_processors[i]])
            end 

            data_taken = 1 
            cliques_for_me = Vector{Vector{Int}}(undef,0)
            while data_taken <= length(p_a2a_sending_to)
                their_data = take!(p_a2a_my_channel)
            
                append!(cliques_for_me,their_data)
                data_taken += 1 
            end 
        end

        (_,profiling["clique_reduction2"]) = @timed reduce_to_unique_cliques!(cliques_for_me)


        #
        #   Compute the total number of cliques found and reuse the gather channels to reduce 
        #
        clique_count = [[length(cliques_for_me)]]
        (_,profiling["clique_count_reduction"]) = @timed begin 
            for channel in reverse(gather_receiving_from)
                their_counts = take!(channel)
                clique_count[1][1] += their_counts[1][1]
            end

            if gather_sending_to !== nothing
                put!(gather_sending_to,clique_count)
            end
        end

        #
        #  Create the ssten file serially 
        #
        (_,profiling["ssten_writing"]) = @timed begin 
            if gather_sending_to === nothing

                write_ssten(cliques_for_me, size(A,1), ssten_filename,clique_count[1][1])
                put!(serialized_write_channel,[[1]])

            else #wait for a message from the p_a2a channel before appending 

                take!(p_a2a_my_channel)
                append_to_ssten(cliques_for_me,ssten_filename)
                if serialized_write_channel !== nothing 
                    put!(serialized_write_channel,[[1]])
                end 

            end
        end


    end
    return profiling

end


function distributed_clique_sample(pids, matrix_file, ssten_filename, order, samples,profiling=false)

    collection_idx = 1

    samples_per_proc = Int(ceil(samples/length(pids)))

    gather_sending_to, gather_receiving_from  = gather_communication(pids,collection_idx,[1])
    bcast_sending_to, bcast_receiving_from = broadcast_communication(pids,collection_idx,[[1]])
    pa2a_sending_to, pa2a_channels = personalized_all_to_all_communication(pids,[[1]])

    futures = []
    for p = 1:length(pids)
        if profiling 
            if p == collection_idx
                future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc_profiled(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                                                                nothing, gather_receiving_from[p],
                                                                                                                bcast_sending_to[p],  nothing,
                                                                                                                pa2a_sending_to[p],  pa2a_channels[p],
                                                                                                                pa2a_channels[p+1])
            else
                if p == length(pids)
                    future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc_profiled(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                                                                    gather_sending_to[p], gather_receiving_from[p],
                                                                                                                    bcast_sending_to[p],  bcast_receiving_from[p],
                                                                                                                    pa2a_sending_to[p],  pa2a_channels[p],
                                                                                                                    nothing)
                else
                future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc_profiled(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                                                                gather_sending_to[p], gather_receiving_from[p],
                                                                                                                bcast_sending_to[p],  bcast_receiving_from[p],
                                                                                                                pa2a_sending_to[p],  pa2a_channels[p],
                                                                                                                pa2a_channels[p+1])
                end
            end
        else
            if p == collection_idx
                future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                                                    nothing, gather_receiving_from[p],
                                                                                                    bcast_sending_to[p],  nothing,
                                                                                                    pa2a_sending_to[p],  pa2a_channels[p],
                                                                                                    pa2a_channels[p+1])
            else
                if p == length(pids)
                    future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                            gather_sending_to[p], gather_receiving_from[p],
                                                                            bcast_sending_to[p],  bcast_receiving_from[p],
                                                                            pa2a_sending_to[p],  pa2a_channels[p],
                                                                            nothing)
                else
                    future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc(matrix_file, order, samples_per_proc,pids,ssten_filename,
                                                                                                            gather_sending_to[p], gather_receiving_from[p],
                                                                                                            bcast_sending_to[p],  bcast_receiving_from[p],
                                                                                                            pa2a_sending_to[p],  pa2a_channels[p],
                                                                                                            pa2a_channels[p+1])
                end
            end
        end
        push!(futures,future)
    end
    all_vals = []

    for (i,future) in enumerate(futures)
        push!(all_vals,fetch(future))
    end    

    if profiling
        return all_vals
    end


end


function sample_cliques(matrix_file,output_path,orders,sample_counts)

    #check file format

    A = nothing
    
    file_ext = split(matrix_file,".")[end]
    if (file_ext == "smat")
        A = MatrixNetworks.readSMAT(matrix_file)
    elseif (file_ext == "csv")
    	A = parse_csv(matrix_file)
    end
    
    A = max.(A,A')

    seed = 4
    Random.seed!(seed)
    root_name = split(split(matrix_file,"/")[end],".")[1]

    for k in orders

    	#create a subfolder for given order
        order_folder = "order:$(k)/"
        local_folder = output_path*order_folder
	    #Note: assuming that output_path ends in '/'
		  
	if !isdir(output_path*order_folder)
	   mkdir(output_path*order_folder)
	end

        log_file = root_name*"-order:$(k)-seed:$(seed)-log.jld"

        runtimes = []
        unique_cliques = Set()

        prev_sample_count = 0
	    prev_motif_count = 0

        for samples in sort(sample_counts)
     	    #  -- find unique cliques  --  #
    	    ((_,cliques),t) = @timed TuranShadow(A,k,(samples-prev_sample_count))
	        prev_sample_count = samples

	    
            cliques = [sort(clique) for clique in cliques]
            sort!(cliques) #lexicographic ordered helps eliminate repeats
	
	    for clique in cliques
 	        push!(unique_cliques,clique)
	    end

	    #  terminate sample search if no new motifs found this iteration

	    motif_count = length(unique_cliques)
	    if (prev_motif_count == motif_count)
	       println("no new cliques found from last iter. motif_count=$(motif_count)")
	       break
	    else
	        prev_motif_count = motif_count
	    end
	    
	

	    tensor_name = split(root_name,".")[1]*"-order:$(k)-sample:$(samples)-seed:$(seed).ssten"
	        #NOTE: assuming matrix_file is of the form rootName.smat

   	    write_tensor(collect(unique_cliques),output_path*order_folder*tensor_name)
	
	    #  --  update log  --  #
	    push!(runtimes,(samples,t))
	    save(output_path*order_folder*log_file,"runtimes",runtimes)
	end
    end



end

#TODO: 
function reduce_to_unique_cliques!(cliques::Array{Array{T,1},1}) where {T <: Int}

    if length(cliques) < 2 
        return
    end
    order = length(cliques[1])

    for i = 1:length(cliques)
        sort!(cliques[i])
    end

    sort!(cliques)
    
    ix_drop = Array{Int,1}(undef,0)

    current_clique_idx = 1
    for i =2:length(cliques)
        
        if cliques[i] == cliques[current_clique_idx] #found a repeated clique, mark for removal
            push!(ix_drop,i)
        else #found a new clique, update ptr
            current_clique_idx = i 
        end

    end

    deleteat!(cliques,ix_drop)
    #return cliques, ix_drop
    #remove all repeated cliques
    #cliques = cliques[setdiff(begin:end, ix_drop)]

    
end

