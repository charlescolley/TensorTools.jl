abstract type endBehavior end
struct returnToSpawner <: endBehavior end
struct writeFile <: endBehavior end

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

#
#  Process Functions 
#


function parallel_clique_sampling_proc(smat_file,order,samples,pids, ssten_filename,
                                                   gather_comm, bcast_comm, pa2a_comm,
                                                   serialized_write_channel,
                                                   atAnd::endBehavior=writeFile())

    proc_count = length(pids)
    A = MatrixNetworks.readSMAT(smat_file)
    _, cliques::Vector{Vector{Int64}} = TuranShadow(A,order,samples)
    reduce_to_unique_cliques!(cliques)
    splitting_indices = compute_splitting_idx(length(cliques),proc_count)


    #
    #  --  collect clique splitting samples  --  #
    #
    all_data = cliques[splitting_indices]
    all_data = gather(all_data,gather_comm)


    if gather_comm.sending_to === nothing
        #process the correct splittings and broadcast them 
        all_data = vcat(all_data...)
        sort!(all_data)
        splitting_indices = compute_splitting_idx(length(all_data),proc_count)
        final_splittings = all_data[splitting_indices]
    end

    #
    #  --  send final clique splittings  --  #  
    #
    if bcast_comm.receiving_from === nothing
        broadcast(final_splittings,bcast_comm)
    else
        final_splittings = broadcast(nothing,bcast_comm)
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
    
    #method doens't use built in function to avoid duplicating things in memory 
    for (i,(idx,channel)) in enumerate(pa2a_comm.sending_to)
        put!(channel,(pa2a_comm.my_idx,cliques[start_idx_for_processors[idx]:end_idx_for_processors[idx]]))
    end 

    data_taken = 1 
    cliques_for_me = cliques[start_idx_for_processors[pa2a_comm.my_idx]:end_idx_for_processors[pa2a_comm.my_idx]]
    while data_taken <= length(pa2a_comm.sending_to)
        (_,their_data) = take!(pa2a_comm.my_channel)
        append!(cliques_for_me,their_data)
        data_taken += 1 
    end 

    reduce_to_unique_cliques!(cliques_for_me)

    if typeof(atAnd) === returnToSpawner
        idx = pa2a_comm.my_idx

        if idx == 1 
            start_cliques  = cliques_for_me[1]
            end_cliques  = final_splittings[1]
        elseif idx == length(pids)
            start_cliques = final_splittings[end]
            end_cliques  = cliques_for_me[end]
        else
            start_cliques  =  final_splittings[idx-1]
            end_cliques  = final_splittings[idx]
        end

        return cliques_for_me, (start_cliques , end_cliques)
    elseif typeof(atAnd) === writeFile
        #
        #   Compute the total number of cliques found and reuse the gather channels to reduce 
        #

        clique_count = [[[length(cliques_for_me)]]]
        for channel in reverse(gather_comm.receiving_from)
            their_counts = take!(channel)
            clique_count[1][1][1] += their_counts[1][1][1]
        end

        if gather_comm.sending_to !== nothing
            put!(gather_comm.sending_to,clique_count)
        end



        #
        #  Create the ssten file serially 
        #
        if pa2a_comm.my_idx == 1
            write_ssten(cliques_for_me, size(A,1), ssten_filename,clique_count[1][1][1])
            put!(serialized_write_channel,(1,[[1]]))

        else #wait for a message from the p_a2a channel before appending 

            take!(pa2a_comm.my_channel)
            append_to_ssten(cliques_for_me,ssten_filename)
            if pa2a_comm.my_idx < length(pids) 
                put!(serialized_write_channel,(1,[[1]]))
            end 

        end
        return
    end

end


function parallel_clique_sampling_proc_profiled(smat_file,order,samples,pids, ssten_filename,
                                                   gather_comm, bcast_comm, pa2a_comm,
                                                   serialized_write_channel,
                                                   atAnd::endBehavior=writeFile())

    profiling = Dict()
    start_time = time_ns()
    duration = timestamp->Float64(time_ns()-timestamp)*1e-9

    proc_count = length(pids)
    A = MatrixNetworks.readSMAT(smat_file)
    TS_start_time = time_ns() 
    _, cliques::Vector{Vector{Int64}} = TuranShadow(A,order,samples)
    profiling["TuranShadow"] = duration(TS_start_time)

    clique_reduction1_start_time = time_ns()
    reduce_to_unique_cliques!(cliques)
    profiling["clique_reduction1"] = duration(clique_reduction1_start_time)
    
    splitting_indices = compute_splitting_idx(length(cliques),proc_count)

    #
    #  --  collect clique splitting samples  --  #
    #
    all_data = cliques[splitting_indices]
    gather_start_time = time_ns()
    all_data = gather(all_data,gather_comm)
    profiling["gather_runtime"] = duration(gather_start_time)
    if gather_comm.sending_to === nothing
        #process the correct splittings and broadcast them 
        all_data = vcat(all_data...)
        sort!(all_data)
        splitting_indices = compute_splitting_idx(length(all_data),proc_count)
        final_splittings = all_data[splitting_indices]
    end

    #
    #  --  send final clique splittings  --  #  
    #
    bcast_start_time = time_ns()
    if bcast_comm.receiving_from === nothing
        broadcast(final_splittings,bcast_comm)
    else
        final_splittings = broadcast(nothing,bcast_comm)
    end
    profiling["bcast_runtime"] =duration(bcast_start_time)

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
    
    pa2a_start_time = time_ns()
    #method doens't use built in function to avoid duplicating things in memory 
    for (i,(idx,channel)) in enumerate(pa2a_comm.sending_to)
        put!(channel,(pa2a_comm.my_idx,cliques[start_idx_for_processors[idx]:end_idx_for_processors[idx]]))
    end 

    data_taken = 1 
    cliques_for_me = Vector{Vector{Int}}(undef,0)
    while data_taken <= length(pa2a_comm.sending_to)
        (_,their_data) = take!(pa2a_comm.my_channel)
        append!(cliques_for_me,their_data)
        data_taken += 1 
    end 
    profiling["pa2a_runtime"] = duration(pa2a_start_time)

    clique_reduction2_start_time = time_ns()
    reduce_to_unique_cliques!(cliques_for_me)
    profiling["clique_reduction2"] = duration(clique_reduction2_start_time)

    if typeof(atAnd) === returnToSpawner
        idx = pa2a_comm.my_idx
        return cliques_for_me, (cliques[start_idx_for_processors[idx]],cliques[end_idx_for_processors[idx]])
    elseif typeof(atAnd) === writeFile

        #
        #   Compute the total number of cliques found and reuse the gather channels to reduce 
        #

        clique_count_reduction_start_time = time_ns()
        clique_count = [[[length(cliques_for_me)]]]
        for channel in reverse(gather_comm.receiving_from)
            their_counts = take!(channel)
            clique_count[1][1][1] += their_counts[1][1][1]
        end

        if gather_comm.sending_to !== nothing
            put!(gather_comm.sending_to,clique_count)
        end
        profiling["clique_count_reduction"] = duration(clique_count_reduction_start_time)



        #
        #  Create the ssten file serially 
        #
        ssten_writing_start_time = time_ns()
        if pa2a_comm.my_idx == 1

            write_ssten(cliques_for_me, size(A,1), ssten_filename,clique_count[1][1][1])
            put!(serialized_write_channel,(1,[[1]]))

        else #wait for a message from the p_a2a channel before appending 

            take!(pa2a_comm.my_channel)
            append_to_ssten(cliques_for_me,ssten_filename)
            if pa2a_comm.my_idx < length(pids) 
                put!(serialized_write_channel,(1,[[1]]))
            end 

        end
        profiling["ssten_writing"] = duration(ssten_writing_start_time)
    end

    profiling["internal_runtime"] = duration(start_time)

    return profiling
end


function sample_smat_files_proc(smat_files, output_path, orders, samples, pids, profile, comms...)
    @assert output_path[end] == '/'

    if profile
        all_profiling = []
    end
    for smat_file in smat_files 
        smat_prefix = split(split(smat_file,"/")[end],".smat")[1]
        for order in orders 
            for TS_sample_count in samples 
                ssten_file = output_path*smat_prefix*"_order:$(order)_samples:$(TS_sample_count*length(pids)).ssten"
                if profile
                    profiling = parallel_clique_sampling_proc_profiled(smat_file,order,TS_sample_count,pids,ssten_file,comms...)
                    push!(all_profiling,(smat_file,order,TS_sample_count,profiling))
                else
                    parallel_clique_sampling_proc(smat_file,order,TS_sample_count,pids,ssten_file,comms...)
                end

            end
        end 
    end 

    if profile 
        return all_profiling
    end
end

#
#  Spawning Drivers
#

function distributed_sample_smat_files(pids, smat_files, output_path, orders,samples,profile=false)

    collection_idx = 1
    gather_comm = gather_communication(pids,collection_idx,[[1]])
    bcast_comm = broadcast_communication(pids,collection_idx,[[1]])
    pa2a_comm = personalized_all_to_all_communication(pids,[[1]])

    samples_per_proc = [Int(ceil(s/length(pids))) for s in samples]

    futures = []
    for p = 1:length(pids)

        if p == length(pids)
            serialization_channel = nothing 
        else
            serialization_channel = pa2a_comm[p+1].my_channel
        end 

        future = @spawnat pids[p] sample_smat_files_proc(
                                    smat_files, output_path, orders,
                                    samples_per_proc, pids, profile,
                                    gather_comm[p],bcast_comm[p],pa2a_comm[p],
                                    serialization_channel,writeFile())
        push!(futures,future)
    end

    for future in futures
        f = fetch(future)
        if isa(f,RemoteException)
            throw(f)
        end 
    end

end

function distributed_clique_sample(pids, matrix_file, ssten_filename, order, samples,profile=false)

    collection_idx = 1

    samples_per_proc = Int(ceil(samples/length(pids)))

    gather_comm = gather_communication(pids,collection_idx,[[1]])
    bcast_comm = broadcast_communication(pids,collection_idx,[[1]])
    pa2a_comm = personalized_all_to_all_communication(pids,[[1]])

    futures = []
    for p = 1:length(pids)
        if p == length(pids)
            serialization_channel = nothing 
        else
            serialization_channel = pa2a_comm[p+1].my_channel
        end 
        if profile 
            future = @spawnat pids[p] parallel_clique_sampling_proc_profiled(matrix_file, order, samples_per_proc, pids, ssten_filename,
                                                                    gather_comm[p], bcast_comm[p], pa2a_comm[p],
                                                                    serialization_channel)
        else
            future = @spawnat pids[p] parallel_clique_sampling_proc(matrix_file, order, samples_per_proc, pids, ssten_filename,
                                                                            gather_comm[p], bcast_comm[p], pa2a_comm[p],
                                                                            serialization_channel)
        end
        push!(futures,future)
    end
    all_vals = []

    for future in futures
        f = fetch(future)
        if isa(f,RemoteException)
            throw(f)
        end
        push!(all_vals,f)
    end    

    return all_vals
end



#
#  Serial Codes + Helper Functions
#

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

