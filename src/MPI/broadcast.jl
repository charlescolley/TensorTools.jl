function broadcast_communication(pids,bcast_pididx,channel_type::T) where T

    if bcast_pididx != 1 
        temp = pids[1]
        pids[1] = pids[bcast_pididx]
        pids[bcast_pididx] = temp
    end

    #initialize memory


    receiving_from = Vector{RemoteChannel{Channel{T}}}(undef,length(pids))
    sending_to = Vector{Vector{RemoteChannel{Channel{T}}}}(undef,length(pids))
    for p in 1:length(pids)
        sending_to[p] = Vector{RemoteChannel{Channel{T}}}(undef,0)
    end


    PowOT_batches = PowOT_process_breakdown(pids)


    if length(PowOT_batches) > 1

        batch_offset = length(PowOT_batches[1]) + 1

        for i = 2:length(PowOT_batches)
            channel = RemoteChannel(()->Channel{T}(1),PowOT_batches[i][1])
            push!(sending_to[1],channel)
            receiving_from[batch_offset] = channel

            batch_offset += length(PowOT_batches[i])
        end
    end

    batch_offset = 0 
    for batch in PowOT_batches

        broadcast_PowOT_communication!(batch,batch_offset,
                                       sending_to, receiving_from,
                                       channel_type)
        batch_offset += length(batch)
    end


    return  sending_to, receiving_from
end

function broadcast_PowOT_communication!(pids, batch_offset, sending_to,receiving_from,channel_type::T) where T
    #accumulates on pids[1]

    max_depth = Int(round(log2(length(pids))))
    pididx = findfirst(isequal(myid()), pids)


    #
    #    Coordinate communication
    #
  
    receiving_from_idx = zeros(Int,2^max_depth)

    sending = [1]

    for l=1:max_depth

        offset = Int(floor(2^(max_depth-l)))

        new_to_send = []
        for pididx in sending 
            #sendto
            channel = RemoteChannel(()->Channel{T}(1),pids[pididx + offset])
                                                        #channels should exist on receiving nodes
            #push!(channels,RemoteChannel(()->Channel{Matrix{Float64}}(1),pids[pididx]))

            if receiving_from_idx[pididx+offset] == 0
                receiving_from_idx[pididx+offset] = pididx
                receiving_from[batch_offset + pididx+offset] = channel
            end
            
            push!(sending_to[batch_offset + pididx],channel)
            push!(new_to_send,pididx + offset)

        end 

        append!(sending,new_to_send)
        
    end 

end

function broadcast_PowOT_communication(pids,channel_type::T) where T 
    #accumulates on pids[1]

    max_depth = Int(round(log2(length(pids))))
    pididx = findfirst(isequal(myid()), pids)


    #
    #    Coordinate communication
    #
  
    receiving_from_idx = zeros(Int,2^max_depth)
    receiving_from = Vector{RemoteChannel{Channel{T}}}(undef,2^max_depth)

    sending_to = Vector{Vector{RemoteChannel{Channel{T}}}}(undef,2^max_depth)
    for p in 1:2^max_depth
        sending_to[p] = Vector{RemoteChannel{Channel{T}}}(undef,0)
    end

    sending = [1]

    for l=1:max_depth

        offset = Int(floor(2^(max_depth-l)))

        new_to_send = []
        for pididx in sending 
            #sendto
            channel = RemoteChannel(()->Channel{T}(1),pids[pididx + offset])
                                                        #channels should exist on receiving nodes
            #push!(channels,RemoteChannel(()->Channel{Matrix{Float64}}(1),pids[pididx]))

            if receiving_from_idx[pididx+offset] == 0
                receiving_from_idx[pididx+offset] = pididx
                receiving_from[pididx+offset] = channel
            end
            
            push!(sending_to[pididx],channel)
            push!(new_to_send,pididx + offset)

        end 

        append!(sending,new_to_send)
        
    end 

    return sending_to, receiving_from
end

@everywhere function broadcast_proc_profiled(sending_to,receiving_from,n)

    #println("proc $(myid())")
    start_time = now()
    if receiving_from === nothing 
        Random.seed!(0)
        data = rand(Float64,n,n)
    end
    put_timings = []

    internal_t = @timed begin 

    
        if receiving_from === nothing 
            take_time = -1
            take_timing = -1
        else
            take_time = now()
            data,take_timing = @timed take!(receiving_from)
        end

        for channel in sending_to
            push_time = now()
            t = @timed put!(channel,data)
            push!(put_timings,(t.time,push_time))
        end
    end
    return data, (take_timing,take_time), put_timings, internal_t.time ,start_time

end


function profile_broadcast_proc(pids, trials, n_sizes,save=false)
    #assuming a length(pids) is a power of 2

    max_depth = Int(round(log2(length(pids))))

    runtimes = zeros(Float64,max_depth,length(n_sizes),trials)

    for p=1:max_depth
        for (i,n) in enumerate(n_sizes) 
            test_broadcast_proc_v2(pids[1:2^p],n)
            for t = 1:trials
                time = @timed test_broadcast_proc(pids[1:2^p],n)
                runtimes[p,i,t] = time.time 
            end
        end 
    end

    if save 
        filename = "bcast_profile_n:$(n_sizes)_maxProcs:$(length(pids))_trials:$(trials)_results.json"
    end
    return runtimes 
end