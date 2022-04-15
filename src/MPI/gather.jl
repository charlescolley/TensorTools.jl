function gather_communication(pids,gather_pididx,channel_type::T) where T 

    if gather_pididx != 1 
        temp = pids[1]
        pids[1] = pids[gather_pididx]
        pids[gather_pididx] = temp
    end

    #initialize memory
    channels = Vector{RemoteChannel{Channel{Vector{T}}}}(undef,0)


    sending_to = Vector{RemoteChannel{Channel{Vector{T}}}}(undef,length(pids))
    receiving_from = Vector{Vector{RemoteChannel{Channel{Vector{T}}}}}(undef,length(pids))
    for p in 1:length(pids)
        receiving_from[p] = Vector{RemoteChannel{Channel{Vector{T}}}}(undef,0)
    end


    PowOT_batches = PowOT_process_breakdown(pids)

    #gather the results from the first nodes in the Power of Two batches
    if length(PowOT_batches) > 1

        batch_offset = length(pids) + 1

        for i = length(PowOT_batches):-1:2
            batch_offset -= length(PowOT_batches[i])
            push!(channels,RemoteChannel(()->Channel{Vector{T}}(1),PowOT_batches[i][1]))
            
            push!(receiving_from[1],channels[end])
            sending_to[batch_offset] = channels[end]

        end
    end


    batch_offset = length(pids) 
    
    for batch in reverse(PowOT_batches)
        batch_offset -= length(batch)

        gather_PowOT_communication!(batch,batch_offset, 
                                    sending_to, receiving_from,
                                    channel_type)

        
    end


    return  sending_to, receiving_from

end

function gather_PowOT_communication!(pids, batch_offset, sending_to,receiving_from,channel_type::T) where T
    #accumulates on pids[1]

    max_depth = Int(round(log2(length(pids))))
    pididx = findfirst(isequal(myid()), pids)


    #
    #    Coordinate communication
    #
  
    sending_to_idx = zeros(Int,2^max_depth)

    receiving = [1]

    for l=1:max_depth

        offset = Int(floor(2^(max_depth-l)))

        new_to_receive = []
        for pididx in receiving
            #sendto
            channel = RemoteChannel(()->Channel{Vector{T}}(1),pids[pididx + offset])
                                                        #channels should exist on sending nodes
            #push!(channels,RemoteChannel(()->Channel{Matrix{Float64}}(1),pids[pididx]))

            if sending_to_idx[pididx+offset] == 0
                sending_to_idx[pididx+offset] = pididx
                sending_to[batch_offset + pididx+offset] = channel 
            end
            
            push!(receiving_from[batch_offset + pididx],channel)
            push!(new_to_receive,pididx + offset)

        end 

        append!(receiving,new_to_receive)
        
    end 

end

