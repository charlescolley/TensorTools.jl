function personalized_all_to_all_communication(pids,args...)
    if length(pids) == 2^Int(round(log2(length(pids))))
        return personalized_all_to_all_communication_PoWOT(pids,args...)
    else
        return personalized_all_to_all_communication_naive(pids,args...)
    end
end

function personalized_all_to_all_communication_naive(pids,channel_type::T) where T 
    # have all processes send to the same process at every iteration

    # each process gets its own channel 
    process_channels = Vector{RemoteChannel{Channel{T}}}(undef,length(pids))
    sending_to = Vector{Vector{RemoteChannel{Channel{T}}}}(undef,length(pids))

    for (i,pid) in enumerate(pids)
        process_channels[i] = RemoteChannel(()->Channel{T}(length(pids)-1),pid)
        sending_to[i] = Vector{RemoteChannel{Channel{Tuple{Int,T}}}}(undef,length(pids)-1)
    end 

    for i = 1:length(pids)
        sending_to[i] = process_channels[1:end .!= i]
    end 

    return sending_to, process_channels
end

function personalized_all_to_all_communication_PoWOT(pids,channel_type::T) where T 
    # have all processes send to the same process at every iteration

    # each process gets its own channel 
    process_channels = Vector{RemoteChannel{Channel{T}}}(undef,length(pids))
    sending_to = Vector{Vector{RemoteChannel{Channel{T}}}}(undef,length(pids))

    for i = 1:length(pids)
        process_channels[i] = RemoteChannel(()->Channel{T}(length(pids)-1),pids[i])
        sending_to[i] = Vector{RemoteChannel{Channel{T}}}(undef,length(pids)-1)
    end 

    for i = 1:length(pids)
        sending_to[i] = Vector{RemoteChannel{Channel{T}}}(undef,length(pids)-1)
        for j = 1:(length(pids)-1)
            sending_idx = ((i - 1) ⊻ j) + 1
            sending_to[i][j] = process_channels[sending_idx]
        end
    end 

    return sending_to, process_channels
end



@everywhere function personalized_communication_proc_profiled(sending_to,my_channel,seed,my_idx,n)

    seed!(seed)
    all_data = Vector{Matrix{Float64}}(undef,length(sending_to))
    data_for_me = Vector{Matrix{Float64}}(undef,length(sending_to)+1)
                                    # expecting length(pids) - 1 data points  
    data_for_me[my_idx] = rand(Float64,n,n)
    for i = 1:length(sending_to)
             # generate data for all_other_process
        all_data[i] = rand(Float64,n,n)
    end 

    putting_time = 0.0
    for (i,channel) in enumerate(sending_to)
        _,t = @timed put!(channel,(my_idx,all_data[i]))
        putting_time += t
    end 

   
    data_taken = 1 
    taking_time = 0.0
    while data_taken <= length(sending_to)
        (idx,their_data),t = @timed take!(my_channel)
        taking_time += t 
        data_for_me[idx] = their_data
        data_taken += 1 
    end 

    return data_for_me, putting_time, taking_time

end 


