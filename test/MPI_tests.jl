addprocs(4)
using DistributedTensorConstruction: broadcast_communication, gather_communication, 
                                     personalized_all_to_all_communication,
                                     all_to_all_reduction_communication
@everywhere using DistributedTensorConstruction: all_to_all_reduce
@everywhere using Random: seed!



@testset "MPI Tests" begin

    pids = workers()
    seed!(0)
    seeds = rand(UInt,length(pids))
    n = 1
   
    
    @testset "Broadcast Test" begin
        
        @everywhere function broadcast_test_proc(n,communication)

            if communication.receiving_from === nothing 
                seed!(0)
                data = rand(Float64,n,n)
                broadcast(data,communication)
            else
                data = broadcast(nothing ,communication)
            end 

            if communication.receiving_from === nothing 
                broadcast(data,communication)
                profiled_data = data
            else
                profiled_data,_,_,_ = broadcast_profiled(nothing ,communication)
            end 

            return data, profiled_data      
        end
        

        bcast_pididx = 1
        #
        #  Stage the Communication
        #
        @inferred broadcast_communication(pids,bcast_pididx,zeros(Float64,0,0))
        communication = broadcast_communication(pids,bcast_pididx,zeros(Float64,0,0))

        futures = []

        #
        #  Start the processors
        #

        for p = 1:length(pids)
            future = @spawnat pids[p] broadcast_test_proc(communication[p],n)
        end

        bcast_vals = []
        bcast_profiled_vals = []

        
        # collect and aggregate the results 
        for future in futures
            bcast_data, bcast_profiled_data = fetch(future)
            push!(bcast_vals,bcast_data)
            push!(bcast_profiled_vals,bcast_profiled_data)
        end    

        @test all([v == bcast_vals[1] for v in bcast_vals])
        @test all([v == bcast_profiled_vals[1] for v in bcast_profiled_vals])

    end

    @testset "Gather Test" begin
    
        @everywhere function gather_proc(sending_to,receiving_from,seed,n)


            seed!(seed)
            my_data = rand(Float64,n,n)
        
            all_data = Vector{Matrix{Float64}}(undef,1)
            all_data[1] = my_data
        
            for channel in reverse(receiving_from)#reverse()
                           # communication patterns come from the inverse of the broadcast code, 
                           # so receiving channels need to be reverse.  
                           # TODO: may be good to reconsider this version
        
        
                their_data = take!(channel)
        
                append!(all_data,their_data)
            end
        
            if sending_to === nothing
                return all_data
            else 
                put!(sending_to,all_data)
            end
        
        
        end


        gather_pididx = 1
        
        #
        #  Stage the Communication
        #
        @inferred gather_communication(pids,gather_pididx,zeros(Float64,0,0))
        sending_to, receiving_from  = gather_communication(pids,gather_pididx,zeros(Float64,0,0))

        futures = []


        #
        #  Start the processors
        #

        for p = 1:length(pids)
            if p == gather_pididx
                future = @spawnat pids[p] gather_proc(nothing,receiving_from[p],seeds[p],n)
            else
                future = @spawnat pids[p] gather_proc(sending_to[p],receiving_from[p],seeds[p],n)
            end
            push!(futures,future)
        end



        all_vals = []

        
        # collect and aggregate the results 
        for future in futures
            push!(all_vals,fetch(future))
        end    

        parallel_generated = fetch(futures[gather_pididx])
        
        serial_generated = []
        for seed in seeds
            seed!(seed)
            push!(serial_generated,rand(Float64,n,n))
        end

        @test parallel_generated == serial_generated
    end
    =#
    @testset "All to All Reduction" begin

        @inferred all_to_all_reduction_communication(pids,1)
        communication = all_to_all_reduction_communication(pids,1)
    

        @everywhere function all_to_all_reduction_proc(my_data,proc_communication)

            reduction_f = (x,y) -> x + y 
            reduced_data = all_to_all_reduce(reduction_f,my_data,proc_communication)

            return reduced_data       
        end


        test_vals = rand(1:100,length(pids))

        futures = []
    
        for p = 1:length(pids)
            future = @spawnat pids[p] all_to_all_reduction_proc(test_vals[p],communication[p])
            push!(futures,future)
        end

        all_vals = [] 
        for future in futures 
            push!(all_vals,fetch(future))
        end 

        @test sum(test_vals) == all_vals[1]
        @test all([v == all_vals[1] for v in all_vals])
        
    end
    #=
    @testset "Personalized All to All" begin

        @everywhere function personalized_communication_proc(sending_to,my_channel,seed,my_idx,n)

            seed!(seed)
            all_data = Vector{Matrix{Float64}}(undef,length(sending_to))
            data_for_me = Vector{Matrix{Float64}}(undef,length(sending_to)+1)
                                            # expecting length(pids) - 1 data points  
            data_for_me[my_idx] = rand(Float64,n,n)
            for i = 1:length(sending_to)
                     # generate data for all_other_process
                all_data[i] = rand(Float64,n,n)
            end 
        
            for (i,channel) in enumerate(sending_to)
                put!(channel,(my_idx,all_data[i]))
            end 
        
            data_taken = 1 
            while data_taken <= length(sending_to)
                (idx,their_data) = take!(my_channel)
        
                data_for_me[idx] = their_data
                data_taken += 1 
            end 
        
            return data_for_me
        
        end 
        

        #  --  Stage the Communication  --  #
        @inferred personalized_all_to_all_communication(pids,(1,zeros(Float64,0,0)))
        sending_to, channels = personalized_all_to_all_communication(pids,(1,zeros(Float64,0,0)))
        futures = []
    
    
        #
        #  Start the processors
        #
    
        for p = 1:length(pids)

            future = @spawnat pids[p] personalized_communication_proc(sending_to[p], channels[p],seeds[p],p,n)
            push!(futures,future)
        end
    

        all_vals = Array{Matrix{Float64}}(undef,length(pids),length(pids))
        
        # collect and aggregate the results 
        for (i,future) in enumerate(futures)
            all_vals[i,:] = fetch(future)
        end    
        
    
        serial_generated = Array{Matrix{Float64}}(undef,length(pids),length(pids))
        for (p_i,seed) in enumerate(seeds)
            seed!(seed)
    
            serial_generated[p_i,p_i] = rand(Float64,n,n)
            if length(pids) == 2^Int(round(log2(length(pids))))
                for p_j=1:length(pids)-1
                    serial_generated[((p_i - 1) ⊻ p_j) + 1,p_i] = rand(Float64,n,n)
                end
            else
                for p_j=1:length(pids)
                    if p_j != p_i 
                        serial_generated[p_j,p_i] = rand(Float64,n,n)
                    end
                end
            end
    
        end
        @test all_vals == serial_generated
            #power of two serial data is mapped incorrectly 
   

    end 

end