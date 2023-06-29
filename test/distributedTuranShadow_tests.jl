@everywhere using TensorTools: parallel_clique_sampling_proc, parallel_clique_sampling_proc_profiled, 
                                     parallel_find_all_clique_proc

@testset "Distributed Turan Shadow Tests" begin

    pids = workers()
    samples=10000
    
    @testset "Distributed Sample Sort Test" begin


        collection_idx = 1
        gather_comm = T.gather_communication(pids,collection_idx,[[1]])
        bcast_comm = T.broadcast_communication(pids,collection_idx,[[1]])
        pa2a_comm = T.personalized_all_to_all_communication(pids,[[1]])

        futures = []
        for p = 1:length(pids)

            if p == length(pids)
                serialization_channel = nothing 
            else
                serialization_channel = pa2a_comm[p+1].my_channel
            end 

            future = @spawnat pids[p] parallel_clique_sampling_proc(test_smat_file,order,samples,pids, "test.ssten",
                                                                    gather_comm[p],bcast_comm[p],pa2a_comm[p],
                                                                    serialization_channel,returnToSpawner())
            push!(futures,future)
        end



        all_splittings = []
        cliques_were_correctly_places = Vector{Bool}(undef,length(pids))
        correct_range = true 
        for (p,future) in enumerate(futures)
            f  = fetch(future)
            if isa(f,RemoteException)
                throw(f)
            end 
            cliques,(lower_clique,upper_clique) = f 
            for clique in cliques 
                if p == length(pids)
                    if !(lower_clique <= clique)
                        #last process only needs lower bound 
                        println("proc:$(p)  clique:$(clique) not in range ($(lower_clique):Inf)")
                        correct_range = false
                    end
                else 
                    if !(lower_clique <= clique && clique < upper_clique)
                        println("proc:$(p)  clique:$(clique) not in range ($(lower_clique):$(upper_clique))")
                        correct_range = false 
                    end 
                end
            end 
        end

        @test correct_range
    end 

    @testset "Distributed Find All Cliques" begin
        
        @testset "Helper Functions" begin
            
        end 
      
        @testset "Full Procedure" begin

            collection_idx = 1
            init_samples_per_proc =1000
            gather_comm = T.gather_communication(pids,collection_idx,[[1]])
            bcast_comm = T.broadcast_communication(pids,collection_idx,[[1]])
            pa2a_comm = T.personalized_all_to_all_communication(pids,[[1]])
            ra2a_comm = T.all_to_all_reduction_communication(pids,true)

            futures = []
            for p = 1:length(pids)

                if p == length(pids)
                    serialization_channel = nothing 
                else
                    serialization_channel = pa2a_comm[p+1].my_channel
                end 
                #=
                future = @spawnat pids[p] DistributedTensorConstruction.parallel_clique_sampling_proc(
                                                test_smat_file,order,samples,pids, "test.ssten",
                                                gather_comm[p],bcast_comm[p],pa2a_comm[p],
                                                serialization_channel,returnToSpawner())
                =#
                future = @spawnat pids[p] parallel_find_all_clique_proc(test_smat_file, order, init_samples_per_proc, pids, "test.ssten",
                                                gather_comm[p], bcast_comm[p], pa2a_comm[p],ra2a_comm[p],
                                                serialization_channel,returnToSpawner())
                push!(futures,future)
            end



            all_splittings = []

            correct_range = true 
            total_cliques_found = 0 
            for (p,future) in enumerate(futures)
                f  = fetch(future)
                if isa(f,RemoteException)
                    throw(f)
                end 
                cliques,(lower_clique,upper_clique) = f 
        
                total_cliques_found += length(cliques)

                for clique in cliques 
                    if p == length(pids)
                        if !(lower_clique <= clique)
                            #last process only needs lower bound 
                            #println("proc:$(p)  clique:$(clique) not in range ($(lower_clique):Inf)")
                            correct_range = false
                        end
                    else 
                        if !(lower_clique <= clique && clique < upper_clique)
                           #println("proc:$(p)  clique:$(clique) not in range ($(lower_clique):$(upper_clique))")
                            correct_range = false 
                        end 
                    end
                end
            end

            mat_n = parse(Int,split(filter(f->occursin("n:",f), split(test_smat_file,"_"))[1],"n:")[end])
            @test binomial(mat_n,order) == total_cliques_found
                  # test matrices are fully connected networks
            @test correct_range
        end
        
    end



    @testset "Distributed Sample Sort Driver Test" begin

        proc_count = length(pids)
        samples = 10000*proc_count

        ssten_filename = "test.ssten"

        for profile in [false, true]
            @test_nothrow distributed_clique_sample(pids, test_smat_file, ssten_filename, order, samples, profile)
        end
        @test_nothrow distributed_sample_smat_files(pids, [test_smat_file], "./", [3,4],[1000*proc_count,2000*proc_count])
    end 

end