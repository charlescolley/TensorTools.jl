@testset "Motif Sampling" begin

    _, cliques::Array{Array{Int64,1},1} = DTC.TuranShadow(A,order,trials)

    @testset "clique enumeration/sorting" begin

        original_cliques = copy(cliques)
        DTC.reduce_to_unique_cliques!(cliques)

        original_cliques = [sort(clique) for clique in original_cliques]
        original_cliques = Set(original_cliques)

        @test issetequal(original_cliques,Set(cliques))
        @test length(cliques) == length(Set(cliques))
    end

    @testset "cycle enumeration" begin 

        LG_T = tensors_from_graph(A,[3],100000,Cycle())
                                #sample count is overkill to find all triangles
        
        LG_Triangles = Set(eachcol(LG_T[1].indices))
        @test all([collect(t) in LG_Triangles for t in collect(triangles(A))])



    end

    @testset "matrix Contructors" begin

        @inferred tensor_from_graph(A, clique_size, trials,Clique())
        @inferred tensors_from_graph(A, [3,4], trials,Clique())
        @inferred tensors_from_graph(A, [3,4], [1000,10000],Clique())

        @inferred tensor_from_graph(A, clique_size, trials,Cycle())
        @inferred tensors_from_graph(A, [3,4], trials,Cycle())
        @inferred tensors_from_graph(A, 5, trials, Cycle())
    end 

end
