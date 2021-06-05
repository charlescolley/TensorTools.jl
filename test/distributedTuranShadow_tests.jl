@testset "Motif Sampling" begin


    _, cliques::Array{Array{Int64,1},1} = DTC.TuranShadow(A,order,trials)

    @testset "clique enumeration/sorting" begin

        original_cliques = copy(cliques)
        DTC.reduce_to_unique_motifs!(cliques)

        original_cliques = [sort(clique) for clique in original_cliques]
        original_cliques = Set(original_cliques)

        @test issetequal(original_cliques,Set(cliques))
        @test length(cliques) == length(Set(cliques))
    end

    @testset "matrix Contructors" begin

        @inferred tensor_from_graph(A, clique_size, trials)
        @inferred tensors_from_graph(A, [3,4], trials)
        @inferred tensors_from_graph(A, [3,4], [1000,10000])

    end

end
