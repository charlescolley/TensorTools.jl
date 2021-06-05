@testset "Motif Sampling" begin

    seed!(54321)
    n= 25
    trials = 1000
    order = 5
    clique_size = 3

    A = sparse(erdos_renyi_undirected(n,.9999)) # use a small clique, 
                                    #   p=1.0 interprets as d_avg = 1
    A = max.(A,A')
    #export SymTensorUnweighted

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
