
import TensorTools: tensor_from_graph_profiled
@testset "Motif Sampling" begin

    _, cliques::Array{Array{Int64,1},1} = T.TuranShadow(A,order,trials)

    @testset "clique enumeration/sorting" begin

        original_cliques = copy(cliques)
        T.reduce_to_unique_cliques!(cliques)

        original_cliques = [sort(clique) for clique in original_cliques]
        original_cliques = Set(original_cliques)

        @test issetequal(original_cliques,Set(cliques))
        @test length(cliques) == length(Set(cliques))
    end

    @testset "cycle enumeration" begin 

        LG_T = tensors_from_graph(A,[3],200000,Cycle())[1]
                                #sample count is overkill to find all triangles
        
        LG_Triangles = Set(eachcol(LG_T.indices))
        MN_Triangles = collect(triangles(A))
        
        @test length(LG_Triangles) == length(MN_Triangles)
        @test all([collect(t) in LG_Triangles for t in MN_Triangles])

    
        @testset "unique_cycles" begin 



            @testset "cycle hashing" begin 
            
                l = [rand(UInt)]
                rev_l = [l[i] for i =length(l):-1:1]
    
                @test T.cycle_hash(l) == T.cycle_hash(rev_l)
                @test T.cycle_hash2(l) == T.cycle_hash2(rev_l)

                @test_broken T.cycle_hash([2, 9, 12]) != T.cycle_hash([3,4,18])
                #hash fails when i*j*k = i'*j'*k'
            end

        end

    end

    @testset "matrix Contructors" begin

        @inferred tensor_from_graph(A, clique_size, trials, Clique())
        @inferred tensor_from_graph(A, clique_size, 0, Clique())

        @inferred tensor_from_graph(A, clique_size, Clique())
        @suppress begin
            @inferred tensor_from_graph_profiled(A, clique_size, Clique())
        end
        @inferred tensors_from_graph(A, [3,4], trials,Clique())
        @inferred tensors_from_graph(A, [3,4], [1000,10000],Clique())
        @inferred tensors_from_graph(A, [3,4], 0,Clique())

        @inferred tensor_from_graph(A, clique_size, trials,Cycle())
        @inferred tensor_from_graph(A, clique_size, 0,Cycle())

        @inferred tensors_from_graph(A, [3,4], trials,Cycle())
        @inferred tensors_from_graph(A, 5, trials, Cycle())
        @inferred tensors_from_graph(A, 5, 0, Cycle())
    end 

end


function is_oriented_opposite(cycle1,cycle2)
    l = length(cycle1)
    if l != length(cycle2)
        return false
    end

    for i = 1:l
    end

end