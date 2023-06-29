
include("hard_coded_contractions.jl")


@testset "Type Stability" begin

    orders = [3,4]
    clique_unweighted_tensors = tensors_from_graph(A,orders,1000,Clique())
    cycle_unweighted_tensors = tensors_from_graph(A,orders,100000,Cycle())
    clique_weighted_tensors = [SymTensor{Clique}(A_ten.n,A_ten.order,A_ten.indices,ones(size(A_ten.indices,2))) for A_ten in clique_unweighted_tensors]

    x  = ones(n)
    y = zeros(n)

    #NOTE: testing allocating functions higher level functions stability depends on ! versions

    @inferred contraction_divide_out(clique_unweighted_tensors[end],x)
    @inferred contraction_divide_out(clique_weighted_tensors[end],x)
    @inferred contraction(clique_unweighted_tensors[end],x)
    @inferred contraction(clique_weighted_tensors[end],x)

    #@inferred embedded_contraction(clique_tensors, x)
    @inferred embedded_contraction!(clique_unweighted_tensors[1], x,y,4)

    @inferred contract_to_mat(clique_unweighted_tensors[end],x)
    @inferred contract_to_mat(clique_weighted_tensors[end],x)
    @inferred contract_to_mat_divide_out(clique_unweighted_tensors[end],x)
    @inferred contract_to_mat_divide_out(clique_weighted_tensors[end],x)

    #@inferred T.contraction_divide_out!(cycle_tensors[end],x,y_divide_out)
    @inferred contraction(cycle_unweighted_tensors[end],x)
    @inferred embedded_contraction!(cycle_unweighted_tensors[1], x,y,4)

    

end

@testset "Contraction Comparisons" begin 

    orders = [3,4,5]
    unweighted_tensors = tensors_from_graph(A,orders,1000,Clique())
    const_weighted_tensors = [SymTensor{Clique}(A_ten.n,A_ten.order,A_ten.indices,ones(size(A_ten.indices,2))) for A_ten in unweighted_tensors]
    rand_weighted_tensors = [SymTensor{Clique}(A_ten.n,A_ten.order,A_ten.indices,rand(size(A_ten.indices,2))) for A_ten in unweighted_tensors]
    x  = ones(n)
    x2 = rand(n)
    
    @testset "single tensor contractions" begin 
        for tensors in [const_weighted_tensors, unweighted_tensors]
            for tensor in tensors
                y_naive = zeros(n)
                y_divide_out = zeros(n)

                T.contraction_divide_out!(tensor,x,y_divide_out)
                T.contraction!(tensor,x,y_naive)

                @test norm(y_divide_out - y_naive)/norm(y_divide_out) < TOL

                y_naive = zeros(n)
                y_divide_out = zeros(n)
                T.contraction_divide_out!(tensor,x2,y_divide_out)
                T.contraction!(tensor,x2,y_naive)
                @test norm(y_divide_out - y_naive)/norm(y_divide_out) < TOL
            end
        end
    end

    @testset "hard coded unweighted contractions" begin 

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(unweighted_tensors[1].indices, x,y_hc)
        T.contraction_divide_out!(unweighted_tensors[1], x,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        fourth_order_contraction!(unweighted_tensors[2].indices, x,y_hc)
        T.contraction_divide_out!(unweighted_tensors[2], x,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(unweighted_tensors[1].indices, x2,y_hc)
        T.contraction_divide_out!(unweighted_tensors[1], x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        fourth_order_contraction!(unweighted_tensors[2].indices, x2,y_hc)
        T.contraction_divide_out!(unweighted_tensors[2], x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

    end

    @testset "single mode contraction" begin 

       

        A_ten = rand_weighted_tensors[end]
    
        B_ten = single_mode_ttv(A_ten,x)
        for _ in 1:(A_ten.order-2)
            B_ten = single_mode_ttv(B_ten,x)
        end

        y = contraction(A_ten,x)
        y2 = zeros(A_ten.n)

        for (i,w) in zip(B_ten.indices,B_ten.weights)
            y2[i] = w 
        end

        @test norm(y - y2)/norm(y) < TOL

    end

    @testset "permutation contraction" begin 

        U = rand(rand_weighted_tensors[2].n,5)
        contraction_comps = contract_all_unique_permutations(rand_weighted_tensors[2],U)
        contraction_comps_hc = fourth_order_contract_all_pairs(rand_weighted_tensors[2],U)
        @test contraction_comps_hc  == contraction_comps

        #TODO: add in fifth order hard coded test to check extra step of recursion
    end

    @testset "hard coded weighted contractions" begin 

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(rand_weighted_tensors[1].indices,rand_weighted_tensors[1].weights, x,y_hc)
        T.contraction_divide_out!(rand_weighted_tensors[1], x,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        fourth_order_contraction!(rand_weighted_tensors[2].indices,rand_weighted_tensors[2].weights, x,y_hc)
        T.contraction_divide_out!(rand_weighted_tensors[2], x,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(rand_weighted_tensors[1].indices,rand_weighted_tensors[1].weights, x2,y_hc)
        T.contraction_divide_out!(rand_weighted_tensors[1], x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        fourth_order_contraction!(rand_weighted_tensors[2].indices,rand_weighted_tensors[2].weights, x2,y_hc)
        T.contraction_divide_out!(rand_weighted_tensors[2], x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

    end

    @testset "contract to mat" begin 

        for tensor in [unweighted_tensors[1],const_weighted_tensors[1]]
            A = contract_to_mat(tensor,x)
            B = contract_to_mat_divide_out(tensor,x)

            @test norm(A - B)/norm(A) < TOL 

            # check vector contraction code coincides
            y = contraction(tensor,x)
            @test norm(y - A*x)/norm(y) < TOL 
        end

    end  

    @testset "embedded contractions" begin 
        y_hc = zeros(n)
        y = zeros(n)

        T.embedded_contraction!(unweighted_tensors[1], x,y,4)
        third_order_encoded_into_fourth_order_contraction!(unweighted_tensors[1].indices,x,y_hc)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)

        T.embedded_contraction!(unweighted_tensors[1], x2,y,4)
        third_order_encoded_into_fourth_order_contraction!(unweighted_tensors[1].indices,x2,y_hc)
        @test norm(y - y_hc)/norm(y) < TOL
    end

    @testset "multi motif contractions" begin
    
        y_hc = zeros(n)
        y = zeros(n)

        contract_multi_motif_tensor!(unweighted_tensors[1].indices, unweighted_tensors[2].indices,x,y_hc)
        T.contraction_divide_out!(unweighted_tensors[1:2],x,y)
        @test norm(y - y_hc)/norm(y) < TOL


        y_hc = zeros(n)
        y = zeros(n)

        contract_multi_motif_tensor!(unweighted_tensors[1].indices, unweighted_tensors[2].indices,x2,y_hc)
        T.contraction_divide_out!(unweighted_tensors[1:2],x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

    end

end

