
include("hard_coded_contractions.jl")


@testset "Type Stability" begin

    orders = [3,4]
    clique_tensors = tensors_from_graph(A,orders,1000,Clique())
    cycle_tensors = tensors_from_graph(A,orders,100000,Cycle())

    x  = ones(n)
    y = zeros(n)

    #NOTE: testing allocating functions higher level functions stability depends on ! versions

    @inferred contraction_divide_out(clique_tensors[end],x)
    @inferred contraction(clique_tensors[end],x)
    #@inferred embedded_contraction(clique_tensors, x)
    @inferred embedded_contraction!(clique_tensors[1], x,y,4)
        

    #@inferred DTC.contraction_divide_out!(cycle_tensors[end],x,y_divide_out)
    @inferred contraction(cycle_tensors[end],x)
    @inferred embedded_contraction!(cycle_tensors[1], x,y,4)

end

@testset "Contraction Comparisons" begin 

    orders = [3,4,5]
    tensors = tensors_from_graph(A,orders,1000,Clique())
    
    x  = ones(n)
    x2 = rand(n)
    
    @testset "single tensor contractions" begin 
        for tensor in tensors
            y_naive = zeros(n)
            y_divide_out = zeros(n)

            DTC.contraction_divide_out!(tensor,x,y_divide_out)
            DTC.contraction!(tensor,x,y_naive)

            @test norm(y_divide_out - y_naive)/norm(y_divide_out) < TOL

            y_naive = zeros(n)
            y_divide_out = zeros(n)
            DTC.contraction_divide_out!(tensor,x2,y_divide_out)
            DTC.contraction!(tensor,x2,y_naive)
            @test norm(y_divide_out - y_naive)/norm(y_divide_out) < TOL
        end
    end

    @testset "hard coded contractions" begin 

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(tensors[1].indices, x,y_hc)
        DTC.contraction_divide_out!(tensors[1], x,y)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)
        third_order_contraction!(tensors[1].indices, x2,y_hc)
        DTC.contraction_divide_out!(tensors[1], x2,y)
        @test norm(y - y_hc)/norm(y) < TOL
    end

    @testset "embedded contractions" begin 
        y_hc = zeros(n)
        y = zeros(n)

        DTC.embedded_contraction!(tensors[1], x,y,4)
        third_order_encoded_into_fourth_order_contraction!(tensors[1].indices,x,y_hc)
        @test norm(y - y_hc)/norm(y) < TOL

        y_hc = zeros(n)
        y = zeros(n)

        DTC.embedded_contraction!(tensors[1], x2,y,4)
        third_order_encoded_into_fourth_order_contraction!(tensors[1].indices,x2,y_hc)
        @test norm(y - y_hc)/norm(y) < TOL
    end

    @testset "multi motif contractions" begin
    
        y_hc = zeros(n)
        y = zeros(n)

        contract_multi_motif_tensor!(tensors[1].indices, tensors[2].indices,x,y_hc)
        DTC.contraction_divide_out!(tensors[1:2],x,y)
        @test norm(y - y_hc)/norm(y) < TOL


        y_hc = zeros(n)
        y = zeros(n)

        contract_multi_motif_tensor!(tensors[1].indices, tensors[2].indices,x2,y_hc)
        DTC.contraction_divide_out!(tensors[1:2],x2,y)
        @test norm(y - y_hc)/norm(y) < TOL

    end

end