@testset "fileio" begin

    @inferred T.load_ssten("test_tensors/test_tensorA.ssten",'\t')
    for motifType in [Clique(),Cycle()]
        @inferred load_SymTensorUnweighted("test_tensors/test_tensorA.ssten",motifType,'\t')
    end
end