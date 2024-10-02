using Test
using Suppressor
using Distributed

using SparseArrays
#import Random:seed!
import MatrixNetworks:erdos_renyi_undirected, triangles
import LinearAlgebra:norm



#assuming being run from test/ folder
#include("../src/LambdaTAME.jl")

using TensorTools
const T = TensorTools

T.seed!(54321)
n= 10
trials = 1000
order = 4
clique_size = 3
TOL= 1e-15

A = sparse(erdos_renyi_undirected(n,.9999)) # use a small clique, 
                                #   p=1.0 interprets as d_avg = 1
A = max.(A,A')
for i = 1:size(A,1)
    A[i,i] = 0.0
end
dropzeros!(A)

test_smat_file = "test_tensors/testsymER_n:25_p:9999e-4_seed:54321.smat"

# src: https://github.com/JuliaLang/julia/issues/18780#issuecomment-863505347
struct NoException <: Exception end
macro test_nothrow(ex)
    esc(:(@test_throws NoException ($(ex); throw(NoException()))))
end

addprocs(7)
@everywhere using TensorTools
using MAT

include("distributedTuranShadow_tests.jl")
include("fileio_tests.jl")
include("tensorConstruction_tests.jl")
include("eigenvector_algorithms_tests.jl")

include("contraction_tests.jl")
#include("MPI_tests.jl")





