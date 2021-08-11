using Test
using Suppressor

using SparseArrays
import Random:seed!
import MatrixNetworks:erdos_renyi_undirected, triangles
import LinearAlgebra:norm



#assuming being run from test/ folder
#include("../src/LambdaTAME.jl")

using DistributedTensorConstruction
const DTC = DistributedTensorConstruction

seed!(54321)
n= 25
trials = 1000
order = 5
clique_size = 3
TOL= 1e-15

A = sparse(erdos_renyi_undirected(n,.9999)) # use a small clique, 
                                #   p=1.0 interprets as d_avg = 1
A = max.(A,A')
for i = 1:size(A,1)
    A[i,i] = 0.0
end
dropzeros!(A)


#include("distributedTuranShadow_tests.jl")
include("tensorConstruction_tests.jl")
include("contraction_tests.jl")






