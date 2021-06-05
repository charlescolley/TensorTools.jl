using Test
using Suppressor

using SparseArrays
import Random:seed!
import MatrixNetworks:erdos_renyi_undirected



#assuming being run from test/ folder
#include("../src/LambdaTAME.jl")

using DistributedTensorConstruction
const DTC = DistributedTensorConstruction


include("distributedTuranShadow_tests.jl")
include("contraction_tests.jl")






