
module DistributedTensorConstruction 

include("../externalSoftware/TuranShadow.jl")   #local
#include("/homes/ccolley/Documents/Software/GLANCE/TuranShadow.jl")    #server


# -- fileio -- #
#using NumbersFromText
using FileIO: save
using JLD: save,load
using CSV

# -- fileio -- #
using Random

using MatrixNetworks
using LightGraphs
using Distributed

# -- contractions.jl -- #
using Combinatorics:integer_partitions,permutations, multinomial

#using PyCall
#@pyimport pickle

#NOTE: keeping @pyimport currently triggers compilation error "ERROR: LoadError: Evaluation into the closed module `__anon__`..."

# struct SymTensorUnweighted
#     n::Int
#     order::Int
#     indices::Array{Int,2}
# end

abstract type Motif  end
struct Clique <: Motif end
struct Cycle <: Motif end

struct SymTensorUnweighted{T <: Motif}
    n::Int
    order::Int
    indices::Array{Int,2}
end

#TODO: rethink naming

#struct TensorComplexUnweighted
#    tensors::Array{SymTensorUnweighted,1}
#end


#LVGNA_data = "/Users/ccolley/PycharmProjects/LambdaTAME/data/sparse_matrices/LVGNA/"  #local
#LVGNA_data = "/homes/ccolley/Documents/Research/heresWaldo/data/MultiMagna_TAME"       #server
#TENSOR_PATH = "/homes/ccolley/Documents/Research/TensorConstruction/tensors/"   #server

include("fileio.jl")
include("tensorConstruction.jl")
include("distributedTuranShadow.jl")
#include("communication.jl")
include("contraction.jl")

#TODO: move to Experiments.jl?

export SymTensorUnweighted, TensorComplexUnweighted
export Motif, Clique, Cycle 

export tensor_from_graph, tensors_from_graph
export load_SymTensorUnweighted
export contraction_divide_out!, embedded_contraction!, contraction!, contraction_divide_out, embedded_contraction, contraction


end #module end 