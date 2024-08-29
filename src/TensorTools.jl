
module TensorTools

include("externalSoftware/TuranShadow.jl")


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
using RemoteChannelCollectives

# -- contractions.jl -- #
using Combinatorics:integer_partitions,permutations, multinomial, combinations,  partitions, multiset_permutations

#using PyCall
#@pyimport pickle

#NOTE: keeping @pyimport currently triggers compilation error "ERROR: LoadError: Evaluation into the closed module `__anon__`..."

abstract type Motif  end
struct Clique <: Motif end
struct Cycle <: Motif end

struct SymTensorUnweighted{T <: Motif}
    n::Int
    order::Int
    indices::Array{Int,2}
end

struct SymTensor{T <: Motif,S}
    n::Int
    order::Int
    indices::Matrix{Int}
    weights::Vector{S}
    SymTensor{T}(n::Int,order::Int,indices::Matrix{Int},weights::Vector{S}) where {T<:Motif,S} = new{T,S}(n,order,indices,weights)
end

#struct TensorComplexUnweighted
#    tensors::Array{SymTensorUnweighted,1}
#end


#LVGNA_data = "/Users/ccolley/PycharmProjects/LambdaTAME/data/sparse_matrices/LVGNA/"  #local
#LVGNA_data = "/homes/ccolley/Documents/Research/heresWaldo/data/MultiMagna_TAME"       #server
#TENSOR_PATH = "/homes/ccolley/Documents/Research/TensorConstruction/tensors/"   #server


include("fileio.jl")
include("tensorConstruction.jl")
include("contraction.jl")
export partitions
include("distributedTuranShadow.jl")


#TODO: move to Experiments.jl?

export SymTensorUnweighted, SymTensor, TensorComplexUnweighted
export Motif, Clique, Cycle 

export endBehavior,returnToSpawner, writeFile
export distributed_clique_sample, distributed_sample_smat_files, distributed_find_all_cliques

export tensor_from_graph, tensors_from_graph
export load_SymTensorUnweighted
export contraction_divide_out!, embedded_contraction!, contraction!, contraction_divide_out, embedded_contraction, contraction
export contract_to_mat, contract_to_mat!, contract_to_mat_divide_out
export single_mode_ttv
export contract_all_unique_permutations


end #module end 