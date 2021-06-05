"""-----------------------------------------------------------------------------
    These hard coded contraction operations have been tested against small 
  examples generated in Tensor Toolbox. They serve as some simple sanity checks 
  for the general contraction routines. 
-----------------------------------------------------------------------------"""

function contract_embedded_multi_motif_tensor!(third_order_Indices::Array{Integer,2}, fourth_order_Indices::Array{Integer,2},
    x::Array{Float64,1},y::Array{Float64,1})

    fourth_order_contraction!(fourth_order_Indices::Array{Integer,2}, x::Array{Float64,1},y::Array{Float64,1})   
    third_order_encoded_into_forth_order_contraction!(third_order_Indices::Array{Integer,2}, x::Array{Float64,1},y::Array{Float64,1})                        

end

function contract_multi_motif_tensor!(second_order_Indices::Array{Int,2},third_order_Indices::Array{Int,2}, fourth_order_Indices::Array{Int,2},
    x::Array{Float64,1},y::Array{Float64,1})

    fourth_order_contraction!(fourth_order_Indices, x,y)   
    third_order_contraction!(third_order_Indices,   x,y)  
    second_order_contraction!(second_order_Indices, x,y)                      

end

function contract_embedded_multi_motif_tensor!(second_order_Indices::Array{Int,2},third_order_Indices::Array{Int,2}, fourth_order_Indices::Array{Int,2},
    x::Array{Float64,1},y::Array{Float64,1})

    fourth_order_contraction!(fourth_order_Indices, x,y)   
    third_order_encoded_into_forth_order_contraction!(third_order_Indices, x,y)
    second_order_encoded_into_forth_order_contraction!(second_order_Indices, x,y)                         

end

function second_order_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1}) where T

    #TODO: needs more robust testing. 
    #      seems to work on small examples compared to Tensor Toolbox

    order,nnz= size(indices)
    @assert order == 2

    #
    for idx=1:nnz
        i,j,k = indices[:,idx]

        y[i] += x[j]
        y[j] += x[i]
    end

end

function third_order_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1}) where T

    #TODO: needs more robust testing. 
    #      seems to work on small examples compared to Tensor Toolbox

    order,nnz= size(indices)
    @assert order == 3

    #
    for idx=1:nnz
        i,j,k = indices[:,idx]

        y[i] += 2*x[j]*x[k]
        y[j] += 2*x[i]*x[k]
        y[k] += 2*x[i]*x[j]
    end

end

#Assuming that y is zeroed out and that weight on edges is 1. 
function fourth_order_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1}) where T

    #TODO: needs more robust testing. 
    #      seems to work on small examples compared to Tensor Toolbox

    order,nnz= size(indices)
    @assert order == 4

    #
    for idx=1:nnz
        i,j,k,l = indices[:,idx]

        y[i] += 6*x[j]*x[k]*x[l]
        y[j] += 6*x[i]*x[k]*x[l]
        y[k] += 6*x[i]*x[j]*x[l]
        y[l] += 6*x[i]*x[j]*x[k]

    end

end

#Assuming that y is zeroed out and that weight on edges is 1. 
function third_order_encoded_into_forth_order_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1}) where T

    #TODO: needs more robust testing. 
    #      seems to work on small examples compared to Tensor Toolbox
    #TODO: double check math 

    order,nnz= size(indices)
    @assert order == 3

    #
    for idx=1:nnz
        i,j,k = indices[:,idx]

        y[i] += 6*x[j]*(x[k]^2)
        y[j] += 6*x[i]*(x[k])^2
        y[k] += 12*x[i]*x[j]*x[k]

        y[i] += 6*(x[j]^2)*x[k]
        y[j] += 12*x[i]*x[k]*x[j]
        y[k] += 6*x[i]*x[j]^2

        y[i] += 12*x[j]*x[k]*x[i]
        y[j] += 6*(x[i]^2)*x[k]
        y[k] += 6*(x[i]^2)*x[j]

    end

end

function second_order_encoded_into_forth_order_contraction!(indices::Array{Int,2}, x::Array{T,1},y::Array{T,1}) where T

    #TODO: needs more robust testing. 
    #      seems to work on small examples compared to Tensor Toolbox

    order,nnz= size(indices)
    @assert order == 2

    #
    for idx=1:nnz
        i,j = indices[:,idx]

        y[i] += 6*x[j]^3
        y[j] += 18*x[i]*x[j]^2

        y[i] += 18*x[i]^2*x[j]
        y[j] += 6*x[i]^3

    end

end

function contract_multi_motif_tensor!(third_order_Indices::Array{Int,2}, fourth_order_Indices::Array{Int,2},
    x::Array{Float64,1},y::Array{Float64,1})

    fourth_order_contraction!(fourth_order_Indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})   
    third_order_contraction!(third_order_Indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})                        

end

function contract_embedded_multi_motif_tensor!(third_order_Indices::Array{Int,2}, fourth_order_Indices::Array{Int,2},
    x::Array{Float64,1},y::Array{Float64,1})

    fourth_order_contraction!(fourth_order_Indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})   
    third_order_encoded_into_forth_order_contraction!(third_order_Indices::Array{Int,2}, x::Array{Float64,1},y::Array{Float64,1})                        

end
