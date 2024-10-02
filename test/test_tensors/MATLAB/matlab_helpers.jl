#= NOTE: this was moved to the main package and exported. 
function eigenspace_residual(A,x)
    
    Axᵏ⁻¹ = contraction(A,x)
    λ =  Axᵏ⁻¹' * x 

    return norm(Axᵏ⁻¹ - λ * x)
end 

function eigenspace_residual(A,λ,x)
    
    Axᵏ⁻¹ = contraction(A,x)
    computed_λ =  Axᵏ⁻¹' * x 

    return abs(computed_λ - λ), norm(Axᵏ⁻¹ - λ * x)
end 
=#

function convert_mat_to_SymTensorUnweighted(A_mat;zero_tol::Float64=1e-15)

    R = CartesianIndices(A_mat)
    Ifirst, Ilast = first(R), last(R)

    n = reduce(max,Ilast.:I)
    order = length(Ilast.:I)
    max_entries = reduce(*,Ilast.:I) #upper bound on unique new entries
    

    coo_indices = Array{Int,2}(undef,order,max_entries)

    new_edge_idx = 0
    for I in R 
        if abs(A_mat[I]) < zero_tol
            new_edge_idx += 1
            coo_indices[:,new_edge_idx] .= I.:I
        end
    end 

    #only keep needed indices
    coo_indices = coo_indices[:,1:new_edge_idx]
    
    return SymTensorUnweighted{Clique}(n, order, unique_edges!(coo_indices))  
end 

function convert_mat_to_SymTensor(A_mat;zero_tol::Float64=1e-15)

    R = CartesianIndices(A_mat)
    Ifirst, Ilast = first(R), last(R)

    n = reduce(max,Ilast.:I)
    order = length(Ilast.:I)
    max_entries = reduce(*,Ilast.:I) #upper bound on unique new entries
    
    coo_indices = Array{Int,2}(undef,order,max_entries)
    coo_weights = Array{Float64,1}(undef,max_entries)

    new_edge_idx = 0
    for I in R 
        if abs(A_mat[I]) > zero_tol
            new_edge_idx += 1
            coo_indices[:,new_edge_idx] .= I.:I
            coo_weights[new_edge_idx] = A_mat[I]
        end
    end 
    
    #only keep needed indices
    coo_indices = coo_indices[:,1:new_edge_idx]
    coo_weights = coo_weights[1:new_edge_idx]


    #return coo_indices, coo_weights
    coo_indices, coo_weights = unique_edges!(coo_indices,coo_weights)
    return SymTensor{Clique}(n, order,coo_indices, coo_weights)  
end 


function unique_edges!(edge_indices)
    #TODO: Is there a proper in place solution?

    #  -- sort lexicographically

    foreach(sort!,eachslice(edge_indices,dims=2)) #sort each column in place
    Base.permutecols!!(edge_indices,sortperm(eachcol(edge_indices))) #sort columns by found permutation 
    #TODO: can this be done without running sortperm? 

    #  -- remove duplicate columns 

    head = 1 #head is last column with unique col
    for j = 2:size(edge_indices,2)
        if edge_indices[:,head] == edge_indices[:,j]
            continue 
        else 
            head += 1 
            edge_indices[:,head] .= edge_indices[:,j]
        end 
    end 

    return edge_indices[:,1:head]
end 

function unique_edges!(edge_indices,edge_weights)
    #TODO: Is there a proper in place solution?

    #  -- sort lexicographically
    foreach(sort!,eachslice(edge_indices,dims=2)) #sort each column in place

    p = sortperm(eachcol(edge_indices))
    permute!(edge_weights,p)
    Base.permutecols!!(edge_indices,p) #sort columns by found permutation 


    #  -- remove duplicate columns 

    head = 1 #head is last column with unique col
    for j = 2:size(edge_indices,2)
        if edge_indices[:,head] == edge_indices[:,j]
            continue 
        else 
            head += 1 
            edge_indices[:,head] .= edge_indices[:,j]
            edge_weights[head] = edge_weights[j]
        end 
    end 

    return edge_indices[:,1:head], edge_weights[1:head]
end 

