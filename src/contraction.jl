


#TODO: test
function contraction!(A::SymTensorUnweighted, x::Array{Float64,1},y::Array{Float64,1})

    order, edges = size(A.indices)

    scaling_factor = factorial(order-1)

    for i=1:edges
        for j=1:order

            val = scaling_factor

            for k=1:j-1
                val *= x[A.indices[k,i]]
            end

            for k=j+1:order
                val *= x[A.indices[k,i]]
            end
            y[A.indices[j,i]] += val
        end
    end
end

#TODO: test
function contraction_divide_out!(A::SymTensorUnweighted, x::Array{Float64,1},y::Array{Float64,1})

    order, edges = size(A.indices)
    scaling_factor = factorial(order-1)

    for i=1:edges
        val = scaling_factor
        for j=1:order
            val *= x[A.indices[j,i]]
        end

        for j=1:order
            y[A.indices[j,i]] += (val/x[A.indices[j,i]])
        end
    end
end

function contraction_divide_out!(simplicial_complexes::Array{SymTensorUnweighted,1},x, y)
    #assuming simplicial_complexes are sorted by their motif order

    for i=1:length(simplicial_complexes)
        contraction_divide_out!(simplicial_complexes[i],x,y)
    end
end


function embedded_contraction!(A::SymTensorUnweighted, x::Array{T,1},y::Array{T,1},embedded_mode::Int) where T

    order,edges = size(A.indices)
    #@assert order == 3

    # compute contraction template
    partition = [ x for x in integer_partitions(embedded_mode - order) if length(x) <= order]
    #println(partition)
    partition = [length(y) < order ? vcat(y,zeros(Int,order - length(y))) : y  for y in partition]
                     #pad the entries to all have the same length
    partition = [x .+= 1 for x in partition]
    
    multiplicities::Array{Array{Integer,1},1} = unique(hcat([collect(permutations(y)) for y in partition]...))

    scaling_factors = Array{Integer,2}(undef,order,length(multiplicities))
    for i =1:length(multiplicities)
        for j=1:order
            row = copy(multiplicities[i])
            row[j] -= 1
            scaling_factors[j,i] = multinomial(row...)
        end
        #push!(scaling_factors,row_factors)
    end

    for idx =1:edges

        for idx2 =1:length(multiplicities)
            for idx3 =1:order
                val::Float64 = scaling_factors[idx3,idx2]

                for idx4=1:idx3-1
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end
                val *= x[A.indices[idx3,idx]]^(multiplicities[idx2][idx3] -1)
                for idx4=idx3+1:order
                    val *= x[A.indices[idx4,idx]]^multiplicities[idx2][idx4]
                end

                y[A.indices[idx3,idx]] += val
            end
        end
    
    end
end

function embedded_contraction!(simplicial_complexes::Array{SymTensorUnweighted,1},x, y)
    #assuming simplicial_complexes are sorted by their motif order

    max_order = size(simplicial_complexes[end].indices,1)

    for i=1:length(simplicial_complexes)
        if i == length(simplicial_complexes)
            contraction_divide_out!(simplicial_complexes[i],x,y)
        else
            embedded_contraction!(simplicial_complexes[i],x,y,max_order)
        end
    end
end

