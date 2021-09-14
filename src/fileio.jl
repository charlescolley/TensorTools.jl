

function process_matrix(file)
    M = readmatrix(file)
    n = Int(M[1,1])

    #reduce to only symmetric edges(no diagonal elements)
    edges = round.(Int,hcat(filter(row->(row[1] < row[2]),collect(eachrow(M[2:end,1:2])))...))
    edges .+= 1

    return  edges,n
end

function write_smat(A::SparseMatrixCSC{T,Int},filename::String) where T

    @assert filename[end-4:end] == ".smat"

    n,m = size(A)
    edges = nnz(A)
    open(filename,"w") do f 
    
        write(f, "$n $m $edges\n")

        for (i,j,w) in zip(findnz(A)...)
            write(f,"$(i-1) $(j-1) $w\n")
        end
        
    end
        
end

function write_vector(x::Array{T,1}, filename::String) where T

    n = length(x)

    open(filename,"w") do f 
    
        for element in x
            write(f,"$element\n")
        end        
    end
end

#TODO: determine types
function write_tensor(edge_set,outputfile)

    k = length(edge_set[1])
    indices = zeros(k,length(edge_set))
    idx = 1
    n = -1
    for clique in edge_set
   
        indices[:,idx] = clique
        n_c = maximum(clique)
        if n < n_c
            n = n_c
        end
        idx += 1;
    end

   write_ssten(round.(Int,indices),n,outputfile)

end

function write_ssten(indices::Array{Int,2},n, filepath::String)
    file = open(filepath, "w")
    order = size(indices,1)

    header = "$(order) $(n) $(size(indices,2))\n"
    write(file,header);

    for (edge) in eachcol(indices)
	edge_string=""
        for v_i in edge
	     edge_string *= string(v_i," ")
        end
 	edge_string *= "\n"
	write(file,edge_string)
    end

    close(file)
end

#TODO: add saving in binary flag, parse v_i
function load_ssten(filepath,delimiter=' ')

	#check path validity
	@assert filepath[end-5:end] == ".ssten"

    open(filepath) do file
		#preallocate from the header line
		order, n, m =
			[parse(Int,elem) for elem in split(chomp(readline(file)),delimiter)]
        #@assert order == 3

		indices = Array{Int,2}(undef,order,m)
		#values = Array{Float64,1}(undef,m)


		i = 1
		@inbounds for line in eachline(file)
			entries = split(chomp(line),delimiter)

			indices[:,i] = [parse(Int,elem) for elem in entries[1:end-1]]
			#values[i] = parse(Float64,entries[end])
			i += 1
		end

		#check for 0 indexing
		zero_indexed = false

		@inbounds for i in 1:m
		    if indices[1,i] == 0
    			zero_indexed = true
				break
			end
	    end


		if zero_indexed
			indices .+= 1
		end

        return indices, n
    end
end

function load_SymTensorUnweighted(filepath,motif::M=Clique(),args...) where {M <: Motif}

    indices, n = load_ssten(filepath,args...)
    #TODO: make this more robust
    SymTensorUnweighted{M}(n,size(indices,1),indices)
end

#routine used for reading in edge files stored as .csv
function parse_edges_csv(file)
     
     f = CSV.File(file,header=false, types=[Int,Int])
     B = zeros(length(f)-1,2) #ignore old header 
     B[:,1] = f.Column1[2:end]
     B[:,2] = f.Column2[2:end]
     B .+= 1

     A = sparse(B[:,1],B[:,2],1,maximum(B),maximum(B)) 
     A = max.(A,A')

     return A
end
