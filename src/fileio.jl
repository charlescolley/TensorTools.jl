

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

#routine used for reading in edge files stored as .csv
function parse_csv(file)
     
     f = CSV.File(file,header=false, types=[Int,Int])
     B = zeros(length(f)-1,2) #ignore old header 
     B[:,1] = f.Column1[2:end]
     B[:,2] = f.Column2[2:end]
     B .+= 1

     A = sparse(B[:,1],B[:,2],1,maximum(B),maximum(B)) 
     A = max.(A,A')

     return A
end
