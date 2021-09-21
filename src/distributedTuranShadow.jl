
function distributed_clique_sample(matrix_file,output_path, orders, sample_counts)


    # load in matrix 

    A = nothing
    
    file_ext = split(matrix_file,".")[end]
    if (file_ext == "smat")
        A = MatrixNetworks.readSMAT(matrix_file)
    elseif (file_ext == "csv")
    	A = parse_csv(matrix_file)
    end
    
    #A = max.(A,A')

    # find unique local cliques 

    # partition distributed lexicographic list 

    # collectively build the final output file


end


function sample_cliques(matrix_file,output_path,orders,sample_counts)

    #check file format

    A = nothing
    
    file_ext = split(matrix_file,".")[end]
    if (file_ext == "smat")
        A = MatrixNetworks.readSMAT(matrix_file)
    elseif (file_ext == "csv")
    	A = parse_csv(matrix_file)
    end
    
    A = max.(A,A')

    seed = 4
    Random.seed!(seed)
    root_name = split(split(matrix_file,"/")[end],".")[1]

    for k in orders

    	#create a subfolder for given order
        order_folder = "order:$(k)/"
        local_folder = output_path*order_folder
	    #Note: assuming that output_path ends in '/'
		  
	if !isdir(output_path*order_folder)
	   mkdir(output_path*order_folder)
	end

        log_file = root_name*"-order:$(k)-seed:$(seed)-log.jld"

        runtimes = []
        unique_cliques = Set()

        prev_sample_count = 0
	    prev_motif_count = 0

        for samples in sort(sample_counts)
     	    #  -- find unique cliques  --  #
    	    ((_,cliques),t) = @timed TuranShadow(A,k,(samples-prev_sample_count))
	        prev_sample_count = samples

	    
            cliques = [sort(clique) for clique in cliques]
            sort!(cliques) #lexicographic ordered helps eliminate repeats
	
	    for clique in cliques
 	        push!(unique_cliques,clique)
	    end

	    #  terminate sample search if no new motifs found this iteration

	    motif_count = length(unique_cliques)
	    if (prev_motif_count == motif_count)
	       println("no new cliques found from last iter. motif_count=$(motif_count)")
	       break
	    else
	        prev_motif_count = motif_count
	    end
	    
	

	    tensor_name = split(root_name,".")[1]*"-order:$(k)-sample:$(samples)-seed:$(seed).ssten"
	        #NOTE: assuming matrix_file is of the form rootName.smat

   	    write_tensor(collect(unique_cliques),output_path*order_folder*tensor_name)
	
	    #  --  update log  --  #
	    push!(runtimes,(samples,t))
	    save(output_path*order_folder*log_file,"runtimes",runtimes)
	end
    end



end

#TODO: 
function reduce_to_unique_cliques!(cliques::Array{Array{T,1},1}) where {T <: Int}

    if length(cliques) < 2 
        return
    end
    order = length(cliques[1])

    for i = 1:length(cliques)
        sort!(cliques[i])
    end

    sort!(cliques)
    
    ix_drop = Array{Int,1}(undef,0)

    current_clique_idx = 1
    for i =2:length(cliques)
        
        if cliques[i] == cliques[current_clique_idx] #found a repeated clique, mark for removal
            push!(ix_drop,i)
        else #found a new clique, update ptr
            current_clique_idx = i 
        end

    end

    deleteat!(cliques,ix_drop)
    #return cliques, ix_drop
    #remove all repeated cliques
    #cliques = cliques[setdiff(begin:end, ix_drop)]

    
end

