function [A] = generate_random_k_regular_tensor(n, order, seed)

    rng(seed);

    % A = symtensor(@(x,y) double(rand(x,y)>.01), order, m);  % unweighted
    A = symtensor(@(x,y) round(randn(x,y),1), order, n)
    I = indices(A);

    unique_edges = size(I,1);

    for idx = 1:unique_edges
        % keep edges which have no repeating indices

        unique_indices = false;
        edge = I(idx,:);
        if length(unique(edge)) < order 
            A(edge) = 0; 
        end 
    end 

    A = full(A)
end