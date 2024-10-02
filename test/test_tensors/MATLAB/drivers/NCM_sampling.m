function [output] = NCM_sampling(A,delta,max_iter,trials,seed,method)

    if nargin < 6
       method = "NCM"
    end

    m = size(A,1);
    order = length(size(A));


    lambdas = zeros(trials,1);
    eigvecs = zeros(m,trials);
    startvecs = zeros(m,trials);
    runtimes = zeros(trials,1);
    iterations = zeros(trials,1);
    A_exp_idx = 0;

    if trials > 100000
        fprintf("WARNING : Increase max sample_seed range.");
    end

    sample_seeds = round((100000-1)*rand(trials,1) + 1);

    for idx = 1:trials
        rng(sample_seeds(idx));

        startvecs(:,idx) = randn(n,1);
        startvecs(:,idx)  = startvecs(:,idx)/norm(startvecs(:,idx) );

        if method == "NCM"
            [x,lambda,ctr,runtime,converge] = newton_correction_method(A,max_iter,delta,startvecs(:,idx));
        elseif method == "ONCM"
            [x,lambda,ctr,runtime,converge] = orthogonal_newton_correction_method(A,max_iter,delta,startvecs(:,idx));
        end

        if converge == 1 
    
            A_exp_idx = A_exp_idx + 1;
            lambdas(A_exp_idx) = lambda;
            eigvecs(:,A_exp_idx) = x;
            startvecs(:,A_exp_idx) = startvecs(:,idx);
            runtimes(A_exp_idx) = runtime;
            iterations(A_exp_idx) = ctr; 
        end
    end


    %
    %   Consolidate to unique eigepairs of A 
    %

    [vals,A_final_idx] = uniquetol(abs(lambdas(1:A_exp_idx)),delta);

    
    output = struct(...,
        "eigvals",lambdas(A_final_idx),...
        "eigvecs",eigvecs(:,A_final_idx),...
        "startvecs",startvecs(:,A_final_idx),...
        "exp_runtimes",runtimes(A_final_idx),...
        "iterations_needed",iterations(A_final_idx)...
    );
    
end