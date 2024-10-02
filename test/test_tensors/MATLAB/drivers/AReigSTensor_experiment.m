function [output] = AReigSTensor_experiment(A,options)

    p=2; 
    f_c = [];
    n = size(A,1);
    order = length(size(A));


    if isfield(options,'Nmax')
        Nmax = options.Nmax;
    else
        Nmax = 10;
    end
    if isfield(options,'dt0')
        dt0 = options.dt0;
    else
        dt0 = .05;
    end
    if isfield(options,'tol')
        tol = options.tol;
    else
        tol = 1e-6;
    end

    mpol('x',n,1);

    f_A = symtensor2poly(A,x,order,n);

    tic;
    [lmd_A, eigvec_A,  info_A] = AReigSTensors(f_A, f_c, x, order, p, options)
    A_rt = toc;

    output = struct(...,
        "eigvals",lmd_A,...
        "eigvecs", eigvec_A',...
        "sta_hist", info_A,...
        "runtime", A_rt...
    );

end