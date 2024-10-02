
TENSOR_TOOLS_PKG_LOC = "./"
if TENSOR_TOOLS_PKG_LOC == "./"
    disp("`TENSOR_TOOLS_PKG_LOC` is set to default, please update to current package location.")
end

MATLAB_PKG_LOC = strcat(TENSOR_TOOLS_PKG_LOC,"test/test_tensors/MATLAB/external/")
NCM_METHOD_LOC = strcat(MATLAB_PKG_LOC,"NCM/")
TENSOR_TOOLBOX_LOC = strcat(MATLAB_PKG_LOC,"tensor_toolbox-v3.1/")
%AREST_LOC = strcat(MATLAB_PKG_LOC,"AReigSTensors/")

MY_SCRIPTS_LOC = strcat(TENSOR_TOOLS_PKG_LOC,"test/test_tensors/MATLAB/MATLAB_drivers/")

addpath(genpath(MATLAB_PKG_LOC))
addpath(NCM_METHOD_LOC,TENSOR_TOOLBOX_LOC,AREST_LOC,MY_SCRIPTS_LOC)

testing = true 

if testing 
    %[n,order,seed]
    examples = {
        [10,3,8435], ...
    };
else
    examples = {
        [2,3,3333], ...
        [2,3,4444], ...
        [2,3,956],  ...
        [2,3,9250], ...
        [2,4,956],  ...
        [2,4,3105], ...
        [2,4,9250], ...
        [2,4,3333], ...
    };
end

output_folder = strcat(MY_SCRIPTS_LOC,"output_files/")
if not(isfolder(output_folder))
    mkdir(output_folder)
end

%  - (O)NCM parameters

delta = 1e-10
max_iter = 1000
trials = 1000 

%  - AReigSTensor Parameters

arest_options= struct("Nmax",10,"dt0",.05,"tol",1e-4);

for i =1:length(examples)

    n = examples{i}(1);
    order = examples{i}(2);
    seed = examples{i}(3);

    [A] = generate_random_k_regular_tensor(n, order, seed)
    filename = strcat(output_folder,method,sprintf("A_n:%d_order:%d_seed:%d.mat",n,order,seed));
    save(filename,"A");

    [ncm_output] = NCM_sampling(A,delta,max_iter,trials,seed,"NCM")
    [oncm_output] = NCM_sampling(A,delta,max_iter,trials,seed,"ONCM")

    filename = strcat(output_folder,method,sprintf("ncm_sampling_n:%d_order:%d_seed:%d.mat",n,order,seed));
    save(filename,"ncm_output","oncm_output""delta","max_iter","trials");

    %[arest_output] = AReigSTensor_experiment(A,arest_options)
    %filename = strcat(output_folder,method,sprintf("arest_n:%d_order:%d_seed:%d.mat",n,order,seed));
    %save(filename,"arest_output","arest_options");

end