
REMOTE_COMPUTED_SOLUTION = "test_tensors/MATLAB/"
include("test_tensors/MATLAB/matlab_helpers.jl")

try 
    using MATLAB
    #mat"addpath('../../Software/empericalZ/julia/rsynced_packages/TensorTools.jl/test/test_tensors/MATLAB/drivers/zeig_experiment_driver.m')]"
catch e
    println("unable to load MATLAB.jl, relying on previously computed results included in package.")
end


@testset "Eigenvector Algorithms" begin

    A_ten,_ = T.tensor_from_graph(A,5,1000,Clique())
    A_ten = TensorTools.remap_to_touched_vertices!(A_ten).tensor

    @testset "Newton Correction Methods" begin 

        # TODO: make this iterate over multiple files
        filename = "NCM_sampling_n:4_order:3_seed:8435.mat"
        f = matopen(REMOTE_COMPUTED_SOLUTION*filename)
        A_mat_info = read(f,"A")
        A = convert_mat_to_SymTensor(A_mat_info["data"])
        
        #arest_output = read(f,"arest_output")

        trials = 0
        y = zeros(A.n)

        @testset "Newton Correction Method" begin 

            ncm_output = read(f,"ncm_output")
            max_iter = Int(read(f,"max_iter"))
            δ = read(f,"delta")
            ncm_eigs = T.unique_eigenvalues(abs.(ncm_output["eigvals"][:]),δ)

            for solve_method in [MINRES(), GMRES(), Backslash()]
                @inferred T.NCM!(A, max_iter, δ, randn(A.n), y, solve_method;verbose=false)

                for t = 1:trials 
                    λ, R, iter  = T.NCM!(A, max_iter, δ, randn(A.n), y, solve_method;verbose=false)
                    test_val = round(abs(λ),digits=-Int(log10(δ)))
                    λ_diff, eig_res = eigenspace_residual(A,λ,y)

                    #println(λ," ", R," ", iter," ",λ_diff," ",eig_res)
                    @test haskey(ncm_eigs,test_val) || (λ_diff < δ*A.n && eig_res < δ*A.n)
                                    # norm checks are scaled by n to account for round off error
                                    # see 2.7.6 of 4th edition of Matrix Computations for more.
                end    
            end
            
            # getting errors with MINRES when tested on a ge
            @test_broken T.NCM!(A_ten, max_iter, δ, randn(A_ten.n), y, MINRES();verbose=false)
        end 

        @testset "Orthorgonal Newton Correction Method" begin 
            output = read(f,"oncm_output")
            max_iter = Int(read(f,"max_iter"))
            δ = read(f,"delta")
            eigs = T.unique_eigenvalues(abs.(output["eigvals"][:]),δ)

            @inferred T.ONCM!(A, max_iter, δ, randn(A.n), y, MINRES(); verbose=false)

            for t = 1:trials 
                
                λ, R, iter  = T.ONCM!(A, max_iter, δ, randn(A.n), y, MINRES(); verbose=false)
                test_val = round(abs(λ),digits=-Int(log10(δ)))
                eig_res = eigenspace_residual(A,y)

                #println(λ," ", R," ", iter," ")
                #println(λ_diff," ",eig_res," haskey:",haskey(eigs,test_val))
                @test haskey(eigs,test_val) || eig_res < δ*A.n
            end    

        end 

        close(f)



        @testset "Sherman Morrison Inverse" begin 
            
            n = 1000
            B = sprand(n,n,.2)
            B = max.(B,B')

            u = rand(n)
            v = rand(n)
            b = rand(n)

            explicit_mat = B + u*v' 
            sol1 = explicit_mat \ b 
        
            residual = x-> norm(B*x + u*(v'*x) - b)

            sol2 = zeros(n)
            
            T.sherman_morrison_inverse!(B,u,v,b,sol2,Backslash())
                                                     # Krylov methods tend to struggle with uniform 
                                                     # random graphs so we test this with Backslash.
            @test residual(sol2) < TOL * n^2
            @test norm(sol2 - sol1) < TOL * n^2

        end     

    end 

    @testset "Dynamical Systems Methods" begin
        
        x₀ = rand(A_ten.n)

        @inferred T.dynamical_systems(A_ten, :LM, forwardEuler(.01), 10, 1e-6, true, copy(x₀))
        @inferred T.dynamical_systems_profiled(A_ten, :LM, forwardEuler(.01), 10, 1e-6, true, copy(x₀))
        
        for eigenmap in [:LM, :SM, :SA ,:LA]

            @test_nothrow T.dynamical_systems(A_ten, eigenmap, forwardEuler(.01), 10, 1e-6, true, copy(x₀))
            @test_nothrow T.dynamical_systems_profiled(A_ten, eigenmap, forwardEuler(.01), 10, 1e-6, true, copy(x₀))

            # symeigs fails when the vector is unnormalized 
            @test_broken T.dynamical_systems(A_ten, eigenmap, forwardEuler(.01), 10, 1e-6, false, copy(x₀))
        end

        # certain choices of `which` fail
        for eigenmap in [:SR, :LR]
            @test_broken T.dynamical_systems(A_ten, eigenmap, forwardEuler(.01), 10, 1e-6, true, copy(x₀))
        end
    
    end 

    @testset "Top Level Drivers" begin

        #sample_eigenvectors(A_ten,10)
        exp_args = [
            NCM(10,.1,Backslash()),
            ONCM(10,.1,GMRES()),
            DynamicalSystems(10,.1,forwardEuler(.01),:LM, true)
        ]
        
        for exp_arg in exp_args
            println("testing $(exp_arg)")
            @test_nothrow sample_eigenvectors(A_ten,2,nothing,exp_arg)
            @inferred sample_eigenvectors(A_ten,2,nothing,exp_arg)

            @test_nothrow sample_eigenvectors_profiled(A_ten,2,nothing,exp_arg)
            @inferred sample_eigenvectors_profiled(A_ten,2,nothing,exp_arg)
            
            seed = UInt(93329832)
            output = sample_eigenvectors(A_ten,1,seed,exp_arg)
            output2 = sample_eigenvectors(A_ten,1,seed,exp_arg)
            @test output.vecs == output2.vecs && output.vals == output2.vals && output.unique_eigs == output2.unique_eigs 

        end 
    end
end