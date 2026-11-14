using Test, LinearAlgebra, Random
include("../src/ColmatLab.jl")
using .ColmatLab
BLAS.set_num_threads(1)
rng=MersenneTwister(84)
executors=[Serial(),Team(Threads.nthreads(:default))]
relative(A,B)=norm(A-B)/max(norm(A),floatmin(Float64))
@testset "QR: shapes, two panels and tail, rank deficiency, scale" begin
    for (m,n) in [(0,0),(0,3),(3,0),(1,1),(7,4),(4,7),(35,19),(67,33)], kind in 1:3
        A=randn(rng,m,n)
        kind==2 && fill!(A,0.0)
        kind==3 && n>1 && (A[:,end].=A[:,1])
        for ex in executors, block in [1,8,16]
            F=qr_lab(A,ex;block)
            @test relative(A,F.Q*F.R)<2e-13
            @test norm(F.Q'F.Q-I)<2e-12
        end
    end
    for s in [1e-200,1e200], ex in executors
        A=randn(rng,17,9); F=qr_lab(A*s,ex)
        @test relative(A,F.Q*(F.R/s))<2e-13
    end
end
@testset "Eigh: residual, orthogonality, ordering, scales" begin
    for n in [0,1,2,17,35,67], kind in 1:4
        X=randn(rng,n,n); A=(X+X')/2
        kind==2 && fill!(A,0.0)
        kind==3 && (A=Matrix{Float64}(I,n,n)+1e-8*A)
        kind==4 && (A=Matrix(Diagonal([10.0^(-12*(i-1)/max(1,n-1)) for i in 1:n]))+1e-14*A)
        for ex in executors
            F=eigh_lab(A,ex)
            @test relative(A*F.vectors,F.vectors*Diagonal(F.values))<3e-12
            @test norm(F.vectors'F.vectors-I)<3e-12
            @test issorted(F.values)
            @test norm(F.values-eigvals(Symmetric(A)))<1e-12*max(norm(A),floatmin(Float64))
        end
    end
    for s in [1e-200,1e200], ex in executors
        X=randn(rng,19,19); A=(X+X')/2; F=eigh_lab(A*s,ex)
        @test relative(A*F.vectors,F.vectors*Diagonal(F.values/s))<2e-12
    end
end
@testset "SVD: blocked two-sided reduction and back transforms" begin
    for (m,n) in [(0,0),(0,3),(3,0),(1,1),(7,4),(4,7),(35,19),(67,33)], kind in 1:3
        A=randn(rng,m,n)
        kind==2 && fill!(A,0.0)
        kind==3 && n>1 && (A[:,end].=A[:,1])
        for ex in executors, block in [1,8,16]
            F=svd_lab(A,ex;block)
            @test relative(A,F.U*Diagonal(F.S)*F.V')<3e-12
            @test norm(F.U'F.U-I)<3e-12
            @test norm(F.V'F.V-I)<3e-12
            @test issorted(F.S;rev=true) && all(>=(0),F.S)
        end
    end
    for s in [1e-200,1e200], ex in executors
        A=randn(rng,19,13); F=svd_lab(A*s,ex)
        @test relative(A,F.U*Diagonal(F.S/s)*F.V')<3e-12
    end
end
@testset "SVD: Rust block floor, two panels plus tail, padded leading dimension" begin
    for n in [128,129,169], ex in executors
        A=randn(rng,n+3,n); F=svd_lab(A,ex)
        @test relative(A,F.U*Diagonal(F.S)*F.V')<3e-12
        @test norm(F.U'F.U-I)<3e-12
        @test norm(F.V'F.V-I)<3e-12
    end
    A=randn(rng,128,17)
    @test norm(svd_lab(A;pad=true).S-svd_lab(A;pad=false).S)<1e-12
end
@testset "Repair: fixed scalar order, secular residual, deflation" begin
    for (n,k) in [(0,0),(1,0),(17,13),(256,239),(512,465)]
        f=repair_fixture(n;k); results=[]
        for ex in executors
            z=copy(f.z); U=fill(NaN,n,n)
            repair_weights!(z,f.d,f.mus,f.shifts,k,ex)
            repair_vectors!(U,f.order,f.rows,f.d,z,f.mus,f.shifts,k,ex)
            push!(results,(z,U))
            @test all(isfinite,U)
            @test norm(U'U-I)<2e-10
            M=Matrix(Diagonal(f.d)); k>0 && (M[1:k,1:k].+=f.z[1:k]*f.z[1:k]')
            vals=(f.mus.+f.shifts)[f.order]
            @test relative(M*U,U*Diagonal(vals))<2e-12
        end
        @test reinterpret(UInt64,results[1][1])==reinterpret(UInt64,results[2][1])
        @test reinterpret(UInt64,vec(results[1][2]))==reinterpret(UInt64,vec(results[2][2]))
    end
end
@testset "Contracts and inference" begin
    @test_throws ArgumentError Team(0)
    @test_throws ArgumentError eigh_lab([1.0 2.0;3.0 4.0])
    @test_throws ArgumentError qr_lab(fill(Inf,2,2))
    @test_throws ArgumentError QRWorkspace(3,2;block=0)
    A=randn(rng,17,9); ws=QRWorkspace(17,9)
    @test (@inferred qr_factor!(copy(A),ws,Serial())) isa Matrix{Float64}
    ws2=BidiagWorkspace(17,9)
    @test (@inferred bidiagonalize!(copy(A),ws2,Serial())) isa Matrix{Float64}
    B=randn(rng,9,9); B=B+B'; ws3=TridiagWorkspace(9)
    @test (@inferred tridiagonalize!(B,ws3,Serial())) isa Matrix{Float64}
end
@testset "Representative timed inputs (same seed, shapes and draw order)" begin
    timed_rng=MersenneTwister(84)
    for n in [256,512]
        X=randn(timed_rng,n,n); H=(X+X')/2
        A=randn(timed_rng,4n,n÷2)
        for ex in executors
            E=eigh_lab(H,ex)
            @test relative(H*E.vectors,E.vectors*Diagonal(E.values))<3e-12
            @test norm(E.vectors'E.vectors-I)<3e-12
            @test issorted(E.values)
            Q=qr_lab(A,ex)
            @test relative(A,Q.Q*Q.R)<3e-12
            @test norm(Q.Q'Q.Q-I)<3e-12
            S=svd_lab(X,ex)
            @test relative(X,S.U*Diagonal(S.S)*S.V')<3e-12
            @test norm(S.U'S.U-I)<3e-12
            @test norm(S.V'S.V-I)<3e-12
            @test issorted(S.S;rev=true)
        end
    end
end
