using Test,LinearAlgebra,Random
include("../src/ColmatLab.jl")
using .ColmatLab
BLAS.set_num_threads(1)
rng=MersenneTwister(84)
@testset "Recursive inverse-WY QR and zero-prefix Q" begin
 for (m,n) in [(0,0),(3,0),(0,3),(1,1),(17,9),(9,17),(97,61),(257,129),(2048,256)]
  for kind in (m==2048 ? [1] : [1,2,3])
   A=randn(rng,m,n)
   kind==2 && fill!(A,0.0)
   kind==3 && n>1 && (A[:,end].=A[:,1])
   for ex in [Serial(),Team(8)], prefix in [false,true], colgrain in [1,8]
    F=qr_colmat_lab(A,ex;prefix,colgrain)
    @test norm(A-F.Q*F.R)<=1e-12*max(norm(A),floatmin(Float64))
    @test norm(F.Q'F.Q-I)<=1e-11
   end
  end
 end
 for scale in [1e-200,1e200], prefix in [false,true]
  A=randn(rng,67,33);F=qr_colmat_lab(A*scale,Team(8);prefix)
  @test norm(A-F.Q*(F.R/scale))<=1e-12*norm(A)
 end
end

@testset "Padded basis and poisoned unused triangle" begin
 A=randn(rng,128,65)
 for padbasis in [false,true]
  F=qr_colmat_lab(A,Team(8);prefix=true,colgrain=8,padbasis)
  @test norm(A-F.Q*F.R)<1e-12*norm(A)
 end
 B=copy(A);ws=ColmatLab.QRColmatWorkspace(128,65,32,8,true)
 fill!(ws.factors,NaN);fill!(ws.H,NaN);fill!(ws.W,NaN)
 ColmatLab.qr_colmat_recursive!(B,ws,1,65,32,Team(8),true)
 Q=ColmatLab.qr_colmat_basis(B,ws,Team(8),Val(true))
 @test norm(A-Q*triu(B[1:65,:]))<1e-12*norm(A)
 @test norm(Q'Q-I)<1e-11
end

@testset "Larger serial panel leaves" begin
 for leaf in [6,12,24,48], (m,n) in [(67,33),(2048,256)]
  A=randn(rng,m,n);F=qr_colmat_lab(A,Team(8);prefix=true,colgrain=4,leaf)
  @test norm(A-F.Q*F.R)<1e-12*norm(A)
  @test norm(F.Q'F.Q-I)<1e-11
 end
end

@testset "Owning QR without whole-matrix prescaling" begin
 for scale in [1.0,1e-200,1e200]
  A=randn(rng,257,129);F=qr_colmat_lab(A*scale,Team(8);prefix=true,colgrain=4,leaf=32,block=32,prescale=false)
  @test norm(A-F.Q*(F.R/scale))<1e-12*norm(A)
  @test norm(F.Q'F.Q-I)<1e-11
 end
end

@testset "Identity structure and weighted column ownership" begin
 for (m,n) in [(0,3),(3,0),(1,1),(17,9),(9,17),(97,61),(257,129),(2048,256)], zero in [false,true]
  A=randn(rng,m,n);zero && fill!(A,0.0)
  for ex in [Serial(),Team(8)]
   F=qr_colmat_lab(A,ex;block=32,leaf=32,colgrain=4,prescale=false,structured=true)
   @test norm(A-F.Q*F.R)<=1e-12*max(norm(A),floatmin(Float64))
   @test norm(F.Q'F.Q-I)<1e-11
  end
 end
end

@testset "Independent factor and Q panel widths" begin
 for (m,n) in [(0,3),(3,0),(1,1),(97,61),(257,129),(2048,256)], ex in [Serial(),Team(8)], basisblock in [32,64,96]
  A=randn(rng,m,n)
  F=qr_colmat_lab(A,ex;block=16,leaf=16,colgrain=4,prescale=false,structured=true,basisblock)
  @test norm(A-F.Q*F.R)<=1e-12*max(norm(A),floatmin(Float64))
  @test norm(F.Q'F.Q-I)<1e-11
 end
end
