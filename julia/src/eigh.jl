struct TridiagWorkspace
    tau::Vector{Float64}
    v::Vector{Float64}
    w::Vector{Float64}
end
TridiagWorkspace(n::Int)=TridiagWorkspace(zeros(max(0,n-1)),zeros(n),zeros(n))
"""Symmetric Householder reduction, full stored triangle, packed reflector tails below subdiagonal.
Unlike Rust's fused lower-triangle kernel, the reference uses separate matvec/rank-2 passes.
"""
function tridiagonalize!(A::Matrix{Float64}, ws::TridiagWorkspace, ex::Execution=Serial())
    n=size(A,1); size(A,2)==n || throw(DimensionMismatch())
    length(ws.v)==n || throw(DimensionMismatch())
    for j in 1:n-1
        τ=reflector!(@view A[j+1:n,j]); ws.tau[j]=τ
        v=@view ws.v[j+1:n]; w=@view ws.w[j+1:n]
        v[1]=1.0; @views v[2:end] .= A[j+2:n,j]
        C=@view A[j+1:n,j+1:n]
        if τ!=0.0
            matvec!(w,C,v,ex,τ)
            c=-0.5*τ*dot(v,w)
            @. w=w+c*v
            ranges(size(C,2),ex,length(C)) do lo,hi
                for col in lo:hi
                    vc=v[col]; wc=w[col]
                    @inbounds @simd for row in axes(C,1)
                        C[row,col]-=v[row]*wc+w[row]*vc
                    end
                end
            end
        end
    end
    A
end
function tridiag_basis(A,tau,ex)
    householder_basis(A,tau,size(A,1),size(A,1),1,Val(:column),ex)
end
"""Symmetric EVD: Julia reduction + standard SymTridiagonal eigensolver + back transform."""
function eigh_lab(A::AbstractMatrix{Float64},ex::Execution=Serial())
    issymmetric(A) || throw(ArgumentError("symmetric input required"))
    B,s=normalized(A); n=size(B,1)
    n==0 && return (;values=Float64[],vectors=zeros(0,0),times=(reduction=0.0,solver=0.0,backtransform=0.0))
    ws=TridiagWorkspace(n)
    tr=@elapsed tridiagonalize!(B,ws,ex)
    T=SymTridiagonal(diag(B),diag(B,-1))
    ts=@elapsed F=eigen(T)
    tb=@elapsed begin
        Q=tridiag_basis(B,ws.tau,ex)
        vectors=similar(Q); product!(vectors,Q,F.vectors,ex)
    end
    (;values=F.values .* s,vectors,times=(reduction=tr,solver=ts,backtransform=tb))
end
