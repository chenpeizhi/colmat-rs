# Deferred two-sided block update: A_current = A_base - V*Y' - X*U'.
# Same block-reduction design used by colmat's bidiag_blocked_panel; independently
# expressed here using conventional tau reflectors, with packed row reflectors.
struct BidiagWorkspace
    left_tau::Vector{Float64}
    right_tau::Vector{Float64}
    V::Matrix{Float64}
    Y::Matrix{Float64}
    X::Matrix{Float64}
    U::Matrix{Float64}
    row::Vector{Float64}
    floor::Int
end
function BidiagWorkspace(m::Int,n::Int;block::Int=16, floor::Int=128)
    m>=n || throw(ArgumentError("bidiagonalization expects m >= n; transpose wide input"))
    block>0 || throw(ArgumentError("positive block required"))
    floor>=0 || throw(ArgumentError("nonnegative floor required"))
    BidiagWorkspace(zeros(n),zeros(max(0,n-1)),zeros(m,block),zeros(n,block),
                   zeros(m,block),zeros(n,block),zeros(n),floor)
end
function bidiagonalize!(A::StridedMatrix{Float64},ws::BidiagWorkspace,ex::Execution=Serial())
    m,n=size(A); m>=n || throw(DimensionMismatch()); b=size(ws.V,2)
    size(ws.V,1)==m && length(ws.row)==n || throw(DimensionMismatch())
    p=1
    while p<=n
        width=n-p+1<=ws.floor ? 1 : b
        q=min(p+width-1,n); nb=q-p+1
        V=@view ws.V[:,1:nb]; Y=@view ws.Y[:,1:nb]
        X=@view ws.X[:,1:nb]; U=@view ws.U[:,1:nb]
        fill!(V,0.0); fill!(Y,0.0); fill!(X,0.0); fill!(U,0.0)
        for j in p:q
            c=j-p+1
            # Bring the column up to date, then form its left reflector.
            @inbounds for i in j:m
                z=A[i,j]
                for l in 1:c-1; z-=V[i,l]*Y[j,l]+X[i,l]*U[j,l]; end
                A[i,j]=z
            end
            τ=reflector!(@view A[j:m,j]); ws.left_tau[j]=τ
            V[j,c]=1.0; @views V[j+1:m,c] .= A[j+1:m,j]
            if j<n
                # Y_j = tau * (A_current' * v_j), future columns only.
                @views matvec!(Y[j+1:n,c],transpose(A[j:m,j+1:n]),V[j:m,c],ex,τ)
                for l in 1:c-1
                    a=τ*dot(@view(V[j:m,l]),@view(V[j:m,c]))
                    d=τ*dot(@view(X[j:m,l]),@view(V[j:m,c]))
                    @inbounds for i in j+1:n; Y[i,c]-=Y[i,l]*a+U[i,l]*d; end
                end
                # Gather the updated strided row before forming the right reflector.
                @inbounds for i in j+1:n
                    z=A[j,i]
                    for l in 1:c-1; z-=V[j,l]*Y[i,l]+X[j,l]*U[i,l]; end
                    ws.row[i]=z-Y[i,c]
                end
                τr=reflector!(@view ws.row[j+1:n]); ws.right_tau[j]=τr
                A[j,j+1]=ws.row[j+1]
                U[j+1,c]=1.0
                @views U[j+2:n,c] .= ws.row[j+2:n]
                @views A[j,j+2:n] .= ws.row[j+2:n]
                # X_j = tau_r * ((A_current - v_j*y_j') * u_j).
                @views matvec!(X[j+1:m,c],A[j+1:m,j+1:n],U[j+1:n,c],ex,τr)
                for l in 1:c
                    a=τr*dot(@view(Y[j+1:n,l]),@view(U[j+1:n,c]))
                    d=l<c ? τr*dot(@view(U[j+1:n,l]),@view(U[j+1:n,c])) : 0.0
                    @inbounds for i in j+1:m; X[i,c]-=V[i,l]*a; end
                    if l<c
                        @inbounds for i in j+1:m; X[i,c]-=X[i,l]*d; end
                    end
                end
            end
        end
        if q<n
            trail=@view A[q+1:m,q+1:n]
            @views product!(trail,V[q+1:m,:],transpose(Y[q+1:n,:]),ex,-1.0,1.0)
            @views product!(trail,X[q+1:m,:],transpose(U[q+1:n,:]),ex,-1.0,1.0)
        end
        p=q+1
    end
    A
end
function bidiag_bases(A,ws,ex)
    m,n=size(A)
    L=householder_basis(A,ws.left_tau,m,n,0,Val(:column),ex)
    R=householder_basis(A,ws.right_tau,n,n,1,Val(:row),ex)
    L,R
end
function svd_tall(A,ex,block,floor,pad)
    B,s=normalized(A); m,n=size(B)
    n==0 && return (;U=zeros(m,0),S=Float64[],V=zeros(0,0),times=(reduction=0.0,solver=0.0,backtransform=0.0))
    if pad && (m*sizeof(Float64))%1024==0
        storage=Matrix{Float64}(undef,m+8,n)
        copyto!(@view(storage[1:m,:]),B)
        B=@view storage[1:m,:]
    end
    ws=BidiagWorkspace(m,n;block,floor)
    tr=@elapsed bidiagonalize!(B,ws,ex)
    D=Bidiagonal(diag(B),diag(B,1),:U)
    ts=@elapsed F=svd(D)
    tb=@elapsed begin
        L,R=bidiag_bases(B,ws,ex)
        U=similar(L); V=similar(R)
        product!(U,L,F.U,ex); product!(V,R,F.V,ex)
    end
    (;U,S=F.S .* s,V,times=(reduction=tr,solver=ts,backtransform=tb))
end
"""Thin SVD: blocked two-sided reduction + standard bidiagonal solver + back transforms."""
function svd_lab(A::AbstractMatrix{Float64},ex::Execution=Serial();block::Int=16, floor::Int=128, pad::Bool=true)
    if size(A,1)>=size(A,2)
        svd_tall(A,ex,block,floor,pad)
    else
        F=svd_tall(Matrix(transpose(A)),ex,block,floor,pad)
        (;U=F.V,S=F.S,V=F.U,times=F.times)
    end
end
