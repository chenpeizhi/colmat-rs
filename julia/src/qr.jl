struct QRWorkspace
    tau::Vector{Float64}
    V::Matrix{Float64}
    T::Matrix{Float64}
    W::Matrix{Float64}
    Z::Matrix{Float64}
end
function QRWorkspace(m::Int,n::Int; block::Int=16)
    block > 0 || throw(ArgumentError("positive block required"))
    QRWorkspace(zeros(min(m,n)), zeros(m,block), zeros(block,block),
                zeros(block,n), zeros(block,n))
end
"""Packed blocked Householder QR. Workspace and A are caller owned; returns A."""
function qr_factor!(A::Matrix{Float64}, ws::QRWorkspace, ex::Execution=Serial())
    m,n = size(A); k=min(m,n); b=size(ws.V,2)
    length(ws.tau)==k && size(ws.V,1)==m && size(ws.W,2)==n || throw(DimensionMismatch())
    for p in 1:b:k
        q=min(p+b-1,k); nb=q-p+1
        V=@view ws.V[p:m,1:nb]; T=@view ws.T[1:nb,1:nb]
        fill!(V,0.0); fill!(T,0.0)
        for j in p:q
            τ=reflector!(@view A[j:m,j]); ws.tau[j]=τ
            c=j-p+1
            V[c,c]=1.0
            @views V[c+1:end,c] .= A[j+1:m,j]
            left!(@view(A[j:m,j+1:q]), @view(V[c:end,c]), τ, Serial())
            # Compact WY: H_p ... H_j = I - V T V'.
            for i in 1:c-1
                ws.W[i,1] = -τ*dot(@view(V[:,i]), @view(V[:,c]))
            end
            for i in 1:c-1
                s=0.0
                for l in i:c-1; s+=T[i,l]*ws.W[l,1]; end
                T[i,c]=s
            end
            T[c,c]=τ
        end
        if q<n
            W=@view ws.W[1:nb,1:n-q]; Z=@view ws.Z[1:nb,1:n-q]
            trail=@view A[p:m,q+1:n]
            product!(W, transpose(V), trail, ex)
            product!(Z, transpose(T), W, ex)
            product!(trail, V, Z, ex, -1.0, 1.0)
        end
    end
    A
end
function qr_basis(A,tau,ex;block=16)
    householder_basis(A,tau,size(A,1),min(size(A)...),0,Val(:column),ex;block)
end
"""Thin QR with stage times; no pivoting. Supports rectangular and rank-deficient input."""
function qr_lab(A::AbstractMatrix{Float64}, ex::Execution=Serial(); block::Int=16)
    B,s=normalized(A); m,n=size(B); k=min(m,n)
    ws=QRWorkspace(m,n;block)
    tf=@elapsed qr_factor!(B,ws,ex)
    tq=@elapsed Q=qr_basis(B,ws.tau,ex;block)
    R=triu(B[1:k,:]); R .*= s
    (;Q,R,times=(factor=tf,basis=tq))
end
