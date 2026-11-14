# QR competition: colmat-style recursive panels and inverse compact-WY factors.
# H = inv(T): upper(V'V) off diagonal, 1/tau on the diagonal; Q=I-V/H*V'.
struct QRColmatWorkspace
    tau::Vector{Float64}
    V::Matrix{Float64}
    H::Matrix{Float64}
    W::Matrix{Float64}
    factors::Matrix{Float64}
    colgrain::Int
    leaf::Int
    seed::Matrix{Float64}
end
function qr_colmat_block(m,n)
    p=m*n
    min(max(1,min(m,n)),p>8192^2 ? 256 : p>2048^2 ? 128 : p>1024^2 ? 64 : p>512^2 ? 48 : p>128^2 ? 32 : p>32^2 ? 8 : p>16^2 ? 4 : 1)
end
function QRColmatWorkspace(m,n,b,colgrain,padbasis=false,leaf=3)
    leaf>0 || throw(ArgumentError("positive leaf required"))
    colgrain>0 || throw(ArgumentError("positive column grain required"))
    b>0 || throw(ArgumentError("positive block required"))
    QRColmatWorkspace(zeros(min(m,n)),zeros(m+(padbasis && m*sizeof(Float64)%1024==0 ? 8 : 0),b),zeros(b,b),zeros(b,max(n,min(m,n))),Matrix{Float64}(undef,b,min(m,n)),colgrain,leaf,Matrix{Float64}(undef,b,b))
end
function qr_colmat_vectors!(A,ws,p,q)
    m=size(A,1); b=q-p+1
    V=@view ws.V[p:m,1:b]; H=@view ws.H[1:b,1:b]
    fill!(V,0.0)
    for j in p:q
        c=j-p+1; V[c,c]=1.0
        @views V[c+1:end,c].=A[j+1:m,j]
    end
    V
end
function qr_colmat_block!(A,ws,p,q)
    V=qr_colmat_vectors!(A,ws,p,q); b=q-p+1; H=@view ws.H[1:b,1:b]
    BLAS.syrk!('U','T',1.0,V,0.0,H)
    for j in p:q; H[j-p+1,j-p+1]=inv(ws.tau[j]); end
    V,H
end
panel_execution(ex::Serial,n,grain)=ex
panel_execution(ex::Team,n,grain)=Team(min(ex.threads,max(1,n÷grain));minwork=ex.minwork)
function qr_colmat_apply!(C,V,H,ws,ex,::Val{TRANS}) where {TRANS}
    # A single section owns each complete target column through GEMM/TRSM/GEMM.
    ranges(size(C,2),panel_execution(ex,size(C,2),ws.colgrain),size(V,1)*size(V,2)*size(C,2)) do lo,hi
        target=@view C[:,lo:hi]; W=@view ws.W[1:size(V,2),lo:hi]
        mul!(W,transpose(V),target)
        BLAS.trsm!('L','U',TRANS,'N',1.0,H,W)
        mul!(target,V,W,-1.0,1.0)
    end
    C
end
function qr_colmat_leaf!(A,tau,p,stop)
    m=size(A,1)
    t=reflector!(@view A[p:m,p]); tau[p]=t
    if t!=0.0
        v=@view A[p+1:m,p]
        for j in p+1:stop
            s=t*(A[p,j]+dot(v,@view A[p+1:m,j]))
            A[p,j]-=s
            @inbounds @simd for i in p+1:m; A[i,j]-=A[i,p]*s; end
        end
    end
end
function qr_colmat_panel!(A,ws,p,q,ex)
    nb=q-p+1
    if nb<=ws.leaf
        for j in p:q; qr_colmat_leaf!(A,ws.tau,j,q); end
        V,H=qr_colmat_block!(A,ws,p,q)
        for j in 1:nb, i in 1:j; ws.factors[i,p+j-1]=H[i,j]; end
        return
    end
    left=nb÷2; mid=p+left-1; right=nb-left
    qr_colmat_panel!(A,ws,p,mid,ex)
    V=qr_colmat_vectors!(A,ws,p,mid); H=@view ws.factors[1:left,p:mid]
    qr_colmat_apply!(@view(A[p:end,mid+1:q]),V,H,ws,ex,Val('T'))
    qr_colmat_panel!(A,ws,mid+1,q,ex)
    # Move the second child's upper triangle down before overwriting its old rows.
    for j in 1:right, i in j:-1:1
        ws.factors[left+i,mid+j]=ws.factors[i,mid+j]
    end
    V=qr_colmat_vectors!(A,ws,p,q)
    @views mul!(ws.factors[1:left,mid+1:q],transpose(V[left+1:end,1:left]),V[left+1:end,left+1:nb])
    nothing
end
function qr_colmat_recursive!(A,ws,start,stop,b,ex,save=false)
    k=min(size(A,1),stop)
    if b==1
        for p in start:k
            qr_colmat_leaf!(A,ws.tau,p,stop)
            ws.factors[1,p]=inv(ws.tau[p])
        end
        return A
    end
    for p in start:b:k
        q=min(p+b-1,k)
        qr_colmat_panel!(A,ws,p,q,ex)
        if q<stop
            V=qr_colmat_vectors!(A,ws,p,q); H=@view ws.factors[1:q-p+1,p:q]
            qr_colmat_apply!(@view(A[p:end,q+1:stop]),V,H,ws,ex,Val('T'))
        end
    end
    A
end
function qr_colmat_basis(A,ws,ex,::Val{PREFIX}) where {PREFIX}
    m,n=size(A); k=min(m,n); b=size(ws.V,2)
    Q=Matrix{Float64}(I,m,k)
    for p in reverse(1:b:k)
        q=min(p+b-1,k); V=qr_colmat_vectors!(A,ws,p,q); H=@view ws.factors[1:q-p+1,p:q]
        firstcol=PREFIX ? p : 1
        qr_colmat_apply!(@view(Q[p:m,firstcol:k]),V,H,ws,ex,Val('N'))
    end
    Q
end
# Identity-seeded Q: new panel columns use V0' directly; later columns have
# zero head rows. Work intervals weight a later column twice a new one.
function qr_colmat_basis_structured(A,ws,ex;block=size(ws.V,2),reblock=false)
    m,n=size(A);k=min(m,n)
    Q=Matrix{Float64}(undef,m,k)
    for p in reverse(1:block:k)
        q=min(p+block-1,k);b=q-p+1
        V=qr_colmat_vectors!(A,ws,p,q);H=reblock ? last(qr_colmat_block!(A,ws,p,q)) : @view(ws.factors[1:b,p:q])
        seed=@view ws.seed[1:b,1:b]
        @views copyto!(seed,transpose(V[1:b,:]))
        BLAS.trsm!('L','U','N','N',1.0,H,seed)
        C=@view Q[p:m,p:k];nc=size(C,2);units=b+2*(nc-b)
        boundary(u)=u<=b ? u : b+(u-b)÷2
        ranges(units,panel_execution(ex,nc,ws.colgrain),size(V,1)*b*nc) do first,last
            lo=boundary(first-1)+1;hi=boundary(last)
            if hi>=max(lo,b+1)
                rlo=max(lo,b+1);target=@view C[:,rlo:hi];W=@view ws.W[1:b,rlo:hi]
                @views fill!(target[1:b,:],0.0)
                @views mul!(W,transpose(V[b+1:end,:]),target[b+1:end,:])
                BLAS.trsm!('L','U','N','N',1.0,H,W)
                mul!(target,V,W,-1.0,1.0)
            end
            if lo<=min(hi,b)
                ownhi=min(hi,b)
                @views mul!(C[:,lo:ownhi],V,seed[:,lo:ownhi],-1.0,0.0)
                for j in lo:ownhi;C[j,j]+=1.0;end
            end
        end
    end
    Q
end
"""Recursive inverse-WY QR experiment. prefix=true skips identically-zero Q columns."""
function qr_colmat_lab(A::AbstractMatrix{Float64},ex::Execution=Serial();block::Int=qr_colmat_block(size(A)...),prefix::Bool=false,colgrain::Int=1,padbasis::Bool=false,leaf::Int=3,prescale::Bool=true,structured::Bool=false,basisblock::Int=block)
    B,s=prescale ? normalized(A) : (Matrix(A),1.0); m,n=size(B); k=min(m,n)
    basisblock>0 || throw(ArgumentError("positive basis block required"))
    basisblock!=block && !structured && throw(ArgumentError("reblocking requires structured Q"))
    ws=QRColmatWorkspace(m,n,max(block,basisblock),colgrain,padbasis,leaf)
    tf=@elapsed qr_colmat_recursive!(B,ws,1,n,block,ex,true)
    tq=@elapsed Q=structured ? qr_colmat_basis_structured(B,ws,ex;block=basisblock,reblock=basisblock!=block) : qr_colmat_basis(B,ws,ex,Val(prefix))
    R=triu(B[1:k,:]); prescale && (R .*= s)
    (;Q,R,times=(factor=tf,basis=tq))
end
