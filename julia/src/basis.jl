# Apply complete blocks of packed reflectors to an identity basis. Keeping this
# blocked avoids a serial per-reflector Q construction becoming an artificial limit.
function gather_basis_tail!(V,A,c,j,dim,offset,::Val{:column})
    @views V[c+1:end,c] .= A[j+offset+1:dim,j]
end
function gather_basis_tail!(V,A,c,j,dim,offset,::Val{:row})
    @views V[c+1:end,c] .= A[j,j+offset+1:dim]
end
function householder_basis(A,tau,dim,columns,offset,orientation,ex;block::Int=16)
    Q=Matrix{Float64}(I,dim,columns)
    length(tau)==0 && return Q
    ws=QRWorkspace(dim,columns;block)
    for p in reverse(1:block:length(tau))
        q=min(p+block-1,length(tau)); nb=q-p+1
        V=@view ws.V[p+offset:dim,1:nb]; T=@view ws.T[1:nb,1:nb]
        fill!(V,0.0); fill!(T,0.0)
        for j in p:q
            c=j-p+1; V[c,c]=1.0
            gather_basis_tail!(V,A,c,j,dim,offset,orientation)
            for i in 1:c-1
                ws.W[i,1]=-tau[j]*dot(@view(V[:,i]),@view(V[:,c]))
            end
            for i in 1:c-1
                t=0.0
                for l in i:c-1; t+=T[i,l]*ws.W[l,1]; end
                T[i,c]=t
            end
            T[c,c]=tau[j]
        end
        target=@view Q[p+offset:dim,:]
        W=@view ws.W[1:nb,:]; Z=@view ws.Z[1:nb,:]
        product!(W,transpose(V),target,ex)
        product!(Z,T,W,ex)
        product!(target,V,Z,ex,-1.0,1.0)
    end
    Q
end
