# Direct translation of the arithmetic/order in colmat's tridiag_evd/repair.rs,
# b292f474, MIT (repository LICENSE); indices changed from zero to one based.
# One weight or complete column belongs to one worker. No cross-worker reductions.
function repair_weights!(z,d,mus,shifts,k::Int,ex::Execution=Serial())
    ranges(k,ex,k*k) do lo,hi
        for i in lo:hi
            di=d[i]; prod=mus[i]+(shifts[i]-di)
            for j in 1:i-1; prod*=(mus[j]+(shifts[j]-di))/(d[j]-di); end
            for j in i+1:k; prod*=(mus[j]+(shifts[j]-di))/(d[j]-di); end
            a=sqrt(abs(prod)); z[i]=z[i]<0.0 ? -a : a
        end
    end
    z
end
function repair_vectors!(U,order,rows,d,z,mus,shifts,k::Int,ex::Execution=Serial())
    n=size(U,1)
    ranges(n,ex,n*k) do lo,hi
        for j in lo:hi
            pj=order[j]
            if pj>k
                @views fill!(U[:,j],0.0)
                U[rows[pj],j]=1.0
            else
                for i in 1:k; U[rows[i],j]=z[i]/((d[i]-shifts[pj])-mus[pj]); end
                for i in k+1:n; U[rows[i],j]=0.0; end
                invnorm=inv(norm(@view U[:,j]))
                @inbounds @simd for i in 1:n; U[i,j]*=invnorm; end
            end
        end
    end
    U
end
"""Untimed rank-one secular fixture with optional deflated tail, no random state.
Reference eigenvalues are from eigen(Diagonal(d)+z*z'); this does not implement root finding.
"""
function repair_fixture(n::Int;k::Int=n)
    0<=k<=n || throw(ArgumentError("require 0 <= k <= n"))
    d=Float64.(1:n); z=[(isodd(i) ? -1.0 : 1.0)/sqrt(max(1,k)) for i in 1:n]
    mus=zeros(n); shifts=copy(d)
    if k>0
        roots=eigvals(Symmetric(Diagonal(d[1:k])+z[1:k]*z[1:k]'))
        mus[1:k].=roots.-d[1:k]
    end
    order=sortperm(mus.+shifts); rows=collect(1:n)
    (;d,z,mus,shifts,order,rows,k)
end
