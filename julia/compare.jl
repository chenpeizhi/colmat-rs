# Same-process, same-input comparison. All three routes return complete thin factors.
using LinearAlgebra, Random, SHA
include("src/ColmatLab.jl")
using .ColmatLab
const RUST_LIB=get(ENV,"COLMAT_COMPARE_LIB",joinpath(@__DIR__,"..","target","release","libcolmat_julia_compare.so"))

team_new(width)=ccall((:compare_team_new,RUST_LIB),Ptr{Cvoid},(Csize_t,),width)
team_free(p)=ccall((:compare_team_free,RUST_LIB),Cvoid,(Ptr{Cvoid},),p)
result_free(p)=ccall((:compare_result_free,RUST_LIB),Cvoid,(Ptr{Cvoid},),p)
function rust_call(team,A,kind)
    p=ccall((:compare_run,RUST_LIB),Ptr{Cvoid},(Ptr{Cvoid},Ptr{Float64},Csize_t,Csize_t,UInt32),team,A,size(A,1),size(A,2),kind)
    p==C_NULL && error("Rust decomposition failed")
    p
end
function copy_factor(p,which,dest)
    ccall((:compare_copy,RUST_LIB),Cvoid,(Ptr{Cvoid},UInt32,Ptr{Float64}),p,which,dest)
    dest
end
function rust_outputs(p,A,::Val{:eigh})
    n=size(A,1)
    (;values=copy_factor(p,0,fill(NaN,n)),vectors=copy_factor(p,1,fill(NaN,n,n)))
end
function rust_outputs(p,A,::Val{:qr})
    m,n=size(A); k=min(m,n)
    (;Q=copy_factor(p,1,fill(NaN,m,k)),R=copy_factor(p,2,fill(NaN,k,n)))
end
function rust_outputs(p,A,::Val{:svd})
    m,n=size(A); k=min(m,n)
    (;U=copy_factor(p,1,fill(NaN,m,k)),S=copy_factor(p,0,fill(NaN,k)),V=copy_factor(p,2,fill(NaN,n,k)))
end
lab(A,ex,::Val{:eigh})=eigh_lab(A,ex)
lab(A,ex,::Val{:qr})=qr_lab(A,ex)
lab(A,ex,::Val{:svd})=svd_lab(A,ex)
standard(A,::Val{:eigh})=eigen(Symmetric(A))
function standard(A,::Val{:qr})
    F=qr(A)
    (;Q=Matrix(F.Q),R=Matrix(F.R))
end
function standard(A,::Val{:svd})
    F=svd(A)
    (;U=F.U,S=F.S,V=Matrix(F.V))
end
family_name(::Val{F}) where {F}=String(F)
kind(::Val{:eigh})=UInt32(0)
kind(::Val{:qr})=UInt32(1)
kind(::Val{:svd})=UInt32(2)
function errors(A,F,::Val{:eigh})
    r=norm(A*F.vectors-F.vectors*Diagonal(F.values))/max(norm(A),floatmin(Float64))
    o=norm(F.vectors'F.vectors-I)
    issorted(F.values) || error("unordered eigenvalues")
    (r,o)
end
function errors(A,F,::Val{:qr})
    (norm(A-F.Q*F.R)/max(norm(A),floatmin(Float64)),norm(F.Q'F.Q-I))
end
function errors(A,F,::Val{:svd})
    issorted(F.S;rev=true) && all(>=(0),F.S) || error("invalid singular values")
    (norm(A-F.U*Diagonal(F.S)*F.V')/max(norm(A),floatmin(Float64)),max(norm(F.U'F.U-I),norm(F.V'F.V-I)))
end
function check(A,F,family)
    r,o=errors(A,F,family)
    tol=100*max(size(A)...)*eps(Float64)
    isfinite(r) && isfinite(o) && r<=tol && o<=tol || error("$family residual=$r orthogonality=$o tolerance=$tol")
    r,o
end
# Keep validation at one BLAS thread and outside the measured call.
function checked_rust(team,A,family)
    p=rust_call(team,A,kind(family))
    try
        F=rust_outputs(p,A,family)
        check(A,F,family)
    finally
        result_free(p)
    end
end
function verify(team,ex,A,family,backends)
    BLAS.set_num_threads(1)
    original=copy(A)
    "colmat" in backends && checked_rust(team,A,family)
    "julia_lab" in backends && check(A,lab(A,ex,family),family)
    "julia_stdlib" in backends && check(A,standard(A,family),family)
    A==original || error("input modified")
end
function sample(team,ex,A,family,backend,width)
    BLAS.set_num_threads(backend=="julia_stdlib" ? width : 1)
    GC.gc(); sleep(0.02) # settle bounded spins of the other runtime, outside timing
    if backend=="colmat"
        result=@timed rust_call(team,A,kind(family))
        p=result.value
        try
            F=rust_outputs(p,A,family)
            BLAS.set_num_threads(1)
            residual,orth=check(A,F,family)
        finally
            result_free(p)
        end
        return result.time,missing,result.gctime,residual,orth
    else
        result=backend=="julia_lab" ? (@timed lab(A,ex,family)) : (@timed standard(A,family))
        BLAS.set_num_threads(1)
        residual,orth=check(A,result.value,family)
        return result.time,result.bytes,result.gctime,residual,orth
    end
end
function cell(io,team,ex,A,family,width,reps,backends)
    # Verify and warm all paths before measuring this exact input.
    verify(team,ex,A,family,backends)
    for _ in 1:2
        for backend in backends
            sample(team,ex,A,family,backend,width)
        end
    end
    fingerprint=bytes2hex(sha256(reinterpret(UInt8,vec(A))))
    original=copy(A)
    for rep in 1:reps
        order=circshift(collect(backends),mod(rep-1,length(backends)))
        iseven(rep) && reverse!(order)
        for (position,backend) in enumerate(order)
            time,bytes,gc,residual,orth=sample(team,ex,A,family,backend,width)
            println(io,join((family_name(family),size(A,1),size(A,2),width,backend,rep,position,time,bytes,gc,residual,orth,fingerprint),','))
            flush(io)
        end
    end
    A==original || error("timed call modified input")
end
function main(outdir,width,reps,backend="all")
    backends=backend=="all" ? ("colmat","julia_lab","julia_stdlib") : (backend,)
    all(b->b in ("colmat","julia_lab","julia_stdlib"),backends) || error("unknown backend")
    ispath(outdir) && error("output directory already exists")
    width==Threads.nthreads(:default) || error("one fresh Julia process per budget required")
    mkpath(outdir)
    ex=width==1 ? Serial() : Team(width)
    team="colmat" in backends ? team_new(width) : C_NULL
    "colmat" in backends && team==C_NULL && error("cannot create Rust team")
    try
        BLAS.set_num_threads(1)
        # FFI coverage beyond timing inputs: rectangular thin outputs, zero/rank deficiency.
        rng=MersenneTwister(391)
        for (m,n) in [(17,17),(67,19),(19,31)], zero in (false,true)
            A=zero ? zeros(m,n) : randn(rng,m,n)
            verify(team,ex,A,Val(:qr),backends); verify(team,ex,A,Val(:svd),backends)
            m==n && verify(team,ex,Matrix(Symmetric(A)),Val(:eigh),backends)
        end
        rng=MersenneTwister(84)
        open(joinpath(outdir,"samples.csv"),"w") do io
            println(io,"family,m,n,width,backend,rep,position,total_s,julia_alloc_bytes,gc_s,residual,orthogonality,input_sha256")
            for n in parse.(Int,split(get(ENV,"SIZES","256,512"),','))
                X=randn(rng,n,n); H=(X+X')/2; A=randn(rng,4n,n÷2)
                cell(io,team,ex,H,Val(:eigh),width,reps,backends)
                cell(io,team,ex,A,Val(:qr),width,reps,backends)
                cell(io,team,ex,X,Val(:svd),width,reps,backends)
            end
        end
        open(joinpath(outdir,"metadata.txt"),"w") do io
            println(io,"Julia=$VERSION; width=$width; reps=$reps; backend=$backend; seed=84; BLAS=",BLAS.get_config())
            println(io,"Rust lib=",abspath(RUST_LIB))
            println(io,"Allocation-inclusive owning factors. Team startup, output extraction/destruction, validation, GC preparation and20ms spin-settle are outside timing.")
            println(io,"Rust allocation bytes are not visible to Julia @timed; missing is intentional.")
            println(io,"Custom Julia terminal solvers use one BLAS thread. Standard dense Julia uses BLAS=width.")
            println(io,"head=",readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
            print(io,read("/proc/self/status",String))
        end
    finally
        team_free(team)
        BLAS.set_num_threads(1)
    end
    println("Saved ",abspath(outdir))
end
length(ARGS) in (3,4) || error("usage: compare.jl NEW_OUTPUT_DIR WIDTH REPS [BACKEND]")
main(ARGS[1],parse(Int,ARGS[2]),parse(Int,ARGS[3]),length(ARGS)==4 ? ARGS[4] : "all")
