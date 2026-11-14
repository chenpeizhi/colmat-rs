include("compare_common.jl")
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
