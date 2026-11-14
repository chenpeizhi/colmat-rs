# User-requested local laboratory driver; separate from the Rust acceptance campaign.
using LinearAlgebra, Random, Statistics
include("src/ColmatLab.jl")
using .ColmatLab

function stage_times(value)
    if value isa NamedTuple && hasproperty(value,:times)
        t=values(value.times)
        return length(t)==2 ? (t[1],t[2],0.0) : Tuple(t)
    end
    (0.0,0.0,0.0)
end
function measure!(io,family,m,n,width,mode,call,reset,reps)
    # Compile and warm both the numerical path and persistent workers, outside samples.
    for _ in 1:2; reset(); call(); end
    for rep in 1:reps
        reset(); GC.gc()
        result=@timed call()
        a,b,c=stage_times(result.value)
        println(io,join((family,m,n,width,mode,rep,result.time,result.bytes,result.gctime,a,b,c),','))
        flush(io)
    end
end
function family!(io,name,A,driver,makeworkspace,kernelfn,ex,width,reps)
    m,n=size(A)
    measure!(io,name,m,n,width,"julia",()->driver(A,ex),()->nothing,reps)
    B=if name=="svd" && m*sizeof(Float64)%1024==0
        storage=Matrix{Float64}(undef,m+8,n)
        result=@view storage[1:m,:]
        copyto!(result,A)
        result
    else
        copy(A)
    end
    ws=makeworkspace(m,n)
    measure!(io,name*"_kernel",m,n,width,"julia",()->kernelfn(B,ws,ex),()->copyto!(B,A),reps)
end
function main(outdir; reps=parse(Int,get(ENV,"REPS","5")))
    reps>0 || error("REPS must be positive")
    widths=parse.(Int,split(get(ENV,"WIDTHS","1,4,8"),','))
    all(w->1<=w<=Threads.nthreads(:default),widths) || error("launch with -t at least maximum(WIDTHS),0")
    ispath(outdir) && error("output directory already exists: $outdir")
    mkpath(outdir)
    rng=MersenneTwister(84)
    sizes=parse.(Int,split(get(ENV,"SIZES","256,512"),','))
    metadata=open(joinpath(outdir,"metadata.txt"),"w")
    println(metadata,"Julia: ",VERSION,"; threads: ",Threads.nthreads(:default))
    println(metadata,"CPU: ",Sys.cpu_info()[1].model)
    println(metadata,"BLAS: ",BLAS.get_config())
    println(metadata,"commit: ",readchomp(`git -C $(@__DIR__) rev-parse HEAD`))
    println(metadata,"widths=$widths sizes=$sizes reps=$reps; seed=84; local reference scan")
    println(metadata,"Custom paths: BLAS=1. Standard dense references: BLAS=width.")
    println(metadata,"Whole calls include workspace/output allocations; *_kernel reuse workspaces, reset outside timing.")
    println(metadata,"Stage a/b/c: QR=factor/basis/0, eigh/SVD=reduction/terminal/backtransform.")
    println(metadata,"Kernel and repair stages are 0; total_s is their kernel time.")
    println(metadata,"ThreadTeams uses bounded default spin, no parallel_phase; task/worker startup is warmed.")
    isfile("/proc/self/status") && print(metadata,read("/proc/self/status",String))
    close(metadata)
    open(joinpath(outdir,"samples.csv"),"w") do io
        println(io,"family,m,n,width,mode,rep,total_s,bytes,gc_s,stage_a_s,stage_b_s,stage_c_s")
        # Interleave budgets in forward/reverse order in separate processes for formal comparisons.
        for n in sizes
            X=randn(rng,n,n); H=(X+X')/2
            tall=randn(rng,4n,max(1,n÷2))
            for width in widths
                ex=width==1 ? Serial() : Team(width)
                BLAS.set_num_threads(1)
                family!(io,"eigh",H,eigh_lab,(m,n)->TridiagWorkspace(n),tridiagonalize!,ex,width,reps)
                family!(io,"qr",tall,qr_lab,QRWorkspace,qr_factor!,ex,width,reps)
                family!(io,"svd",X,svd_lab,BidiagWorkspace,bidiagonalize!,ex,width,reps)
                f=repair_fixture(n;k=max(0,n-n÷10)); z=copy(f.z); U=Matrix{Float64}(undef,n,n)
                weight=()->repair_weights!(z,f.d,f.mus,f.shifts,f.k,ex)
                measure!(io,"repair_weights",n,n,width,"julia",weight,()->copyto!(z,f.z),reps)
                weight()
                columns=()->repair_vectors!(U,f.order,f.rows,f.d,z,f.mus,f.shifts,f.k,ex)
                measure!(io,"repair_vectors",n,n,width,"julia",columns,()->nothing,reps)
                BLAS.set_num_threads(width)
                measure!(io,"eigh",n,n,width,"stdlib",()->eigen(Symmetric(H)),()->nothing,reps)
                measure!(io,"svd",n,n,width,"stdlib",()->svd(X),()->nothing,reps)
                measure!(io,"qr",size(tall)...,width,"stdlib",()->begin
                    F=qr(tall); (Matrix(F.Q),Matrix(F.R))
                end,()->nothing,reps)
            end
        end
    end
    BLAS.set_num_threads(1)
    println("Saved ",abspath(outdir))
end
length(ARGS)==1 || error("usage: julia --project=julia -t N,0 julia/measure.jl NEW_OUTPUT_DIR")
main(ARGS[1])
