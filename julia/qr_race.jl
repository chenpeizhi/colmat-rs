# Run each runtime in a separate process; shared variants use BLAS parallelism.
include("compare_common.jl")
using Statistics
const QR_RUST_LEAF=parse(Int,get(ENV,"QR_RUST_LEAF","1"))
const QR_RACE_WIDTH=parse(Int,get(ENV,"QR_WIDTH","8"))
const QR_RACE_M=parse(Int,get(ENV,"QR_M","2048"))
const QR_RACE_N=parse(Int,get(ENV,"QR_N","256"))
const QR_RACE_BASISBLOCK=parse(Int,get(ENV,"QR_BASISBLOCK","0"))
const QR_RACE_PRESCALE=get(ENV,"QR_PRESCALE","1")=="1"
const QR_RACE_LEAF=parse(Int,get(ENV,"QR_LEAF","3"))
const QR_RACE_PAD=get(ENV,"QR_PAD_BASIS","0")=="1"
const QR_RACE_GRAIN=parse(Int,get(ENV,"QR_COLGRAIN","1"))
function target_input()
    (QR_RACE_M,QR_RACE_N)!=(2048,256) && return randn(MersenneTwister(84),QR_RACE_M,QR_RACE_N)
    rng=MersenneTwister(84)
    randn(rng,256,256);randn(rng,1024,128);randn(rng,512,512)
    randn(rng,2048,256)
end
invoke_race(::Val{:leaf1},team,A,ex,block)=rust_call(team,A,UInt32(101))
invoke_race(::Val{:leaf4},team,A,ex,block)=rust_call(team,A,UInt32(104))
invoke_race(::Val{:leaf8},team,A,ex,block)=rust_call(team,A,UInt32(108))
invoke_race(::Val{:leaf12},team,A,ex,block)=rust_call(team,A,UInt32(112))
invoke_race(::Val{:leaf16},team,A,ex,block)=rust_call(team,A,UInt32(116))
invoke_race(::Val{:leaf24},team,A,ex,block)=rust_call(team,A,UInt32(124))
invoke_race(::Val{:leaf48},team,A,ex,block)=rust_call(team,A,UInt32(148))
invoke_race(::Val{:rust_leaf},team,A,ex,b)=rust_call(team,A,UInt32(100+QR_RUST_LEAF))
invoke_race(::Val{:rust_original},team,A,ex,b)=rust_call(team,A,UInt32(3))
invoke_race(::Val{:rust},team,A,ex,b)=rust_call(team,A,UInt32(1))
invoke_race(::Val{:rust_control},team,A,ex,b)=rust_call(team,A,UInt32(1))
invoke_race(::Val{:old},team,A,ex,b)=qr_lab(A,ex;block=16)
invoke_race(::Val{:aligned},team,A,ex,b)=qr_colmat_lab(A,ex;block=b,prefix=false,colgrain=QR_RACE_GRAIN,padbasis=QR_RACE_PAD,leaf=QR_RACE_LEAF,prescale=QR_RACE_PRESCALE)
invoke_race(::Val{:structured},team,A,ex,b)=qr_colmat_lab(A,ex;block=b,prefix=true,colgrain=QR_RACE_GRAIN,padbasis=QR_RACE_PAD,leaf=QR_RACE_LEAF,prescale=QR_RACE_PRESCALE,structured=true,basisblock=QR_RACE_BASISBLOCK==0 ? b : QR_RACE_BASISBLOCK)
invoke_race(::Val{:structured_shared},team,A,ex,b)=qr_colmat_lab(A,Serial();block=b,prefix=true,colgrain=QR_RACE_GRAIN,padbasis=QR_RACE_PAD,leaf=QR_RACE_LEAF,prescale=QR_RACE_PRESCALE,structured=true,basisblock=QR_RACE_BASISBLOCK==0 ? b : QR_RACE_BASISBLOCK)
invoke_race(::Val{:shared},team,A,ex,b)=qr_colmat_lab(A,Serial();block=b,prefix=true,colgrain=QR_RACE_GRAIN,padbasis=QR_RACE_PAD,leaf=QR_RACE_LEAF,prescale=QR_RACE_PRESCALE)
invoke_race(::Val{:prefix},team,A,ex,b)=qr_colmat_lab(A,ex;block=b,prefix=true,colgrain=QR_RACE_GRAIN,padbasis=QR_RACE_PAD,leaf=QR_RACE_LEAF,prescale=QR_RACE_PRESCALE)
function trial(mode,team,A,ex,b)
    GC.gc();sleep(.02)
    t=@timed invoke_race(mode,team,A,ex,b)
    if t.value isa Ptr{Cvoid}
        try
            check(A,rust_outputs(t.value,A,Val(:qr)),Val(:qr))
            if mode isa Val{:rust_leaf} || startswith(string(typeof(mode).parameters[1]),"leaf")
                stages=zeros(2)
                ccall((:compare_copy,RUST_LIB),Cvoid,(Ptr{Cvoid},UInt32,Ptr{Float64}),t.value,UInt32(0),stages)
                return t.time,stages[1],stages[2]
            end
        finally
            result_free(t.value)
        end
        return t.time,NaN,NaN
    end
    check(A,t.value,Val(:qr))
    t.time,t.value.times.factor,t.value.times.basis
end
function run_race(outdir,reps,block,modes)
    ispath(outdir) && error("new output directory required")
    any(m->m in (:shared,:structured_shared),modes) && length(modes)!=1 && error("shared BLAS requires its own process")
    mkpath(outdir);BLAS.set_num_threads(any(m->m in (:shared,:structured_shared),modes) ? QR_RACE_WIDTH : 1)
    A=target_input();inputhash=bytes2hex(sha256(reinterpret(UInt8,vec(A))))
    ex=Team(QR_RACE_WIDTH;minwork=parse(Int,get(ENV,"QR_MINWORK","0")))
    native=any(m->m in (:rust,:rust_control,:rust_original,:rust_leaf) || startswith(string(m),"leaf"),modes)
    team=native ? team_new(QR_RACE_WIDTH) : C_NULL
    native && team==C_NULL && error("Rust team")
    try
        for _ in 1:2, mode in modes;trial(Val(mode),team,A,ex,block);end
        times=zeros(3)
        open(joinpath(outdir,"rust-profile.csv"),"w") do io
            println(io,"rep,factor_s,basis_s,sum_s")
            for rep in 1:(native ? 7 : 0)
                GC.gc();sleep(.02)
                status=ccall((:compare_qr_profile,RUST_LIB),Cint,(Ptr{Cvoid},Ptr{Float64},Csize_t,Csize_t,Ptr{Float64}),team,A,size(A,1),size(A,2),times)
                status==0 || error("QR stage profile failed")
                println(io,join((rep,times...),','))
            end
        end
        open(joinpath(outdir,"samples.csv"),"w") do io
            println(io,"mode,block,rep,position,total_s,factor_s,basis_s,input_sha256")
            for rep in 1:reps
                order=circshift(modes,mod(rep-1,length(modes)))
                iseven(rep) && reverse!(order)
                for (position,mode) in enumerate(order)
                    t,f,q=trial(Val(mode),team,A,ex,block)
                    println(io,join((mode,block,rep,position,t,f,q,inputhash),','));flush(io)
                end
            end
        end
        open(joinpath(outdir,"metadata.txt"),"w") do io
            println(io,"Target $(size(A)), thin Q/R,$QR_RACE_WIDTH threads, BLAS=",BLAS.get_num_threads(),", Julia=",VERSION)
            println(io,"Rust binary=",RUST_LIB,"; input_sha256=",inputhash)
            println(io,"modes=$modes; block=$block; reps=$reps; grain=$QR_RACE_GRAIN; leaf=$QR_RACE_LEAF; rust_leaf=$QR_RUST_LEAF; basisblock=$QR_RACE_BASISBLOCK; pad=$QR_RACE_PAD; prescale=$QR_RACE_PRESCALE; owning-call allocations included")
            print(io,read("/proc/self/status",String))
        end
    finally
        team_free(team)
    end
    println("Saved ",outdir)
end
length(ARGS)>=1 || error("usage: qr_race.jl NEW_OUTPUT_DIR [REPS BLOCK MODES]")
run_race(ARGS[1],length(ARGS)>1 ? parse(Int,ARGS[2]) : 11,length(ARGS)>2 ? parse(Int,ARGS[3]) : 48,
         Symbol.(split(length(ARGS)>3 ? ARGS[4] : "rust,rust_control,old,aligned,prefix",',')))
