"""
Float64 algorithm laboratory, mapped to colmat b292f474.
Householder algorithms follow A = Q R, A = Q T Q', A = U B V'.
This is independent Julia code, not a binding or a bitwise Rust translation.
See ../README.md for the explicit LAPACK terminal-solver boundary.
"""
module ColmatLab
using LinearAlgebra
using ThreadTeams
export Serial, Team, qr_lab, eigh_lab, svd_lab, qr_factor!, tridiagonalize!,
       bidiagonalize!, QRWorkspace, BidiagWorkspace, TridiagWorkspace,
       repair_weights!, repair_vectors!, repair_fixture

abstract type Execution end
struct Serial <: Execution end
struct Team <: Execution
    threads::Int
    minwork::Int
    function Team(threads::Int=Threads.nthreads(:default); minwork::Int=0)
        1 <= threads <= Threads.nthreads(:default) || throw(ArgumentError("start Julia with -t N,0 for this budget"))
        minwork >= 0 || throw(ArgumentError("minwork must be nonnegative"))
        new(threads, minwork)
    end
end
function ranges(f, n::Int, ::Serial, work::Int)
    n > 0 && f(1, n)
    nothing
end
function ranges(f, n::Int, ex::Team, work::Int)
    n == 0 && return nothing
    if ex.threads == 1 || work < ex.minwork
        f(1, n)
    else
        ThreadTeams.foreach_range(f, n; threads=ex.threads)
    end
    nothing
end

# BLAS is single threaded; the laboratory splits disjoint output columns.
# No per-worker full matrix or partial-result reduction is introduced.
function product!(C, A, B, ex::Execution, α=1.0, β=0.0)
    size(C, 2) == 0 && return C
    ranges(size(C, 2), ex, size(A,1)*size(A,2)*size(B,2)) do lo, hi
        @views mul!(C[:,lo:hi], A, B[:,lo:hi], α, β)
    end
    C
end
function matvec!(y, A, x, ex::Execution, α=1.0, β=0.0)
    ranges(length(y), ex, length(A)) do lo, hi
        @views mul!(y[lo:hi], A[lo:hi,:], x, α, β)
    end
    y
end

# LAPACK's conventional reflector H = I - tau v v', with v[1] = 1 implicit.
# The public drivers scale the entire input before reduction.
function reflector!(x)
    isempty(x) && return 0.0
    tail = norm(@view x[2:end])
    tail == 0.0 && return 0.0
    β = -copysign(hypot(x[1], tail), x[1])
    τ = (β-x[1])/β
    denom = x[1]-β
    @views x[2:end] ./= denom
    x[1] = β
    τ
end
function left!(A, v, τ, ex::Execution)
    τ == 0.0 && return A
    ranges(size(A,2), ex, length(A)) do lo, hi
        for j in lo:hi
            s = 0.0
            @inbounds for i in eachindex(v); s += v[i]*A[i,j]; end
            s *= τ
            @inbounds for i in eachindex(v); A[i,j] -= v[i]*s; end
        end
    end
    A
end
function right!(A, v, τ, work, ex::Execution)
    τ == 0.0 && return A
    matvec!(work, A, v, ex, τ)
    ranges(size(A,2), ex, length(A)) do lo, hi
        for j in lo:hi
            c = v[j]
            @inbounds @simd for i in axes(A,1); A[i,j] -= work[i]*c; end
        end
    end
    A
end
function identity!(A)
    fill!(A, 0.0)
    for i in 1:min(size(A)...); A[i,i] = 1.0; end
    A
end
function normalized(A::AbstractMatrix{Float64})
    all(isfinite,A) || throw(ArgumentError("finite Float64 input required"))
    s = maximum(abs, A; init=0.0)
    B = Matrix(A)
    s != 0.0 && (B ./= s)
    B, (s == 0.0 ? 1.0 : s)
end
include("qr.jl")
include("basis.jl")
include("eigh.jl")
include("svd.jl")
include("repair.jl")
end
