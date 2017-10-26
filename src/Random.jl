# Floyd's algorithm
# http://stackoverflow.com/questions/2394246/algorithm-to-select-a-single-random-combination-of-values
# http://delivery.acm.org/10.1145/320000/315746/p754-bentley.pdf?ip=131.215.123.169&id=315746
# &acc=ACTIVE%20SERVICE&key=1FCBABC0A756505B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35
# &CFID=457620382&CFTOKEN=52570986&__acm__=1416526957_98e38bc71903d7b401b9715ef079c775
function random_combination(N::Int, M::Int)
    @assert N > 0
    @assert M >= 0 # M = 0 returns an emtpy Vector{Int}
    @assert N >= M

    # this isn't actually necessary, but is more explicit
    if M == 0
        return Int[]
    end

    S = Set{Int}()
    V = Int[]
    for J in N-M+1:N
        T = rand(1:J)
        a = !in(T, S) ? T : J
        push!(S, a)
        push!(V, a)
    end

    return V
end

# version of println which flushes the stream after every call
function printlnf(args...)
    println(args...)
    flush(STDOUT)
end
