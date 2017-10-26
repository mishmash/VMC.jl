import Base: +, -, *, /

immutable Site2D
    # NB: origin is at (x,y) = (0,0)
    x::Int
    y::Int

    Lx::Int
    Ly::Int
    index::Int

    function Site2D(x::Int, y::Int, Lx::Int, Ly::Int)
        # we don't want these asserts so as to have PBC capabilities
        # @assert 0 <= x < Lx
        # @assert 0 <= y < Ly
        index = mod(x, Lx) * Ly + mod(y, Ly) + 1
        return new(div(index-1, Ly), mod(index-1, Ly), Lx, Ly, index)
    end

    function Site2D(index::Int, Lx::Int, Ly::Int)
        @assert 1 <= index <= Lx * Ly
        return new(div(index-1, Ly), mod(index-1, Ly), Lx, Ly, index)
    end
end

function +(r1::Site2D, r2::Site2D)
    @assert r1.Lx == r2.Lx
    @assert r1.Ly == r2.Ly
    return Site2D(r1.x + r2.x, r1.y + r2.y, r1.Lx, r1.Ly)
end

function -(r1::Site2D, r2::Site2D)
    @assert r1.Lx == r2.Lx
    @assert r1.Ly == r2.Ly
    return Site2D(r1.x - r2.x, r1.y - r2.y, r1.Lx, r1.Ly)
end

*(c::Number, r::Site2D) = Site2D(convert(Int, c * r.x), convert(Int, c * r.y), r.Lx, r.Ly)
*(r::Site2D, c::Number) = Site2D(convert(Int, c * r.x), convert(Int, c * r.y), r.Lx, r.Ly)
/(r::Site2D, c::Number) = Site2D(convert(Int, r.x / c), convert(Int, r.y / c), r.Lx, r.Ly)

export +, -, *, /

unitx(Lx::Int, Ly::Int) = Site2D(1, 0, Lx, Ly)
unity(Lx::Int, Ly::Int) = Site2D(0, 1, Lx, Ly)

# rectangular, contiguous subsystem LAx/LAy sites long in x/y direction
# whose bottom-/left-most site is located at (x0, y0)
function make_subsystem(LAx::Int, LAy::Int, Lx::Int, Ly::Int, x0::Int = 0, y0::Int = 0)
    @assert 0 < LAx <= Lx
    @assert 0 < LAy <= Ly
    @assert 0 <= x0 < Lx
    @assert 0 <= y0 < Ly

    origin = Site2D(x0, y0, Lx, Ly)

    subsystem = Int[]
    for i in 1:LAx
        bottom_site = origin + (i-1)*unitx(Lx, Ly) 
        for j in 1:LAy
            push!(subsystem, (bottom_site + (j-1)*unity(Lx, Ly)).index)
        end
    end

    return subsystem
end

#########################################################
#################### Neighbor tables ####################
#########################################################

function zigzag_strip_periodic_nn(Lx::Int)
    xhat = unitx(Lx, 1)
    Nsites = Lx
    table = Vector{Int}[]
    for i in 1:Nsites
        r = Site2D(i, Lx, 1) # actually a site on a 1D chain..
        push!(table, [(r + 2xhat).index, 
                      (r + xhat).index,
                      (r - xhat).index,
                      (r - 2xhat).index])
    end
    return table
end

function kagome_strip_periodic_nn(Lx::Int)
    xhat = unitx(Lx, 3)
    yhat = unity(Lx, 3)
    Nsites = 3Lx
    table = Vector{Int}[]
    for i in 1:Nsites
        r = Site2D(i, Lx, 3)
        if r.y == 0
            push!(table, [(r + xhat).index,
                          (r + xhat + yhat).index,
                          (r + yhat).index,
                          (r - xhat).index])
        elseif r.y == 1
            push!(table, [(r + yhat).index,
                          (r - xhat + yhat).index,
                          (r - xhat - yhat).index,
                          (r - yhat).index])
        elseif r.y == 2
            push!(table, [(r + xhat).index,
                          (r - xhat).index,
                          (r - yhat).index,
                          (r + xhat - yhat).index])
        else
            @assert false
        end
    end
    return table
end

function kagome_strip_open_nn(Lx::Int)
    xhat = unitx(Lx, 3)
    yhat = unity(Lx, 3)
    Nsites = 3Lx-2
    table = Vector{Int}[]
    for i in 1:Nsites-1
        r = Site2D(i, Lx, 3)
        if r.x > 0 && r.x < Lx-2
            if r.y == 0
                push!(table, [(r + xhat).index,
                              (r + xhat + yhat).index,
                              (r + yhat).index,
                              (r - xhat).index])
            elseif r.y == 1
                push!(table, [(r + yhat).index,
                              (r - xhat + yhat).index,
                              (r - xhat - yhat).index,
                              (r - yhat).index])
            elseif r.y == 2
                push!(table, [(r + xhat).index,
                              (r - xhat).index,
                              (r - yhat).index,
                              (r + xhat - yhat).index])
            else
                @assert false
            end
        elseif r.x == 0
            if r.y == 0
                push!(table, [(r + xhat).index,
                              (r + xhat + yhat).index,
                              (r + yhat).index])
            elseif r.y == 1
                push!(table, [(r + yhat).index,
                              (r - yhat).index])
            elseif r.y == 2
                push!(table, [(r + xhat).index,
                              (r - yhat).index,
                              (r + xhat - yhat).index])
            else
                @assert false
            end
        elseif r.x == Lx-2
            if r.y == 0
                push!(table, [Nsites,
                              (r + yhat).index,
                              (r - xhat).index])
            elseif r.y == 1
                push!(table, [(r + yhat).index,
                              (r - xhat + yhat).index,
                              (r - xhat - yhat).index,
                              (r - yhat).index])
            elseif r.y == 2
                push!(table, [(r - xhat).index,
                              (r - yhat).index,
                              Nsites])
            else
                @assert false
            end
        else
            @assert false
        end
    end
    push!(table, [Nsites-1, Nsites-3]) # deal with the last site separately
    return table
end

function triangular_ladder_PBCx_PBCy_nn(Lx::Int, Ly::Int)
    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Lx * Ly
    table = Vector{Int}[]
    for i in 1:Nsites
        r = Site2D(i, Lx, Ly)
        push!(table, [(r + xhat).index,
                      (r + xhat + yhat).index,
                      (r + yhat).index,
                      (r - xhat).index,
                      (r - xhat - yhat).index,
                      (r - yhat).index])
    end
    return table
end

function triangular_ladder_subsystem_wrap_nn(Lx::Int, Ly::Int, LAx::Int, LAy::Int)
    @assert LAy == Ly # require cylindrical subsystem for now
    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Lx * Ly

    subsystemA = make_subsystem(LAx, LAy, Lx, Ly)
    tableA = Vector{Int}[]
    for i in subsystemA
        r = Site2D(i, Lx, Ly)
        push!(tableA, [in((r + xhat).index, subsystemA) ? (r + xhat).index : (r - (LAx-1)*xhat).index,
                       in((r + xhat + yhat).index, subsystemA) ? (r + xhat + yhat).index : (r - (LAx-1)*xhat + yhat).index,
                       (r + yhat).index, # this guy's always still in subsystemA given the assert
                       in((r - xhat).index, subsystemA) ? (r - xhat).index : (r + (LAx-1)*xhat).index,
                       in((r - xhat - yhat).index, subsystemA) ? (r - xhat - yhat).index : (r + (LAx-1)*xhat - yhat).index,
                       (r - yhat).index]) # this guy's always still in subsystemA given the assert
        @assert issubset(tableA[end], subsystemA)
    end

    subsystemB = setdiff(collect(1:Nsites), subsystemA)
    tableB = Vector{Int}[]
    for i in subsystemB
        r = Site2D(i, Lx, Ly)
        push!(tableB, [in((r + xhat).index, subsystemB) ? (r + xhat).index : (r - (LAx-1)*xhat).index,
                       in((r + xhat + yhat).index, subsystemB) ? (r + xhat + yhat).index : (r - (LAx-1)*xhat + yhat).index,
                       (r + yhat).index, # this guy's always still in subsystemB given the assert
                       in((r - xhat).index, subsystemB) ? (r - xhat).index : (r + (LAx-1)*xhat).index,
                       in((r - xhat - yhat).index, subsystemB) ? (r - xhat - yhat).index : (r + (LAx-1)*xhat - yhat).index,
                       (r - yhat).index]) # this guy's always still in subsystemB given the assert
        @assert issubset(tableB[end], subsystemB)
    end

    return tableA, tableB
end

function square_lattice_PBCx_PBCy_nn(Lx::Int, Ly::Int)
    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Lx * Ly
    table = Vector{Int}[]
    for i in 1:Nsites
        r = Site2D(i, Lx, Ly)
        push!(table, [(r + xhat).index,
                      (r + yhat).index,
                      (r - xhat).index,
                      (r - yhat).index])
    end
    return table
end
