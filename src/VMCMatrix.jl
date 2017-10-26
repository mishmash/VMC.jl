# This is the backbone of the VMC code.  Both determinants and Pfaffians are available.

abstract VMCMatrix{T}

#####################################################################
#################### Determinantal wavefunctions ####################
#####################################################################

type CeperleyMatrix{T} <: VMCMatrix{T}
    # FIXME: make current_state an Enum, i.e., enforce that it's one of a certain set of states
    current_state::Symbol

    detmat::Matrix{T}
    inverse::Matrix{T}
    
    det::T
    old_det::T

    detrat::T
    Q_inv::Matrix{T}

    # pending indices
    cs::Vector{Int}
    rs::Vector{Int}

    # these apply equally well to single column or row updates
    pending_cols::Matrix{T}
    pending_rows::Matrix{T}

    # these are only necessary for combined column and row updates,
    # and are determined from pending_cols and pending_rows
    cols_tilde::Matrix{T}
    rows_tilde::Matrix{T}

    CeperleyMatrix() = new(:UNINITIALIZED)

    function CeperleyMatrix(mat::Matrix{T})
        @assert size(mat)[1] == size(mat)[2]
        LUmat = lufact(mat)
        detval = det(LUmat)
        return new(:READY_FOR_UPDATE, mat, inv(LUmat), detval, detval)
    end
end

CeperleyMatrix(mat::Matrix) = CeperleyMatrix{eltype(mat)}(mat)
CeperleyMatrix() = CeperleyMatrix{Float64}() # default to Float64

# this should be unnecessary to call anywhere below,
# unless there was some rogue action on the detmat and inverse fields from the outside
function consistent_Ceperley_dims(cmat::CeperleyMatrix)
    return size(cmat.detmat)[1] == size(cmat.detmat)[2] &&
           size(cmat.inverse)[1] == size(cmat.inverse)[2] &&
           size(cmat.detmat)[1] == size(cmat.inverse)[1]
end

# check inverse for numerical error due to SMW
function check_inverse{T}(cmat::CeperleyMatrix{T}, tol::Float64 = 1e-10)
    @assert cmat.current_state == :READY_FOR_UPDATE

    error = sum(abs(cmat.detmat * cmat.inverse - eye(T, size(cmat.detmat)[1]))) / size(cmat.detmat)[1]
    is_good = error < tol

    return is_good, error
end

function fix_inverse!(cmat::CeperleyMatrix)
    LU = lufact(cmat.detmat)
    cmat.inverse = inv(LU)
    cmat.det = cmat.old_det = det(LU)
    return nothing
end

# FIXME: should be able to get name of passed argument from argument itself..?
function check_and_fix_inverse!(cmat::CeperleyMatrix, name::ASCIIString)
    cmat_error = check_inverse(cmat)
    if !cmat_error[1]
        printlnf("Recomputing $(name).inverse from scratch!  Error = $(cmat_error[2])")
        fix_inverse!(cmat)
    end
    # FIXME: have this return something so that we can keep track of how often errors occurred
    return nothing
end

function check_and_reset_inverse_and_det!(cmat::CeperleyMatrix, name::ASCIIString,
                                          inv_tol::Float64 = 1e-10, det_tol::Float64 = 1e-10)
    @assert cmat.current_state == :READY_FOR_UPDATE

    LU = lufact(cmat.detmat)
    inv_exact = inv(LU)
    det_exact = det(LU)

    inv_error = sum(abs(cmat.inverse - inv_exact)) / size(inv_exact)[1]
    det_error = abs((cmat.det - det_exact) / det_exact)

    if inv_error > inv_tol
        printlnf("Inverse error for $(name) = $(inv_error)")
    end
    if det_error > det_tol
        printlnf("Determinant error for $(name) = $(det_error)")
    end

    # reset inverse and determinant to exact values
    cmat.inverse = inv_exact
    cmat.det = cmat.old_det = det_exact

    # do we want/need to return this?
    return inv_error, det_error
end

function get_detrat(cmat::CeperleyMatrix)
    @assert cmat.current_state == :COLUMNS_UPDATE_PENDING ||
            cmat.current_state == :ROWS_UPDATE_PENDING ||
            cmat.current_state == :COLUMNS_AND_ROWS_UPDATE_PENDING
    return cmat.detrat
end

# I don't think it's appropriate for another class to get det and detmat during an update (see asserts)
function get_det(cmat::CeperleyMatrix)
    @assert cmat.current_state == :READY_FOR_UPDATE
    return cmat.det
end

function get_detmat(cmat::CeperleyMatrix)
    @assert cmat.current_state == :READY_FOR_UPDATE
    return cmat.detmat
end

function get_det_exact(cmat::CeperleyMatrix)
    @assert cmat.current_state == :READY_FOR_UPDATE
    return det(cmat.detmat)
end

function relative_determinant_error(cmat::CeperleyMatrix)
    @assert cmat.current_state == :READY_FOR_UPDATE
    det_exact = det(cmat.detmat)
    return abs((cmat.det - det_exact) / det_exact)
end

###########################################################
#################### Swap columns/rows ####################
###########################################################

function swap_columns!{T}(A::Matrix{T}, c1::Int, c2::Int)
    m, n = size(A)
    if (1 <= c1 <= n) && (1 <= c2 <= n)
        @inbounds begin
            for i in 1:m
                A[i, c1], A[i, c2] = A[i, c2], A[i, c1]
            end
        end
        return A
    else
        throw(BoundsError())
    end
end

function swap_columns{T}(A::Matrix{T}, c1::Int, c2::Int)
    m, n = size(A)
    Anew = copy(A)
    if (1 <= c1 <= n) && (1 <= c2 <= n)
        @inbounds begin
            for i in 1:m
                Anew[i, c1], Anew[i, c2] = A[i, c2], A[i, c1]
            end
        end
        return Anew
    else
        throw(BoundsError())
    end
end

function swap_rows!{T}(A::Matrix{T}, r1::Int, r2::Int)
    m, n = size(A)
    if (1 <= r1 <= m) && (1 <= r2 <= m)
        @inbounds begin
            for j in 1:n
                A[r1, j], A[r2, j] = A[r2, j], A[r1, j] 
            end
        end
        return A
    else
        throw(BoundsError())
    end
end

function swap_rows{T}(A::Matrix{T}, r1::Int, r2::Int)
    m, n = size(A)
    Anew = copy(A)
    if (1 <= r1 <= m) && (1 <= r2 <= m)
        @inbounds begin
            for j in 1:n
                Anew[r1, j], Anew[r2, j] = A[r2, j], A[r1, j] 
            end
        end
        return Anew
    else
        throw(BoundsError())
    end
end

function swap_columns!(cmat::CeperleyMatrix, c1::Int, c2::Int)
    @assert c1 <= size(cmat.detmat)[2]
    @assert c2 <= size(cmat.detmat)[2]
    @assert c1 != c2
    @assert cmat.current_state == :READY_FOR_UPDATE

    swap_columns!(cmat.detmat, c1, c2)
    swap_rows!(cmat.inverse, c1, c2)

    cmat.det *= -1
    # no need to do anything to detrat since it's only relevant while an update is pending

    return cmat
end

function swap_rows!(cmat::CeperleyMatrix, r1::Int, r2::Int)
    @assert r1 <= size(cmat.detmat)[1]
    @assert r2 <= size(cmat.detmat)[1]
    @assert r1 != r2
    @assert cmat.current_state == :READY_FOR_UPDATE

    swap_rows!(cmat.detmat, r1, r2)
    swap_columns!(cmat.inverse, r1, r2)

    cmat.det *= -1
    # no need to do anything to detrat since it's only relevant while an update is pending

    return cmat
end

#########################################################################
#################### Start updates (compute detrats) ####################
#########################################################################

# NB: In the detrat_from_* functions, we keep the type parameter in the definition to assert that the
# type of the input CeperleyMatrix object and the cols / rows Matrix objects have the same type T.

# single column
function detrat_from_columns_update!{T}(cmat::CeperleyMatrix{T}, c::Int, col::Matrix{T})
    @assert size(col)[2] == 1
    @assert c <= size(cmat.detmat)[2]
    @assert size(col)[1] == size(cmat.detmat)[1]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    cmat.detrat = (cmat.inverse[c, :] * col)[1]
    cmat.Q_inv = [1. / cmat.detrat].'

    cmat.old_det = cmat.det
    cmat.det *= cmat.detrat

    cmat.cs = [c]
    cmat.pending_cols = col
    cmat.current_state = :COLUMNS_UPDATE_PENDING

    return cmat
end

# single column (no bang)
function detrat_from_columns_update{T}(cmat::CeperleyMatrix{T}, c::Int, col::Matrix{T})
    @assert size(col)[2] == 1
    @assert c <= size(cmat.detmat)[2]
    @assert size(col)[1] == size(cmat.detmat)[1]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    return (cmat.inverse[c, :] * col)[1]
end

# multiple columns
function detrat_from_columns_update!{T}(cmat::CeperleyMatrix{T}, cs::Vector{Int}, cols::Matrix{T})
    @assert length(cs) == size(cols)[2]
    @assert length(cs) <= size(cmat.detmat)[2]
    # these asserts are somewhat costly (the first more so)
    # @assert all(cs .<= size(cmat.detmat)[2])
    # @assert cs == unique(cs)
    @assert size(cols)[1] == size(cmat.detmat)[1]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # generalized matrix determinant lemma
    Q = zeros(T, (length(cs), length(cs))) # should we even allocate here?
    Q = cmat.inverse[cs, :] * cols
    LUQ = lufact(Q) # = lufact(cmat.inverse[cs, :] * cols)
    cmat.detrat = det(LUQ)
    cmat.Q_inv = inv(LUQ)

    cmat.old_det = cmat.det
    cmat.det *= cmat.detrat

    cmat.cs = cs
    cmat.pending_cols = cols
    cmat.current_state = :COLUMNS_UPDATE_PENDING

    return cmat
end

# multiple columns (no bang)
function detrat_from_columns_update{T}(cmat::CeperleyMatrix{T}, cs::Vector{Int}, cols::Matrix{T})
    @assert length(cs) == size(cols)[2]
    @assert length(cs) <= size(cmat.detmat)[2]
    # these asserts are somewhat costly (the first more so)
    # @assert all(cs .<= size(cmat.detmat)[2])
    # @assert cs == unique(cs)
    @assert size(cols)[1] == size(cmat.detmat)[1]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # generalized matrix determinant lemma
    return det(cmat.inverse[cs, :] * cols)
end

# single row
function detrat_from_rows_update!{T}(cmat::CeperleyMatrix{T}, r::Int, row::Matrix{T})
    @assert size(row)[1] == 1
    @assert r <= size(cmat.detmat)[1]
    @assert size(row)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    cmat.detrat = (row * cmat.inverse[:, r])[1]
    cmat.Q_inv = [1. / cmat.detrat].'

    cmat.old_det = cmat.det
    cmat.det *= cmat.detrat

    cmat.rs = [r]
    cmat.pending_rows = row
    cmat.current_state = :ROWS_UPDATE_PENDING

    return cmat
end

# single row (no bang)
function detrat_from_rows_update{T}(cmat::CeperleyMatrix{T}, r::Int, row::Matrix{T})
    @assert size(row)[1] == 1
    @assert r <= size(cmat.detmat)[1]
    @assert size(row)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    return (row * cmat.inverse[:, r])[1]
end

# multiple rows
function detrat_from_rows_update!{T}(cmat::CeperleyMatrix{T}, rs::Vector{Int}, rows::Matrix{T})
    @assert length(rs) == size(rows)[1]
    @assert length(rs) <= size(cmat.detmat)[1]
    # these asserts are somewhat costly (the first more so)
    # @assert all(rs .<= size(cmat.detmat)[1])
    # @assert rs == unique(rs)
    @assert size(rows)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # generalized matrix determinant lemma
    Q = zeros(T, (length(rs), length(rs))) # should we even allocate here?
    Q = rows * cmat.inverse[:, rs]
    LUQ = lufact(Q) # = lufact(rows * cmat.inverse[:, rs])
    cmat.detrat = det(LUQ)
    cmat.Q_inv = inv(LUQ)

    cmat.old_det = cmat.det
    cmat.det *= cmat.detrat

    cmat.rs = rs
    cmat.pending_rows = rows
    cmat.current_state = :ROWS_UPDATE_PENDING

    return cmat
end

# multiple rows (no bang)
function detrat_from_rows_update{T}(cmat::CeperleyMatrix{T}, rs::Vector{Int}, rows::Matrix{T})
    @assert length(rs) == size(rows)[1]
    @assert length(rs) <= size(cmat.detmat)[1]
    # these asserts are somewhat costly (the first more so)
    # @assert all(rs .<= size(cmat.detmat)[1])
    # @assert rs == unique(rs)
    @assert size(rows)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    # generalized matrix determinant lemma
    return det(rows * cmat.inverse[:, rs])
end

# multiple (in general) rows and columns
function detrat_from_columns_and_rows_update!{T}(cmat::CeperleyMatrix{T},
                                                 cs::Vector{Int}, cols::Matrix{T},
                                                 rs::Vector{Int}, rows::Matrix{T})
    @assert length(cs) == size(cols)[2]
    @assert length(rs) == size(rows)[1]
    @assert length(cs) <= size(cmat.detmat)[2]
    @assert length(rs) <= size(cmat.detmat)[1]
    @assert all(cs .<= size(cmat.detmat)[2])
    @assert all(rs .<= size(cmat.detmat)[1])
    @assert cs == unique(cs)
    @assert rs == unique(rs)
    @assert size(cols)[1] == size(cmat.detmat)[1]
    @assert size(rows)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    if cols[rs, :] != rows[:, cs]
        warn("cols[rs, :] != rows[:, cs] in detrat_from_columns_and_rows_update.
Proceeding anyway, putting the columns on top..") # see Q[nc+1:end, 1:nc] assignment below
    end

    cols_tilde = copy(cols)
    rows_tilde = copy(rows)
    cols_tilde[rs, :] = 0.5(cols[rs, :] + cmat.detmat[rs, cs])
    rows_tilde[:, cs] = 0.5(rows[:, cs] + cmat.detmat[rs, cs])

    # generalized matrix determinant lemma
    nc = length(cs); nr = length(rs)
    Q = zeros(T, (nc + nr, nc + nr))
    Q[1:nc, 1:nc] = cmat.inverse[cs, :] * cols_tilde
    Q[1:nc, nc+1:end] = cmat.inverse[cs, rs]
    Q[nc+1:end, nc+1:end] = rows_tilde * cmat.inverse[:, rs]
    Q[nc+1:end, 1:nc] = rows_tilde * cmat.inverse * cols_tilde - 2cols_tilde[rs, :] + cmat.detmat[rs, cs]
    # Q[nc+1:end, 1:nc] = rows_tilde * cmat.inverse * cols_tilde - 2rows_tilde[:, cs] + cmat.detmat[rs, cs]

    LUQ = lufact(Q)
    cmat.detrat = det(LUQ)
    cmat.Q_inv = inv(LUQ)

    cmat.old_det = cmat.det
    cmat.det *= cmat.detrat

    cmat.cs = cs
    cmat.rs = rs
    cmat.pending_cols = cols
    cmat.pending_rows = rows
    cmat.cols_tilde = cols_tilde
    cmat.rows_tilde = rows_tilde
    cmat.current_state = :COLUMNS_AND_ROWS_UPDATE_PENDING

    return cmat
end

# multiple (in general) rows and columns (no bang)
function detrat_from_columns_and_rows_update{T}(cmat::CeperleyMatrix{T},
                                                cs::Vector{Int}, cols::Matrix{T},
                                                rs::Vector{Int}, rows::Matrix{T})
    @assert length(cs) == size(cols)[2]
    @assert length(rs) == size(rows)[1]
    @assert length(cs) <= size(cmat.detmat)[2]
    @assert length(rs) <= size(cmat.detmat)[1]
    @assert all(cs .<= size(cmat.detmat)[2])
    @assert all(rs .<= size(cmat.detmat)[1])
    @assert cs == unique(cs)
    @assert rs == unique(rs)
    @assert size(cols)[1] == size(cmat.detmat)[1]
    @assert size(rows)[2] == size(cmat.detmat)[2]
    @assert cmat.current_state == :READY_FOR_UPDATE

    if cols[rs, :] != rows[:, cs]
        warn("cols[rs, :] != rows[:, cs] in detrat_from_columns_and_rows_update.
Proceeding anyway, putting the columns on top..") # see Q[nc+1:end, 1:nc] assignment below
    end

    cols_tilde = copy(cols)
    rows_tilde = copy(rows)
    cols_tilde[rs, :] = 0.5(cols[rs, :] + cmat.detmat[rs, cs])
    rows_tilde[:, cs] = 0.5(rows[:, cs] + cmat.detmat[rs, cs])

    # generalized matrix determinant lemma
    nc = length(cs); nr = length(rs)
    Q = zeros(T, (nc + nr, nc + nr))
    Q[1:nc, 1:nc] = cmat.inverse[cs, :] * cols_tilde
    Q[1:nc, nc+1:end] = cmat.inverse[cs, rs]
    Q[nc+1:end, nc+1:end] = rows_tilde * cmat.inverse[:, rs]
    Q[nc+1:end, 1:nc] = rows_tilde * cmat.inverse * cols_tilde - 2cols_tilde[rs, :] + cmat.detmat[rs, cs]
    # Q[nc+1:end, 1:nc] = rows_tilde * cmat.inverse * cols_tilde - 2rows_tilde[:, cs] + cmat.detmat[rs, cs]

    return det(Q)
end

# single column, multiple rows (in general)
detrat_from_columns_and_rows_update!{T}(cmat::CeperleyMatrix{T},
                                        c::Int, cols::Matrix{T},
                                        rs::Vector{Int}, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update!(cmat, [c], cols, rs, rows) )
detrat_from_columns_and_rows_update{T}(cmat::CeperleyMatrix{T},
                                       c::Int, cols::Matrix{T},
                                       rs::Vector{Int}, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update(cmat, [c], cols, rs, rows) )

# multiple columns (in general), single row
detrat_from_columns_and_rows_update!{T}(cmat::CeperleyMatrix{T},
                                        cs::Vector{Int}, cols::Matrix{T},
                                        r::Int, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update!(cmat, cs, cols, [r], rows) )
detrat_from_columns_and_rows_update{T}(cmat::CeperleyMatrix{T},
                                       cs::Vector{Int}, cols::Matrix{T},
                                       r::Int, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update(cmat, cs, cols, [r], rows) )

# single column, single row
detrat_from_columns_and_rows_update!{T}(cmat::CeperleyMatrix{T},
                                        c::Int, cols::Matrix{T},
                                        r::Int, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update!(cmat, [c], cols, [r], rows) )
detrat_from_columns_and_rows_update{T}(cmat::CeperleyMatrix{T},
                                        c::Int, cols::Matrix{T},
                                        r::Int, rows::Matrix{T}) = (
    detrat_from_columns_and_rows_update(cmat, [c], cols, [r], rows) )

###########################################################################
#################### Finish updates (compute inverses) ####################
###########################################################################

function finish_columns_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :COLUMNS_UPDATE_PENDING

    cmat.detmat[:, cmat.cs] = cmat.pending_cols

    # Sherman-Morrison-Woodbury
    old_inverse_rows = cmat.inverse[cmat.cs, :]
    cmat.inverse -= (cmat.inverse * cmat.pending_cols) * cmat.Q_inv * cmat.inverse[cmat.cs, :]
    cmat.inverse[cmat.cs, :] = cmat.Q_inv * old_inverse_rows

    cmat.old_det = cmat.det

    cmat.current_state = :READY_FOR_UPDATE

    return cmat
end

function finish_rows_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :ROWS_UPDATE_PENDING

    cmat.detmat[cmat.rs, :] = cmat.pending_rows

    # Sherman-Morrison-Woodbury
    old_inverse_cols = cmat.inverse[:, cmat.rs]
    cmat.inverse -= cmat.inverse[:, cmat.rs] * cmat.Q_inv * (cmat.pending_rows * cmat.inverse)
    cmat.inverse[:, cmat.rs] = old_inverse_cols * cmat.Q_inv

    cmat.old_det = cmat.det

    cmat.current_state = :READY_FOR_UPDATE

    return cmat
end

function finish_columns_and_rows_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :COLUMNS_AND_ROWS_UPDATE_PENDING

    # put the columns down after the rows (see detrat_from_columns_and_rows_update)
    cmat.detmat[cmat.rs, :] = cmat.pending_rows
    cmat.detmat[:, cmat.cs] = cmat.pending_cols

    # Sherman-Morrison-Woodbury
    nc = length(cmat.cs);
    old_inverse = copy(cmat.inverse)
    cmat.inverse -= ( (cmat.inverse * cmat.cols_tilde) * cmat.Q_inv[1:nc, 1:nc] * cmat.inverse[cmat.cs, :]
                     + cmat.inverse[:, cmat.rs] * cmat.Q_inv[nc+1:end, nc+1:end] * (cmat.rows_tilde * cmat.inverse)
                     + cmat.inverse[:, cmat.rs] * cmat.Q_inv[nc+1:end, 1:nc] * cmat.inverse[cmat.cs, :]
                     + (cmat.inverse * cmat.cols_tilde) * cmat.Q_inv[1:nc, nc+1:end] * (cmat.rows_tilde * cmat.inverse) )
    cmat.inverse[cmat.cs, :] = ( cmat.Q_inv[1:nc, 1:nc] * old_inverse[cmat.cs, :]
                               + cmat.Q_inv[1:nc, nc+1:end] * (cmat.rows_tilde * old_inverse) )
    cmat.inverse[:, cmat.rs] = ( old_inverse[:, cmat.rs] * cmat.Q_inv[nc+1:end, nc+1:end]
                               + (old_inverse * cmat.cols_tilde) * cmat.Q_inv[1:nc, nc+1:end] )
    cmat.inverse[cmat.cs, cmat.rs] = -cmat.Q_inv[1:nc, nc+1:end]

    cmat.old_det = cmat.det

    cmat.current_state = :READY_FOR_UPDATE

    return cmat
end

########################################################
#################### Cancel updates ####################
########################################################

function cancel_columns_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :COLUMNS_UPDATE_PENDING
    cmat.det = cmat.old_det
    cmat.current_state = :READY_FOR_UPDATE
    return cmat
end

function cancel_rows_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :ROWS_UPDATE_PENDING
    cmat.det = cmat.old_det
    cmat.current_state = :READY_FOR_UPDATE
    return cmat
end

function cancel_columns_and_rows_update!(cmat::CeperleyMatrix)
    @assert cmat.current_state == :COLUMNS_AND_ROWS_UPDATE_PENDING
    cmat.det = cmat.old_det
    cmat.current_state = :READY_FOR_UPDATE
    return cmat
end


########################################################################
#################### Householder --> exact Pfaffian ####################
########################################################################

is_antisymmetric(A, tol::Float64 = 1e-15) = maximum(abs(A + A.')) < tol

function Householder(x::Vector{Float64})
    absx = norm(x)
    u = copy(x)
    u[1] = x[1] + (x[1] > 0 ? absx : -absx)
    beta = 2 / norm(u)^2
    return u, beta
end

function Householder(x::Vector{Complex128})
    absx = norm(x)
    u = copy(x)
    u1p = x[1] + exp(im*angle(x[1])) * absx
    u1m = x[1] - exp(im*angle(x[1])) * absx
    u[1] = abs(u1p) > abs(u1m) ? u1p : u1m
    beta = 2 / norm(u)^2
    return u, beta
end

function Householder{T}(x::Matrix{T})
    @assert size(x)[2] == 1 || size(x)[1] == 1
    return Householder(vec(x))
end

# FIXME: this implementation is very slow/naive -- use Pfaffian.jl instead
function Pf{T}(A::Matrix{T})
    @assert size(A)[1] == size(A)[2]
    @assert is_antisymmetric(A)
    N = size(A)[1]

    if mod(N, 2) == 1
        return 0.
    end

    for i in 1:2:N-2 # only need to do every other row/col to get actual Pfaffian (see sign in return statement)
        u, beta = Householder(A[i, i+1:end])
        u = [zeros(T, i); u]
        p = beta * A * conj(u)
        K = 0.5beta * (u' * p)[1]
        A = A - p * u.' + u * p.' + 2K * u * u.'
        # @assert is_antisymmetric(A) # this is too harsh
    end

    # = sign from Householder determinants * product of upper diagonal, with only every other element included
    return (-1.)^((N-2)/2) * prod(diag(A, 1)[1:2:end])
end


################################################################
#################### Pfaffian wavefunctions ####################
################################################################

# Pfaffian <--> Ceperley
# pfaff <--> det
# pf <--> c

type PfaffianMatrix{T} #<: VMCMatrix{T}
    current_state::Symbol

    Amat::Matrix{T}
    inverse::Matrix{T}

    # pfaff::T
    pfaffrat::T
    # Q_inv::Matrix{T}

    # pending index (only have single row/column complete updates currently available)
    ks::Vector{Int}

    # pending row/col (only have single row/column complete updates currently available)
    pending_rows::Matrix{T}

    PfaffianMatrix() = new(:UNINITIALIZED)

    function PfaffianMatrix(A::Matrix{T})
        @assert size(A)[1] == size(A)[2]
        @assert is_antisymmetric(A)
        return new(:READY_FOR_UPDATE, A, inv(A))
    end
end

PfaffianMatrix(A::Matrix) = PfaffianMatrix{eltype(A)}(A)
PfaffianMatrix() = PfaffianMatrix{Float64}() # default to Float64

# FIXME: Can we have something like a VMCMatrix abstract base class,
# so we don't need to redefine the following functions from their Ceperley counterparts?

# check inverse for numerical error due to SMW
function check_inverse{T}(pfmat::PfaffianMatrix{T}, tol::Float64 = 1e-10)
    @assert pfmat.current_state == :READY_FOR_UPDATE

    error = sum(abs(pfmat.Amat * pfmat.inverse - eye(T, size(pfmat.Amat)[1]))) / size(pfmat.Amat)[1]
    is_good = error < tol

    return is_good, error
end

function fix_inverse!(pfmat::PfaffianMatrix)
    pfmat.inverse = inv(pfmat.Amat)
    return pfmat
end

function get_pfaffrat(pfmat::PfaffianMatrix)
    @assert pfmat.current_state == :UPDATE_PENDING
    return pfmat.pfaffrat
end

###########################################################################
#################### Start updates (compute pfaffrats) ####################
###########################################################################

# NB: In the pfaffrat_from_* functions, we keep the type parameter in the definition to assert that the
# type of the input PfaffianMatrix object and the cols / rows Matrix objects have the same type T.
# NB: Below, only ever actually use row updates, since antisymmetry makes row and column updates equivalent.

# single row
function pfaffrat_from_rows_update!{T}(pfmat::PfaffianMatrix{T}, k::Int, row::Matrix{T})
    @assert size(row)[1] == 1
    @assert row[1, k] == 0 # so the new A can be antisymmetric
    @assert k <= size(pfmat.Amat)[1]
    @assert size(row)[2] == size(pfmat.Amat)[2]
    @assert pfmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    pfmat.pfaffrat = (row * pfmat.inverse[:, k])[1]

    pfmat.ks = [k]
    pfmat.pending_rows = row
    pfmat.current_state = :UPDATE_PENDING

    return pfmat
end

# single row (no bang)
function pfaffrat_from_rows_update{T}(pfmat::PfaffianMatrix{T}, k::Int, row::Matrix{T})
    @assert size(row)[1] == 1
    @assert row[1, k] == 0 # so the new A can be antisymmetric
    @assert k <= size(pfmat.Amat)[1]
    @assert size(row)[2] == size(pfmat.Amat)[2]
    @assert pfmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    return (row * pfmat.inverse[:, k])[1]
end

# 2 rows (no bang)
function pfaffrat_from_rows_update{T}(pfmat::PfaffianMatrix{T}, ks::Vector{Int}, rows::Matrix{T})
    @assert length(ks) == 2
    @assert size(rows)[1] == 2
    @assert rows[1, ks[1]] == rows[2, ks[2]] == 0 # so the new A can be antisymmetric
    @assert maximum(ks) <= size(pfmat.Amat)[1]
    @assert size(rows)[2] == size(pfmat.Amat)[2]
    @assert pfmat.current_state == :READY_FOR_UPDATE

    # matrix determinant lemma
    return ( rows[2, ks[1]] * pfmat.inverse[ks[1], ks[2]]
          + (rows[2, :] * pfmat.inverse[:, ks[2]])[1] * (rows[1, :] * pfmat.inverse[:, ks[1]])[1]
          - (rows[2, :] * pfmat.inverse[:, ks[1]])[1] * (rows[1, :] * pfmat.inverse[:, ks[2]])[1]
          + pfmat.inverse[ks[1], ks[2]] * (rows[2, :] * pfmat.inverse * rows[1, :].')[1] )
          # note sign/transpose on last factor of last term (see notes)
end

# single column
function pfaffrat_from_columns_update!{T}(pfmat::PfaffianMatrix{T}, k::Int, col::Matrix{T})
    return pfaffrat_from_rows_update!(pfmat, k, -col.')
end

# single column (no bang)
function pfaffrat_from_columns_update{T}(pfmat::PfaffianMatrix{T}, k::Int, col::Matrix{T})
    return pfaffrat_from_rows_update(pfmat, k, -col.')
end

# 2 columns (no bang)
function pfaffrat_from_columns_update{T}(pfmat::PfaffianMatrix{T}, ks::Vector{Int}, cols::Matrix{T})
    return pfaffrat_from_rows_update(pfmat, ks, -cols.')
end

###########################################################################
#################### Finish updates (compute inverses) ####################
###########################################################################

# only single row/col currently available
function finish_update!(pfmat::PfaffianMatrix)
    @assert pfmat.current_state == :UPDATE_PENDING

    pfmat.Amat[pfmat.ks, :] = pfmat.pending_rows
    pfmat.Amat[:, pfmat.ks] = -pfmat.pending_rows.'

    # Sherman-Morrison-Woodbury
    old_inverse_rows = pfmat.inverse[pfmat.ks, :]
    pfmat.inverse -= ( (pfmat.inverse * -pfmat.pending_rows.') * pfmat.inverse[pfmat.ks, :]
                      + pfmat.inverse[:, ks] * (pfmat.pending_rows * pfmat.inverse) ) / pfmat.pfaffrat
    pfmat.inverse[ks, :] = old_inverse_rows / pfmat.pfaffrat
    pfmat.inverse[:, ks] = -pfmat.inverse[ks, :].'
    pfmat.inverse[ks, ks] = 0

    pfmat.current_state = :READY_FOR_UPDATE

    return pfmat
end

########################################################
#################### Cancel updates ####################
########################################################

function cancel_update!(pfmat::PfaffianMatrix)
    @assert pfmat.current_state == :UPDATE_PENDING
    pfmat.current_state = :READY_FOR_UPDATE
    return pfmat
end
