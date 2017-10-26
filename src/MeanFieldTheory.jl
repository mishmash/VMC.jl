# include("Configuration.jl")
# include("Lattice.jl")

###############################################################
#################### Free fermion orbitals ####################
###############################################################

immutable FreeFermionOrbitals{T}
    all::Matrix{T}
    filled::Matrix{T}

    energies::Vector{Float64}
    Fermi_energy::Float64

    no_degeneracies_at_FS::Bool
    degeneracy_tolerance::Float64

    function FreeFermionOrbitals(Hami::Matrix{T}, Nfilled::Int, deg_tol::Float64 = 1e-12)
        @assert ishermitian(Hami)
        @assert Nfilled <= size(Hami)[1]
        eig_data = eigfact(Hami)
        if Nfilled == size(Hami)[1]
            no_deg = true
            warn("Filling up all possible states. Have fun.")
        else
            # printlnf(eig_data[:values][Nfilled+1] - eig_data[:values][Nfilled] > deg_tol)
            if eig_data[:values][Nfilled+1] - eig_data[:values][Nfilled] > deg_tol
                no_deg = true
            else
                no_deg = false
                warn("Degeneracies exist at the Fermi surface. Proceed with caution!")
            end
        end
        return new(eig_data[:vectors],
                   eig_data[:vectors][:, 1:Nfilled],
                   eig_data[:values],
                   eig_data[:values][Nfilled],
                   no_deg, deg_tol)
    end
end

FreeFermionOrbitals(Hami::Matrix, Nfilled::Int) = FreeFermionOrbitals{eltype(Hami)}(Hami, Nfilled)

get_Nsites(orbs::FreeFermionOrbitals) = size(orbs.filled)[1]
get_Nfilled(orbs::FreeFermionOrbitals) = size(orbs.filled)[2]
no_degeneracies_at_FS(orbs::FreeFermionOrbitals) = orbs.no_degeneracies_at_FS

function fill_detmat(orbs::FreeFermionOrbitals, config::LatticeParticlesConfiguration, species::Int = 1)
    @assert get_Nsites(orbs) == get_Nsites(config)
    @assert species <= get_Nspecies(config)
    @assert get_Nfilled(orbs) == get_Nparticles(config)[species]

    detmat = transpose(orbs.filled[config.particle_positions[species], :])

    return detmat
end

function fill_detmats{T}(orbs::FreeFermionOrbitals{T}, config::LatticeParticlesConfiguration)
    detmats = Matrix{T}[]
    for sp in 1:get_Nspecies(config)
        push!(detmats, fill_detmat(orbs, config, sp))
    end
    return detmats
end

function columns_for_detmat(orbs::FreeFermionOrbitals, positions::Vector{Int})
    @assert all(positions .<= get_Nsites(orbs))
    return transpose(orbs.filled[positions, :])
end

columns_for_detmat(orbs::FreeFermionOrbitals, position::Int) = columns_for_detmat(orbs, [position])

# construct free fermion Green's function as the single-particle density matrix corresponding to a filled Fermi sea
function single_particle_density_matrix{T}(orbs::FreeFermionOrbitals{T})
    # spdm = zeros(T, (get_Nsites(orbs), get_Nsites(orbs)))
    # for i in 1:get_Nfilled(orbs)
    #     spdm = spdm + orbs.filled[:, i] * orbs.filled[:, i]'
    # end
    # @assert ishermitian(spdm)
    # NB: the above lines are slow, but emphasize how the SPDM is defined
    spdm = orbs.filled * orbs.filled' # this is obviously much faster
    return transpose(spdm) # NB: transpose because spdm_{ij} = <c_j^dagger c_i>; we want to return <c_i^dagger c_j>
end

# local density in real space
local_density(orbs::FreeFermionOrbitals) = real(diag(single_particle_density_matrix(orbs)))

# compute density-density correlation function <n_i n_j> using Wick's theorem
function density_density_correlation_function(orbs::FreeFermionOrbitals)
    spdm = single_particle_density_matrix(orbs)
    ni = local_density(orbs)
    return real(-abs(spdm).^2 + diagm(ni) + ni * ni.')
end

# core function in AnalysisTools.jl
function local_particle_number_variance(orbs::FreeFermionOrbitals, sites_per_region::Int)
    ninj = density_density_correlation_function(orbs)
    ni = local_density(orbs)
    return local_particle_number_variance(ninj, ni, sites_per_region)
end

# core function in AnalysisTools.jl
function bipartite_number_fluctuations(orbs::FreeFermionOrbitals, sites_added_per_cut::Int)
    ninj = density_density_correlation_function(orbs)
    ni = local_density(orbs)
    return bipartite_number_fluctuations(ninj, ni, sites_added_per_cut)
end

# calculate von Neumann entanglement entropy using correlation matrix technique for many sequential cuts through a system
function bipartite_vonNeumann_entanglement_entropy(orbs::FreeFermionOrbitals, sites_added_per_cut::Int) # = garbage
    @assert sites_added_per_cut < get_Nsites(orbs)

    spdm = single_particle_density_matrix(orbs)

    if mod(get_Nsites(orbs), sites_added_per_cut) == 0
        numcuts = convert(Int, get_Nsites(orbs)/sites_added_per_cut - 1)
    else
        numcuts = convert(Int, floor(get_Nsites(orbs)/sites_added_per_cut))
    end

    S1 = zeros(Float64, (numcuts, 2))
    for x in 1:numcuts
        sites_in_partition = sites_added_per_cut * x
        all_evals = eigfact(spdm[1:sites_in_partition, 1:sites_in_partition])[:values]
        good_evals = all_evals[find(0. .< all_evals .< 1.)]
        S1[x, 1] = sites_in_partition
        S1[x, 2] = -(good_evals.' * log(good_evals) + (1. - good_evals.') * log(1. - good_evals))[1]
    end

    return S1
end

function Renyi_formula(cs::Vector{Float64}, alpha::Int)
    @assert minimum(cs) > 0. && maximum(cs) < 1.

    if alpha == 1
        # S = -((1. - cs.') * log(1. - cs) + cs.' * log(cs))[1]
        S = -((1. - cs.') * log1p(-cs) + cs.' * log(cs))[1]
    elseif alpha == 2
        S = -sum(log((1. - cs).^2 + cs.^2))
        # S = -sum(log1p(2 * cs .* (cs - 1.)))
    elseif alpha >= 3
        S = 1./(1. - alpha) * sum(log((1. - cs).^alpha + cs.^alpha))
    else
        error("Renyi_formula requires alpha > 0.")
    end

    return S
end

function bipartite_entanglement_entropy(orbs::FreeFermionOrbitals, subsystem::Vector{Int}, alpha::Int)
    @assert 0 <= length(subsystem) <= get_Nsites(orbs)
    if length(subsystem) == 0 || length(subsystem) == get_Nsites(orbs)
        return 0.
    end
    @assert minimum(subsystem) >= 1 && maximum(subsystem) <= get_Nsites(orbs)

    C = single_particle_density_matrix(orbs)[subsystem, subsystem] # = correlation matrix
    cs = eigfact(C)[:values]
    cs = cs[find(0. .< cs .< 1.)] # good evals

    return Renyi_formula(cs, alpha)
end

function bipartite_entanglement_entropy(spdm::Matrix, subsystem::Vector{Int}, alpha::Int)
    @assert ishermitian(spdm)
    Nsites = size(spdm)[1]
    @assert 0 <= length(subsystem) <= Nsites
    if length(subsystem) == 0 || length(subsystem) == Nsites
        return 0.
    end
    @assert minimum(subsystem) >= 1 && maximum(subsystem) <= Nsites

    C = spdm[subsystem, subsystem]
    cs = eigfact(C)[:values]
    cs = cs[find(0. .< cs .< 1.)] # good evals

    return Renyi_formula(cs, alpha)
end

##########################################################
#################### Spinful BCS data ####################
##########################################################

immutable SpinfulBCS{T}
    evecs::Matrix{T}
    U::Matrix{T}
    V::Matrix{T}

    energies::Vector{Float64}

    function SpinfulBCS(t::Matrix, Delta::Matrix)
        # both t and Delta should include on-site terms, e.g., t_{ij} = ttilde_{ij} = t_{ij} + mu_i * krondelta_{ij} from notes
        @assert ishermitian(t)
        @assert issym(Delta)
        @assert size(t)[1] == size(Delta)[1]

        Nsites = size(t)[1]

        H = zeros(T, (2Nsites, 2Nsites))
        H[1:Nsites, 1:Nsites] = -t
        H[Nsites+1:2Nsites, Nsites+1:2Nsites] = conj(t) # = t.'
        H[1:Nsites, Nsites+1:2Nsites] = Delta
        H[Nsites+1:2Nsites, 1:Nsites] = conj(Delta)
        @assert ishermitian(H)

        eig_data = eigfact(H)
        energies = eig_data[:values]

        # check that energies come in plus-minus pairs
        @assert maxabs(flipdim(energies[1:Nsites], 1) + energies[Nsites+1:2Nsites]) < 1e-12

        evecs = eig_data[:vectors]

        # U and V are defined by the positive-energy eigenvectors (see notes)
        U = evecs[1:Nsites, Nsites+1:2Nsites]
        V = evecs[Nsites+1:2Nsites, Nsites+1:2Nsites]

        # check that (v, -u)^* are indeed the correct negative-energy eigenvectors
        for j in 1:Nsites
            evec_test = vcat(conj(V[:, j]), -conj(U[:, j]))
            @assert (maxabs(H * evec_test - energies[Nsites-j+1] * evec_test) < 1e-12) "$(j) is bad"
        end

        return new(evecs, U, V, energies)
    end
end

SpinfulBCS(t::Matrix, Delta::Matrix) = SpinfulBCS{promote_type(eltype(t), eltype(Delta))}(t, Delta)

get_Nsites(bcs::SpinfulBCS) = size(bcs.U)[1]

function fill_detmat_for_SL{T}(bcs::SpinfulBCS{T}, config::LatticeParticlesConfiguration)
    @assert get_Nsites(bcs) == get_Nsites(config)
    Nsites = get_Nsites(bcs)
    @assert Nsites/2 == get_Nparticles(config)[1] # hard-core bosons are at half-filling

    detmat = zeros(T, (Nsites, Nsites))
    detmat[:, 1:2:Nsites-1] = transpose(bcs.evecs[config.particle_positions[1], 1:Nsites])
    detmat[:, 2:2:Nsites] = transpose(bcs.evecs[Nsites + config.particle_positions[1], 1:Nsites])

    return detmat
end

function columns_for_detmat_for_SL{T}(bcs::SpinfulBCS{T}, positions::Vector{Int})
    @assert all(positions .<= get_Nsites(bcs))
    Nsites = get_Nsites(bcs)

    cols = zeros(T, (Nsites, 2*length(positions)))
    cols[:, 1:2:2*length(positions)-1] = transpose(bcs.evecs[positions, 1:Nsites])
    cols[:, 2:2:2*length(positions)] = transpose(bcs.evecs[Nsites + positions, 1:Nsites])

    return cols
end

columns_for_detmat_for_SL(bcs::SpinfulBCS, position::Int) = columns_for_detmat_for_SL(bcs, [position])

#######################################################
#################### Hopping Hamis ####################
#######################################################

function zigzag_strip_hopping_Hami(t1::Complex128, t2::Complex128, mu::Float64, BCphase::Complex128, Lx::Int)
    xhat = unitx(Lx, 1)
    Nsites = Lx
    tmat = zeros(Complex128, (Nsites, Nsites))

    # bulk sites t1
    for i in 1:Nsites-1
        tmat[i, (Site2D(i, Lx, 1) + xhat).index] = t1
    end
    # end site t1
    tmat[Nsites, (Site2D(Nsites, Lx, 1) + xhat).index] = BCphase * t1

    # bulk sites t2
    for i in 1:Nsites-2
        tmat[i, (Site2D(i, Lx, 1) + 2xhat).index] = t2
    end
    # end sites t2
    for i in Nsites-1:Nsites
        tmat[i, (Site2D(i, Lx, 1) + 2xhat).index] = BCphase * t2
    end

    tmat = tmat + tmat'

    for i in 1:Nsites
        tmat[i, i] = mu
    end

    H = -tmat

    # convert to a real matrix if we can
    return imag(H) == zeros(size(H)...) ? real(H) : H
end

# the argument n::Int determines the flux per triangle consistent with the kx preserving gauge
function uniform_flux_triangular_ladder_kx_gauge_hopping_Hami(tx::Float64, ty::Float64, td::Float64, n::Int, BCxphase::Number, Lx::Int, Ly::Int)
    # BCxphase should either be of unit modulus or zero (for OBCs)
    # I think having BCxphase::Number in the argument list is good (so it can be either real or complex)
    # @assert abs(BCxphase) == 1 || BCxphase == 0
    # Eh, maybe not.  Perhaps BCxphase should just be renamed BCxfac or something..

    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Ly * Lx
    phi = pi * n / Ly
    tmat = zeros(Complex128, (Nsites, Nsites))

    for i in 1:Nsites
        if Site2D(i, Lx, Ly).x < Lx-1
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = tx * exp(-im * 2 * Site2D(i, Lx, Ly).y * phi)
            tmat[i, (Site2D(i, Lx, Ly) + xhat + yhat).index] = td * exp(-im * (2 * Site2D(i, Lx, Ly).y + 1) * phi)
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = ty
        elseif Site2D(i, Lx, Ly).x == Lx-1
            # terms corresponding to the next two lines cross the vertical boundary at the end of the sample
            # NB: collectively, they don't change the flux piercing the corresponding plaquettes
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = BCxphase * tx * exp(-im * 2 * Site2D(i, Lx, Ly).y * phi)
            tmat[i, (Site2D(i, Lx, Ly) + xhat + yhat).index] = BCxphase * td * exp(-im * (2 * Site2D(i, Lx, Ly).y + 1) * phi)
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = ty
        else
            @assert false
        end
    end

    tmat = tmat + tmat'
    H = -tmat

    # convert to a real matrix if we can
    return imag(H) == zeros(size(H)...) ? real(H) : H
end

# allows for different BCphase factors in both the x and y directions
function uniform_flux_triangular_lattice_kx_gauge_hopping_Hami(tx::Float64, ty::Float64, td::Float64, n::Int,
                                                               BCxphase::Number, BCyphase::Number, Lx::Int, Ly::Int)
# FIXME: build it
end

function kagome_strip_hopping_Hami(tleg::Float64, tcross::Float64, muleg::Float64, mumid::Float64, phi::Float64, BCxfac::Number, Lx::Int)
    @assert abs(BCxfac) == 1 || BCxfac == 0

    Ly = 3
    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Ly * Lx
    tmat = zeros(Complex128, (Nsites, Nsites))

    for i in 1:Nsites
        if Site2D(i, Lx, Ly).x < Lx-1
            if Site2D(i, Lx, Ly).y == 0
                tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = tleg * exp(-im * phi)
                tmat[i, (Site2D(i, Lx, Ly) + xhat + yhat).index] = tcross
                tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = tcross * exp(im * phi)
            elseif Site2D(i, Lx, Ly).y == 1
                tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = tcross
            elseif Site2D(i, Lx, Ly).y == 2
                tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = tleg
                tmat[i, (Site2D(i, Lx, Ly) + xhat - yhat).index] = tcross
            else
                @assert false
            end
        elseif Site2D(i, Lx, Ly).x == Lx-1
            if Site2D(i, Lx, Ly).y == 0
                tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = BCxfac * tleg * exp(-im * phi)
                tmat[i, (Site2D(i, Lx, Ly) + xhat + yhat).index] = BCxfac * tcross
                tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = tcross * exp(im * phi)
            elseif Site2D(i, Lx, Ly).y == 1
                tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = tcross
            elseif Site2D(i, Lx, Ly).y == 2
                tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = BCxfac * tleg
                tmat[i, (Site2D(i, Lx, Ly) + xhat - yhat).index] = BCxfac * tcross
            else
                @assert false
            end
        else
            @assert false
        end
    end

    tmat = tmat + tmat'

    for i in 1:Nsites
        if Site2D(i, Lx, Ly).y == 0
            tmat[i, i] = muleg
        elseif Site2D(i, Lx, Ly).y == 1
            tmat[i, i] = mumid
        elseif Site2D(i, Lx, Ly).y == 2
            tmat[i, i] = muleg
        else
            @assert false
        end
    end

    H = -tmat

    # convert to a real matrix if we can
    return imag(H) == zeros(size(H)...) ? real(H) : H
end

function no_flux_square_lattice_hopping_Hami(tx::Float64, ty::Float64, BCxfac::Number, BCyfac::Number, Lx::Int, Ly::Int)
    @assert abs(BCxfac) == 1 || BCxfac == 0
    @assert abs(BCyfac) == 1 || BCyfac == 0

    xhat = unitx(Lx, Ly)
    yhat = unity(Lx, Ly)
    Nsites = Ly * Lx
    tmat = zeros(Complex128, (Nsites, Nsites))

    for i in 1:Nsites
        if Site2D(i, Lx, Ly).x < Lx-1 && Site2D(i, Lx, Ly).y < Ly-1
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = tx
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = ty
        elseif Site2D(i, Lx, Ly).x < Lx-1 && Site2D(i, Lx, Ly).y == Ly-1
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = tx
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = BCyfac * ty
        elseif Site2D(i, Lx, Ly).x == Lx-1 && Site2D(i, Lx, Ly).y < Ly-1
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = BCxfac * tx
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = ty
        elseif Site2D(i, Lx, Ly).x == Lx-1 && Site2D(i, Lx, Ly).y == Ly-1
            tmat[i, (Site2D(i, Lx, Ly) + xhat).index] = BCxfac * tx
            tmat[i, (Site2D(i, Lx, Ly) + yhat).index] = BCyfac * ty
        else
            @assert false
        end
    end

    tmat = tmat + tmat'
    H = -tmat

    # convert to a real matrix if we can
    return imag(H) == zeros(size(H)...) ? real(H) : H
end
