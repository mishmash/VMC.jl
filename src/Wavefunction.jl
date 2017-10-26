# include("MeanFieldTheory.jl")
# include("Configuration.jl")
# include("VMCMatrix.jl")
# include("Random.jl")
# include("Move.jl")
# include("delegate.jl")

# having the type parameter here is actually pretty stupid, since some wavefunctions can have multiple types (e.g., CFL)
abstract Wavefunction

# spinless fermions or hard-core bosons:
abstract HardcoreWavefunction <: Wavefunction

################################################################
#################### Free spinless fermions ####################
################################################################

type FreeFermionWavefunction{T} <: HardcoreWavefunction
    orbitals::FreeFermionOrbitals{T}
    config::LatticeParticlesConfiguration
    cmat::CeperleyMatrix{T}
    neighbor_table::Vector{Vector{Int}}
    proposed_move::ParticlesMove{LatticeParticleMove}

    function FreeFermionWavefunction(orbs::FreeFermionOrbitals{T}, nbs::Vector{Vector{Int}})
        Nsites = get_Nsites(orbs)
        Nparticles = get_Nfilled(orbs)
        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        positions = random_combination(Nsites, Nparticles)
        config = LatticeParticlesConfiguration(Vector{Int}[positions], Nsites)

        cmat = CeperleyMatrix(fill_detmat(orbs, config))

        return new(orbs, config, cmat, nbs)
    end
end

function FreeFermionWavefunction{T}(orbs::FreeFermionOrbitals{T}, nbs::Vector{Vector{Int}})
    return FreeFermionWavefunction{eltype(orbs.filled)}(orbs, nbs)
end

function FreeFermionWavefunction{T}(orbs::FreeFermionOrbitals{T})
    return FreeFermionWavefunction{eltype(orbs.filled)}(orbs, Vector{Int}[])
end

get_Nspecies(wf::HardcoreWavefunction) = 1 # I think this is what we want
get_Nparticles(wf::HardcoreWavefunction) = get_Nparticles(wf.config)[1]
get_Nsites(wf::HardcoreWavefunction) = get_Nsites(wf.config)

function check_for_numerical_error!(wf::FreeFermionWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat, "cmat")
end

function random_empty_site_move(wf::HardcoreWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    Nempty = get_Nsites(wf) - get_Nparticles(wf)
    proposed_position = setdiff(collect(1:get_Nsites(wf)), wf.config.particle_positions[1])[rand(1:Nempty)]

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function random_neighbor_move(wf::HardcoreWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config.particle_positions[1][particle_to_move.index]
    proposed_position = wf.neighbor_table[current_site][rand(1:end)]

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function assess_move!(wf::FreeFermionWavefunction, move::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(move, wf)

    @assert length(move) == 1 # for now
    if is_occupied(wf.config, move[1].destination, 1)
        # finish_move! could technically still be called after this (i.e., we could try to accept the move),
        # but in practice it never will since probrat = 0.
        # This logic could afford to be a little tighter however..
        wf.cmat.current_state = :COLUMNS_UPDATE_PENDING
        probrat = 0.
    else
        detrat_from_columns_update!(wf.cmat, move[1].particle.index,
            columns_for_detmat(wf.orbitals, move[1].destination))
        probrat = abs2(get_detrat(wf.cmat))
    end

    # nAtest = 4
    # subsystemA = collect(1:4)
    # if get_Nparticles_in_subsystem(update_configuration(wf.config, move), subsystemA)[1] == nAtest
    #     printlnf("Proposing a move with $nAtest particles in subsystem A:  probrat = $probrat")
    # end

    wf.proposed_move = move

    return probrat
end

# FIXME: this actually returns a CeperleyMatrix object..
# Does this then have to be garbage collected etc., leading to slow-down?
# Issue probably crops up everywhere..
function finish_move!(wf::FreeFermionWavefunction)
    update_configuration!(wf.config, wf.proposed_move)
    finish_columns_update!(wf.cmat)
end

function cancel_move!(wf::FreeFermionWavefunction)
    cancel_columns_update!(wf.cmat)
end

##########################################################
#################### SBM wavefunction ####################
##########################################################

type SBMWavefunction{T} <: Wavefunction
    orbitals_up::FreeFermionOrbitals{T}
    orbitals_down::FreeFermionOrbitals{T}

    config::LatticeParticlesConfiguration

    cmat_up::CeperleyMatrix{T}
    cmat_down::CeperleyMatrix{T}

    neighbor_table::Vector{Vector{Int}} # this could be a Vector of neighbor tables for different neighbors

    proposed_move::ParticlesMove{LatticeParticleMove} # empty upon initialization

    function SBMWavefunction(orbs_up::FreeFermionOrbitals{T}, orbs_down::FreeFermionOrbitals{T}, nbs::Vector{Vector{Int}})
        @assert get_Nsites(orbs_up) == get_Nsites(orbs_down)
        Nsites = get_Nsites(orbs_up)
        @assert get_Nfilled(orbs_up) + get_Nfilled(orbs_down) == Nsites # half filling
        if !isempty(nbs)
            @assert length(nbs) == Nsites
        end

        # get random spin configuration
        positions_up = random_combination(get_Nsites(orbs_up), get_Nfilled(orbs_up))
        config = GutzSpinhalfConfiguration(positions_up, get_Nsites(orbs_up))
        @assert is_valid_Gutz_spinhalf_configuration(config) # being anal

        # compute Ceperley matrices
        detmat_up = fill_detmat(orbs_up, config, 1)
        detmat_down = fill_detmat(orbs_down, config, 2)
        cmat_up = CeperleyMatrix(detmat_up)
        cmat_down = CeperleyMatrix(detmat_down)

        return new(orbs_up, orbs_down, config, cmat_up, cmat_down, nbs)
    end
end

# This is annoying, but now I think I understand the issue:
# After defining a parameterized type, no constructors are available without type specification (in the call),
# e.g., mytype(...) is not an available method, so we need to define it as such with the type specified on the RHS.
# Also, presence of the type parameter T in this "outer constructor" ensures commonality amongst
# the orbital types of the different partons.
function SBMWavefunction{T}(orbs_up::FreeFermionOrbitals{T}, orbs_down::FreeFermionOrbitals{T}, nbs::Vector{Vector{Int}})
    return SBMWavefunction{eltype(orbs_up.filled)}(orbs_up, orbs_down, nbs)
end

# if not included in argument list, set neighbor_table to an empty array
function SBMWavefunction{T}(orbs_up::FreeFermionOrbitals{T}, orbs_down::FreeFermionOrbitals{T})
    return SBMWavefunction(orbs_up, orbs_down, Vector{Int}[])
end

@delegate SBMWavefunction.config [ is_valid_Gutz_spinhalf_configuration ]
export is_valid_Gutz_spinhalf_configuration

get_Nspecies(wf::SBMWavefunction) = 2
get_Nparticles(wf::SBMWavefunction) = get_Nparticles(wf.config)
get_Nup(wf::SBMWavefunction) = get_Nparticles(wf)[1]
get_Ndown(wf::SBMWavefunction) = get_Nparticles(wf)[2]
get_Nsites(wf::SBMWavefunction) = get_Nsites(wf.config)

function check_for_numerical_error!(wf::SBMWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_up, "cmat_up")
    check_and_reset_inverse_and_det!(wf.cmat_down, "cmat_down")
end

function random_exchange_move(wf::SBMWavefunction)
    @assert is_valid_Gutz_spinhalf_configuration(wf)

    particle_to_move_up = Particle(rand(1:get_Nup(wf)), 1)
    particle_to_move_down = Particle(rand(1:get_Ndown(wf)), 2)

    proposed_position_up = wf.config.particle_positions[2][particle_to_move_down.index] # current position of down particle
    proposed_position_down = wf.config.particle_positions[1][particle_to_move_up.index] # current position of up particle

    mv_up = LatticeParticleMove(particle_to_move_up, proposed_position_up)
    mv_down = LatticeParticleMove(particle_to_move_down, proposed_position_down)

    return ParticlesMove([mv_up, mv_down])
end

function neighbor_exchange_move(wf::SBMWavefunction)
    @assert is_valid_Gutz_spinhalf_configuration(wf)

    random_site = rand(1:get_Nsites(wf))
    random_neighbor = wf.neighbor_table[random_site][rand(1:end)]

    # printlnf("random_site = " * string(random_site))
    # printlnf("random_neighbor = " * string(random_neighbor))

    if is_occupied(wf.config, random_site, 1) && is_occupied(wf.config, random_neighbor, 1) ||
        is_occupied(wf.config, random_site, 2) && is_occupied(wf.config, random_neighbor, 2)
        return ParticlesMove{LatticeParticleMove}() # spins are parallel, so move is trivial
    else
        if is_occupied(wf.config, random_site, 1) # up particle at random_site
            @assert is_occupied(wf.config, random_neighbor, 2) # down particle at random_neighbor
            particle_to_move_up = Particle(wf.config.lattice_status[1][random_site], 1)
            proposed_position_up = random_neighbor
            particle_to_move_down = Particle(wf.config.lattice_status[2][random_neighbor], 2)
            proposed_position_down = random_site
        elseif is_occupied(wf.config, random_site, 2) # down particle at random_site
            @assert is_occupied(wf.config, random_neighbor, 1) # up particle at random_neighbor
            particle_to_move_up = Particle(wf.config.lattice_status[1][random_neighbor], 1)
            proposed_position_up = random_site
            particle_to_move_down = Particle(wf.config.lattice_status[2][random_site], 2)
            proposed_position_down = random_neighbor
        else
            @assert false
        end
        mv_up = LatticeParticleMove(particle_to_move_up, proposed_position_up)
        mv_down = LatticeParticleMove(particle_to_move_down, proposed_position_down)
        return ParticlesMove([mv_up, mv_down])
    end
end

# similar to Jim's perform_move functions
function assess_move!(wf::SBMWavefunction, move::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(move, wf)

    detrat_from_columns_update!(wf.cmat_up, move[1].particle.index,
        columns_for_detmat(wf.orbitals_up, move[1].destination))
    detrat_from_columns_update!(wf.cmat_down, move[2].particle.index,
        columns_for_detmat(wf.orbitals_down, move[2].destination))

    wf.proposed_move = move

    probrat_up = abs2(get_detrat(wf.cmat_up))
    probrat_down = abs2(get_detrat(wf.cmat_down))
    return probrat_up * probrat_down
end

function finish_move!(wf::SBMWavefunction)
    update_configuration!(wf.config, wf.proposed_move)
    finish_columns_update!(wf.cmat_up)
    finish_columns_update!(wf.cmat_down)
end

function cancel_move!(wf::SBMWavefunction)
    cancel_columns_update!(wf.cmat_up)
    cancel_columns_update!(wf.cmat_down)
end

#########################################################################
#################### Boson CFL = det(nu=1) * det(FS) ####################
#########################################################################

type BosonCFLWavefunction{Td,Tf} <: HardcoreWavefunction
    orbitals_d::FreeFermionOrbitals{Td}
    orbitals_f::FreeFermionOrbitals{Tf}

    config::LatticeParticlesConfiguration

    cmat_d::CeperleyMatrix{Td}
    cmat_f::CeperleyMatrix{Tf}

    neighbor_table::Vector{Vector{Int}} # this could be a Vector of neighbor tables for different neighbors

    proposed_move::ParticlesMove{LatticeParticleMove} # empty upon initialization

    function BosonCFLWavefunction(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf},
                                  nbs::Vector{Vector{Int}}, Ly_for_init::Int)
        @assert Ly_for_init >= 0
        @assert get_Nsites(orbs_d) == get_Nsites(orbs_f)
        @assert get_Nfilled(orbs_d) == get_Nfilled(orbs_f)
        Nsites = get_Nsites(orbs_d)
        Nparticles = get_Nfilled(orbs_d)
        @assert Nparticles <= Nsites
        if !isempty(nbs)
            @assert length(nbs) == Nsites
        end

        # get initial configuration of bosons
        if Ly_for_init == 0
            positions = random_combination(Nsites, Nparticles)
            config = LatticeParticlesConfiguration(Vector{Int}[positions], Nsites)
        else
            # put one particle per rung sequentially until we run out of particles
        end

        # compute Ceperley matrices
        cmat_d = CeperleyMatrix(fill_detmat(orbs_d, config))
        cmat_f = CeperleyMatrix(fill_detmat(orbs_f, config))

        return new(orbs_d, orbs_f, config, cmat_d, cmat_f, nbs)
    end
end

function BosonCFLWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf},
                                     nbs::Vector{Vector{Int}}, Ly_for_init::Int)
    return BosonCFLWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, nbs, Ly_for_init)
end

function BosonCFLWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf}, nbs::Vector{Vector{Int}})
    return BosonCFLWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, nbs, 0)
end

function BosonCFLWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf}, Ly_for_init::Int)
    return BosonCFLWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, Vector{Int}[], Ly_for_init)
end

function BosonCFLWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf})
    return BosonCFLWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, Vector{Int}[], 0)
end

function check_for_numerical_error!(wf::BosonCFLWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d, "cmat_d")
    check_and_reset_inverse_and_det!(wf.cmat_f, "cmat_f")
end

function assess_move!(wf::BosonCFLWavefunction, move::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(move, wf)

    @assert length(move) == 1 # for now
    if is_occupied(wf.config, move[1].destination, 1)
        wf.cmat_d.current_state = :COLUMNS_UPDATE_PENDING
        wf.cmat_f.current_state = :COLUMNS_UPDATE_PENDING
        probrat = 0.
    else
        detrat_from_columns_update!(wf.cmat_d, move[1].particle.index,
            columns_for_detmat(wf.orbitals_d, move[1].destination))
        detrat_from_columns_update!(wf.cmat_f, move[1].particle.index,
            columns_for_detmat(wf.orbitals_f, move[1].destination))
        probrat = abs2(get_detrat(wf.cmat_d)) * abs2(get_detrat(wf.cmat_f))
    end

    wf.proposed_move = move

    return probrat
end

function finish_move!(wf::BosonCFLWavefunction)
    update_configuration!(wf.config, wf.proposed_move)
    finish_columns_update!(wf.cmat_d)
    finish_columns_update!(wf.cmat_f)
end

function cancel_move!(wf::BosonCFLWavefunction)
    cancel_columns_update!(wf.cmat_d)
    cancel_columns_update!(wf.cmat_f)
end

###############################################################################
#################### CFL = det(nu=1) * det(nu=1) * det(FS) ####################
###############################################################################

type CFLWavefunction{Td1,Td2,Tf} <: HardcoreWavefunction
    orbitals_d1::FreeFermionOrbitals{Td1}
    orbitals_d2::FreeFermionOrbitals{Td2}
    orbitals_f::FreeFermionOrbitals{Tf}

    config::LatticeParticlesConfiguration

    cmat_d1::CeperleyMatrix{Td1}
    cmat_d2::CeperleyMatrix{Td2}
    cmat_f::CeperleyMatrix{Tf}

    neighbor_table::Vector{Vector{Int}} # this could be a Vector of neighbor tables for different neighbors

    proposed_move::ParticlesMove{LatticeParticleMove} # empty upon initialization

    function CFLWavefunction(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                             nbs::Vector{Vector{Int}}, Ly_for_init::Int)
        @assert Ly_for_init >= 0
        @assert get_Nsites(orbs_d1) == get_Nsites(orbs_d2) == get_Nsites(orbs_f)
        @assert get_Nfilled(orbs_d1) == get_Nfilled(orbs_d2) == get_Nfilled(orbs_f)
        Nsites = get_Nsites(orbs_d1)
        Nparticles = get_Nfilled(orbs_d1)
        @assert Nparticles <= Nsites
        if !isempty(nbs)
            @assert length(nbs) == Nsites
        end

        # get initial configuration of electrons
        if Ly_for_init == 0
            positions = random_combination(Nsites, Nparticles)
            config = LatticeParticlesConfiguration(Vector{Int}[positions], Nsites)
        else
            # put one particle per rung sequentially until we run out of particles
        end

        # compute Ceperley matrices
        cmat_d1 = CeperleyMatrix(fill_detmat(orbs_d1, config))
        cmat_d2 = CeperleyMatrix(fill_detmat(orbs_d2, config))
        cmat_f = CeperleyMatrix(fill_detmat(orbs_f, config))

        return new(orbs_d1, orbs_d2, orbs_f, config, cmat_d1, cmat_d2, cmat_f, nbs)
    end
end

function CFLWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                         nbs::Vector{Vector{Int}}, Ly_for_init::Int)
    return CFLWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, nbs, Ly_for_init)
end

function CFLWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                         nbs::Vector{Vector{Int}})
    return CFLWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, nbs, 0)
end

function CFLWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                         Ly_for_init::Int)
    return CFLWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, Vector{Int}[], Ly_for_init)
end

function CFLWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf})
    return CFLWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, Vector{Int}[], 0)
end

function check_for_numerical_error!(wf::CFLWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d1, "cmat_d1")
    check_and_reset_inverse_and_det!(wf.cmat_d2, "cmat_d2")
    check_and_reset_inverse_and_det!(wf.cmat_f, "cmat_f")
end

function assess_move!(wf::CFLWavefunction, move::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(move, wf)

    @assert length(move) == 1 # for now
    if is_occupied(wf.config, move[1].destination, 1)
        wf.cmat_d1.current_state = :COLUMNS_UPDATE_PENDING
        wf.cmat_d2.current_state = :COLUMNS_UPDATE_PENDING
        wf.cmat_f.current_state = :COLUMNS_UPDATE_PENDING
        probrat = 0.
    else
        detrat_from_columns_update!(wf.cmat_d1, move[1].particle.index,
            columns_for_detmat(wf.orbitals_d1, move[1].destination))
        detrat_from_columns_update!(wf.cmat_d2, move[1].particle.index,
            columns_for_detmat(wf.orbitals_d2, move[1].destination))
        detrat_from_columns_update!(wf.cmat_f, move[1].particle.index,
            columns_for_detmat(wf.orbitals_f, move[1].destination))
        probrat = abs2(get_detrat(wf.cmat_d1)) * abs2(get_detrat(wf.cmat_d2)) * abs2(get_detrat(wf.cmat_f))
    end

    wf.proposed_move = move

    return probrat
end

function finish_move!(wf::CFLWavefunction)
    update_configuration!(wf.config, wf.proposed_move)
    finish_columns_update!(wf.cmat_d1)
    finish_columns_update!(wf.cmat_d2)
    finish_columns_update!(wf.cmat_f)
end

function cancel_move!(wf::CFLWavefunction)
    cancel_columns_update!(wf.cmat_d1)
    cancel_columns_update!(wf.cmat_d2)
    cancel_columns_update!(wf.cmat_f)
end

#########################################################################################
#################### RVB [single (1) species via p-h transformation] ####################
#########################################################################################

type RVBWavefunction1{T} <: HardcoreWavefunction
    bcs::SpinfulBCS{T}
    config::LatticeParticlesConfiguration
    cmat::CeperleyMatrix{T}
    neighbor_table::Vector{Vector{Int}}
    proposed_move::ParticlesMove{LatticeParticleMove}

    function RVBWavefunction1(bcs::SpinfulBCS{T}, nbs::Vector{Vector{Int}})
        Nsites = get_Nsites(bcs)
        @assert iseven(Nsites) # otherwise spin system will necessarily be polarized, which I'm assuming not to be the case below
        Nparticles = convert(Int, Nsites/2) # = number of effective hard-core bosons
        if !isempty(nbs)
            @assert length(nbs) == Nsites
        end

        positions = random_combination(Nsites, Nparticles) # positions of up spinons
        config = LatticeParticlesConfiguration(Vector{Int}[positions], Nsites)

        cmat = CeperleyMatrix(fill_detmat_for_SL(bcs, config))

        return new(bcs, config, cmat, nbs)
    end
end

function RVBWavefunction1(bcs::SpinfulBCS, nbs::Vector{Vector{Int}})
    return RVBWavefunction1{eltype(bcs.evecs)}(bcs, nbs)
end

function RVBWavefunction1(bcs::SpinfulBCS)
    return RVBWavefunction1{eltype(bcs.evecs)}(bcs, Vector{Int}[])
end

function check_for_numerical_error!(wf::RVBWavefunction1)
    check_and_reset_inverse_and_det!(wf.cmat, "cmat")
end

function assess_move!(wf::RVBWavefunction1, move::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(move, wf)

    @assert length(move) == 1 # for now
    if is_occupied(wf.config, move[1].destination, 1)
        wf.cmat.current_state = :COLUMNS_UPDATE_PENDING
        probrat = 0.
    else
        detrat_from_columns_update!(wf.cmat, [2move[1].particle.index - 1, 2move[1].particle.index],
            columns_for_detmat_for_SL(wf.bcs, move[1].destination))
        probrat = abs2(get_detrat(wf.cmat))
    end

    wf.proposed_move = move

    return probrat
end

function finish_move!(wf::RVBWavefunction1)
    update_configuration!(wf.config, wf.proposed_move)
    finish_columns_update!(wf.cmat)
end

function cancel_move!(wf::RVBWavefunction1)
    cancel_columns_update!(wf.cmat)
end
