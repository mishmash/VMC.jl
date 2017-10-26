# include("MeanFieldTheory.jl")
# include("Configuration.jl")
# include("VMCMatrix.jl")
# include("Random.jl")
# include("Move.jl")
# include("delegate.jl")

# for calculating 2nd Renyi entropy (conserving particle numbers in the subsystems)
abstract SwapWavefunction

# spinless fermions or hard-core bosons:
abstract HardcoreSwapWavefunction <: SwapWavefunction

# FIXME: need to be more general regarding number of species throughout this class

################################################################
#################### Free spinless fermions ####################
################################################################

type FreeFermionSwapWavefunction{T} <: HardcoreSwapWavefunction
    orbitals::FreeFermionOrbitals{T}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B
    nA::Int
    nB::Int

    # Ceperley matrices for swapped and unswapped configurations of the 2 copies
    # NB1: for wavefunctions with a product of n determinants, we'll need 4n Ceperley matrix objects here
    # NB2: not all schemes for swap mod require keeping track of the phibeta's, but I don't think there's much harm in at least initializing them regardless
    cmat_phialpha1::CeperleyMatrix{T}
    cmat_phialpha2::CeperleyMatrix{T}
    cmat_phibeta1::CeperleyMatrix{T}
    cmat_phibeta2::CeperleyMatrix{T}

    # just an ordinary neighbor table
    neighbor_table::Vector{Vector{Int}}

    # moves for the 2 copies
    proposed_move_alpha1::ParticlesMove{LatticeParticleMove}
    proposed_move_alpha2::ParticlesMove{LatticeParticleMove}

    function FreeFermionSwapWavefunction(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}})
        Nsites = get_Nsites(orbs)
        Nparticles = get_Nfilled(orbs)
        Nsites_subsystemA = length(subsystemA)

        @assert 0 < Nsites_subsystemA < Nsites # subsystemA is not the whole system nor empty
        @assert minimum(subsystemA) >= 1 && maximum(subsystemA) <= Nsites
        @assert 0 <= nA <= Nparticles # nA = 0 and Nparticles are physical but somewhat special cases
        @assert nA <= Nsites_subsystemA && Nparticles - nA <= Nsites - Nsites_subsystemA # since we have hard-core particles here
        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        subsystemB = setdiff(collect(1:Nsites), subsystemA)

        # FIXME: ensure that this process doesn't produce configurations giving zero amplitude for the cmat.phi's (might need to try several times)
        positions_alpha1_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha1_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha1 = [positions_alpha1_subsystemA; positions_alpha1_subsystemB]

        positions_alpha2_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha2_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha2 = [positions_alpha2_subsystemA; positions_alpha2_subsystemB]  

        config_alpha1 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha1], Nsites)
        config_alpha2 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha2], Nsites)
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemA)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemB)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemB)[1] == Nparticles - nA

        config_beta1, config_beta2 = swap(config_alpha1, config_alpha2, subsystemA)
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemA)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemB)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemB)[1] == Nparticles - nA

        cmat_phialpha1 = CeperleyMatrix(fill_detmat(orbs, config_alpha1))
        cmat_phialpha2 = CeperleyMatrix(fill_detmat(orbs, config_alpha2))
        cmat_phibeta1 = CeperleyMatrix(fill_detmat(orbs, config_beta1))
        cmat_phibeta2 = CeperleyMatrix(fill_detmat(orbs, config_beta2))

        return new(orbs, config_alpha1, config_alpha2, subsystemA, subsystemB, nA, Nparticles - nA,
                   cmat_phialpha1, cmat_phialpha2, cmat_phibeta1, cmat_phibeta2, nbs)
    end
end

# FIXME: take out {T} here and in other similar places
function FreeFermionSwapWavefunction{T}(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}})
    return FreeFermionSwapWavefunction{eltype(orbs.filled)}(orbs, subsystemA, nA, nbs)
end

function FreeFermionSwapWavefunction{T}(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA::Int)
    return FreeFermionSwapWavefunction{eltype(orbs.filled)}(orbs, subsystemA, nA, Vector{Int}[])
end

function print_wf_info(wf::FreeFermionSwapWavefunction)
    # printlnf("inverse errors = ", check_inverse(wf.cmat_phialpha1)[2], ", ", check_inverse(wf.cmat_phialpha2)[2])
    # printlnf("det errors = ", relative_determinant_error(wf.cmat_phialpha1), ", ", relative_determinant_error(wf.cmat_phialpha2))
    # printlnf("approx det values = ", get_phialpha1(wf), ", ", get_phialpha2(wf))
    # printlnf("exact det values = ", get_phialpha1_exact_from_cmat(wf), ", ", get_phialpha2_exact_from_cmat(wf))
end

get_Nspecies(wf::HardcoreSwapWavefunction) = 1 # I think this is what we want
get_Nparticles(wf::HardcoreSwapWavefunction) = get_Nparticles(wf.config_alpha1)[1]
get_Nsites(wf::HardcoreSwapWavefunction) = get_Nsites(wf.config_alpha1)

# commented out these asserts as they're quite costly
function get_nA(wf::HardcoreSwapWavefunction)
    # @assert get_Nparticles_in_subsystem(wf.config_alpha1, wf.subsystemA)[1] == get_Nparticles_in_subsystem(wf.config_alpha2, wf.subsystemA)[1] == wf.nA
    return wf.nA
end

function get_nB(wf::HardcoreSwapWavefunction)
    # @assert get_Nparticles(wf) - get_nA(wf) == wf.nB
    return wf.nB
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::FreeFermionSwapWavefunction) = get_det(wf.cmat_phialpha1)
get_phialpha2(wf::FreeFermionSwapWavefunction) = get_det(wf.cmat_phialpha2)
get_phibeta1(wf::FreeFermionSwapWavefunction) = get_det(wf.cmat_phibeta1)
get_phibeta2(wf::FreeFermionSwapWavefunction) = get_det(wf.cmat_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::FreeFermionSwapWavefunction) = get_det_exact(wf.cmat_phialpha1)
get_phialpha2_exact_from_cmat(wf::FreeFermionSwapWavefunction) = get_det_exact(wf.cmat_phialpha2)
get_phibeta1_exact_from_cmat(wf::FreeFermionSwapWavefunction) = get_det_exact(wf.cmat_phibeta1) # = probably useless
get_phibeta2_exact_from_cmat(wf::FreeFermionSwapWavefunction) = get_det_exact(wf.cmat_phibeta2) # = probably useless
get_phibeta1_exact_from_swap(wf::FreeFermionSwapWavefunction) = det(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::FreeFermionSwapWavefunction) = det(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

# move a random particle to a random empty site (within the subsystem)
# FIXME: using setdiff here may actually be quite slow..
function random_empty_site_move_alpha1(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha1.particle_positions[1][particle_to_move.index]
    if in(current_site, wf.subsystemA) # particle is in subsystem A
        if get_nA(wf) == length(wf.subsystemA)
            return ParticlesMove{LatticeParticleMove}() # move is trivial since particles are tightly packed
        else
            proposed_position = setdiff(wf.subsystemA, wf.config_alpha1.particle_positions[1][1:get_nA(wf)])[rand(1:length(wf.subsystemA)-get_nA(wf))]
        end
    else # particle is in subsystem B
        if get_nB(wf) == length(wf.subsystemB)
            return ParticlesMove{LatticeParticleMove}() # move is trivial since particles are tightly packed
        else
            proposed_position = setdiff(wf.subsystemB, wf.config_alpha1.particle_positions[1][get_nA(wf)+1:end])[rand(1:length(wf.subsystemB)-get_nB(wf))]
        end
    end

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function random_empty_site_move_alpha2(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha2.particle_positions[1][particle_to_move.index]
    if in(current_site, wf.subsystemA) # particle is in subsystem A
        if get_nA(wf) == length(wf.subsystemA)
            return ParticlesMove{LatticeParticleMove}() # move is trivial since particles are tightly packed
        else
            proposed_position = setdiff(wf.subsystemA, wf.config_alpha2.particle_positions[1][1:get_nA(wf)])[rand(1:length(wf.subsystemA)-get_nA(wf))]
        end
    else # particle is in subsystem B
        if get_nB(wf) == length(wf.subsystemB)
            return ParticlesMove{LatticeParticleMove}() # move is trivial since particles are tightly packed
        else
            proposed_position = setdiff(wf.subsystemB, wf.config_alpha2.particle_positions[1][get_nA(wf)+1:end])[rand(1:length(wf.subsystemB)-get_nB(wf))]
        end
    end

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

# move a random particle to a random nearest neighbor
# (subystem particle number conservation and hard-core condition accounted for later in assess_move functions)
function random_neighbor_move_alpha1(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha1.particle_positions[1][particle_to_move.index]
    proposed_position = wf.neighbor_table[current_site][rand(1:end)]

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function random_neighbor_move_alpha2(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha2.particle_positions[1][particle_to_move.index]
    proposed_position = wf.neighbor_table[current_site][rand(1:end)]

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function random_neighbor_wrap_move_alpha1(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha1.particle_positions[1][particle_to_move.index]
    if in(current_site, wf.subsystemA)
        proposed_position = wf.neighbor_table_A[current_site][rand(1:end)]
    else # particle is in subsystem B
        # FIXME: this assumes that the subsystemA site indices are always less than those of subsystemB, but is efficient (no calls to find)
        proposed_position = wf.neighbor_table_B[current_site - length(wf.subsystemA)][rand(1:end)]
    end

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function random_neighbor_wrap_move_alpha2(wf::HardcoreSwapWavefunction)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    current_site = wf.config_alpha2.particle_positions[1][particle_to_move.index]
    if in(current_site, wf.subsystemA)
        proposed_position = wf.neighbor_table_A[current_site][rand(1:end)]
    else # particle is in subsystem B
        proposed_position = wf.neighbor_table_B[current_site - length(wf.subsystemA)][rand(1:end)]
    end

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
end

function move_for_single_copy(wf::HardcoreSwapWavefunction, move_function_alpha1::Function, move_function_alpha2::Function)
    copy_to_change = rand(1:2)
    if copy_to_change == 1 # move particle in alpha1
        return move_function_alpha1(wf), ParticlesMove{LatticeParticleMove}()
    else # move particle in alpha2
        return ParticlesMove{LatticeParticleMove}(), move_function_alpha2(wf)
    end
end

function move_for_both_copies(wf::HardcoreSwapWavefunction, move_function_alpha1::Function, move_function_alpha2::Function)
    return move_function_alpha1(wf), move_function_alpha2(wf)
end

function assess_move_for_mod!(wf::FreeFermionSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        # the first two statements check if we're attempting to violate the hard-core constraint, i.e., the destination site is occupied
        # the last two statements check if we're attempting to move a particle outside of its subsystem
        # in both cases, we should automatically reject and return probrat = 0.
        # NB: move_alpha1[1].particle.index <= get_nA(wf) means the particle being moved is in subsystem A, since we never allow particles to cross the subsystem boundary,
        # a condition in fact assured by this part of the program (probrat = 0. --> automatically reject)
        # a more direct alternative is the following:  in(wf.config_alpha1.particle_positions[move_alpha1[1].particle.index], wf.subsystemA)
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_phialpha1)) * abs2(get_detrat(wf.cmat_phialpha2))
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
            probrat = abs2(get_detrat(wf.cmat_phialpha1))
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_phialpha2))
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_mod_naive!(wf::FreeFermionSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_phialpha1)
        finish_columns_update!(wf.cmat_phialpha2)
        # need to take care updating the cmat_phibeta's
        # if we didn't want to do SMW on the phibeta's and just compute det(phibeta) exact every time we measure,
        # we could instead try something like the following where appropriate in the next lines, i.e., construct a new CeperleyMatrix based on the orbitals and config:
        #   wf.cmat_phibeta1 = CeperleyMatrix(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
        #   wf.cmat_phibeta2 = CeperleyMatrix(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))
        # FIXME: make a finish_move_for_mod_safe! function implementing this idea, which might even be faster,
        # since currently, for the phibeta's, accepting a transition requires O(N^2) operations while measurements require O(1) operations (det is known),
        # while the safe scheme would require O(1) to accept and O(N^3) for measurement (det needs to be computed from scratch)
        # actually, the above is dumb..!
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # both moves in A
            detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
            finish_columns_update!(wf.cmat_phibeta1)
            finish_columns_update!(wf.cmat_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in B
            detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
            finish_columns_update!(wf.cmat_phibeta1)
            finish_columns_update!(wf.cmat_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            # need 2-column update on cmat_phibeta2, while cmat_phibeta1 doesn't change
            detrat_from_columns_update!(wf.cmat_phibeta2, [wf.proposed_move_alpha1[1].particle.index, wf.proposed_move_alpha2[1].particle.index],
                columns_for_detmat(wf.orbitals, [wf.proposed_move_alpha1[1].destination, wf.proposed_move_alpha2[1].destination]))
            finish_columns_update!(wf.cmat_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
            detrat_from_columns_update!(wf.cmat_phibeta1, [wf.proposed_move_alpha2[1].particle.index, wf.proposed_move_alpha1[1].particle.index],
                columns_for_detmat(wf.orbitals, [wf.proposed_move_alpha2[1].destination, wf.proposed_move_alpha1[1].destination]))
            finish_columns_update!(wf.cmat_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
            finish_columns_update!(wf.cmat_phibeta2)
        else # move in B
            detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
            finish_columns_update!(wf.cmat_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
            finish_columns_update!(wf.cmat_phibeta1)
        else # move in B
            detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
            finish_columns_update!(wf.cmat_phibeta2)
        end
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function finish_move_for_mod!(wf::FreeFermionSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    # in this scheme, we don't touch the phibeta's during the walk; they'll be calculated when necessary for measurement
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_phialpha1)
        finish_columns_update!(wf.cmat_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_phialpha2)
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_mod!(wf::FreeFermionSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_phialpha1)
        cancel_columns_update!(wf.cmat_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_phialpha2)
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

# FIXME: change to using check_and_reset_inverse_and_det! on regular wfs
function check_for_numerical_error_for_mod_naive!(wf::FreeFermionSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_phialpha1, "cmat_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_phialpha2, "cmat_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta1, "cmat_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta2, "cmat_phibeta2")
end

function check_for_numerical_error_for_mod!(wf::FreeFermionSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_phialpha1, "cmat_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_phialpha2, "cmat_phialpha2")
end

function assess_move_for_sign!(wf::FreeFermionSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) ||
               !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
                wf.cmat_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                wf.cmat_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                wf.cmat_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else
                @assert false
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_phialpha1)) * abs(get_detrat(wf.cmat_phialpha2))
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) # both moves in A
                detrat_from_columns_update!(wf.cmat_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta1)) * abs(get_detrat(wf.cmat_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in B
                detrat_from_columns_update!(wf.cmat_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta1)) * abs(get_detrat(wf.cmat_phibeta2))
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                # need 2-column update on cmat_phibeta2, while cmat_phibeta1 doesn't change
                detrat_from_columns_update!(wf.cmat_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals, [move_alpha1[1].destination, move_alpha2[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
                detrat_from_columns_update!(wf.cmat_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals, [move_alpha2[1].destination, move_alpha1[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_phibeta1))
            else
                @assert false
            end
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
            probrat = abs(get_detrat(wf.cmat_phialpha1))
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta2))
            else # move in B
                detrat_from_columns_update!(wf.cmat_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta1))
            end
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_phialpha2))
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta1))
            else # move in B
                detrat_from_columns_update!(wf.cmat_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_phibeta2))
            end
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_sign!(wf::FreeFermionSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_phialpha1)
        finish_columns_update!(wf.cmat_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
            finish_columns_update!(wf.cmat_phibeta1)
            finish_columns_update!(wf.cmat_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            finish_columns_update!(wf.cmat_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
            finish_columns_update!(wf.cmat_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_phibeta2)
        else # move in B
            finish_columns_update!(wf.cmat_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_phibeta1)
        else # move in B
            finish_columns_update!(wf.cmat_phibeta2)
        end
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_sign!(wf::FreeFermionSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_phialpha1)
        cancel_columns_update!(wf.cmat_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
           cancel_columns_update!(wf.cmat_phibeta1)
           cancel_columns_update!(wf.cmat_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            cancel_columns_update!(wf.cmat_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            cancel_columns_update!(wf.cmat_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_phibeta2)
        else # move in B
            cancel_columns_update!(wf.cmat_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_phibeta1)
        else # move in B
            cancel_columns_update!(wf.cmat_phibeta2)
        end
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

function check_for_numerical_error_for_sign!(wf::FreeFermionSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_phialpha1, "cmat_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_phialpha2, "cmat_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta1, "cmat_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta2, "cmat_phibeta2")
end

#########################################################################
#################### Boson CFL = det(nu=1) * det(FS) ####################
#########################################################################

type BosonCFLSwapWavefunction{Td,Tf} <: HardcoreSwapWavefunction
    orbitals_d::FreeFermionOrbitals{Td}
    orbitals_f::FreeFermionOrbitals{Tf}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B
    nA::Int
    nB::Int

    # Ceperley matrices for swapped and unswapped configurations of the 2 copies
    # NB1: for wavefunctions with a product of n determinants, we'll need 4n Ceperley matrix objects here
    # NB2: not all schemes for swap mod require keeping track of the phibeta's, but I don't think there's much harm in at least initializing them regardless
    cmat_d_phialpha1::CeperleyMatrix{Td}
    cmat_f_phialpha1::CeperleyMatrix{Tf}
    cmat_d_phialpha2::CeperleyMatrix{Td}
    cmat_f_phialpha2::CeperleyMatrix{Tf}
    cmat_d_phibeta1::CeperleyMatrix{Td}
    cmat_f_phibeta1::CeperleyMatrix{Tf}
    cmat_d_phibeta2::CeperleyMatrix{Td}
    cmat_f_phibeta2::CeperleyMatrix{Tf}

    # just an ordinary neighbor table
    neighbor_table::Vector{Vector{Int}}

    # moves for the 2 copies
    proposed_move_alpha1::ParticlesMove{LatticeParticleMove}
    proposed_move_alpha2::ParticlesMove{LatticeParticleMove}

    function BosonCFLSwapWavefunction(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf}, subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}})
        @assert get_Nsites(orbs_d) == get_Nsites(orbs_f)
        @assert get_Nfilled(orbs_d) == get_Nfilled(orbs_f)
        Nsites = get_Nsites(orbs_d)
        Nparticles = get_Nfilled(orbs_d)
        Nsites_subsystemA = length(subsystemA)

        @assert 0 < Nsites_subsystemA < Nsites # subsystemA is not the whole system nor empty
        @assert minimum(subsystemA) >= 1 && maximum(subsystemA) <= Nsites
        @assert 0 <= nA <= Nparticles # nA = 0 and Nparticles are physical but somewhat special cases
        @assert nA <= Nsites_subsystemA && Nparticles - nA <= Nsites - Nsites_subsystemA # since we have hard-core particles here
        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        subsystemB = setdiff(collect(1:Nsites), subsystemA)

        # FIXME: ensure that this process doesn't produce configurations giving zero amplitude for the cmat.phi's (might need to try several times)
        positions_alpha1_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha1_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha1 = [positions_alpha1_subsystemA; positions_alpha1_subsystemB]

        positions_alpha2_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha2_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha2 = [positions_alpha2_subsystemA; positions_alpha2_subsystemB]  

        config_alpha1 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha1], Nsites)
        config_alpha2 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha2], Nsites)
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemA)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemB)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemB)[1] == Nparticles - nA

        config_beta1, config_beta2 = swap(config_alpha1, config_alpha2, subsystemA)
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemA)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemB)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemB)[1] == Nparticles - nA

        cmat_d_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d, config_alpha1))
        cmat_f_phialpha1 = CeperleyMatrix(fill_detmat(orbs_f, config_alpha1))
        cmat_d_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d, config_alpha2))
        cmat_f_phialpha2 = CeperleyMatrix(fill_detmat(orbs_f, config_alpha2))
        cmat_d_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d, config_beta1))
        cmat_f_phibeta1 = CeperleyMatrix(fill_detmat(orbs_f, config_beta1))
        cmat_d_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d, config_beta2))
        cmat_f_phibeta2 = CeperleyMatrix(fill_detmat(orbs_f, config_beta2))

        return new(orbs_d, orbs_f, config_alpha1, config_alpha2, subsystemA, subsystemB, nA, Nparticles - nA,
                   cmat_d_phialpha1, cmat_f_phialpha1, cmat_d_phialpha2, cmat_f_phialpha2,
                   cmat_d_phibeta1, cmat_f_phibeta1, cmat_d_phibeta2, cmat_f_phibeta2, nbs)
    end
end

function BosonCFLSwapWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf}, subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}})
    return BosonCFLSwapWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, subsystemA, nA, nbs)
end

function BosonCFLSwapWavefunction{Td,Tf}(orbs_d::FreeFermionOrbitals{Td}, orbs_f::FreeFermionOrbitals{Tf}, subsystemA::Vector{Int}, nA::Int)
    return BosonCFLSwapWavefunction{eltype(orbs_d.filled),eltype(orbs_f.filled)}(orbs_d, orbs_f, subsystemA, nA, Vector{Int}[])
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::BosonCFLSwapWavefunction) = get_det(wf.cmat_d_phialpha1) * get_det(wf.cmat_f_phialpha1)
get_phialpha2(wf::BosonCFLSwapWavefunction) = get_det(wf.cmat_d_phialpha2) * get_det(wf.cmat_f_phialpha2)
get_phibeta1(wf::BosonCFLSwapWavefunction) = get_det(wf.cmat_d_phibeta1) * get_det(wf.cmat_f_phibeta1)
get_phibeta2(wf::BosonCFLSwapWavefunction) = get_det(wf.cmat_d_phibeta2) * get_det(wf.cmat_f_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::BosonCFLSwapWavefunction) = get_det_exact(wf.cmat_d_phialpha1) * get_det_exact(wf.cmat_f_phialpha1)
get_phialpha2_exact_from_cmat(wf::BosonCFLSwapWavefunction) = get_det_exact(wf.cmat_d_phialpha2) * get_det_exact(wf.cmat_f_phialpha2)
get_phibeta1_exact_from_cmat(wf::BosonCFLSwapWavefunction) = get_det_exact(wf.cmat_d_phibeta1) * get_det_exact(wf.cmat_f_phibeta1) # = probably useless
get_phibeta2_exact_from_cmat(wf::BosonCFLSwapWavefunction) = get_det_exact(wf.cmat_d_phibeta2) * get_det_exact(wf.cmat_f_phibeta2) # = probably useless
get_phibeta1_exact_from_swap(wf::BosonCFLSwapWavefunction) = det(fill_detmat(wf.orbitals_d, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                             det(fill_detmat(wf.orbitals_f, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::BosonCFLSwapWavefunction) = det(fill_detmat(wf.orbitals_d, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                             det(fill_detmat(wf.orbitals_f, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

function assess_move_for_mod!(wf::BosonCFLSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        # the first two statements check if we're attempting to violate the hard-core constraint, i.e., the destination site is occupied
        # the last two statements check if we're attempting to move a particle outside of its subsystem
        # in both cases, we should automatically reject and return probrat = 0.
        # NB: move_alpha1[1].particle.index <= get_nA(wf) means the particle being moved is in subsystem A, since we never allow particles to cross the subsystem boundary,
        # a condition in fact assured by this part of the program (probrat = 0. --> automatically reject)
        # a more direct alternative is the following:  in(wf.config_alpha1.particle_positions[move_alpha1[1].particle.index], wf.subsystemA)
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d_phialpha1) * get_detrat(wf.cmat_f_phialpha1)) * abs2(get_detrat(wf.cmat_d_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d_phialpha1) * get_detrat(wf.cmat_f_phialpha1))
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_mod!(wf::BosonCFLSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    # in this scheme, we don't touch the phibeta's during the walk; they'll be calculated when necessary for measurement
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_d_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        finish_columns_update!(wf.cmat_d_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_d_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_d_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_mod!(wf::BosonCFLSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_d_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        cancel_columns_update!(wf.cmat_d_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_d_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_d_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

function check_for_numerical_error_for_mod!(wf::BosonCFLSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d_phialpha1, "cmat_d_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha1, "cmat_f_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d_phialpha2, "cmat_d_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha2, "cmat_f_phialpha2")
end

function assess_move_for_sign!(wf::BosonCFLSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) ||
               !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
                wf.cmat_d_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                wf.cmat_d_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                wf.cmat_d_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else
                @assert false
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_d_phialpha1) * get_detrat(wf.cmat_f_phialpha1)) * abs(get_detrat(wf.cmat_d_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) # both moves in A
                detrat_from_columns_update!(wf.cmat_d_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta1) * get_detrat(wf.cmat_f_phibeta1)) * abs(get_detrat(wf.cmat_d_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in B
                detrat_from_columns_update!(wf.cmat_d_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta1) * get_detrat(wf.cmat_f_phibeta1)) * abs(get_detrat(wf.cmat_d_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                # need 2-column update on cmat_phibeta2, while cmat_phibeta1 doesn't change
                detrat_from_columns_update!(wf.cmat_d_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals_d, [move_alpha1[1].destination, move_alpha2[1].destination]))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals_f, [move_alpha1[1].destination, move_alpha2[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
                detrat_from_columns_update!(wf.cmat_d_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals_d, [move_alpha2[1].destination, move_alpha1[1].destination]))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals_f, [move_alpha2[1].destination, move_alpha1[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            else
                @assert false
            end
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_d_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_d_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            probrat = abs(get_detrat(wf.cmat_d_phialpha1) * get_detrat(wf.cmat_f_phialpha1))
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_d_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            else # move in B
                detrat_from_columns_update!(wf.cmat_d_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            end
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_d_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_d_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_d_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_d_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            else # move in B
                detrat_from_columns_update!(wf.cmat_d_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            end
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_sign!(wf::BosonCFLSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_d_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        finish_columns_update!(wf.cmat_d_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
            finish_columns_update!(wf.cmat_d_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
            finish_columns_update!(wf.cmat_d_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            finish_columns_update!(wf.cmat_d_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
            finish_columns_update!(wf.cmat_d_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_d_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_d_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        else # move in B
            finish_columns_update!(wf.cmat_d_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_d_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_d_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        else # move in B
            finish_columns_update!(wf.cmat_d_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        end
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_sign!(wf::BosonCFLSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_d_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        cancel_columns_update!(wf.cmat_d_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
           cancel_columns_update!(wf.cmat_d_phibeta1)
           cancel_columns_update!(wf.cmat_f_phibeta1)
           cancel_columns_update!(wf.cmat_d_phibeta2)
           cancel_columns_update!(wf.cmat_f_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            cancel_columns_update!(wf.cmat_d_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            cancel_columns_update!(wf.cmat_d_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_d_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_d_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        else # move in B
            cancel_columns_update!(wf.cmat_d_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_d_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_d_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        else # move in B
            cancel_columns_update!(wf.cmat_d_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        end
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

function check_for_numerical_error_for_sign!(wf::BosonCFLSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d_phialpha1, "cmat_d_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha1, "cmat_f_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d_phialpha2, "cmat_d_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha2, "cmat_f_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d_phibeta1, "cmat_d_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phibeta1, "cmat_f_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d_phibeta2, "cmat_d_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phibeta2, "cmat_f_phibeta2")
end

###############################################################################
#################### CFL = det(nu=1) * det(nu=1) * det(FS) ####################
###############################################################################

type CFLSwapWavefunction{Td1,Td2,Tf} <: HardcoreSwapWavefunction
    orbitals_d1::FreeFermionOrbitals{Td1}
    orbitals_d2::FreeFermionOrbitals{Td2}
    orbitals_f::FreeFermionOrbitals{Tf}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B
    nA::Int
    nB::Int

    # Ceperley matrices for swapped and unswapped configurations of the 2 copies
    # NB1: for wavefunctions with a product of n determinants, we'll need 4n Ceperley matrix objects here
    # NB2: not all schemes for swap mod require keeping track of the phibeta's, but I don't think there's much harm in at least initializing them regardless
    cmat_d1_phialpha1::CeperleyMatrix{Td1}
    cmat_d2_phialpha1::CeperleyMatrix{Td2}
    cmat_f_phialpha1::CeperleyMatrix{Tf}
    cmat_d1_phialpha2::CeperleyMatrix{Td1}
    cmat_d2_phialpha2::CeperleyMatrix{Td2}
    cmat_f_phialpha2::CeperleyMatrix{Tf}
    cmat_d1_phibeta1::CeperleyMatrix{Td1}
    cmat_d2_phibeta1::CeperleyMatrix{Td2}
    cmat_f_phibeta1::CeperleyMatrix{Tf}
    cmat_d1_phibeta2::CeperleyMatrix{Td1}
    cmat_d2_phibeta2::CeperleyMatrix{Td2}
    cmat_f_phibeta2::CeperleyMatrix{Tf}

    # just an ordinary neighbor table
    neighbor_table::Vector{Vector{Int}}

    # neighbor tables which "wrap" subsystem
    neighbor_table_A::Vector{Vector{Int}}
    neighbor_table_B::Vector{Vector{Int}}

    # moves for the 2 copies
    proposed_move_alpha1::ParticlesMove{LatticeParticleMove}
    proposed_move_alpha2::ParticlesMove{LatticeParticleMove}

    function CFLSwapWavefunction(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf}, subsystemA::Vector{Int}, nA::Int,
                                 nbs::Vector{Vector{Int}}, nbsA::Vector{Vector{Int}}, nbsB::Vector{Vector{Int}})
        @assert get_Nsites(orbs_d1) == get_Nsites(orbs_d2) == get_Nsites(orbs_f)
        @assert get_Nfilled(orbs_d1) == get_Nfilled(orbs_d2) == get_Nfilled(orbs_f)
        Nsites = get_Nsites(orbs_d1)
        Nparticles = get_Nfilled(orbs_d1)
        Nsites_subsystemA = length(subsystemA)

        @assert 0 < Nsites_subsystemA < Nsites # subsystemA is not the whole system nor empty
        @assert minimum(subsystemA) >= 1 && maximum(subsystemA) <= Nsites
        @assert 0 <= nA <= Nparticles # nA = 0 and Nparticles are physical but somewhat special cases
        @assert nA <= Nsites_subsystemA && Nparticles - nA <= Nsites - Nsites_subsystemA # since we have hard-core particles here
        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        subsystemB = setdiff(collect(1:Nsites), subsystemA)

        # FIXME: ensure that this process doesn't produce configurations giving zero amplitude for the cmat.phi's (might need to try several times)
        positions_alpha1_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha1_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha1 = [positions_alpha1_subsystemA; positions_alpha1_subsystemB]

        positions_alpha2_subsystemA = subsystemA[random_combination(Nsites_subsystemA, nA)]
        positions_alpha2_subsystemB = subsystemB[random_combination(Nsites - Nsites_subsystemA, Nparticles - nA)]
        positions_alpha2 = [positions_alpha2_subsystemA; positions_alpha2_subsystemB]  

        config_alpha1 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha1], Nsites)
        config_alpha2 = LatticeParticlesConfiguration(Vector{Int}[positions_alpha2], Nsites)
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemA)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_alpha1, subsystemB)[1] == get_Nparticles_in_subsystem(config_alpha2, subsystemB)[1] == Nparticles - nA

        config_beta1, config_beta2 = swap(config_alpha1, config_alpha2, subsystemA)
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemA)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemA)[1] == nA
        @assert get_Nparticles_in_subsystem(config_beta1, subsystemB)[1] == get_Nparticles_in_subsystem(config_beta2, subsystemB)[1] == Nparticles - nA

        cmat_d1_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha1))
        cmat_d2_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha1))
        cmat_f_phialpha1 = CeperleyMatrix(fill_detmat(orbs_f, config_alpha1))
        cmat_d1_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha2))
        cmat_d2_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha2))
        cmat_f_phialpha2 = CeperleyMatrix(fill_detmat(orbs_f, config_alpha2))
        cmat_d1_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta1))
        cmat_d2_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta1))
        cmat_f_phibeta1 = CeperleyMatrix(fill_detmat(orbs_f, config_beta1))
        cmat_d1_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta2))
        cmat_d2_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta2))
        cmat_f_phibeta2 = CeperleyMatrix(fill_detmat(orbs_f, config_beta2))

        return new(orbs_d1, orbs_d2, orbs_f, config_alpha1, config_alpha2, subsystemA, subsystemB, nA, Nparticles - nA,
                   cmat_d1_phialpha1, cmat_d2_phialpha1, cmat_f_phialpha1, cmat_d1_phialpha2, cmat_d2_phialpha2, cmat_f_phialpha2,
                   cmat_d1_phibeta1, cmat_d2_phibeta1, cmat_f_phibeta1, cmat_d1_phibeta2, cmat_d2_phibeta2, cmat_f_phibeta2,
                   nbs, nbsA, nbsB)
    end
end

function CFLSwapWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                                         subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}})
    return CFLSwapWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, subsystemA, nA, nbs, Vector{Int}[], Vector{Int}[])
end

function CFLSwapWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                                         subsystemA::Vector{Int}, nA::Int, nbs::Vector{Vector{Int}}, nbsA::Vector{Vector{Int}}, nbsB::Vector{Vector{Int}})
    return CFLSwapWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, subsystemA, nA, nbs, nbsA, nbsB)
end

function CFLSwapWavefunction{Td1,Td2,Tf}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_f::FreeFermionOrbitals{Tf},
                                         subsystemA::Vector{Int}, nA::Int)
    return CFLSwapWavefunction{eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_f.filled)}(orbs_d1, orbs_d2, orbs_f, subsystemA, nA, Vector{Int}[], Vector{Int}[], Vector{Int}[])
end

function print_wf_info(wf::CFLSwapWavefunction)
    # printlnf("d1 inverse errors = ", check_inverse(wf.cmat_d1_phialpha1)[2], ", ", check_inverse(wf.cmat_d1_phialpha2)[2])
    # printlnf("d2 inverse errors = ", check_inverse(wf.cmat_d2_phialpha1)[2], ", ", check_inverse(wf.cmat_d2_phialpha2)[2])
    # printlnf("f inverse errors = ", check_inverse(wf.cmat_f_phialpha1)[2], ", ", check_inverse(wf.cmat_f_phialpha2)[2])
    # printlnf("d1 det errors = ", relative_determinant_error(wf.cmat_d1_phialpha1), ", ", relative_determinant_error(wf.cmat_d1_phialpha2))
    # printlnf("d2 det errors = ", relative_determinant_error(wf.cmat_d2_phialpha1), ", ", relative_determinant_error(wf.cmat_d2_phialpha2))
    # printlnf("f det errors = ", relative_determinant_error(wf.cmat_f_phialpha1), ", ", relative_determinant_error(wf.cmat_f_phialpha2))
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::CFLSwapWavefunction) = get_det(wf.cmat_d1_phialpha1) * get_det(wf.cmat_d2_phialpha1) * get_det(wf.cmat_f_phialpha1)
get_phialpha2(wf::CFLSwapWavefunction) = get_det(wf.cmat_d1_phialpha2) * get_det(wf.cmat_d2_phialpha2) * get_det(wf.cmat_f_phialpha2)
get_phibeta1(wf::CFLSwapWavefunction) = get_det(wf.cmat_d1_phibeta1) * get_det(wf.cmat_d2_phibeta1) * get_det(wf.cmat_f_phibeta1)
get_phibeta2(wf::CFLSwapWavefunction) = get_det(wf.cmat_d1_phibeta2) * get_det(wf.cmat_d2_phibeta2) * get_det(wf.cmat_f_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::CFLSwapWavefunction) = get_det_exact(wf.cmat_d1_phialpha1) * get_det_exact(wf.cmat_d2_phialpha1) * get_det_exact(wf.cmat_f_phialpha1)
get_phialpha2_exact_from_cmat(wf::CFLSwapWavefunction) = get_det_exact(wf.cmat_d1_phialpha2) * get_det_exact(wf.cmat_d2_phialpha2) * get_det_exact(wf.cmat_f_phialpha2)
get_phibeta1_exact_from_cmat(wf::CFLSwapWavefunction) = get_det_exact(wf.cmat_d1_phibeta1) * get_det_exact(wf.cmat_d2_phibeta1) * get_det_exact(wf.cmat_f_phibeta1) # = probably useless
get_phibeta2_exact_from_cmat(wf::CFLSwapWavefunction) = get_det_exact(wf.cmat_d1_phibeta2) * get_det_exact(wf.cmat_d2_phibeta2) * get_det_exact(wf.cmat_f_phibeta2) # = probably useless
get_phibeta1_exact_from_swap(wf::CFLSwapWavefunction) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                        det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                        det(fill_detmat(wf.orbitals_f, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::CFLSwapWavefunction) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                        det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                        det(fill_detmat(wf.orbitals_f, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

function assess_move_for_mod!(wf::CFLSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        # the first two statements check if we're attempting to violate the hard-core constraint, i.e., the destination site is occupied
        # the last two statements check if we're attempting to move a particle outside of its subsystem
        # in both cases, we should automatically reject and return probrat = 0.
        # NB: move_alpha1[1].particle.index <= get_nA(wf) means the particle being moved is in subsystem A, since we never allow particles to cross the subsystem boundary,
        # a condition in fact assured by this part of the program (probrat = 0. --> automatically reject)
        # a more direct alternative is the following:  in(wf.config_alpha1.particle_positions[move_alpha1[1].particle.index], wf.subsystemA)
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d1_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d1_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d1_phialpha1) * get_detrat(wf.cmat_d2_phialpha1) * get_detrat(wf.cmat_f_phialpha1)) * 
                      abs2(get_detrat(wf.cmat_d1_phialpha2) * get_detrat(wf.cmat_d2_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d1_phialpha1) * get_detrat(wf.cmat_d2_phialpha1) * get_detrat(wf.cmat_f_phialpha1))
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs2(get_detrat(wf.cmat_d1_phialpha2) * get_detrat(wf.cmat_d2_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_mod!(wf::CFLSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    # in this scheme, we don't touch the phibeta's during the walk; they'll be calculated when necessary for measurement
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_d1_phialpha1)
        finish_columns_update!(wf.cmat_d2_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        finish_columns_update!(wf.cmat_d1_phialpha2)
        finish_columns_update!(wf.cmat_d2_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_d1_phialpha1)
        finish_columns_update!(wf.cmat_d2_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_d1_phialpha2)
        finish_columns_update!(wf.cmat_d2_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_mod!(wf::CFLSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_d1_phialpha1)
        cancel_columns_update!(wf.cmat_d2_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        cancel_columns_update!(wf.cmat_d1_phialpha2)
        cancel_columns_update!(wf.cmat_d2_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_d1_phialpha1)
        cancel_columns_update!(wf.cmat_d2_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_d1_phialpha2)
        cancel_columns_update!(wf.cmat_d2_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

function check_for_numerical_error_for_mod!(wf::CFLSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha1, "cmat_f_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha2, "cmat_f_phialpha2")
end

function assess_move_for_sign!(wf::CFLSwapWavefunction, move_alpha1::ParticlesMove{LatticeParticleMove}, move_alpha2::ParticlesMove{LatticeParticleMove})
    # @assert is_valid_move(wf, move_alpha1, move_alpha2)

    if is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating alpha1 and alpha2
        # printlnf("assessing alpha1 and alpha2 move")
        @assert length(move_alpha1) == length(move_alpha2) == 1 # only single-particle moves at the moment
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d1_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) ||
               !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
                wf.cmat_d1_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d1_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                wf.cmat_d1_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                wf.cmat_d1_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else
                @assert false
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d1_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_d1_phialpha1) * get_detrat(wf.cmat_d2_phialpha1) * get_detrat(wf.cmat_f_phialpha1)) *
                      abs(get_detrat(wf.cmat_d1_phialpha2) * get_detrat(wf.cmat_d2_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
            if move_alpha1[1].particle.index <= get_nA(wf) && move_alpha2[1].particle.index <= get_nA(wf) # both moves in A
                detrat_from_columns_update!(wf.cmat_d1_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d1_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta1) * get_detrat(wf.cmat_d2_phibeta1) * get_detrat(wf.cmat_f_phibeta1)) *
                           abs(get_detrat(wf.cmat_d1_phibeta2) * get_detrat(wf.cmat_d2_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && !(move_alpha2[1].particle.index <= get_nA(wf)) # both moves in B
                detrat_from_columns_update!(wf.cmat_d1_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d1_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta1) * get_detrat(wf.cmat_d2_phibeta1) * get_detrat(wf.cmat_f_phibeta1)) *
                           abs(get_detrat(wf.cmat_d1_phibeta2) * get_detrat(wf.cmat_d2_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif move_alpha1[1].particle.index <= get_nA(wf) && !(move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
                # need 2-column update on cmat_phibeta2, while cmat_phibeta1 doesn't change
                detrat_from_columns_update!(wf.cmat_d1_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals_d1, [move_alpha1[1].destination, move_alpha2[1].destination]))
                detrat_from_columns_update!(wf.cmat_d2_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals_d2, [move_alpha1[1].destination, move_alpha2[1].destination]))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, [move_alpha1[1].particle.index, move_alpha2[1].particle.index], 
                    columns_for_detmat(wf.orbitals_f, [move_alpha1[1].destination, move_alpha2[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta2) * get_detrat(wf.cmat_d2_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            elseif !(move_alpha1[1].particle.index <= get_nA(wf)) && move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
                # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
                detrat_from_columns_update!(wf.cmat_d1_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals_d1, [move_alpha2[1].destination, move_alpha1[1].destination]))
                detrat_from_columns_update!(wf.cmat_d2_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals_d2, [move_alpha2[1].destination, move_alpha1[1].destination]))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, [move_alpha2[1].particle.index, move_alpha1[1].particle.index],
                    columns_for_detmat(wf.orbitals_f, [move_alpha2[1].destination, move_alpha1[1].destination]))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta1) * get_detrat(wf.cmat_d2_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            else
                @assert false
            end
        end
    elseif is_nontrivial(move_alpha1) && !is_nontrivial(move_alpha2) # updating only alpha1
        # printlnf("assessing alpha1 move")
        @assert length(move_alpha1) == 1
        if is_occupied(wf.config_alpha1, move_alpha1[1].destination, 1) ||
           !in(move_alpha1[1].destination, move_alpha1[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha1.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_d1_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_d1_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
            probrat = abs(get_detrat(wf.cmat_d1_phialpha1) * get_detrat(wf.cmat_d2_phialpha1) * get_detrat(wf.cmat_f_phialpha1))
            if move_alpha1[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_d1_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta2) * get_detrat(wf.cmat_d2_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            else # move in B
                detrat_from_columns_update!(wf.cmat_d1_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha1[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha1[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta1) * get_detrat(wf.cmat_d2_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            end
        end
    elseif !is_nontrivial(move_alpha1) && is_nontrivial(move_alpha2) # updating only alpha2
        # printlnf("assessing alpha2 move")
        @assert length(move_alpha2) == 1
        if is_occupied(wf.config_alpha2, move_alpha2[1].destination, 1) ||
           !in(move_alpha2[1].destination, move_alpha2[1].particle.index <= get_nA(wf) ? wf.subsystemA : wf.subsystemB)
            wf.cmat_d1_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_d2_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            wf.cmat_f_phialpha2.current_state = :COLUMNS_UPDATE_PENDING
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                wf.cmat_d1_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta1.current_state = :COLUMNS_UPDATE_PENDING
            else # move in B
                wf.cmat_d1_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_d2_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
                wf.cmat_f_phibeta2.current_state = :COLUMNS_UPDATE_PENDING
            end
            probrat = 0.
        else
            detrat_from_columns_update!(wf.cmat_d1_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_d2_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
            detrat_from_columns_update!(wf.cmat_f_phialpha2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
            probrat = abs(get_detrat(wf.cmat_d1_phialpha2) * get_detrat(wf.cmat_d2_phialpha2) * get_detrat(wf.cmat_f_phialpha2))
            if move_alpha2[1].particle.index <= get_nA(wf) # move in A
                detrat_from_columns_update!(wf.cmat_d1_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta1, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta1) * get_detrat(wf.cmat_d2_phibeta1) * get_detrat(wf.cmat_f_phibeta1))
            else # move in B
                detrat_from_columns_update!(wf.cmat_d1_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_d2_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, move_alpha2[1].destination))
                detrat_from_columns_update!(wf.cmat_f_phibeta2, move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_f, move_alpha2[1].destination))
                probrat *= abs(get_detrat(wf.cmat_d1_phibeta2) * get_detrat(wf.cmat_d2_phibeta2) * get_detrat(wf.cmat_f_phibeta2))
            end
        end
    else
        warn("Assessing a completely trivial transition.")
        probrat = 1.
    end

    wf.proposed_move_alpha1, wf.proposed_move_alpha2 = move_alpha1, move_alpha2

    return probrat
end

function finish_move_for_sign!(wf::CFLSwapWavefunction)
    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_columns_update!(wf.cmat_d1_phialpha1)
        finish_columns_update!(wf.cmat_d2_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        finish_columns_update!(wf.cmat_d1_phialpha2)
        finish_columns_update!(wf.cmat_d2_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
            finish_columns_update!(wf.cmat_d1_phibeta1)
            finish_columns_update!(wf.cmat_d2_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
            finish_columns_update!(wf.cmat_d1_phibeta2)
            finish_columns_update!(wf.cmat_d2_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            finish_columns_update!(wf.cmat_d1_phibeta2)
            finish_columns_update!(wf.cmat_d2_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            # need 2-column update on cmat_phibeta1, while cmat_phibeta2 doesn't change
            finish_columns_update!(wf.cmat_d1_phibeta1)
            finish_columns_update!(wf.cmat_d2_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_columns_update!(wf.cmat_d1_phialpha1)
        finish_columns_update!(wf.cmat_d2_phialpha1)
        finish_columns_update!(wf.cmat_f_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_d1_phibeta2)
            finish_columns_update!(wf.cmat_d2_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        else # move in B
            finish_columns_update!(wf.cmat_d1_phibeta1)
            finish_columns_update!(wf.cmat_d2_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_columns_update!(wf.cmat_d1_phialpha2)
        finish_columns_update!(wf.cmat_d2_phialpha2)
        finish_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            finish_columns_update!(wf.cmat_d1_phibeta1)
            finish_columns_update!(wf.cmat_d2_phibeta1)
            finish_columns_update!(wf.cmat_f_phibeta1)
        else # move in B
            finish_columns_update!(wf.cmat_d1_phibeta2)
            finish_columns_update!(wf.cmat_d2_phibeta2)
            finish_columns_update!(wf.cmat_f_phibeta2)
        end
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    return nothing
end

function cancel_move_for_sign!(wf::CFLSwapWavefunction)
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("cancelling alpha1 and alpha2 move")
        cancel_columns_update!(wf.cmat_d1_phialpha1)
        cancel_columns_update!(wf.cmat_d2_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        cancel_columns_update!(wf.cmat_d1_phialpha2)
        cancel_columns_update!(wf.cmat_d2_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) ||
           !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # both moves in A or both moves in B
           cancel_columns_update!(wf.cmat_d1_phibeta1)
           cancel_columns_update!(wf.cmat_d2_phibeta1)
           cancel_columns_update!(wf.cmat_f_phibeta1)
           cancel_columns_update!(wf.cmat_d1_phibeta2)
           cancel_columns_update!(wf.cmat_d2_phibeta2)
           cancel_columns_update!(wf.cmat_f_phibeta2)
        elseif wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) && !(wf.proposed_move_alpha2[1].particle.index <= get_nA(wf)) # alpha1 move in A, alpha2 move in B
            cancel_columns_update!(wf.cmat_d1_phibeta2)
            cancel_columns_update!(wf.cmat_d2_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        elseif !(wf.proposed_move_alpha1[1].particle.index <= get_nA(wf)) && wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # alpha1 move in B, alpha2 move in A
            cancel_columns_update!(wf.cmat_d1_phibeta1)
            cancel_columns_update!(wf.cmat_d2_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        else
            @assert false
        end
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("cancelling alpha1 move")
        cancel_columns_update!(wf.cmat_d1_phialpha1)
        cancel_columns_update!(wf.cmat_d2_phialpha1)
        cancel_columns_update!(wf.cmat_f_phialpha1)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_d1_phibeta2)
            cancel_columns_update!(wf.cmat_d2_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        else # move in B
            cancel_columns_update!(wf.cmat_d1_phibeta1)
            cancel_columns_update!(wf.cmat_d2_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("cancelling alpha2 move")
        cancel_columns_update!(wf.cmat_d1_phialpha2)
        cancel_columns_update!(wf.cmat_d2_phialpha2)
        cancel_columns_update!(wf.cmat_f_phialpha2)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move in A
            cancel_columns_update!(wf.cmat_d1_phibeta1)
            cancel_columns_update!(wf.cmat_d2_phibeta1)
            cancel_columns_update!(wf.cmat_f_phibeta1)
        else # move in B
            cancel_columns_update!(wf.cmat_d1_phibeta2)
            cancel_columns_update!(wf.cmat_d2_phibeta2)
            cancel_columns_update!(wf.cmat_f_phibeta2)
        end
    else
        warn("Cancelling a completely trivial tranistion. Good work.")
    end

    return nothing
end

function check_for_numerical_error_for_sign!(wf::CFLSwapWavefunction)
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha1, "cmat_f_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phialpha2, "cmat_f_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta1, "cmat_d1_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta1, "cmat_d2_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_f_phibeta1, "cmat_f_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta2, "cmat_d1_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta2, "cmat_d2_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_f_phibeta2, "cmat_f_phibeta2")
end
