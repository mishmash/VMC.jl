# include("MeanFieldTheory.jl")
# include("Configuration.jl")
# include("VMCMatrix.jl")
# include("Random.jl")
# include("Move.jl")
# include("delegate.jl")

using Retry

# for calculating 2nd Renyi entropy (not conserving particle numbers in the subsystems; Jim's method)
# NS = non-sectored
abstract SwapWavefunctionNS

# this should work for any SwapWavefunctionNS
function cancel_move_for_mod!(wf::SwapWavefunctionNS)
    if !wf.automatically_reject
        if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
            # printlnf("cancelling alpha1 and alpha2 move")
            cancel_phialpha1_update!(wf)
            cancel_phialpha2_update!(wf)
        elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
            # printlnf("cancelling alpha1 move")
            cancel_phialpha1_update!(wf)
        elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
            # printlnf("cancelling alpha2 move")
            cancel_phialpha2_update!(wf)
        else
            warn("Cancelling a completely trivial tranistion. Good work.")
        end
    end

    return nothing
end

# spinless fermions or hard-core bosons:
abstract HardcoreSwapWavefunctionNS <: SwapWavefunctionNS

function assess_move_for_mod!(wf::HardcoreSwapWavefunctionNS, move::ParticlesMove{LatticeParticleMove}, copy_to_base::Int)
    @assert copy_to_base == 1 || copy_to_base == 2
    @assert length(move) == 1 # only single-particle moves at the moment

    # base move around "copy_a" (copy i in the notes); the other copy is "copy_b" (copy ibar in the notes)
    config_copy_a = (copy_to_base == 1) ? wf.config_alpha1 : wf.config_alpha2
    config_copy_b = (copy_to_base == 1) ? wf.config_alpha2 : wf.config_alpha1

    if is_occupied(config_copy_a, move[1].destination, 1) # automatically reject due to Pauli exclusion / hard-core constraint
        wf.automatically_reject = true
        wf.proposed_Delta_nA = 0 # trivially
        # the following is probably not necessary, but I guess we can keep track of what the move is in any case
        if copy_to_base == 1
            # printlnf("automatically rejected move, copy_to_base == 1")
            wf.proposed_move_alpha1 = move
            wf.proposed_move_alpha2 = ParticlesMove{LatticeParticleMove}()
        else # copy_to_base == 2
            # printlnf("automatically rejected move, copy_to_base == 2")
            wf.proposed_move_alpha1 = ParticlesMove{LatticeParticleMove}()
            wf.proposed_move_alpha2 = move
        end
        probrat = 0.
    else
        wf.automatically_reject = false
        # move conserves subystem occupations
        if (in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemA) && in(move[1].destination, wf.subsystemA)) ||
           (in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemB) && in(move[1].destination, wf.subsystemB))
            wf.proposed_Delta_nA = 0
            if copy_to_base == 1
                # printlnf("conserving move, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                wf.proposed_move_alpha2 = ParticlesMove{LatticeParticleMove}()
                probrat = probrat_for_mod_alpha1_move!(wf)
            else # copy_to_base == 2
                # printlnf("conserving move, copy_to_base == 2")
                wf.proposed_move_alpha1 = ParticlesMove{LatticeParticleMove}()
                wf.proposed_move_alpha2 = move
                probrat = probrat_for_mod_alpha2_move!(wf)
            end
        # move hops a particle from A to B
        elseif in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemA) && in(move[1].destination, wf.subsystemB)
            wf.proposed_Delta_nA = -1
            # move a random particle in A to a random empty site in B (in the appropriate copy)
            if copy_to_base == 1
                # printlnf("changing move, A to B, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                particle_to_move = Particle(wf.config_alpha2.lattice_status[1][wf.subsystemA][find(wf.config_alpha2.lattice_status[1][wf.subsystemA])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemB, wf.config_alpha2.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha2 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            else # copy_to_base == 2
                # printlnf("changing move, A to B, copy_to_base == 2")
                wf.proposed_move_alpha2 = move
                particle_to_move = Particle(wf.config_alpha1.lattice_status[1][wf.subsystemA][find(wf.config_alpha1.lattice_status[1][wf.subsystemA])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemB, wf.config_alpha1.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha1 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            end
            forward_particles = get_nA(wf)
            forward_vacancies = length(wf.subsystemB) - get_nB(wf)
            reverse_particles = get_nB(wf) + 1
            reverse_vacancies = length(wf.subsystemA) - get_nA(wf) + 1
            weighting_factor = (forward_particles * forward_vacancies) / (reverse_particles * reverse_vacancies)
            probrat = weighting_factor * probrat_for_mod_alpha1_move!(wf) * probrat_for_mod_alpha2_move!(wf)
        # move hops a particle from B to A
        elseif in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemB) && in(move[1].destination, wf.subsystemA)
            wf.proposed_Delta_nA = 1
            # move a random particle in B to a random empty site in A (in the appropriate copy)
            if copy_to_base == 1
                # printlnf("changing move, B to A, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                particle_to_move = Particle(wf.config_alpha2.lattice_status[1][wf.subsystemB][find(wf.config_alpha2.lattice_status[1][wf.subsystemB])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemA, wf.config_alpha2.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha2 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            else # copy_to_base == 2
                # printlnf("changing move, B to A, copy_to_base == 2")
                wf.proposed_move_alpha2 = move
                particle_to_move = Particle(wf.config_alpha1.lattice_status[1][wf.subsystemB][find(wf.config_alpha1.lattice_status[1][wf.subsystemB])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemA, wf.config_alpha1.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha1 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            end
            forward_particles = get_nB(wf)
            forward_vacancies = length(wf.subsystemA) - get_nA(wf)
            reverse_particles = get_nA(wf) + 1
            reverse_vacancies = length(wf.subsystemB) - get_nB(wf) + 1
            weighting_factor = (forward_particles * forward_vacancies) / (reverse_particles * reverse_vacancies)
            probrat = weighting_factor * probrat_for_mod_alpha1_move!(wf) * probrat_for_mod_alpha2_move!(wf)
        else
            @assert false
        end
    end

    return probrat
end

function finish_move_for_mod!(wf::HardcoreSwapWavefunctionNS)
    @assert wf.automatically_reject == false

    # FIXME: change all wfs in VMC code to update the dets first, then the configs (to be anal)

    # in this scheme, we don't touch the phibeta's during the walk; they'll be calculated when necessary for measurement
    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_phialpha1_update!(wf)
        finish_phialpha2_update!(wf)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_phialpha1_update!(wf)
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_phialpha2_update!(wf)
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    # to be transparent
    nA_before_move = get_nA(wf)

    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    # update wf.nA and wf.nB, and exchange particles to keep those particles with lowest nA indices in subsystem A
    if wf.proposed_Delta_nA == -1
        wf.nA -= 1
        wf.nB += 1
        # printlnf("exchanging for A to B move")
        j1 = wf.proposed_move_alpha1[1].particle.index
        j2 = wf.proposed_move_alpha2[1].particle.index
        @assert j1 <= nA_before_move
        @assert j2 <= nA_before_move
        if j1 < nA_before_move # otherwise we're exchanging a particle with itself
            exchange_phialpha1!(wf, j1, nA_before_move)
            exchange_particles!(wf.config_alpha1, j1, nA_before_move, 1)
        end
        if j2 < nA_before_move # otherwise we're exchanging a particle with itself
            exchange_phialpha2!(wf, j2, nA_before_move)
            exchange_particles!(wf.config_alpha2, j2, nA_before_move, 1)
        end
    elseif wf.proposed_Delta_nA == 1
        wf.nA += 1
        wf.nB -= 1
        # printlnf("exchanging for B to A move")
        j1 = wf.proposed_move_alpha1[1].particle.index
        j2 = wf.proposed_move_alpha2[1].particle.index
        @assert j1 > nA_before_move
        @assert j2 > nA_before_move
        if j1 > nA_before_move+1 # otherwise we're exchanging a particle with itself
            exchange_phialpha1!(wf, j1, nA_before_move+1)
            exchange_particles!(wf.config_alpha1, j1, nA_before_move+1, 1)
        end
        if j2 > nA_before_move+1 # otherwise we're exchanging a particle with itself
            exchange_phialpha2!(wf, j2, nA_before_move+1)
            exchange_particles!(wf.config_alpha2, j2, nA_before_move+1, 1)
        end
    else
        @assert wf.proposed_Delta_nA == 0
        @assert nA_before_move == get_nA(wf)
    end

    return nothing
end

function assess_move_for_sign!(wf::HardcoreSwapWavefunctionNS, move::ParticlesMove{LatticeParticleMove}, copy_to_base::Int)
    @assert copy_to_base == 1 || copy_to_base == 2
    @assert length(move) == 1 # only single-particle moves at the moment

    # base move around "copy_a" (copy i in the notes); the other copy is "copy_b" (copy ibar in the notes)
    config_copy_a = (copy_to_base == 1) ? wf.config_alpha1 : wf.config_alpha2
    config_copy_b = (copy_to_base == 1) ? wf.config_alpha2 : wf.config_alpha1

    if is_occupied(config_copy_a, move[1].destination, 1) # automatically reject due to Pauli exclusion / hard-core constraint
        wf.automatically_reject = true
        wf.proposed_Delta_nA = 0 # trivially
        # the following is probably not necessary, but I guess we can keep track of what the move is in any case
        if copy_to_base == 1
            # printlnf("automatically rejected move, copy_to_base == 1")
            wf.proposed_move_alpha1 = move
            wf.proposed_move_alpha2 = ParticlesMove{LatticeParticleMove}()
        else # copy_to_base == 2
            # printlnf("automatically rejected move, copy_to_base == 2")
            wf.proposed_move_alpha1 = ParticlesMove{LatticeParticleMove}()
            wf.proposed_move_alpha2 = move
        end
        probrat = 0.
    else
        wf.automatically_reject = false
        # move entirely within A
        if in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemA) && in(move[1].destination, wf.subsystemA)
            @assert move[1].particle.index <= get_nA(wf)
            wf.proposed_Delta_nA = 0
            if copy_to_base == 1 # change alpha1 and beta2
                # printlnf("conserving move, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                wf.proposed_move_alpha2 = ParticlesMove{LatticeParticleMove}()
                probrat = probrat_for_sign_alpha1_move_inA!(wf)
            else # copy_to_base == 2 # change alpha2 and beta1
                # printlnf("conserving move, copy_to_base == 2")
                wf.proposed_move_alpha1 = ParticlesMove{LatticeParticleMove}()
                wf.proposed_move_alpha2 = move
                probrat = probrat_for_sign_alpha2_move_inA!(wf)
            end
        # move entirely within B
        elseif in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemB) && in(move[1].destination, wf.subsystemB)
            @assert move[1].particle.index > get_nA(wf)
            wf.proposed_Delta_nA = 0
            if copy_to_base == 1 # change alpha1 and beta1
                # printlnf("conserving move, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                wf.proposed_move_alpha2 = ParticlesMove{LatticeParticleMove}()
                probrat = probrat_for_sign_alpha1_move_inB!(wf)
            else # copy_to_base == 2 # change alpha2 and beta2
                # printlnf("conserving move, copy_to_base == 2")
                wf.proposed_move_alpha1 = ParticlesMove{LatticeParticleMove}()
                wf.proposed_move_alpha2 = move
                probrat = probrat_for_sign_alpha2_move_inB!(wf)
            end
        # move hops a particle from A to B
        elseif in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemA) && in(move[1].destination, wf.subsystemB)
            @assert move[1].particle.index <= get_nA(wf)
            wf.proposed_Delta_nA = -1
            # move a random particle in A to a random empty site in B (in the appropriate copy)
            if copy_to_base == 1
                # printlnf("changing move, A to B, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                particle_to_move = Particle(wf.config_alpha2.lattice_status[1][wf.subsystemA][find(wf.config_alpha2.lattice_status[1][wf.subsystemA])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemB, wf.config_alpha2.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha2 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            else # copy_to_base == 2
                # printlnf("changing move, A to B, copy_to_base == 2")
                wf.proposed_move_alpha2 = move
                particle_to_move = Particle(wf.config_alpha1.lattice_status[1][wf.subsystemA][find(wf.config_alpha1.lattice_status[1][wf.subsystemA])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemB, wf.config_alpha1.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha1 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            end
            forward_particles = get_nA(wf)
            forward_vacancies = length(wf.subsystemB) - get_nB(wf)
            reverse_particles = get_nB(wf) + 1
            reverse_vacancies = length(wf.subsystemA) - get_nA(wf) + 1
            weighting_factor = (forward_particles * forward_vacancies) / (reverse_particles * reverse_vacancies)
            probrat = weighting_factor * probrat_for_sign_two_copy_move_AtoB!(wf)
        # move hops a particle from B to A
        elseif in(config_copy_a.particle_positions[1][move[1].particle.index], wf.subsystemB) && in(move[1].destination, wf.subsystemA)
            @assert move[1].particle.index > get_nA(wf)
            wf.proposed_Delta_nA = 1
            # move a random particle in B to a random empty site in A (in the appropriate copy)
            if copy_to_base == 1
                # printlnf("changing move, B to A, copy_to_base == 1")
                wf.proposed_move_alpha1 = move
                particle_to_move = Particle(wf.config_alpha2.lattice_status[1][wf.subsystemB][find(wf.config_alpha2.lattice_status[1][wf.subsystemB])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemA, wf.config_alpha2.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha2 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            else # copy_to_base == 2
                # printlnf("changing move, B to A, copy_to_base == 2")
                wf.proposed_move_alpha2 = move
                particle_to_move = Particle(wf.config_alpha1.lattice_status[1][wf.subsystemB][find(wf.config_alpha1.lattice_status[1][wf.subsystemB])][rand(1:end)], 1)
                proposed_position = setdiff(wf.subsystemA, wf.config_alpha1.particle_positions[1])[rand(1:end)]
                wf.proposed_move_alpha1 = ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)])
            end
            forward_particles = get_nB(wf)
            forward_vacancies = length(wf.subsystemA) - get_nA(wf)
            reverse_particles = get_nA(wf) + 1
            reverse_vacancies = length(wf.subsystemB) - get_nB(wf) + 1
            weighting_factor = (forward_particles * forward_vacancies) / (reverse_particles * reverse_vacancies)
            probrat = weighting_factor * probrat_for_sign_two_copy_move_BtoA!(wf)
        else
            @assert false
        end
    end

    return probrat
end

function finish_move_for_sign!(wf::HardcoreSwapWavefunctionNS)
    @assert wf.automatically_reject == false

    # FIXME: change all wfs in VMC code to update the dets first, then the configs (to be anal)

    if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
        # printlnf("finishing alpha1 and alpha2 move")
        finish_phialpha1_update!(wf)
        finish_phialpha2_update!(wf)
        finish_phibeta1_update!(wf)
        finish_phibeta2_update!(wf)
    elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
        # printlnf("finishing alpha1 move")
        finish_phialpha1_update!(wf)
        if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move is within A
            # @assert in(wf.config_alpha1.particle_positions[1][wf.proposed_move_alpha1[1].particle.index], wf.subsystemA) && in(wf.proposed_move_alpha1[1].destination, wf.subsystemA) # FIXME: remove this
            finish_phibeta2_update!(wf)
        else # move is within B
            # @assert in(wf.config_alpha1.particle_positions[1][wf.proposed_move_alpha1[1].particle.index], wf.subsystemB) && in(wf.proposed_move_alpha1[1].destination, wf.subsystemB) # FIXME: remove this
            finish_phibeta1_update!(wf)
        end
    elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
        # printlnf("finishing alpha2 move")
        finish_phialpha2_update!(wf)
        if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move is within A
            # @assert in(wf.config_alpha2.particle_positions[1][wf.proposed_move_alpha2[1].particle.index], wf.subsystemA) && in(wf.proposed_move_alpha2[1].destination, wf.subsystemA) # FIXME: remove this
            finish_phibeta1_update!(wf)
        else # move is within B
            # @assert in(wf.config_alpha2.particle_positions[1][wf.proposed_move_alpha2[1].particle.index], wf.subsystemB) && in(wf.proposed_move_alpha2[1].destination, wf.subsystemB) # FIXME: remove this
            finish_phibeta2_update!(wf)
        end
    else
        warn("Finishing a completely trivial tranistion. Congratulations.")
    end

    # to be transparent
    nA_before_move = get_nA(wf)

    # if the move is trivial, update_configuration! doesn't do anything
    update_configuration!(wf.config_alpha1, wf.proposed_move_alpha1)
    update_configuration!(wf.config_alpha2, wf.proposed_move_alpha2)

    # update wf.nA and wf.nB, and exchange particles to keep those particles with lowest nA indices in subsystem A
    if wf.proposed_Delta_nA == -1
        wf.nA -= 1
        wf.nB += 1
        # printlnf("exchanging for A to B move")
        j1 = wf.proposed_move_alpha1[1].particle.index
        j2 = wf.proposed_move_alpha2[1].particle.index
        @assert j1 <= nA_before_move
        @assert j2 <= nA_before_move
        if j1 < nA_before_move # otherwise we're exchanging a particle with itself
            exchange_phialpha1!(wf, j1, nA_before_move)
            exchange_phibeta2!(wf, j1, nA_before_move)
            exchange_particles!(wf.config_alpha1, j1, nA_before_move, 1)
        end
        if j2 < nA_before_move # otherwise we're exchanging a particle with itself
            exchange_phialpha2!(wf, j2, nA_before_move)
            exchange_phibeta1!(wf, j2, nA_before_move)
            exchange_particles!(wf.config_alpha2, j2, nA_before_move, 1)
        end
    elseif wf.proposed_Delta_nA == 1
        wf.nA += 1
        wf.nB -= 1
        # printlnf("exchanging for B to A move")
        j1 = wf.proposed_move_alpha1[1].particle.index
        j2 = wf.proposed_move_alpha2[1].particle.index
        @assert j1 > nA_before_move
        @assert j2 > nA_before_move
        if j1 > nA_before_move+1 # otherwise we're exchanging a particle with itself
            exchange_phialpha1!(wf, j1, nA_before_move+1)
            exchange_phibeta1!(wf, j1, nA_before_move+1)
            exchange_particles!(wf.config_alpha1, j1, nA_before_move+1, 1)
        end
        if j2 > nA_before_move+1 # otherwise we're exchanging a particle with itself
            exchange_phialpha2!(wf, j2, nA_before_move+1)
            exchange_phibeta2!(wf, j2, nA_before_move+1)
            exchange_particles!(wf.config_alpha2, j2, nA_before_move+1, 1)
        end
    else
        @assert wf.proposed_Delta_nA == 0
        @assert nA_before_move == get_nA(wf)
    end

    return nothing
end

function cancel_move_for_sign!(wf::SwapWavefunctionNS)
    if !wf.automatically_reject
        if is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating alpha1 and alpha2
            # printlnf("cancelling alpha1 and alpha2 move")
            cancel_phialpha1_update!(wf)
            cancel_phialpha2_update!(wf)
            cancel_phibeta1_update!(wf)
            cancel_phibeta2_update!(wf)
        elseif is_nontrivial(wf.proposed_move_alpha1) && !is_nontrivial(wf.proposed_move_alpha2) # updating only alpha1
            # printlnf("cancelling alpha1 move")
            cancel_phialpha1_update!(wf)
            if wf.proposed_move_alpha1[1].particle.index <= get_nA(wf) # move is within A
                # @assert in(wf.config_alpha1.particle_positions[1][wf.proposed_move_alpha1[1].particle.index], wf.subsystemA) && in(wf.proposed_move_alpha1[1].destination, wf.subsystemA) # FIXME: remove this
                cancel_phibeta2_update!(wf)
            else # move is within B
                # @assert in(wf.config_alpha1.particle_positions[1][wf.proposed_move_alpha1[1].particle.index], wf.subsystemB) && in(wf.proposed_move_alpha1[1].destination, wf.subsystemB) # FIXME: remove this
                cancel_phibeta1_update!(wf)
            end
        elseif !is_nontrivial(wf.proposed_move_alpha1) && is_nontrivial(wf.proposed_move_alpha2) # updating only alpha2
            # printlnf("cancelling alpha2 move")
            cancel_phialpha2_update!(wf)
            if wf.proposed_move_alpha2[1].particle.index <= get_nA(wf) # move is within A
                # @assert in(wf.config_alpha2.particle_positions[1][wf.proposed_move_alpha2[1].particle.index], wf.subsystemA) && in(wf.proposed_move_alpha2[1].destination, wf.subsystemA) # FIXME: remove this
                cancel_phibeta1_update!(wf)
            else # move is within B
                # @assert in(wf.config_alpha2.particle_positions[1][wf.proposed_move_alpha2[1].particle.index], wf.subsystemB) && in(wf.proposed_move_alpha2[1].destination, wf.subsystemB) # FIXME: remove this
                cancel_phibeta2_update!(wf)
            end
        else
            warn("Cancelling a completely trivial tranistion. Good work.")
        end
    end

    return nothing
end

function random_neighbor_move(wf::HardcoreSwapWavefunctionNS)
    copy_to_move = rand(1:2)
    particle_to_move = Particle(rand(1:get_Nparticles(wf)), 1)

    if copy_to_move == 1
        current_site = wf.config_alpha1.particle_positions[1][particle_to_move.index]
    else # copy_to_move == 2
        current_site = wf.config_alpha2.particle_positions[1][particle_to_move.index]
    end
    proposed_position = wf.neighbor_table[current_site][rand(1:end)]

    return ParticlesMove([LatticeParticleMove(particle_to_move, proposed_position)]), copy_to_move
end

get_Nspecies(wf::HardcoreSwapWavefunctionNS) = 1
get_Nparticles(wf::HardcoreSwapWavefunctionNS) = get_Nparticles(wf.config_alpha1)[1]
get_Nsites(wf::HardcoreSwapWavefunctionNS) = get_Nsites(wf.config_alpha1)

# commented out these asserts as they're quite costly
function get_nA(wf::HardcoreSwapWavefunctionNS)
    # @assert get_Nparticles_in_subsystem(wf.config_alpha1, wf.subsystemA)[1] == get_Nparticles_in_subsystem(wf.config_alpha2, wf.subsystemA)[1] == wf.nA
    return wf.nA
end

function get_nB(wf::HardcoreSwapWavefunctionNS)
    # @assert get_Nparticles(wf) - get_nA(wf) == wf.nB
    return wf.nB
end

################################################################
#################### Free spinless fermions ####################
################################################################

type FreeFermionSwapWavefunctionNS{T} <: HardcoreSwapWavefunctionNS
    orbitals::FreeFermionOrbitals{T}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B (not constants anymore)
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

    # subsystem A occupation change for proposed move
    proposed_Delta_nA::Int

    # keep track if move will be automatically rejected due to Pauli exclusion / hard-core constraint
    automatically_reject::Bool

    function FreeFermionSwapWavefunctionNS(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int)
        Nsites = get_Nsites(orbs)
        Nparticles = get_Nfilled(orbs)

        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        @repeat n_attempts try
            config_alpha1, config_alpha2, config_beta1, config_beta2, subsystemB = initialize_random_hardcore_two_copy_configuration(Nsites, Nparticles, subsystemA, nA_initial)

            cmat_phialpha1 = CeperleyMatrix(fill_detmat(orbs, config_alpha1))
            cmat_phialpha2 = CeperleyMatrix(fill_detmat(orbs, config_alpha2))
            cmat_phibeta1 = CeperleyMatrix(fill_detmat(orbs, config_beta1))
            cmat_phibeta2 = CeperleyMatrix(fill_detmat(orbs, config_beta2))

            return new(orbs, config_alpha1, config_alpha2, subsystemA, subsystemB, nA_initial, Nparticles - nA_initial,
                       cmat_phialpha1, cmat_phialpha2, cmat_phibeta1, cmat_phibeta2, nbs)
        catch e
            printlnf("Caught an exception $e while creating wavefunction. Going to retry..")
            @retry if true end
        end
    end
end

function FreeFermionSwapWavefunctionNS{T}(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int = 20)
    return FreeFermionSwapWavefunctionNS{eltype(orbs.filled)}(orbs, subsystemA, nA_initial, nbs, n_attempts)
end

function FreeFermionSwapWavefunctionNS{T}(orbs::FreeFermionOrbitals{T}, subsystemA::Vector{Int}, nA_initial::Int, n_attempts::Int = 20)
    return FreeFermionSwapWavefunctionNS{eltype(orbs.filled)}(orbs, subsystemA, nA_initial, Vector{Int}[], n_attempts)
end

function print_wf_info(wf::FreeFermionSwapWavefunctionNS)
    # printlnf("inverse errors = ", check_inverse(wf.cmat_phialpha1)[2], ", ", check_inverse(wf.cmat_phialpha2)[2])
    # printlnf("det errors = ", relative_determinant_error(wf.cmat_phialpha1), ", ", relative_determinant_error(wf.cmat_phialpha2))
    # printlnf("approx det values = ", get_phialpha1(wf), ", ", get_phialpha2(wf))
    # printlnf("exact det values = ", get_phialpha1_exact_from_cmat(wf), ", ", get_phialpha2_exact_from_cmat(wf))
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::FreeFermionSwapWavefunctionNS) = get_det(wf.cmat_phialpha1)
get_phialpha2(wf::FreeFermionSwapWavefunctionNS) = get_det(wf.cmat_phialpha2)
get_phibeta1(wf::FreeFermionSwapWavefunctionNS) = get_det(wf.cmat_phibeta1)
get_phibeta2(wf::FreeFermionSwapWavefunctionNS) = get_det(wf.cmat_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::FreeFermionSwapWavefunctionNS) = get_det_exact(wf.cmat_phialpha1)
get_phialpha2_exact_from_cmat(wf::FreeFermionSwapWavefunctionNS) = get_det_exact(wf.cmat_phialpha2)
get_phibeta1_exact_from_swap(wf::FreeFermionSwapWavefunctionNS) = det(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::FreeFermionSwapWavefunctionNS) = det(fill_detmat(wf.orbitals, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

function check_for_numerical_error_for_mod!(wf::FreeFermionSwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_phialpha1, "cmat_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_phialpha2, "cmat_phialpha2")
end

function check_for_numerical_error_for_sign!(wf::FreeFermionSwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_phialpha1, "cmat_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_phialpha2, "cmat_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta1, "cmat_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_phibeta2, "cmat_phibeta2")
end

function probrat_for_mod_alpha1_move!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    return abs2(get_detrat(wf.cmat_phialpha1))
end

function probrat_for_mod_alpha2_move!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    return abs2(get_detrat(wf.cmat_phialpha2))
end

function probrat_for_sign_alpha1_move_inA!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_phialpha1)) * abs(get_detrat(wf.cmat_phibeta2))
end

function probrat_for_sign_alpha2_move_inA!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_phialpha2)) * abs(get_detrat(wf.cmat_phibeta1))
end

function probrat_for_sign_alpha1_move_inB!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_phialpha1)) * abs(get_detrat(wf.cmat_phibeta1))
end

function probrat_for_sign_alpha2_move_inB!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_phialpha2)) * abs(get_detrat(wf.cmat_phibeta2))
end

function probrat_for_sign_two_copy_move_AtoB!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_phialpha1)) * abs(get_detrat(wf.cmat_phialpha2)) * abs(get_detrat(wf.cmat_phibeta1)) * abs(get_detrat(wf.cmat_phibeta2))
end

function probrat_for_sign_two_copy_move_BtoA!(wf::FreeFermionSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_phialpha1)) * abs(get_detrat(wf.cmat_phialpha2)) * abs(get_detrat(wf.cmat_phibeta1)) * abs(get_detrat(wf.cmat_phibeta2))
end

function finish_phialpha1_update!(wf::FreeFermionSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_phialpha1)
end

function finish_phialpha2_update!(wf::FreeFermionSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_phialpha2)
end

function finish_phibeta1_update!(wf::FreeFermionSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_phibeta1)
end

function finish_phibeta2_update!(wf::FreeFermionSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_phibeta2)
end

function cancel_phialpha1_update!(wf::FreeFermionSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_phialpha1)
end

function cancel_phialpha2_update!(wf::FreeFermionSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_phialpha2)
end

function cancel_phibeta1_update!(wf::FreeFermionSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_phibeta1)
end

function cancel_phibeta2_update!(wf::FreeFermionSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_phibeta2)
end

function exchange_phialpha1!(wf::FreeFermionSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_phialpha1, p1, p2)
end

function exchange_phialpha2!(wf::FreeFermionSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_phialpha2, p1, p2)
end

function exchange_phibeta1!(wf::FreeFermionSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_phibeta1, p1, p2)
end

function exchange_phibeta2!(wf::FreeFermionSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_phibeta2, p1, p2)
end

#################################################################################
#################### Gutz2, e.g., bCFL = det(nu=1) * det(FS) ####################
#################################################################################

type Gutz2SwapWavefunctionNS{Td1,Td2} <: HardcoreSwapWavefunctionNS
    orbitals_d1::FreeFermionOrbitals{Td1}
    orbitals_d2::FreeFermionOrbitals{Td2}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B (not constants anymore)
    nA::Int
    nB::Int

    # Ceperley matrices for swapped and unswapped configurations of the 2 copies
    # NB1: for wavefunctions with a product of n determinants, we'll need 4n Ceperley matrix objects here
    # NB2: not all schemes for swap mod require keeping track of the phibeta's, but I don't think there's much harm in at least initializing them regardless
    cmat_d1_phialpha1::CeperleyMatrix{Td1}
    cmat_d1_phialpha2::CeperleyMatrix{Td1}
    cmat_d1_phibeta1::CeperleyMatrix{Td1}
    cmat_d1_phibeta2::CeperleyMatrix{Td1}
    cmat_d2_phialpha1::CeperleyMatrix{Td2}
    cmat_d2_phialpha2::CeperleyMatrix{Td2}
    cmat_d2_phibeta1::CeperleyMatrix{Td2}
    cmat_d2_phibeta2::CeperleyMatrix{Td2}

    # just an ordinary neighbor table
    neighbor_table::Vector{Vector{Int}}

    # moves for the 2 copies
    proposed_move_alpha1::ParticlesMove{LatticeParticleMove}
    proposed_move_alpha2::ParticlesMove{LatticeParticleMove}

    # subsystem A occupation change for proposed move
    proposed_Delta_nA::Int

    # keep track if move will be automatically rejected due to Pauli exclusion / hard-core constraint
    automatically_reject::Bool

    function Gutz2SwapWavefunctionNS(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2},
                                     subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int)
        @assert get_Nsites(orbs_d1) == get_Nsites(orbs_d2)
        @assert get_Nfilled(orbs_d1) == get_Nfilled(orbs_d2)
        Nsites = get_Nsites(orbs_d1)
        Nparticles = get_Nfilled(orbs_d1)

        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        @repeat n_attempts try
            config_alpha1, config_alpha2, config_beta1, config_beta2, subsystemB = initialize_random_hardcore_two_copy_configuration(Nsites, Nparticles, subsystemA, nA_initial)

            cmat_d1_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha1))
            cmat_d1_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha2))
            cmat_d1_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta1))
            cmat_d1_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta2))
            cmat_d2_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha1))
            cmat_d2_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha2))
            cmat_d2_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta1))
            cmat_d2_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta2))

            return new(orbs_d1, orbs_d2,
                       config_alpha1, config_alpha2, subsystemA, subsystemB, nA_initial, Nparticles - nA_initial,
                       cmat_d1_phialpha1, cmat_d1_phialpha2, cmat_d1_phibeta1, cmat_d1_phibeta2,
                       cmat_d2_phialpha1, cmat_d2_phialpha2, cmat_d2_phibeta1, cmat_d2_phibeta2,
                       nbs)
        catch e
            printlnf("Caught an exception $e while creating wavefunction. Going to retry..")
            @retry if true end
        end
    end
end

function Gutz2SwapWavefunctionNS{Td1,Td2}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2},
                                          subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int = 20)
    return Gutz2SwapWavefunctionNS{eltype(orbs_d1.filled),eltype(orbs_d2.filled)}(orbs_d1, orbs_d2, subsystemA, nA_initial, nbs, n_attempts)
end

function Gutz2SwapWavefunctionNS{Td1,Td2}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2},
                                          subsystemA::Vector{Int}, nA_initial::Int, n_attempts::Int = 20)
    return Gutz2SwapWavefunctionNS{eltype(orbs_d1.filled),eltype(orbs_d2.filled)}(orbs_d1, orbs_d2, subsystemA, nA_initial, Vector{Int}[], n_attempts)
end

function print_wf_info(wf::Gutz2SwapWavefunctionNS)
    # printlnf("d1 inverse errors = ", check_inverse(wf.cmat_d1_phialpha1)[2], ", ", check_inverse(wf.cmat_d1_phialpha2)[2])
    # printlnf("d2 inverse errors = ", check_inverse(wf.cmat_d2_phialpha1)[2], ", ", check_inverse(wf.cmat_d2_phialpha2)[2])
    # printlnf("d1 det errors = ", relative_determinant_error(wf.cmat_d1_phialpha1), ", ", relative_determinant_error(wf.cmat_d1_phialpha2))
    # printlnf("d2 det errors = ", relative_determinant_error(wf.cmat_d2_phialpha1), ", ", relative_determinant_error(wf.cmat_d2_phialpha2))
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::Gutz2SwapWavefunctionNS) = get_det(wf.cmat_d1_phialpha1) * get_det(wf.cmat_d2_phialpha1)
get_phialpha2(wf::Gutz2SwapWavefunctionNS) = get_det(wf.cmat_d1_phialpha2) * get_det(wf.cmat_d2_phialpha2)
get_phibeta1(wf::Gutz2SwapWavefunctionNS) = get_det(wf.cmat_d1_phibeta1) * get_det(wf.cmat_d2_phibeta1)
get_phibeta2(wf::Gutz2SwapWavefunctionNS) = get_det(wf.cmat_d1_phibeta2) * get_det(wf.cmat_d2_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::Gutz2SwapWavefunctionNS) = get_det_exact(wf.cmat_d1_phialpha1) * get_det_exact(wf.cmat_d2_phialpha1)
get_phialpha2_exact_from_cmat(wf::Gutz2SwapWavefunctionNS) = get_det_exact(wf.cmat_d1_phialpha2) * get_det_exact(wf.cmat_d2_phialpha2)
get_phibeta1_exact_from_swap(wf::Gutz2SwapWavefunctionNS) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                            det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::Gutz2SwapWavefunctionNS) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                            det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

function check_for_numerical_error_for_mod!(wf::Gutz2SwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
end

function check_for_numerical_error_for_sign!(wf::Gutz2SwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta1, "cmat_d1_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta2, "cmat_d1_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta1, "cmat_d2_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta2, "cmat_d2_phibeta2")
end

function probrat_for_mod_alpha1_move!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    return abs2(get_detrat(wf.cmat_d1_phialpha1)) *
           abs2(get_detrat(wf.cmat_d2_phialpha1))
end

function probrat_for_mod_alpha2_move!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    return abs2(get_detrat(wf.cmat_d1_phialpha2)) *
           abs2(get_detrat(wf.cmat_d2_phialpha2))
end

function probrat_for_sign_alpha1_move_inA!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phibeta2))
end

function probrat_for_sign_alpha2_move_inA!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) *
           abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1))
end

function probrat_for_sign_alpha1_move_inB!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phibeta1)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phibeta1))
end

function probrat_for_sign_alpha2_move_inB!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta2))
end

function probrat_for_sign_two_copy_move_AtoB!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1)) * abs(get_detrat(wf.cmat_d2_phibeta2))
end

function probrat_for_sign_two_copy_move_BtoA!(wf::Gutz2SwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1)) * abs(get_detrat(wf.cmat_d2_phibeta2))
end

function finish_phialpha1_update!(wf::Gutz2SwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d1_phialpha1)
    finish_columns_update!(wf.cmat_d2_phialpha1)
end

function finish_phialpha2_update!(wf::Gutz2SwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d1_phialpha2)
    finish_columns_update!(wf.cmat_d2_phialpha2)
end

function finish_phibeta1_update!(wf::Gutz2SwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d1_phibeta1)
    finish_columns_update!(wf.cmat_d2_phibeta1)
end

function finish_phibeta2_update!(wf::Gutz2SwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d1_phibeta2)
    finish_columns_update!(wf.cmat_d2_phibeta2)
end

function cancel_phialpha1_update!(wf::Gutz2SwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d1_phialpha1)
    cancel_columns_update!(wf.cmat_d2_phialpha1)
end

function cancel_phialpha2_update!(wf::Gutz2SwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d1_phialpha2)
    cancel_columns_update!(wf.cmat_d2_phialpha2)
end

function cancel_phibeta1_update!(wf::Gutz2SwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d1_phibeta1)
    cancel_columns_update!(wf.cmat_d2_phibeta1)
end

function cancel_phibeta2_update!(wf::Gutz2SwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d1_phibeta2)
    cancel_columns_update!(wf.cmat_d2_phibeta2)
end

function exchange_phialpha1!(wf::Gutz2SwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phialpha1, p1, p2)
    swap_columns!(wf.cmat_d2_phialpha1, p1, p2)
end

function exchange_phialpha2!(wf::Gutz2SwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phialpha2, p1, p2)
    swap_columns!(wf.cmat_d2_phialpha2, p1, p2)
end

function exchange_phibeta1!(wf::Gutz2SwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phibeta1, p1, p2)
    swap_columns!(wf.cmat_d2_phibeta1, p1, p2)
end

function exchange_phibeta2!(wf::Gutz2SwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phibeta2, p1, p2)
    swap_columns!(wf.cmat_d2_phibeta2, p1, p2)
end

############################################################################################
#################### Gutz3, e.g., CFL = det(nu=1) * det(nu=1) * det(FS) ####################
############################################################################################

macro make_Gutz3SwapWavefunctionNS(name)
quote

type $(esc(name)){Td1,Td2,Td3} <: HardcoreSwapWavefunctionNS
    orbitals_d1::FreeFermionOrbitals{Td1}
    orbitals_d2::FreeFermionOrbitals{Td2}
    orbitals_d3::FreeFermionOrbitals{Td3}

    # configurations for the 2 copies
    config_alpha1::LatticeParticlesConfiguration
    config_alpha2::LatticeParticlesConfiguration

    # vectors indicating which sites are in subsystems A and B
    subsystemA::Vector{Int}
    subsystemB::Vector{Int} # all sites minus those in A

    # number of particles in subsystems A and B (not constants anymore)
    nA::Int
    nB::Int

    # Ceperley matrices for swapped and unswapped configurations of the 2 copies
    # NB1: for wavefunctions with a product of n determinants, we'll need 4n Ceperley matrix objects here
    # NB2: not all schemes for swap mod require keeping track of the phibeta's, but I don't think there's much harm in at least initializing them regardless
    # NB3: same goes for the d1 and d2 dets in the "presign" wavefunctions
    cmat_d1_phialpha1::CeperleyMatrix{Td1}
    cmat_d1_phialpha2::CeperleyMatrix{Td1}
    cmat_d1_phibeta1::CeperleyMatrix{Td1}
    cmat_d1_phibeta2::CeperleyMatrix{Td1}
    cmat_d2_phialpha1::CeperleyMatrix{Td2}
    cmat_d2_phialpha2::CeperleyMatrix{Td2}
    cmat_d2_phibeta1::CeperleyMatrix{Td2}
    cmat_d2_phibeta2::CeperleyMatrix{Td2}
    cmat_d3_phialpha1::CeperleyMatrix{Td3}
    cmat_d3_phialpha2::CeperleyMatrix{Td3}
    cmat_d3_phibeta1::CeperleyMatrix{Td3}
    cmat_d3_phibeta2::CeperleyMatrix{Td3}

    # just an ordinary neighbor table
    neighbor_table::Vector{Vector{Int}}

    # moves for the 2 copies
    proposed_move_alpha1::ParticlesMove{LatticeParticleMove}
    proposed_move_alpha2::ParticlesMove{LatticeParticleMove}

    # subsystem A occupation change for proposed move
    proposed_Delta_nA::Int

    # keep track if move will be automatically rejected due to Pauli exclusion / hard-core constraint
    automatically_reject::Bool

    function $(esc(name))(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_d3::FreeFermionOrbitals{Td3},
                          subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int)
        @assert get_Nsites(orbs_d1) == get_Nsites(orbs_d2) == get_Nsites(orbs_d3)
        @assert get_Nfilled(orbs_d1) == get_Nfilled(orbs_d2) == get_Nfilled(orbs_d3)
        Nsites = get_Nsites(orbs_d1)
        Nparticles = get_Nfilled(orbs_d1)

        if ~isempty(nbs)
            @assert length(nbs) == Nsites
        end

        @repeat n_attempts try
            config_alpha1, config_alpha2, config_beta1, config_beta2, subsystemB = initialize_random_hardcore_two_copy_configuration(Nsites, Nparticles, subsystemA, nA_initial)

            cmat_d1_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha1))
            cmat_d1_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d1, config_alpha2))
            cmat_d1_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta1))
            cmat_d1_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d1, config_beta2))
            cmat_d2_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha1))
            cmat_d2_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d2, config_alpha2))
            cmat_d2_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta1))
            cmat_d2_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d2, config_beta2))
            cmat_d3_phialpha1 = CeperleyMatrix(fill_detmat(orbs_d3, config_alpha1))
            cmat_d3_phialpha2 = CeperleyMatrix(fill_detmat(orbs_d3, config_alpha2))
            cmat_d3_phibeta1 = CeperleyMatrix(fill_detmat(orbs_d3, config_beta1))
            cmat_d3_phibeta2 = CeperleyMatrix(fill_detmat(orbs_d3, config_beta2))

            return new(orbs_d1, orbs_d2, orbs_d3,
                       config_alpha1, config_alpha2, subsystemA, subsystemB, nA_initial, Nparticles - nA_initial,
                       cmat_d1_phialpha1, cmat_d1_phialpha2, cmat_d1_phibeta1, cmat_d1_phibeta2,
                       cmat_d2_phialpha1, cmat_d2_phialpha2, cmat_d2_phibeta1, cmat_d2_phibeta2,
                       cmat_d3_phialpha1, cmat_d3_phialpha2, cmat_d3_phibeta1, cmat_d3_phibeta2,
                       nbs)
        catch e
            printlnf("Caught an exception $e while creating wavefunction. Going to retry..")
            @retry if true end
        end
    end
end

function $(esc(name)){Td1,Td2,Td3}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_d3::FreeFermionOrbitals{Td3},
                                   subsystemA::Vector{Int}, nA_initial::Int, nbs::Vector{Vector{Int}}, n_attempts::Int = 20)
    return $(esc(name)){eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_d3.filled)}(orbs_d1, orbs_d2, orbs_d3, subsystemA, nA_initial, nbs, n_attempts)
end

function $(esc(name)){Td1,Td2,Td3}(orbs_d1::FreeFermionOrbitals{Td1}, orbs_d2::FreeFermionOrbitals{Td2}, orbs_d3::FreeFermionOrbitals{Td3},
                                   subsystemA::Vector{Int}, nA_initial::Int, n_attempts::Int = 20)
    return $(esc(name)){eltype(orbs_d1.filled),eltype(orbs_d2.filled),eltype(orbs_d3.filled)}(orbs_d1, orbs_d2, orbs_d3, subsystemA, nA_initial, Vector{Int}[], n_attempts)
end

end # quote
end # macro

@make_Gutz3SwapWavefunctionNS Gutz3SwapWavefunctionNS # c = d1 * d2 * d3
@make_Gutz3SwapWavefunctionNS Gutz3premodSwapWavefunctionNS # c = |d1 * d2| * d3
@make_Gutz3SwapWavefunctionNS Gutz3presignSwapWavefunctionNS # c = sign(d1 * d2) * d3

function print_wf_info(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    # printlnf("d1 inverse errors = ", check_inverse(wf.cmat_d1_phialpha1)[2], ", ", check_inverse(wf.cmat_d1_phialpha2)[2])
    # printlnf("d2 inverse errors = ", check_inverse(wf.cmat_d2_phialpha1)[2], ", ", check_inverse(wf.cmat_d2_phialpha2)[2])
    # printlnf("d3 inverse errors = ", check_inverse(wf.cmat_d3_phialpha1)[2], ", ", check_inverse(wf.cmat_d3_phialpha2)[2])
    # printlnf("d1 det errors = ", relative_determinant_error(wf.cmat_d1_phialpha1), ", ", relative_determinant_error(wf.cmat_d1_phialpha2))
    # printlnf("d2 det errors = ", relative_determinant_error(wf.cmat_d2_phialpha1), ", ", relative_determinant_error(wf.cmat_d2_phialpha2))
    # printlnf("d3 det errors = ", relative_determinant_error(wf.cmat_d3_phialpha1), ", ", relative_determinant_error(wf.cmat_d3_phialpha2))
end

function print_wf_info(wf::Gutz3presignSwapWavefunctionNS)
    # printlnf("d3 inverse errors = ", check_inverse(wf.cmat_d3_phialpha1)[2], ", ", check_inverse(wf.cmat_d3_phialpha2)[2])
    # printlnf("d3 det errors = ", relative_determinant_error(wf.cmat_d3_phialpha1), ", ", relative_determinant_error(wf.cmat_d3_phialpha2))
end

# these wavefunction amplitudes are used in the measurements
get_phialpha1(wf::Gutz3SwapWavefunctionNS) = get_det(wf.cmat_d1_phialpha1) * get_det(wf.cmat_d2_phialpha1) * get_det(wf.cmat_d3_phialpha1)
get_phialpha2(wf::Gutz3SwapWavefunctionNS) = get_det(wf.cmat_d1_phialpha2) * get_det(wf.cmat_d2_phialpha2) * get_det(wf.cmat_d3_phialpha2)
get_phibeta1(wf::Gutz3SwapWavefunctionNS) = get_det(wf.cmat_d1_phibeta1) * get_det(wf.cmat_d2_phibeta1) * get_det(wf.cmat_d3_phibeta1)
get_phibeta2(wf::Gutz3SwapWavefunctionNS) = get_det(wf.cmat_d1_phibeta2) * get_det(wf.cmat_d2_phibeta2) * get_det(wf.cmat_d3_phibeta2)

get_phialpha1(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det(wf.cmat_d1_phialpha1)) * abs(get_det(wf.cmat_d2_phialpha1)) * get_det(wf.cmat_d3_phialpha1)
get_phialpha2(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det(wf.cmat_d1_phialpha2)) * abs(get_det(wf.cmat_d2_phialpha2)) * get_det(wf.cmat_d3_phialpha2)
get_phibeta1(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det(wf.cmat_d1_phibeta1)) * abs(get_det(wf.cmat_d2_phibeta1)) * get_det(wf.cmat_d3_phibeta1)
get_phibeta2(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det(wf.cmat_d1_phibeta2)) * abs(get_det(wf.cmat_d2_phibeta2)) * get_det(wf.cmat_d3_phibeta2)

get_phialpha1(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, wf.config_alpha1))) *
                                                    sign(det(fill_detmat(wf.orbitals_d2, wf.config_alpha1))) *
                                                    get_det(wf.cmat_d3_phialpha1)
get_phialpha2(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, wf.config_alpha2))) *
                                                    sign(det(fill_detmat(wf.orbitals_d2, wf.config_alpha2))) *
                                                    get_det(wf.cmat_d3_phialpha2)
get_phibeta1(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                   sign(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                   get_det(wf.cmat_d3_phibeta1)
get_phibeta2(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                   sign(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                   get_det(wf.cmat_d3_phibeta2)

# safe(r) versions
get_phialpha1_exact_from_cmat(wf::Gutz3SwapWavefunctionNS) = get_det_exact(wf.cmat_d1_phialpha1) * get_det_exact(wf.cmat_d2_phialpha1) * get_det_exact(wf.cmat_d3_phialpha1)
get_phialpha2_exact_from_cmat(wf::Gutz3SwapWavefunctionNS) = get_det_exact(wf.cmat_d1_phialpha2) * get_det_exact(wf.cmat_d2_phialpha2) * get_det_exact(wf.cmat_d3_phialpha2)
get_phibeta1_exact_from_swap(wf::Gutz3SwapWavefunctionNS) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                            det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1])) *
                                                            det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::Gutz3SwapWavefunctionNS) = det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                            det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2])) *
                                                            det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

get_phialpha1_exact_from_cmat(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det_exact(wf.cmat_d1_phialpha1)) * abs(get_det_exact(wf.cmat_d2_phialpha1)) * get_det_exact(wf.cmat_d3_phialpha1)
get_phialpha2_exact_from_cmat(wf::Gutz3premodSwapWavefunctionNS) = abs(get_det_exact(wf.cmat_d1_phialpha2)) * abs(get_det_exact(wf.cmat_d2_phialpha2)) * get_det_exact(wf.cmat_d3_phialpha2)
get_phibeta1_exact_from_swap(wf::Gutz3premodSwapWavefunctionNS) = abs(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                                  abs(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                                  det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::Gutz3premodSwapWavefunctionNS) = abs(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                                  abs(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                                  det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

get_phialpha1_exact_from_cmat(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, wf.config_alpha1))) *
                                                                    sign(det(fill_detmat(wf.orbitals_d2, wf.config_alpha1))) *
                                                                    get_det_exact(wf.cmat_d3_phialpha1)
get_phialpha2_exact_from_cmat(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, wf.config_alpha2))) *
                                                                    sign(det(fill_detmat(wf.orbitals_d2, wf.config_alpha2))) *
                                                                    get_det_exact(wf.cmat_d3_phialpha2)
get_phibeta1_exact_from_swap(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                                   sign(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))) *
                                                                   det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[1]))
get_phibeta2_exact_from_swap(wf::Gutz3presignSwapWavefunctionNS) = sign(det(fill_detmat(wf.orbitals_d1, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                                   sign(det(fill_detmat(wf.orbitals_d2, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))) *
                                                                   det(fill_detmat(wf.orbitals_d3, swap(wf.config_alpha1, wf.config_alpha2, wf.subsystemA)[2]))

function check_for_numerical_error_for_mod!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha1, "cmat_d3_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha2, "cmat_d3_phialpha2")
end

function check_for_numerical_error_for_mod!(wf::Gutz3presignSwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha1, "cmat_d3_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha2, "cmat_d3_phialpha2")
end

function check_for_numerical_error_for_sign!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha1, "cmat_d1_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phialpha2, "cmat_d1_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta1, "cmat_d1_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d1_phibeta2, "cmat_d1_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha1, "cmat_d2_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phialpha2, "cmat_d2_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta1, "cmat_d2_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d2_phibeta2, "cmat_d2_phibeta2")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha1, "cmat_d3_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha2, "cmat_d3_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phibeta1, "cmat_d3_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phibeta2, "cmat_d3_phibeta2")
end

function check_for_numerical_error_for_sign!(wf::Gutz3presignSwapWavefunctionNS)
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha1, "cmat_d3_phialpha1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phialpha2, "cmat_d3_phialpha2")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phibeta1, "cmat_d3_phibeta1")
    check_and_reset_inverse_and_det!(wf.cmat_d3_phibeta2, "cmat_d3_phibeta2")
end

function probrat_for_mod_alpha1_move!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs2(get_detrat(wf.cmat_d1_phialpha1)) *
           abs2(get_detrat(wf.cmat_d2_phialpha1)) *
           abs2(get_detrat(wf.cmat_d3_phialpha1))
end

function probrat_for_mod_alpha1_move!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs2(get_detrat(wf.cmat_d3_phialpha1))
end

function probrat_for_mod_alpha2_move!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs2(get_detrat(wf.cmat_d1_phialpha2)) *
           abs2(get_detrat(wf.cmat_d2_phialpha2)) *
           abs2(get_detrat(wf.cmat_d3_phialpha2))
end

function probrat_for_mod_alpha2_move!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs2(get_detrat(wf.cmat_d3_phialpha2))
end

function probrat_for_sign_alpha1_move_inA!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phibeta2)) *
           abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_alpha1_move_inA!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_alpha2_move_inA!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) *
           abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1)) *
           abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1))
end

function probrat_for_sign_alpha2_move_inA!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1))
end

function probrat_for_sign_alpha1_move_inB!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phibeta1)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phibeta1)) *
           abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phibeta1))
end

function probrat_for_sign_alpha1_move_inB!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phibeta1))
end

function probrat_for_sign_alpha2_move_inB!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta2)) *
           abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_alpha2_move_inB!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_two_copy_move_AtoB!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1)) * abs(get_detrat(wf.cmat_d2_phibeta2)) *
           abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_two_copy_move_AtoB!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_two_copy_move_BtoA!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    detrat_from_columns_update!(wf.cmat_d1_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d1_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d1, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d2_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d2, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d1_phialpha1)) * abs(get_detrat(wf.cmat_d1_phialpha2)) * abs(get_detrat(wf.cmat_d1_phibeta1)) * abs(get_detrat(wf.cmat_d1_phibeta2)) *
           abs(get_detrat(wf.cmat_d2_phialpha1)) * abs(get_detrat(wf.cmat_d2_phialpha2)) * abs(get_detrat(wf.cmat_d2_phibeta1)) * abs(get_detrat(wf.cmat_d2_phibeta2)) *
           abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function probrat_for_sign_two_copy_move_BtoA!(wf::Gutz3presignSwapWavefunctionNS)
    detrat_from_columns_update!(wf.cmat_d3_phialpha1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phialpha2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta1, wf.proposed_move_alpha1[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha2[1].destination))
    detrat_from_columns_update!(wf.cmat_d3_phibeta2, wf.proposed_move_alpha2[1].particle.index, columns_for_detmat(wf.orbitals_d3, wf.proposed_move_alpha1[1].destination))
    return abs(get_detrat(wf.cmat_d3_phialpha1)) * abs(get_detrat(wf.cmat_d3_phialpha2)) * abs(get_detrat(wf.cmat_d3_phibeta1)) * abs(get_detrat(wf.cmat_d3_phibeta2))
end

function finish_phialpha1_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    finish_columns_update!(wf.cmat_d1_phialpha1)
    finish_columns_update!(wf.cmat_d2_phialpha1)
    finish_columns_update!(wf.cmat_d3_phialpha1)
end

function finish_phialpha1_update!(wf::Gutz3presignSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d3_phialpha1)
end

function finish_phialpha2_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    finish_columns_update!(wf.cmat_d1_phialpha2)
    finish_columns_update!(wf.cmat_d2_phialpha2)
    finish_columns_update!(wf.cmat_d3_phialpha2)
end

function finish_phialpha2_update!(wf::Gutz3presignSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d3_phialpha2)
end

function finish_phibeta1_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    finish_columns_update!(wf.cmat_d1_phibeta1)
    finish_columns_update!(wf.cmat_d2_phibeta1)
    finish_columns_update!(wf.cmat_d3_phibeta1)
end

function finish_phibeta1_update!(wf::Gutz3presignSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d3_phibeta1)
end

function finish_phibeta2_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    finish_columns_update!(wf.cmat_d1_phibeta2)
    finish_columns_update!(wf.cmat_d2_phibeta2)
    finish_columns_update!(wf.cmat_d3_phibeta2)
end

function finish_phibeta2_update!(wf::Gutz3presignSwapWavefunctionNS)
    finish_columns_update!(wf.cmat_d3_phibeta2)
end

function cancel_phialpha1_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    cancel_columns_update!(wf.cmat_d1_phialpha1)
    cancel_columns_update!(wf.cmat_d2_phialpha1)
    cancel_columns_update!(wf.cmat_d3_phialpha1)
end

function cancel_phialpha1_update!(wf::Gutz3presignSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d3_phialpha1)
end

function cancel_phialpha2_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    cancel_columns_update!(wf.cmat_d1_phialpha2)
    cancel_columns_update!(wf.cmat_d2_phialpha2)
    cancel_columns_update!(wf.cmat_d3_phialpha2)
end

function cancel_phialpha2_update!(wf::Gutz3presignSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d3_phialpha2)
end

function cancel_phibeta1_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    cancel_columns_update!(wf.cmat_d1_phibeta1)
    cancel_columns_update!(wf.cmat_d2_phibeta1)
    cancel_columns_update!(wf.cmat_d3_phibeta1)
end

function cancel_phibeta1_update!(wf::Gutz3presignSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d3_phibeta1)
end

function cancel_phibeta2_update!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS})
    cancel_columns_update!(wf.cmat_d1_phibeta2)
    cancel_columns_update!(wf.cmat_d2_phibeta2)
    cancel_columns_update!(wf.cmat_d3_phibeta2)
end

function cancel_phibeta2_update!(wf::Gutz3presignSwapWavefunctionNS)
    cancel_columns_update!(wf.cmat_d3_phibeta2)
end

function exchange_phialpha1!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS}, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phialpha1, p1, p2)
    swap_columns!(wf.cmat_d2_phialpha1, p1, p2)
    swap_columns!(wf.cmat_d3_phialpha1, p1, p2)
end

function exchange_phialpha1!(wf::Gutz3presignSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d3_phialpha1, p1, p2)
end

function exchange_phialpha2!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS}, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phialpha2, p1, p2)
    swap_columns!(wf.cmat_d2_phialpha2, p1, p2)
    swap_columns!(wf.cmat_d3_phialpha2, p1, p2)
end

function exchange_phialpha2!(wf::Gutz3presignSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d3_phialpha2, p1, p2)
end

function exchange_phibeta1!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS}, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phibeta1, p1, p2)
    swap_columns!(wf.cmat_d2_phibeta1, p1, p2)
    swap_columns!(wf.cmat_d3_phibeta1, p1, p2)
end

function exchange_phibeta1!(wf::Gutz3presignSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d3_phibeta1, p1, p2)
end

function exchange_phibeta2!(wf::Union{Gutz3SwapWavefunctionNS, Gutz3premodSwapWavefunctionNS}, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d1_phibeta2, p1, p2)
    swap_columns!(wf.cmat_d2_phibeta2, p1, p2)
    swap_columns!(wf.cmat_d3_phibeta2, p1, p2)
end

function exchange_phibeta2!(wf::Gutz3presignSwapWavefunctionNS, p1::Int, p2::Int)
    swap_columns!(wf.cmat_d3_phibeta2, p1, p2)
end
