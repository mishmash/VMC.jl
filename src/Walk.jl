# include("Move.jl")
# include("Wavefunction.jl")
# include("SwapWavefunction.jl")

abstract Walk

is_nontrivial_transition(walk::Walk) = walk.transition_is_nontrivial

function get_probability_ratio(walk::Walk)
    if !walk.transition_is_nontrivial
        warn("Asking for the probability ratio ( = 1) of a trivial transition.")
        return 1.
    else
        return walk.probability_ratio
    end
end

# Lots of code copying below since Julia does't really have abstract base classes.  There has to be a better way..
# I could have one class with finish_move! and cancel_move! functions passed to the constructor and held as members,
# but that seems less safe and less explicit.

# standard walk for ordinary wavefunctions / measurements
type StandardWalk <: Walk
    wf::Wavefunction
    transition_function::Function
    probability_ratio::Float64
    transition_is_nontrivial::Bool
    # transition_in_progress::Bool
    StandardWalk(wf::Wavefunction, transition_function::Function) = new(wf, transition_function)
end

function propose_and_assess_random_transition!(walk::StandardWalk)
    move = walk.transition_function(walk.wf)

    if !is_nontrivial(move)
        walk.transition_is_nontrivial = false
    else
        probrat = assess_move!(walk.wf, move)
        walk.transition_is_nontrivial = true
        walk.probability_ratio = probrat
    end
end

accept_transition!(walk::StandardWalk) = finish_move!(walk.wf)
reject_transition!(walk::StandardWalk) = cancel_move!(walk.wf)

check_for_numerical_error!(walk::StandardWalk) = check_for_numerical_error!(walk.wf)


# walk for Renyi mod measurement
type RenyiModWalk <: Walk
    wf::Union{SwapWavefunction, SwapWavefunctionNS}
    transition_function::Function
    probability_ratio::Float64
    transition_is_nontrivial::Bool
    # transition_in_progress::Bool
    RenyiModWalk(wf::Union{SwapWavefunction, SwapWavefunctionNS}, transition_function::Function) = new(wf, transition_function)
end

function propose_and_assess_random_transition!(walk::RenyiModWalk)
    move = walk.transition_function(walk.wf)
    # move here is either a 2-tuple of moves for the different copies (SwapWavefunction) or a Tuple{ParticlesMove{LatticeParticleMove},Int64} (SwapWavefunctionNS),
    # where the Int here is the copy_to_move
    @assert length(move) == 2

    if isa(move[2], Int) # for SwapWavefunctionNS
        trivial_condition = !is_nontrivial(move[1])
    else # for SwapWavefunction
        trivial_condition = all(map(!, map(is_nontrivial, move)))
    end

    if trivial_condition
        walk.transition_is_nontrivial = false
    else
        probrat = assess_move_for_mod!(walk.wf, move...)
        walk.transition_is_nontrivial = true
        walk.probability_ratio = probrat
    end
end

accept_transition!(walk::RenyiModWalk) = finish_move_for_mod!(walk.wf)
reject_transition!(walk::RenyiModWalk) = cancel_move_for_mod!(walk.wf)

check_for_numerical_error!(walk::RenyiModWalk) = check_for_numerical_error_for_mod!(walk.wf)


# walk for Renyi sign measurement
type RenyiSignWalk <: Walk
    wf::Union{SwapWavefunction, SwapWavefunctionNS}
    transition_function::Function
    probability_ratio::Float64
    transition_is_nontrivial::Bool
    # transition_in_progress::Bool
    RenyiSignWalk(wf::Union{SwapWavefunction, SwapWavefunctionNS}, transition_function::Function) = new(wf, transition_function)
end

function propose_and_assess_random_transition!(walk::RenyiSignWalk)
    move = walk.transition_function(walk.wf)
    @assert length(move) == 2 # a 2-tuple of moves for the different copies

    if isa(move[2], Int) # for SwapWavefunctionNS
        trivial_condition = !is_nontrivial(move[1])
    else # for SwapWavefunction
        trivial_condition = all(map(!, map(is_nontrivial, move)))
    end

    if trivial_condition
        walk.transition_is_nontrivial = false
    else
        probrat = assess_move_for_sign!(walk.wf, move...)
        walk.transition_is_nontrivial = true
        walk.probability_ratio = probrat
    end
end

accept_transition!(walk::RenyiSignWalk) = finish_move_for_sign!(walk.wf)
reject_transition!(walk::RenyiSignWalk) = cancel_move_for_sign!(walk.wf)

check_for_numerical_error!(walk::RenyiSignWalk) = check_for_numerical_error_for_sign!(walk.wf)


# type RenyiWalk <: Walk
#     wf::SwapWavefunction
#     transition_function::Function
#     probability_ratio::Float64
#     transition_is_nontrivial::Bool
#     # transition_in_progress::Bool
#     RenyiWalk(wf::SwapWavefunction, transition_function::Function) = new(wf, transition_function)
# end

# type RenyiModWalk
#     walk::RenyiWalk
#     RenyiModWalk(wf::SwapWavefunction, transition_function::Function) = new(RenyiWalk(wf, transition_function))
# end

# How to then make, e.g., RenyiModWalk.wf invoke RenyiModWalk.walk.wf?
