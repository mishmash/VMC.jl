# include("Walk.jl")

abstract Simulation

type MetropolisSimulation <: Simulation
    walk::Walk

    accepted_steps::Int
    rejected_steps::Int
    # automatically_rejected_steps::Int # FIXME: add this
    trivial_steps::Int
    total_steps::Int

    # beeps::Int # FIXME: add this to keep track of numerical error rate

    measurements::Dict{ASCIIString, Measurement}

    function MetropolisSimulation(walk::Walk)
        return new(walk, 0, 0, 0, 0, Dict{ASCIIString, Measurement}())
    end
end

function add_accepted_step!(sim::MetropolisSimulation)
    sim.accepted_steps += 1
    sim.total_steps += 1
end

function add_rejected_step!(sim::MetropolisSimulation)
    sim.rejected_steps += 1
    sim.total_steps += 1
end

# FIXME: add this
# function add_automatically_rejected_step(sim::MetropolisSimulation)
#     sim.rejected_steps += 1
#     sim.automatically_rejected_steps +=1 
#     sim.total_steps += 1
# end

function add_trivial_step!(sim::MetropolisSimulation)
    sim.trivial_steps += 1
    sim.total_steps += 1
end

function consistent_MCinfo(sim::MetropolisSimulation)
    return sim.accepted_steps + sim.rejected_steps + sim.trivial_steps == sim.total_steps
end

# FIXME: instead, just report one acceptance rate, with separate functions giving trival and auto reject %s
function acceptance_rate_1(sim::MetropolisSimulation)
    @assert consistent_MCinfo(sim)
    return (sim.accepted_steps + sim.trivial_steps) / sim.total_steps
end

function acceptance_rate_2(sim::MetropolisSimulation)
    @assert consistent_MCinfo(sim)
    return sim.accepted_steps / (sim.accepted_steps + sim.rejected_steps)
end

function acceptance_rate_3(sim::MetropolisSimulation)
    @assert consistent_MCinfo(sim)
    return sim.accepted_steps / sim.total_steps
end

# by default this will do nothing, but it can be overwritten by a given wavefunction implementation
function print_wf_info(wf)
end

function iterate!(sim::MetropolisSimulation, steps_to_iterate::Int, steps_per_check::Int = 100, print_rate::Float64 = 0.05)

    printlnf("Beginning to iterate a Metropolis simulation $(steps_to_iterate) steps!")
    for k in 1:steps_to_iterate

        if mod(k, print_rate*steps_to_iterate) == 0
        # if true
            printlnf("Starting step #$(k)/$(steps_to_iterate) --> $(100k/steps_to_iterate)%")
            print_wf_info(sim.walk.wf)
        end

        # check for error periodically
        if mod(sim.total_steps, steps_per_check) == 0
            # printlnf("Checking for numerical error!")
            check_for_numerical_error!(sim.walk)
        end

        propose_and_assess_random_transition!(sim.walk)
        if !is_nontrivial_transition(sim.walk)
            add_trivial_step!(sim)
            # printlnf("Step is trivial!")
            # printlnf()
            # FIXME: for trivial transitions, we shouldn't have to recompute the measurements!
            for k in keys(sim.measurements)
                possibly_measure(sim.walk.wf, sim.measurements[k].data, sim.measurements[k].measure_function)
            end
            continue
        else
            probrat = get_probability_ratio(sim.walk)
        end
        # printlnf("probrat = $(probrat)")

        # Metropolis!
        if probrat >= 1.0 || rand() < probrat # accept transition
            accept_transition!(sim.walk)
            add_accepted_step!(sim)
            # printlnf("Step accepted!")
        else # reject transition
            reject_transition!(sim.walk)
            add_rejected_step!(sim)
            # printlnf("Step rejected!")
        end
        # printlnf()

        for k in keys(sim.measurements)
            possibly_measure(sim.walk.wf, sim.measurements[k].data, sim.measurements[k].measure_function)
        end

    end
end
