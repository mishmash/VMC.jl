abstract Configuration

immutable Particle
    index::Int
    species::Int
end

abstract ParticlesConfiguration <: Configuration

function get_Nspecies(config::ParticlesConfiguration)
    @assert length(config.particle_positions) == length(config.lattice_status)
    return length(config.particle_positions)
end

function get_Nparticles(config::ParticlesConfiguration)
    return map(length, config.particle_positions)
end

function get_Ntotal(config::ParticlesConfiguration)
    return sum(get_Nparticles(config))
end

type LatticeParticlesConfiguration <: ParticlesConfiguration
    particle_positions::Vector{Vector{Int}}
    lattice_status::Vector{Vector{Int}}

    function LatticeParticlesConfiguration(particle_positions::Vector{Vector{Int}}, Nsites::Int)
        Nparticles = map(length, particle_positions)
        lattice_status = Vector{Int}[]
        for j in 1:length(particle_positions) # = Nspecies
            @assert findmin(particle_positions[j])[1] >= 1
            @assert findmax(particle_positions[j])[1] <= Nsites
            push!(lattice_status, zeros(Int, Nsites))
            for i in 1:Nparticles[j]
                lattice_status[j][particle_positions[j][i]] = i
            end
        end
        return new(particle_positions, lattice_status)
    end
end

function get_Nsites(config::LatticeParticlesConfiguration)
    @assert length(config.lattice_status) != 0
    # @assert all(length(config.lattice_status[1]) .== map(length, config.lattice_status[2:end]))
    return length(config.lattice_status[1])
end

# this is probably unnecessary?
function is_valid_configuration(config::LatticeParticlesConfiguration)
    guy = true
    #FIXME: loop over species and do some checks..
    return guy
end

function is_valid_particle(config::LatticeParticlesConfiguration, particle::Particle)
    return (particle.species <= get_Nspecies(config) &&
            particle.index <= get_Nparticles(config)[particle.species])
end

function is_occupied(config::LatticeParticlesConfiguration, position::Int, species::Int)
    return config.lattice_status[species][position] != 0
end

function GutzSpinhalfConfiguration(positions_up::Vector{Int}, Nsites)
    @assert all(positions_up .<= Nsites)
    positions_down = setdiff(collect(1:Nsites), positions_up)
    return LatticeParticlesConfiguration(Vector{Int}[positions_up, positions_down], Nsites)
end

function is_valid_Gutz_spinhalf_configuration(config::LatticeParticlesConfiguration)
    guy = true
    # guy &= get_Nspecies(config) == 2 # spin-1/2 spinons
    # guy &= dot(config.lattice_status[1], config.lattice_status[2]) == 0 # no double occupancy [this is an O(N) operations check..]
    # guy &= isempty(setdiff([1:config.Nsites], union(config.particle_positions[1], config.particle_positions[1]))) # particles cover the lattice
    # the above is fairly time-consuming, so use this somewhat weaker check instead
    # guy &= get_Ntotal(config) == get_Nsites(config)
    return guy
end

# takes config_alpha1 and config_alpha2 and returns a tuple of the 2 swapped configurations (beta1, beta2)
# specific to swaps with equivalent particle numbers in the subsystem (for each species), which should be all we ever need..
function swap(config_alpha1::LatticeParticlesConfiguration, config_alpha2::LatticeParticlesConfiguration,
              subsystemA::Vector{Int})
    @assert get_Nspecies(config_alpha1) == get_Nspecies(config_alpha2)
    @assert get_Nparticles(config_alpha1) == get_Nparticles(config_alpha1)
    @assert get_Nsites(config_alpha1) == get_Nsites(config_alpha2)
    Nsites = get_Nsites(config_alpha1)

    @assert 0 < length(subsystemA) < Nsites # subsystemA is not the whole system nor empty
    @assert minimum(subsystemA) >= 1 && maximum(subsystemA) <= Nsites

    # check that there are nA particles in subsystem A for both configs
    nA = get_Nparticles_in_subsystem(config_alpha1, subsystemA)
    @assert nA == get_Nparticles_in_subsystem(config_alpha2, subsystemA)

    swapped_particle_positions_beta1 = Vector{Int}[]
    swapped_particle_positions_beta2 = Vector{Int}[]
    for s in 1:get_Nspecies(config_alpha1)
        push!(swapped_particle_positions_beta1, [config_alpha2.particle_positions[s][1:nA[s]]; config_alpha1.particle_positions[s][nA[s]+1:end]])
        push!(swapped_particle_positions_beta2, [config_alpha1.particle_positions[s][1:nA[s]]; config_alpha2.particle_positions[s][nA[s]+1:end]])
    end
    @assert all(map(length, swapped_particle_positions_beta1) .== map(length, swapped_particle_positions_beta2) .== get_Nparticles(config_alpha1))

    return LatticeParticlesConfiguration(swapped_particle_positions_beta1, Nsites),
           LatticeParticlesConfiguration(swapped_particle_positions_beta2, Nsites)
end

function get_Nparticles_in_subsystem(config::LatticeParticlesConfiguration, subsystem::Vector{Int})
    @assert 0 <= length(subsystem) <= get_Nsites(config)
    if length(subsystem) == 0
        return 0
    else
        return map(v -> length(find(v[subsystem] .!= 0)), config.lattice_status)
    end
end

function initialize_random_hardcore_two_copy_configuration(Nsites::Int, Nparticles::Int, subsystemA::Vector{Int}, nA::Int)
    Nsites_subsystemA = length(subsystemA)

    @assert Nparticles <= Nsites
    @assert 0 < Nsites_subsystemA < Nsites # subsystemA is not the whole system nor empty
    @assert minimum(subsystemA) >= 1 && maximum(subsystemA) <= Nsites
    @assert 0 <= nA <= Nparticles
    @assert nA <= Nsites_subsystemA && Nparticles - nA <= Nsites - Nsites_subsystemA # since we have hard-core particles here

    subsystemB = setdiff(collect(1:Nsites), subsystemA)

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

    return config_alpha1, config_alpha2, config_beta1, config_beta2, subsystemB
end

function exchange_particles!(config::LatticeParticlesConfiguration, particle1_ind::Int, particle2_ind::Int, species::Int)
    @assert species <= get_Nspecies(config)
    @assert particle1_ind <= get_Nparticles(config)[species]
    @assert particle2_ind <= get_Nparticles(config)[species]
    @assert particle1_ind != particle2_ind

    particle1_orig_pos = config.particle_positions[species][particle1_ind]
    particle2_orig_pos = config.particle_positions[species][particle2_ind]

    config.particle_positions[species][particle1_ind], config.particle_positions[species][particle2_ind] = particle2_orig_pos, particle1_orig_pos
    config.lattice_status[species][particle1_orig_pos], config.lattice_status[species][particle2_orig_pos] = particle2_ind, particle1_ind

    return config
end
