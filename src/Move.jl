# include("Configuration.jl")
# include("delegate.jl")
import Base: getindex, findfirst, length, size, start, done, next, isempty

abstract ParticleMove

immutable LatticeParticleMove <: ParticleMove
    particle::Particle
    destination::Int
end

# see Base.Cartesian for better multidimensional capabilities..
immutable Continuum1DParticleMove <: ParticleMove
    particle::Particle
    destination::Float64
end

# immutable Continuum2DParticleMove <: ParticleMove
#     particle::Particle
#     @compat destination::Tuple{Float64, Float64}
# end

abstract SystemMove

immutable ParticlesMove{T<:ParticleMove} <: SystemMove
    nontrivial_move::Bool
    move_vec::Vector{T}
    ParticlesMove(move_vec::Vector{T}) = new(true, move_vec)
    ParticlesMove() = new(false) # need to specify T via ParticlesMove{T}() when called
end

# which form is most appropriate?  (issue shows up in other places too)
ParticlesMove{T<:ParticleMove}(move_vec::Vector{T}) = ParticlesMove{eltype(move_vec)}(move_vec)
# ParticlesMove(move_vec) = ParticlesMove{eltype(move_vec)}(move_vec)
# ParticlesMove(move_vec::Vector) = ParticlesMove{eltype(move_vec)}(move_vec)
# ParticlesMove(move_vec::Vector{ParticleMove}) = ParticlesMove{eltype(move_vec)}(move_vec)

is_nontrivial(move::SystemMove) = move.nontrivial_move

# to make ParticlesMove objects effectively behave like Vector{T<:ParticleMove} objects
# (without explictly accessing the move_vec field)
@delegate ParticlesMove.move_vec [ getindex, length, size, start, done, next, isempty ]
export getindex, length, size, start, done, next, isempty

function update_configuration!(config::LatticeParticlesConfiguration, move::ParticlesMove{LatticeParticleMove})
    # if the move is trivial (not actually a change to the configuration space) then we don't do anything to config
    if is_nontrivial(move)
        for sp_move in move
            config.particle_positions[sp_move.particle.species][sp_move.particle.index] = sp_move.destination
        end
        config.lattice_status = LatticeParticlesConfiguration(config.particle_positions, get_Nsites(config)).lattice_status
    end
    return config
end

function update_configuration(config::LatticeParticlesConfiguration, move::ParticlesMove{LatticeParticleMove})
    new_config = deepcopy(config)
    if is_nontrivial(move)
        for sp_move in move
            new_config.particle_positions[sp_move.particle.species][sp_move.particle.index] = sp_move.destination
        end
        new_config.lattice_status = LatticeParticlesConfiguration(new_config.particle_positions, get_Nsites(new_config)).lattice_status
    end
    return new_config
end
