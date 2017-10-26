# include("Wavefunction.jl")
# include("SwapWavefunction.jl")
# include("SwapWavefunctionNS.jl")

type Estimate{T}
    sum::T
    n::Int
    avg::T
    # Estimate() = new(T(0.), 0, T(0.))
    Estimate() = new(convert(T, 0.), 0, convert(T, 0.))
end

function tally!{T}(est::Estimate{T}, value::T)
    est.sum += value
    est.n += 1
end

function average!(est::Estimate)
    est.avg = est.sum / est.n
end


abstract MeasurementData{T}

# for a single estimate
type MeasurementDataSimple{T} <: MeasurementData{T}
    description::ASCIIString
    steps_per_measurement::Int # a parameter
    measurement_steps::Int # a counter
    estimate::Estimate{T}

    function MeasurementDataSimple(description::ASCIIString, spm::Int)
        @assert spm > 0
        return new(description, spm, 0, Estimate{T}())
    end
end

# for many estimates corresponding to, say, different lattice site combinations in a correlation function
type MeasurementDataLattice{T} <: MeasurementData{T}
    description::ASCIIString
    steps_per_measurement::Int # a parameter
    measurement_steps::Int # a counter
    # FIXME: estimates and site_labels should be more closely tied together, through an ordered dict, say?
    # e.g., estimates[site_labels] = our estimate
    site_labels::Vector{Vector{Int}} # the part specific to being on a lattice
    estimates::Vector{Estimate{T}}

    function MeasurementDataLattice(description::ASCIIString, spm::Int)
        @assert spm > 0
        return new(description, spm, 0, Vector{Int}[], Estimate{T}[])
    end
end

# for holding subsystem occupation number probability measurements, \delta_{nA, NA}, for a single species
type MeasurementDataSubsystemOccupationProbability <: MeasurementData{Float64}
    description::ASCIIString
    steps_per_measurement::Int # a parameter
    measurement_steps::Int # a counter
    n::Vector{Int} # the subsystem occupation numbers
    species::Int
    subsystem::Vector{Int}
    estimates::Vector{Estimate{Float64}}

    function MeasurementDataSubsystemOccupationProbability(description::ASCIIString, spm::Int, n::Vector{Int}, species::Int, subsystem::Vector{Int})
        @assert spm > 0
        estimates = Estimate{Float64}[]
        for nn in n
            push!(estimates, Estimate{Float64}())
        end
        return new(description, spm, 0, n, species, subsystem, estimates)
    end
end

type Measurement
    data::MeasurementData
    measure_function::Function
end

# lots of average! functions, nice!
average!(meas::MeasurementDataSimple) = average!(meas.estimate)

function average!(meas::Union{MeasurementDataLattice, MeasurementDataSubsystemOccupationProbability})
    for e in meas.estimates
        average!(e)
    end
end

average!(measurement::Measurement) = average!(measurement.data)

function average!(measurements::Vector{Measurement})
    for m in measurements
        average!(m)
    end
end

function average!(measurements::Dict{ASCIIString, Measurement})
    for k in keys(measurements)
        average!(measurements[k])
    end
end

# the following two functions get MeasurementDataLattice into a form suitable for HDF5 and plotting
function package_averaged_data(data::MeasurementDataLattice)
    @assert length(data.site_labels) == length(data.estimates)
    out = Any[]
    for p in 1:length(data.site_labels)
        push!(out, Any[data.site_labels[p], data.estimates[p].avg])
    end
    return out
end

# data::Vector{Any}) should be of the form outputted by package_average_data
function corr_func_matrix(data::Vector{Any})
    N = sqrt(length(data))
    @assert floor(N) == N
    out = zeros(typeof(data[1][2]), (convert(Int, N), convert(Int, N)))
    for p in data
        out[p[1][1], p[1][2]] = p[2]
    end
    return out
end

# the following function gets MeasurementDataSubsystemOccupationProbability into a form suitable for HDF5 and plotting
function subsystem_occupatation_probabililities_matrix(data::MeasurementDataSubsystemOccupationProbability)
    @assert length(data.n) == length(data.estimates)
    out = zeros(Float64, (length(data.n), 2))
    for i in 1:length(data.n)
        out[i, :] = [data.n[i], data.estimates[i].avg].'
    end
    return out
end

# the following few functions are essentially constructors for appropriate MeasurementData* types
function initialize_2ptcorr_lattice_measurement(wf::Wavefunction, description::ASCIIString, T::Type, spm::Int)
    corr = MeasurementDataLattice{T}(description, spm)
    for i in 1:get_Nsites(wf)
        for j in 1:get_Nsites(wf)
            push!(corr.site_labels, [i, j])
            push!(corr.estimates, Estimate{T}())
        end
    end
    return corr
end

function initialize_1ptcorr_lattice_measurement(wf::Wavefunction, description::ASCIIString, T::Type, spm::Int)
    corr = MeasurementDataLattice{T}(description, spm)
    for i in 1:get_Nsites(wf)
        push!(corr.site_labels, [i])
        push!(corr.estimates, Estimate{T}())
    end
    return corr
end

function initialize_subsystem_occupation_probability_measurement(wf::Wavefunction, description::ASCIIString,
                                                                 spm::Int, subsystem::Vector{Int}, species::Int = 1)
    @assert 0 < length(subsystem) <= get_Nsites(wf)
    @assert minimum(subsystem) >= 1 && maximum(subsystem) <= get_Nsites(wf)
    n = collect(0:get_Nparticles(wf)[species])
    meas = MeasurementDataSubsystemOccupationProbability(description, spm, n, species, subsystem)
end

# this is what gets called by iterate! in the Simulation class
# FIXME: take current Wavefunction --> RegularWavefunction, and make RegularWavefunction and SwapWavefunction subtypes of Wavefunction
function possibly_measure(wf::Union{Wavefunction, SwapWavefunction, SwapWavefunctionNS}, corr::MeasurementData, measure_func::Function)
    corr.measurement_steps += 1
    if mod(corr.measurement_steps, corr.steps_per_measurement) == 0
        measure_func(wf, corr)
    end
end

# FIXME: put bangs on the names of the measure functions:  measure_* --> measure_*!

# FIXME: move this stuff to the Wavefunction class,
# but first need to separate out specific wf implementations from Wavefunction.jl (e.g., SBM)
function measure_SzSz(wf::SBMWavefunction, corr::MeasurementDataLattice{Float64})
    @assert is_valid_Gutz_spinhalf_configuration(wf)
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][1]
        j = corr.site_labels[p][2]
        Szi = (wf.config.lattice_status[1][i] != 0) ? 0.5 : -0.5
        Szj = (wf.config.lattice_status[1][j] != 0) ? 0.5 : -0.5
        tally!(corr.estimates[p], Szi * Szj)
    end
end

function measure_SpSm{T}(wf::SBMWavefunction{T}, corr::MeasurementDataLattice{T})
    @assert is_valid_Gutz_spinhalf_configuration(wf)
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][1]
        j = corr.site_labels[p][2]
        if i == j
            tally!(corr.estimates[p], (wf.config.lattice_status[1][i] != 0) ? T(1.) : T(0.))
            # tally!(corr.estimates[p], (wf.config.lattice_status[1][i] != 0) ? convert(T, 1.) : convert(T, 0.))
        elseif wf.config.lattice_status[1][i] != 0 && wf.config.lattice_status[2][j] != 0
            m = wf.config.lattice_status[1][i] # = index of up particle at site i
            n = wf.config.lattice_status[2][j] # = index of down particle at site j
            tally!(corr.estimates[p],
                -1. * detrat_from_columns_update(wf.cmat_up, m, columns_for_detmat(wf.orbitals_up, j))
                    * detrat_from_columns_update(wf.cmat_down, n, columns_for_detmat(wf.orbitals_down, i)))
        else
            tally!(corr.estimates[p], 0.)
        end
    end
end

function measure_ninj(wf::HardcoreWavefunction, corr::MeasurementDataLattice{Float64})
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][1]
        j = corr.site_labels[p][2]
        ni = is_occupied(wf.config, i, 1) ? 1. : 0.
        nj = is_occupied(wf.config, j, 1) ? 1. : 0.
        tally!(corr.estimates[p], ni * nj)
    end
end

function measure_ni(wf::HardcoreWavefunction, corr::MeasurementDataLattice{Float64})
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][1]
        ni = is_occupied(wf.config, i, 1) ? 1. : 0.
        tally!(corr.estimates[p], ni)
    end
end

function measure_subsystem_occupation_probability(wf::Wavefunction, meas::MeasurementDataSubsystemOccupationProbability)
    for i in 1:length(meas.n)
        # FIXME: gross...this is why estimates should be a dict or something, but who cares
        tally!(meas.estimates[i], get_Nparticles_in_subsystem(wf.config, meas.subsystem)[meas.species] == meas.n[i] ? 1. : 0.)
    end
end

function measure_Greens_function{T}(wf::FreeFermionWavefunction{T}, corr::MeasurementDataLattice{T})
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][1]
        j = corr.site_labels[p][2]
        if i == j
            # tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? T(1.) : T(0.))
            tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? convert(T, 1.) : convert(T, 0.))
        elseif is_occupied(wf.config, i, 1) && !is_occupied(wf.config, j, 1)
            m = wf.config.lattice_status[1][i]
            tally!(corr.estimates[p], detrat_from_columns_update(wf.cmat, m, columns_for_detmat(wf.orbitals, j)))
        else
            # tally!(corr.estimates[p], T(0.))
            tally!(corr.estimates[p], convert(T, 0.))
        end
    end
end

function measure_Greens_function{T}(wf::CFLWavefunction{T}, corr::MeasurementDataLattice{T})
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][]
        j = corr.site_labels[p][2]
        if i == j
            # tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? T(1.) : T(0.))
            tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? convert(T, 1.) : convert(T, 0.))
        elseif is_occupied(wf.config, i, 1) && !is_occupied(wf.config, j, 1)
            m = wf.config.lattice_status[1][i]
            tally!(corr.estimates[p],
                detrat_from_columns_update(wf.cmat_d1, m, columns_for_detmat(wf.orbitals_d1, j))
              * detrat_from_columns_update(wf.cmat_d2, m, columns_for_detmat(wf.orbitals_d2, j))
              * detrat_from_columns_update(wf.cmat_f, m, columns_for_detmat(wf.orbitals_f, j)))
        else
            # tally!(corr.estimates[p], T(0.))
            tally!(corr.estimates[p], convert(T, 0.))
        end
    end
end

function measure_Greens_function{Td,Tf}(wf::BosonCFLWavefunction{Td,Tf}, corr::MeasurementDataLattice{Td})
    for p in 1:length(corr.site_labels)
        i = corr.site_labels[p][]
        j = corr.site_labels[p][2]
        if i == j
            # tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? Td(1.) : Td(0.))
            tally!(corr.estimates[p], is_occupied(wf.config, i, 1) ? convert(Td, 1.) : convert(Td, 0.))
        elseif is_occupied(wf.config, i, 1) && !is_occupied(wf.config, j, 1)
            m = wf.config.lattice_status[1][i]
            tally!(corr.estimates[p],
                detrat_from_columns_update(wf.cmat_d, m, columns_for_detmat(wf.orbitals_d, j))
              * detrat_from_columns_update(wf.cmat_f, m, columns_for_detmat(wf.orbitals_f, j)))
        else
            # tally!(corr.estimates[p], Td(0.))
            tally!(corr.estimates[p], convert(Td, 0.))
        end
    end
end

# should be pretty robust
function measure_swap_sign(wf::Union{SwapWavefunction, SwapWavefunctionNS}, meas::MeasurementDataSimple{Complex128})
    theta = 0.
    theta -= angle(get_phialpha1(wf))
    theta -= angle(get_phialpha2(wf))
    theta += angle(get_phibeta1(wf))
    theta += angle(get_phibeta2(wf))
    tally!(meas.estimate, exp(im * theta))
    return nothing
end

# gets the phibeta's exactly from the alpha configs and orbitals
function measure_swap_mod_exact(wf::Union{SwapWavefunction, SwapWavefunctionNS}, meas::MeasurementDataSimple{Float64})
    tally!(meas.estimate,
        abs((get_phibeta1_exact_from_swap(wf) / get_phialpha1(wf))
          * (get_phibeta2_exact_from_swap(wf) / get_phialpha2(wf))))
    return nothing
end

# pretty much garbage
function measure_swap_mod_naive(wf::SwapWavefunction, meas::MeasurementDataSimple{Float64})
    tally!(meas.estimate,
        abs((get_phibeta1(wf) / get_phialpha1(wf))
          * (get_phibeta2(wf) / get_phialpha2(wf))))
    return nothing
end

# Could alternatively have functions in SwapWavefunction.jl which return the detrats,
# and have measure_swap_mod_from_SMW(wf::SwapWavefunction, ...) take that function as an argument.
# This applies equally well to, e.g., measure_Greens_function, above.
# Do we want to put the specificity of things with the measurements (here) or with the wavefunctions?
function measure_swap_mod_from_SMW(wf::Union{FreeFermionSwapWavefunction, FreeFermionSwapWavefunctionNS}, meas::MeasurementDataSimple{Float64})
    config_beta1, config_beta2 = swap(wf.config_alpha1, wf.config_alpha2, subsystemA)
    # FIXME: check for sites which are occupied in both the swapped and unswapped configs, so we can get away with moving less particles;
    # since we take abs(detrats), shouldn't need to worry at all about signs popping up in that process.
    tally!(meas.estimate, 
        abs(detrat_from_columns_update(wf.cmat_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals, config_beta1.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals, config_beta2.particle_positions[1][1:get_nA(wf)]))))
    return nothing
end

function measure_swap_mod_from_SMW(wf::BosonCFLSwapWavefunction, meas::MeasurementDataSimple{Float64})
    config_beta1, config_beta2 = swap(wf.config_alpha1, wf.config_alpha2, subsystemA)
    tally!(meas.estimate, 
        abs(detrat_from_columns_update(wf.cmat_d_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d, config_beta1.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_f_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_f, config_beta1.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_d_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d, config_beta2.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_f_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_f, config_beta2.particle_positions[1][1:get_nA(wf)]))))
    return nothing
end

function measure_swap_mod_from_SMW(wf::CFLSwapWavefunction, meas::MeasurementDataSimple{Float64})
    config_beta1, config_beta2 = swap(wf.config_alpha1, wf.config_alpha2, subsystemA)
    tally!(meas.estimate, 
        abs(detrat_from_columns_update(wf.cmat_d1_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d1, config_beta1.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_d2_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d2, config_beta1.particle_positions[1][1:get_nA(wf)]))  
          * detrat_from_columns_update(wf.cmat_f_phialpha1, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_f, config_beta1.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_d1_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d1, config_beta2.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_d2_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_d2, config_beta2.particle_positions[1][1:get_nA(wf)]))
          * detrat_from_columns_update(wf.cmat_f_phialpha2, collect(1:get_nA(wf)), columns_for_detmat(wf.orbitals_f, config_beta2.particle_positions[1][1:get_nA(wf)]))))
    return nothing
end
