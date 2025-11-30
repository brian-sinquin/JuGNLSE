using Test
using JuGNLSE

@testset "Raman Effect Tests" begin
    # Common parameters
    N = 256
    time_window = 5e-12  # 5 ps
    λ0 = 1550e-9  # 1550 nm
    grid = create_grid(N, time_window, λ0)
    
    # Short fiber for testing
    L = 0.01  # 1 cm
    gamma = 0.001  # 1/W/m
    beta2 = -1e-26  # ps²/m (anomalous dispersion)
    medium = Medium(L, gamma, [beta2], 0.0, λ0)
    
    # Gaussian pulse
    T0 = 50e-15  # 50 fs
    P0 = 1e4  # 10 kW peak power
    pulse = gaussian_pulse(grid, T0, P0, 1.0)
    
    input_energy = sum(abs2.(pulse.At)) * grid.dt
    
    @testset "SSFM with Raman" begin
        @testset "SSFM without Raman" begin
            params_no_raman = SimParams(medium=medium, raman=false, N=N)
            z, At_no_raman, Aw_no_raman = propagate_ssfm(pulse, params_no_raman, progress=false)
            
            output_energy = sum(abs2.(At_no_raman[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test length(z) > 0
            @test size(At_no_raman, 1) == N
            @test energy_loss < 0.01  # Less than 1% energy loss
        end
        
        @testset "SSFM with Raman" begin
            params_raman = SimParams(medium=medium, raman=true, raman_model=BlowWood(), N=N)
            z, At_raman, Aw_raman = propagate_ssfm(pulse, params_raman, progress=false)
            
            output_energy = sum(abs2.(At_raman[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test length(z) > 0
            @test size(At_raman, 1) == N
            @test energy_loss < 0.01  # Less than 1% energy loss
            @test !any(isnan, At_raman)  # No NaN values
            @test !any(isinf, At_raman)  # No Inf values
        end
    end
    
    @testset "ERK4IP with Raman" begin
        @testset "ERK4IP without Raman" begin
            params_no_raman = SimParams(medium=medium, raman=false)
            z, At_no_raman, Aw_no_raman = propagate_erk4ip(pulse, params_no_raman, progress=false)
            
            output_energy = sum(abs2.(At_no_raman[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test length(z) > 0
            @test size(At_no_raman, 1) == N
            @test energy_loss < 0.01  # Less than 1% energy loss
        end
        
        @testset "ERK4IP with Raman (scalar gamma)" begin
            params_raman = SimParams(medium=medium, raman=true, raman_model=BlowWood())
            z, At_raman, Aw_raman = propagate_erk4ip(pulse, params_raman, progress=false)
            
            output_energy = sum(abs2.(At_raman[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test length(z) > 0
            @test size(At_raman, 1) == N
            @test energy_loss < 0.01  # Less than 1% energy loss
            @test !any(isnan, At_raman)  # No NaN values
            @test !any(isinf, At_raman)  # No Inf values
        end
        
        @testset "ERK4IP with Raman (frequency-dependent gamma)" begin
            # Create frequency-dependent gamma
            gamma_vec = gamma .* ones(N)
            medium_freq = Medium(L, gamma_vec, [beta2], 0.0, λ0)
            params_raman = SimParams(medium=medium_freq, raman=true, raman_model=BlowWood())
            
            z, At_raman, Aw_raman = propagate_erk4ip(pulse, params_raman, progress=false)
            
            output_energy = sum(abs2.(At_raman[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test length(z) > 0
            @test size(At_raman, 1) == N
            @test energy_loss < 0.01  # Less than 1% energy loss
            @test !any(isnan, At_raman)  # No NaN values
            @test !any(isinf, At_raman)  # No Inf values
        end
    end
    
    @testset "Raman Effect Comparison (SSFM vs ERK4IP)" begin
        # Compare SSFM and ERK4IP with Raman enabled
        params_raman = SimParams(medium=medium, raman=true, raman_model=BlowWood(), N=N)
        
        z_ssfm, At_ssfm, Aw_ssfm = propagate_ssfm(pulse, params_raman, progress=false)
        z_erk4ip, At_erk4ip, Aw_erk4ip = propagate_erk4ip(pulse, params_raman, progress=false)
        
        # Both should conserve energy similarly
        energy_ssfm = sum(abs2.(At_ssfm[:, end])) * grid.dt
        energy_erk4ip = sum(abs2.(At_erk4ip[:, end])) * grid.dt
        
        energy_loss_ssfm = abs(energy_ssfm - input_energy) / input_energy
        energy_loss_erk4ip = abs(energy_erk4ip - input_energy) / input_energy
        
        @test energy_loss_ssfm < 0.01
        @test energy_loss_erk4ip < 0.01
        
        # Both should produce similar output spectra (within reasonable tolerance)
        spectrum_ssfm = abs2.(Aw_ssfm[:, end])
        spectrum_erk4ip = abs2.(Aw_erk4ip[:, end])
        
        # Normalize spectra
        spectrum_ssfm ./= maximum(spectrum_ssfm)
        spectrum_erk4ip ./= maximum(spectrum_erk4ip)
        
        # Check spectral similarity (allowing for some numerical differences)
        spectral_difference = sum(abs.(spectrum_ssfm .- spectrum_erk4ip)) / N
        @test spectral_difference < 0.1  # Less than 10% average difference
    end
    
    @testset "Raman with Self-Steepening" begin
        @testset "SSFM with Raman and Shock" begin
            params = SimParams(medium=medium, raman=true, shock=true, raman_model=BlowWood(), N=N)
            z, At, Aw = propagate_ssfm(pulse, params, progress=false)
            
            output_energy = sum(abs2.(At[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test energy_loss < 0.01
            @test !any(isnan, At)
            @test !any(isinf, At)
        end
        
        @testset "ERK4IP with Raman and Shock" begin
            params = SimParams(medium=medium, raman=true, shock=true, raman_model=BlowWood())
            z, At, Aw = propagate_erk4ip(pulse, params, progress=false)
            
            output_energy = sum(abs2.(At[:, end])) * grid.dt
            energy_loss = abs(output_energy - input_energy) / input_energy
            
            @test energy_loss < 0.01
            @test !any(isnan, At)
            @test !any(isinf, At)
        end
    end
    
    @testset "Different Raman Models" begin
        @testset "LinAgrawal Model" begin
            params = SimParams(medium=medium, raman=true, raman_model=LinAgrawal(), N=N)
            z, At, Aw = propagate_ssfm(pulse, params, progress=false)
            
            @test !any(isnan, At)
            @test !any(isinf, At)
        end
        
        @testset "Hollenbeck Model" begin
            params = SimParams(medium=medium, raman=true, raman_model=Hollenbeck(), N=N)
            z, At, Aw = propagate_ssfm(pulse, params, progress=false)
            
            @test !any(isnan, At)
            @test !any(isinf, At)
        end
    end
end
