/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it andor modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file init methods for macro electrons

#pragma once

#include "picongpu/simulation_defines.hpp"
// need physicalConstants.param

#include <pmacc/algorithms/math/PowerFunction.hpp>
#include <pmacc/math/Matrix.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::initElectrons
{
    struct InitElectronAs
    {
        using Matrix_3x3 = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<3u, 3u>>;
        using MatrixVector = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<3u, 1u>>;

    private:
        /** clone all add attributes that exist in both electron and ion species
         *
         * excludes:
         *  - particleId, new particle --> new ID required, init by default
         *  - multiMask, faster to set hard than copy, set in Kernel directly
         *  - momentum, is mass dependent and therefore always changes
         */
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE static void cloneAdditionalAttributes(
            T_IonParticle& ion,
            // cannot be const even though we do not write to the ion
            T_ElectronParticle& electron)
        {
            namespace partOp = pmacc::particles::operations;

            auto targetElectronClone = partOp::deselect<pmacc::mp_list<multiMask, momentum>>(electron);

            // otherwise this deselect will create incomplete type compile error
            partOp::assign(targetElectronClone, partOp::deselect<particleId>(ion));
        }

        //! fill space components of a Lorentz boots matrix
        HDINLINE static void fillLorentzMatrix(
            MatrixVector beta,
            float_64 gammaIonSystem,
            float_64 normBetaSquared,
            Matrix_3x3& lorentzMatrix)
        {
// general contributions
#pragma unroll
            for(uint32_t i = 0u; i < 3u; i++)
            {
#pragma unroll
                for(uint32_t j = static_cast<uint32_t>(0u); j < static_cast<uint32_t>(3u); j++)
                {
                    lorentzMatrix.element(i, j) = (gammaIonSystem - 1.) * beta.element(i, static_cast<uint32_t>(0u))
                        * beta.element(j, static_cast<uint32_t>(0u)) / normBetaSquared;
                }
            }

// diagonal only contributions
#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < static_cast<uint32_t>(3u); i++)
                lorentzMatrix.element(i, i) += 1.;
        }

    public:
        template<typename T_IonParticle, typename T_ElectronParticle>
        HDINLINE static void coMoving(T_IonParticle& ion, T_ElectronParticle& electron)
        {
            cloneAdditionalAttributes<T_IonParticle, T_ElectronParticle>(ion, electron);

            constexpr float_X massElectronPerMassIon
                = picongpu::traits::frame::getMass<typename T_ElectronParticle::FrameType>()
                / picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>();

            // init electron as co-moving with ion
            electron[momentum_] = ion[momentum_] * massElectronPerMassIon;
        }

        /** init electron according to inelastic relativistic collision of initially
         *  co-moving particles with deltaEnergy equal to transition energy
         *
         * @param ion Particle, view of ion frame slot
         * @param electron Particle, view of electron frame slot
         * @param deltaEnergyTransition  in eV, energy difference between initial and
         *  final state of transition
         * @param rngFactory factory for uniformly distributed random number generator
         *
         * @attention may be numerically unstable for highly relativistic ion/electrons
         * @attention assumes float3_X for momentum
         */
        template<typename T_IonParticle, typename T_ElectronParticle, typename T_RngGeneratorFloat>
        HDINLINE static void decayByInelastic2BodyCollision(
            T_IonParticle& ion,
            // cannot be const even though we do not write to the ion
            T_ElectronParticle& electron,
            float_X const deltaEnergyTransition, // eV
            T_RngGeneratorFloat& rngGenerator /// const?, @todo Brian Marre, 2023
        )
        {
            cloneAdditionalAttributes<T_IonParticle, T_ElectronParticle>(ion, electron);

            /* setting new electron and ion momentum
             *
             * see Brian Marre, notebook 01.06.2022-?, p.78-87 for full derivation
             *
             * Reference:
             * - Def.: Ion-system ... frame of reference co-moving with original ion speed
             * - Def.: Lab-system ... frame of reference PIC-simulation
             * - *Star* after inelastic collision, otherwise before
             */

            // special case dE <= 0
            if(deltaEnergyTransition <= 0.)
            {
                /// @todo generalize the error message,  Brian Marre, 2023
                if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_SPAWN_IONIZATION_ELECTRONS_HOT_DEBUG)
                    if(deltaEnergyTransition < 0.)
                        printf("atomicPhysics ERROR: deltaEnergy autonomous Ionization < 0!\n");

                electron[momentum_] = float3_X(0._X, 0._X, 0._X);
                ion[momentum_] = float3_X(0._X, 0._X, 0._X);
                return;
            }

            //      get electron/ion, momentum/Lorentz factor in IonSystem after ionization
            // always have equal weight, since copied from ion to electron

            // UNIT_MASS, not scaled
            constexpr float_X massElectron
                = picongpu::traits::frame::getMass<typename T_ElectronParticle::FrameType>();
            // UNIT_MASS, not scaled
            constexpr float_X massIon = picongpu::traits::frame::getMass<typename T_IonParticle::FrameType>();

            // unitless
            constexpr float_64 mI_mE = static_cast<float_64>(massIon / massElectron);
            // unitless
            constexpr float_64 mE_mI = static_cast<float_64>(massElectron / massIon);

            // kg * m^2/s^2 * keV/J * 1e3 = J/J * eV = eV
            constexpr float_64 restEnergyElectron = picongpu::SI::ELECTRON_MASS_SI
                * pmacc::math::cPow(picongpu::SI::SPEED_OF_LIGHT_SI, 2u) * picongpu::UNITCONV_Joule_to_keV * 1.e3;
            // eV
            constexpr float_64 restEnergyIon = static_cast<float_X>(massIon) * UNIT_MASS
                * pmacc::math::cPow(picongpu::SI::SPEED_OF_LIGHT_SI, 2u) * picongpu::UNITCONV_Joule_to_keV * 1.e3;

            // unitless + kg/kg * eV/eV = unitless
            float_64 const A_e = 0.25 + 1. + mI_mE * deltaEnergyTransition / restEnergyElectron;
            // unitless
            float_64 const A_i = 0.25 + 1. + mE_mI * deltaEnergyTransition / restEnergyIon;

            // Lorentz factor after inelastic collision in Ion-System
            // unitless + sqrt(unitless + kg/kg * eV) = unitless
            float_64 const gammaStarElectron_IonSystem = -0.5 + math::sqrt(A_e);
            // unitless
            float_64 const gammaStarIon_IonSystem = -0.5 + math::sqrt(A_i);

            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, not weighted
            constexpr float_64 mcElectron = picongpu::SI::ELECTRON_MASS_SI * picongpu::SI::SPEED_OF_LIGHT_SI
                / (UNIT_MASS * UNIT_LENGTH / UNIT_TIME);
            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, not weighted
            constexpr float_64 mcIon = massIon * picongpu::SI::SPEED_OF_LIGHT_SI / (UNIT_LENGTH / UNIT_TIME);

            // UNIT_MASS * UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
            constexpr float_64 mSquaredCSquaredIon
                = pmacc::math::cPow(massIon * picongpu::SI::SPEED_OF_LIGHT_SI / (UNIT_LENGTH / UNIT_TIME), 2u);

            // norm of electron momentum after inelastic collision in Ion-System
            // weight * (unitless^2 - unitless) * (UNIT_MASS * UNIT_LENGTH/UNIT_TIME)
            // = UNIT_MASS * UNIT_LENGTH / UNIT_TIME, weighted
            float_64 const normMomentumStarElectron_IonSystem = static_cast<float_64>(electron[weighting_])
                * math::sqrt((pmacc::math::cPow(gammaStarElectron_IonSystem, 2u) - 1._X)) * mcElectron;

            //      choose direction
            float_X const u = rngGenerator();
            float_X const v = rngGenerator();

            float_X const cosTheta = 1._X - 2._X * v;
            float_X const sinTheta = math::sqrt(v * (2._X - v));
            float_X const phi = 2._X * static_cast<float_X>(picongpu::PI) * u;

            float3_64 const directionVector
                = float3_64(sinTheta * math::cos(phi), sinTheta * math::sin(phi), cosTheta);
            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, weighted
            auto momentumStarElectron_IonSystem = MatrixVector(normMomentumStarElectron_IonSystem * directionVector);

            //      Lorentz transformation from IonSystem to LabSystem
            // square of original momentum of the ion in Lab-System
            // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
            float_64 const momentumSquaredIon_LabSystem
                = static_cast<float_64>(pmacc::math::l2norm2(ion[momentum_]) / ion[weighting_]);

            // square of original beta of the ion in Lab-System
            // beta^2 = 1/(1 + (m^2*c^2)/p^2)
            //  unitless + (UNIT_MASS^2 * UNIT_SPEED^2)(not weighted)
            //  /( UNIT_MASS^2 * UNIT_SPEED^2)(not weighted) = unitless
            float_64 const normSquaredBetaIon_LabSystem
                = 1._X / (1 + mSquaredCSquaredIon / momentumSquaredIon_LabSystem);

            // unitless
            float3_64 const betaIonSystem =
                // magnitude
                math::sqrt(normSquaredBetaIon_LabSystem)
                // direction
                * static_cast<float3_64>(ion[momentum_] / pmacc::math::l2norm(ion[momentum_]));

            // unitless
            auto beta = MatrixVector(betaIonSystem);

            float_64 const gammaIonSystem
                = math::sqrt(1. / (1. - pmacc::math::cPow(normSquaredBetaIon_LabSystem, 2u)));

            // lower 3x3 block of Lorentz transformation matrix for transformation from
            // Ion-System to Lab-System
            Matrix_3x3 lorentzMatrix;
            fillLorentzMatrix(beta, gammaIonSystem, normSquaredBetaIon_LabSystem, lorentzMatrix);

            //      space components of Lorentz boost
            MatrixVector momentumStarElectron_LabSystem;
            MatrixVector momentumStarIon_LabSystem;

            lorentzMatrix.mMul(momentumStarElectron_IonSystem, momentumStarElectron_LabSystem);

            //      calculate ion momentum after ionization
            // UNIT_MASS * UNIT_LENGTH/UNIT_TIME, weighted
            momentumStarIon_LabSystem = (momentumStarElectron_LabSystem.sMul(-1.))
                + (beta.sMul(gammaIonSystem * gammaStarIon_IonSystem * mcIon));

            // set to particle
#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < static_cast<uint32_t>(3u); i++)
                electron[momentum_][i]
                    = static_cast<float_X>(momentumStarIon_LabSystem.element(i, static_cast<uint32_t>(0u)));

            //      calculate ionization electron momentum after ionization
            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, weighted
            momentumStarElectron_LabSystem = momentumStarElectron_LabSystem
                + (beta.sMul(gammaIonSystem * gammaStarElectron_IonSystem * mcElectron));
            // set to particle
#pragma unroll
            for(uint32_t i = static_cast<uint32_t>(0u); i < static_cast<uint32_t>(3u); i++)
                electron[momentum_][i]
                    = static_cast<float_X>(momentumStarElectron_LabSystem.element(i, static_cast<uint32_t>(0u)));
        }
    };
} // namespace picongpu::particles::atomicPhysics2::initElectrons
