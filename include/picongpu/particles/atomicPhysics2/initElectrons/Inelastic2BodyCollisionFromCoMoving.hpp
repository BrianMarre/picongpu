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

//! @file implements init of macro electron as inelastic collision of co-moving electron with ion

#pragma once

#include "picongpu/simulation_defines.hpp"
// need physicalConstants.param

#include "picongpu/particles/atomicPhysics2/initElectrons/CloneAdditionalAttributes.hpp"

#include <pmacc/algorithms/math/PowerFunction.hpp>
#include <pmacc/math/Matrix.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::initElectrons
{

    struct Inelastic2BodyCollisionFromCoMoving
    {
    private:
        using Matrix_DxD = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<picongpu::simDim, picongpu::simDim>>;
        using MatrixVector = pmacc::math::Matrix<float_64, pmacc::math::CT::UInt32<picongpu::simDim, 1u>>;

        /** fill space components of a Lorentz boots matrix
         *
         * @attention normBetaSquared, beta and gamma must be consistent!
         */
        HDINLINE static void fillLorentzMatrix(
            MatrixVector beta,
            float_64 gamma,
            float_64 normBetaSquared,
            Matrix_DxD& lorentzMatrix)
        {
            /// @detail not that readable but faster than checking for every component
            // general contributions
            if(normBetaSquared == 0.)
            {
                // special case non moving system

#pragma unroll
                for(uint32_t i = 0u; i < picongpu::simDim; i++)
                {
#pragma unroll
                    for(uint32_t j = 0u; j < picongpu::simDim; j++)
                    {
                        lorentzMatrix.element(i, j) = 0.;
                    }
                }
            }
            else
            {
                // standard case

#pragma unroll
                for(uint32_t i = 0u; i < picongpu::simDim; i++)
                {
#pragma unroll
                    for(uint32_t j = 0u; j < picongpu::simDim; j++)
                    {
                        lorentzMatrix.element(i, j) = (gamma - 1.) * beta.element(i, static_cast<uint32_t>(0u))
                            * beta.element(j, static_cast<uint32_t>(0u)) / normBetaSquared;
                    }
                }
            }

            // diagonal only contributions

#pragma unroll
            for(uint32_t i = 0u; i < picongpu::simDim; i++)
                lorentzMatrix.element(i, i) += 1.;
        }

    public:
        /** init electron according to inelastic relativistic collision of initially
         *  co-moving particles with deltaEnergy
         *
         * @param ion Particle, view of ion frame slot
         * @param electron Particle, view of electron frame slot
         * @param deltaEnergy in eV, energy difference between initial and
         *  final state of transition
         * @param rngFactory factory for uniformly distributed random number generator
         *
         * @attention numerically unstable for highly relativistic ion/electrons( and MeV+ deltaEnergys)
         */
        template<typename T_IonParticle, typename T_ElectronParticle, typename T_RngGeneratorFloat>
        HDINLINE static void init(
            T_IonParticle& ion,
            // cannot be const even though we do not write to the ion
            T_ElectronParticle& electron,
            // eV
            float_X const deltaEnergy,
            /// const?, @todo Brian Marre, 2023
            T_RngGeneratorFloat& rngGenerator)
        {
            CloneAdditionalAttributes::init<T_IonParticle, T_ElectronParticle>(ion, electron);

            /* setting new electron and ion momentum
             *
             * see Brian Marre, notebook 01.06.2022-?, p.78-87 for full derivation
             *
             * Naming Legend:
             * - Def.: Ion-system ... frame of reference co-moving with original ion speed
             * - Def.: Lab-system ... frame of reference PIC-simulation
             * - *Star* ... after inelastic collision, otherwise before
             */

            // special case dE <= 0
            if(deltaEnergy <= 0._X)
            {
                /// @todo generalize the error message,  Brian Marre, 2023
                if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_SPAWN_IONIZATION_ELECTRONS_HOT_DEBUG)
                    if(deltaEnergy < 0._X)
                        printf("atomicPhysics ERROR: inelastic ionization with deltaEnergy Ionization < 0!\n");

#pragma unroll
                for (uint8_t i = 0u; i < picongpu::simDim; i++)
                {
                    electron[momentum_][i] = 0._X;
                    ion[momentum_][i] = 0._X;
                }
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
            float_64 const A_e = 0.25 + 1. + mI_mE * deltaEnergy / restEnergyElectron;
            // unitless
            float_64 const A_i = 0.25 + 1. + mE_mI * deltaEnergy / restEnergyIon;

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

            float_X sinPhi;
            float_X cosPhi;
            pmacc::math::sincos(phi, sinPhi, cosPhi);

            floatD_64 directionVector;
            if constexpr (picongpu::simDim == 2u)
                directionVector = floatD_64(cosPhi, sinTheta);
            if constexpr (picongpu::simDim == 3u)
                directionVector = floatD_64(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, weighted
            auto momentumStarElectron_IonSystem = MatrixVector(normMomentumStarElectron_IonSystem * directionVector);

            //      Lorentz transformation from IonSystem to LabSystem
            // square of original momentum of the ion in Lab-System
            // UNIT_MASS^2 * UNIT_LENGTH^2 / UNIT_TIME^2, not weighted
            float_64 const momentumSquaredIon_LabSystem
                = static_cast<float_64>(pmacc::math::l2norm2(ion[momentum_]) / (ion[weighting_] * ion[weighting_]));

            // unitless
            floatD_64 betaIonSystem;
            // unitless
            float_64 normSquaredBetaIon_LabSystem;

            if(momentumSquaredIon_LabSystem == 0._X)
            {
                if constexpr(picongpu::simDim == 2u)
                    betaIonSystem = floatD_64(0., 0.);
                if constexpr(picongpu::simDim == 3u)
                    betaIonSystem = floatD_64(0., 0., 0.);

                normSquaredBetaIon_LabSystem = 0.;
            }
            else
            {
                // square of original beta of the ion in Lab-System
                // unitless
                // beta^2 = 1/(1 + (m^2*c^2)/p^2)
                //  unitless + (UNIT_MASS^2 * UNIT_SPEED^2)(not weighted)
                //  /( UNIT_MASS^2 * UNIT_SPEED^2)(not weighted) = unitless
                normSquaredBetaIon_LabSystem = 1. / (1. + mSquaredCSquaredIon / momentumSquaredIon_LabSystem);

                betaIonSystem =
                    // magnitude
                    math::sqrt(normSquaredBetaIon_LabSystem)
                    // direction
                    * static_cast<floatD_64>(ion[momentum_] / pmacc::math::l2norm(ion[momentum_]));
            }

            // unitless
            auto beta = MatrixVector(betaIonSystem);

            float_64 const gammaIonSystem = math::sqrt(momentumSquaredIon_LabSystem / mSquaredCSquaredIon + 1.);

            // lower 3x3 block of Lorentz transformation matrix for transformation from
            // Ion-System to Lab-System
            Matrix_DxD lorentzMatrix;
            fillLorentzMatrix(beta, gammaIonSystem, normSquaredBetaIon_LabSystem, lorentzMatrix);

            //      space components of Lorentz boost
            MatrixVector momentumStarElectron_LabSystem;
            MatrixVector momentumStarIon_LabSystem;

            lorentzMatrix.mMul(momentumStarElectron_IonSystem, momentumStarElectron_LabSystem);

            //      calculate momenta after collision
            //          ion
            // UNIT_MASS * UNIT_LENGTH/UNIT_TIME, weighted
            momentumStarIon_LabSystem = (momentumStarElectron_LabSystem.sMul(-1.))
                + (beta.sMul(gammaIonSystem * gammaStarIon_IonSystem * mcIon));

            //          electron
            // UNIT_MASS * UNIT_LENGTH / UNIT_TIME, weighted
            momentumStarElectron_LabSystem = momentumStarElectron_LabSystem
                + (beta.sMul(gammaIonSystem * gammaStarElectron_IonSystem * mcElectron));

            // set to particle
#pragma unroll
            for(uint32_t i = 0u; i < picongpu::simDim; i++)
                ion[momentum_][i]
                    = static_cast<float_X>(momentumStarIon_LabSystem.element(i, static_cast<uint32_t>(0u)));

                // set to particle
#pragma unroll
            for(uint32_t i = 0u; i < picongpu::simDim; i++)
                electron[momentum_][i]
                    = static_cast<float_X>(momentumStarElectron_LabSystem.element(i, static_cast<uint32_t>(0u)));
        }
    };
} // namespace picongpu::particles::atomicPhysics2::initElectrons
