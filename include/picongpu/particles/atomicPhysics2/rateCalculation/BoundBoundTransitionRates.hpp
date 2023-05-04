/* Copyright 2022-2023 Brian Marre, Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
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

#pragma once

#include "picongpu/simulation_defines.hpp"
/* need the following .param files
 *  - atomicPhysics2_Debug.param      debug check switches
 *  - physicalConstants.param         physical constants, namespace picongpu::SI
 *  - unit.param                      unit of time for normalization
 */

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics2/rateCalculation/CollisionalRate.hpp"
#include "picongpu/particles/atomicPhysics2/rateCalculation/Multiplicities.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cmath>
#include <cstdint>


/** @file implements calculation of rates for bound-bound atomic physics transitions
 *
 * this includes:
 *  - electron-interaction based processes
 *  - spontaneous photon emission
 *  @todo photon interaction based processes, Brian Marre, 2022
 *

 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 * - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 * - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Density Physics 3, 342-352 (2007)
 *
 * - and https://en.wikipedia.org/wiki/Einstein_coefficients for the SI version
 */

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    /** compilation of static rate- and crossSection-calc. methods for all processes
     * working on bound-bound transition data
     *
     * @tparam T_numberLevels maximum principal quantum number of atomic states of ion species
     *
     * @attention atomic data box input data is assumed to be in eV
     */
    template<uint8_t T_numberLevels>
    struct BoundBoundTransitionRates
    {
    private:
        template<typename T_Type>
        HDINLINE static bool relativeTest(T_Type trueValue, T_Type testValue, T_Type errorLimit)
        {
            return math::abs((testValue - trueValue) / trueValue) > errorLimit;
        }

        /** gaunt factor suppression of cross sections
         *
         * @param U = energyElectron / energyDifference, unitless
         * @param energyElectron energy of electron, [eV]
         * @param indexTransition internal index of transition in atomicDataBox
         *      use findIndexTransition method of atomicDataBox and screen for not found value
         * @param boundBoundTransitionDataBox transition data box
         *
         * @attention no range check for indexTransition outside debug compile, invalid memory access otherwise
         *
         * @return unit: unitless
         */
        template<typename T_BoundBoundTransitionDataBox>
        HDINLINE static float_X gauntFactor(
            float_X const U, // unitless
            uint32_t const collectionIndexTransition,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            // get gaunt coefficients, unitless
            float_X const A = boundBoundTransitionDataBox.cxin1(collectionIndexTransition);
            float_X const B = boundBoundTransitionDataBox.cxin2(collectionIndexTransition);
            float_X const C = boundBoundTransitionDataBox.cxin3(collectionIndexTransition);
            float_X const D = boundBoundTransitionDataBox.cxin4(collectionIndexTransition);
            float_X const a = boundBoundTransitionDataBox.cxin5(collectionIndexTransition);

            // calculate gaunt Factor
            float_X const g = A * math::log(U) + B + C / (U + a) + D / ((U + a) * (U + a)); // unitless

            if(U > 1.0_X)
                return g; // unitless
            else
                return 0._X; // unitless
        }

        //! check for NaNs and casting overflows in Ratio
        template<typename T_AtomicStateDataBox>
        HDINLINE static void debugChecksMultiplicity(
            float_X Ratio,
            uint32_t const lowerStateClctIdx,
            uint32_t const upperStateClctIdx,
            T_AtomicStateDataBox const atomicStateDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            // Ratio not NaN
            if(!(Ratio == Ratio)) // only true if nan
                printf(
                    "atomicPhysics ERROR: NaN multiplicityConfigNumber ratio\n"
                    "   upperStateClctIdx %u ,lowerStateClctIdx %u ,(energy electron/energyDifference) %f\n",
                    upperStateClctIdx,
                    lowerStateClctIdx,
                    Ratio);

            // no overflow in float_X cast
            if(relativeTest(
                   (multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(lowerStateClctIdx))
                    / multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(upperStateClctIdx))),
                   float_64(float_X(
                       multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(lowerStateClctIdx))
                       / multiplicityConfigNumber<S_ConfigNumber>(
                           atomicStateDataBox.configNumber(upperStateClctIdx)))),
                   1e-7))
                printf("atomicPhysics ERROR: overflow in multiplicityConfigNumber-ratio cast to float_X\n");
        }

    public:
        /** collisional cross section for a given bound-bound transition and electron energy
         *
         * @tparam T_AtomicStateDataBox }instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         * @tparam T_excitation true =^= excitation, false =^= deexcitation, direction of transition
         *
         * @param energyElectron kinetic energy of interacting free electron(/electron bin), [eV]
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 10^6*b = 10^(-22)m^2; (b == barn = 10^(-28) m^2)
         *
         * @attention assumes that atomicStateDataBox and boundBoundTransitionDataBox belong
         *      to the same AtomicData instance
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox, bool T_excitation>
        HDINLINE static float_X collisionalBoundBoundCrossSection(
            float_X const energyElectron, // [eV]
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            uint32_t const upperStateClctIdx
                = boundBoundTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);
            uint32_t const lowerStateClctIdx
                = boundBoundTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            float_X const energyDifference = static_cast<float_X>(
                atomicStateDataBox.energy(upperStateClctIdx) - atomicStateDataBox.energy(lowerStateClctIdx)); // [eV]

            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CALCULATION_HOT_DEBUG)
                if(energyDifference < 0._X)
                {
                    printf("atomicPhysics ERROR: upper and lower state inverted in "
                           "collisionalBoundBoundCrossSection() call\n");
                    return 0._X;
                }

            // check whether electron has enough kinetic energy
            if constexpr(T_excitation)
            {
                // transition physical impossible, insufficient electron energy
                if(energyDifference > energyElectron)
                    return 0._X;
            }

            // unitless * unitless = unitless
            float_X const collisionalOscillatorStrength = static_cast<float_X>(
                boundBoundTransitionDataBox.collisionalOscillatorStrength(transitionCollectionIndex)); // unitless

            /* formula: scalingConstant * E_Ry^2/deltaE_Trans^2 * f * deltaE_Trans/E_kin * g
             *
             * E_Ry         ... Rydberg Energy
             * deltaE_Trans ... (energy upper state - energy lower state)
             * f            ... oscillator strength
             * E_kin        ... kinetic energy of interacting electron
             * g            ... gauntFactor
             */

            // (unitless * m)^2 / (unitless * m^2/1e6b) = m^2 / m^2 * 1e6b = 1e6b
            constexpr float_X scalingConstant = static_cast<float_X>(
                8. * pmacc::math::cPow(picongpu::PI * picongpu::SI::BOHR_RADIUS, static_cast<uint8_t>(2u))
                / (1.e-22)); // [1e6b], ~ 2211,01 * 1e6b
            // 1e6b
            constexpr float_X constantPart
                = scalingConstant * pmacc::math::cPow(picongpu::SI::RYDBERG_ENERGY, static_cast<uint8_t>(2u));
            // [1e6b * (eV)^2]

            // 1e6b*(eV)^2 / (eV)^2 * unitless * (eV)/(eV) * unitless = 1e6b
            float_X crossSection_butGaunt = constantPart / math::sqrt(3.)
                / pmacc::math::cPow(energyDifference, static_cast<uint8_t>(2u)) * collisionalOscillatorStrength
                * (energyDifference / energyElectron);
            // [1e6b]

            // safeguard against negative cross sections due to imperfect approximations
            if(crossSection_butGaunt < 0._X)
            {
                return 0._X;
            }

            if constexpr(T_excitation)
            {
                // excitation
                return crossSection_butGaunt
                    * gauntFactor(
                           energyElectron / energyDifference,
                           transitionCollectionIndex,
                           boundBoundTransitionDataBox); // [1e6b]
            }
            else
            {
                // deexcitation

                //      different multiplicityConfigNumber for deexcitation
                float_X const ratio = static_cast<float_X>(
                    multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(lowerStateClctIdx))
                    / multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(upperStateClctIdx)));
                // unitless

                if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CALCULATION_HOT_DEBUG)
                    debugChecksMultiplicity(ratio, lowerStateClctIdx, upperStateClctIdx, atomicStateDataBox);

                return ratio * crossSection_butGaunt
                    * gauntFactor(
                           (energyElectron + energyDifference) / energyDifference,
                           transitionCollectionIndex,
                           boundBoundTransitionDataBox); // [1e6b]
            }
        }

        /** rate for collisional bound-bound transition of ion with free electron bin
         *
         * uses second order integration(bin middle)
         *
         * @todo implement higher order integrations, Brian Marre, 2022
         *
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         * @tparam T_excitation true =^= excitation, false =^= deexcitation, direction of transition
         *
         * @param energyElectron kinetic energy of interacting electron(/electron bin), [eV]
         * @param energyElectronBinWidth energy width of electron bin, [eV]
         * @param densityElectrons [1/(m^3 * eV)]
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/UNIT_TIME
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox, bool T_excitation>
        HDINLINE static float_X rateCollisionalBoundBoundTransition(
            float_X const energyElectron, // [eV]
            float_X const energyElectronBinWidth, // [eV]
            float_X const densityElectrons, // [1/(UNIT_LENGTH^3*eV)]
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            float_X const sigma = collisionalBoundBoundCrossSection<
                T_AtomicStateDataBox,
                T_BoundBoundTransitionDataBox,
                T_excitation>(
                energyElectron, // [eV]
                transitionCollectionIndex,
                atomicStateDataBox,
                boundBoundTransitionDataBox); // [1e6*b]

            return picongpu::particles2::atomicPhysics2::rateCalculation::collisionalRate(
                energyElectron,
                energyElectronBinWidth,
                densityElectrons,
                sigma);
        }

        /** rate of spontaneous photon emission for a given bound-bound transition
         *
         * @tparam T_AtomicStateDataBox instantiated type of dataBox
         * @tparam T_BoundBoundTransitionDataBox instantiated type of dataBox
         *
         * @param transitionCollectionIndex index of transition in boundBoundTransitionDataBox
         * @param atomicStateDataBox access to atomic state property data
         * @param boundBoundTransitionDataBox access to bound-bound transition data
         *
         * @return unit: 1/UNIT_TIME, usually Delta_T_SI ... PIC time step length
         */
        template<typename T_AtomicStateDataBox, typename T_BoundBoundTransitionDataBox>
        HDINLINE static float_X rateSpontaneousRadiativeDeexcitation(
            uint32_t const transitionCollectionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_BoundBoundTransitionDataBox const boundBoundTransitionDataBox)
        {
            using S_ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            // short hands for constants in SI
            constexpr float_64 c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
            constexpr float_64 m_e_SI = picongpu::SI::ELECTRON_MASS_SI; // unit: kg, SI
            constexpr float_64 e_SI = picongpu::SI::ELECTRON_CHARGE_SI; // unit: C, SI

            constexpr float_64 mue0_SI = picongpu::SI::MUE0_SI; // unit: C/(Vm), SI
            constexpr float_64 pi = picongpu::PI; // unit: unitless
            constexpr float_64 hbar_SI = picongpu::SI::HBAR_SI; // unit: Js, SI

            // constexpr float_64 au_SI = picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: J, SI

            uint32_t const upperStateClctIdx
                = boundBoundTransitionDataBox.upperStateCollectionIndex(transitionCollectionIndex);
            uint32_t const lowerStateClctIdx
                = boundBoundTransitionDataBox.lowerStateCollectionIndex(transitionCollectionIndex);

            float_X deltaEnergyTransition = static_cast<float_X>(
                atomicStateDataBox.energy(upperStateClctIdx) - atomicStateDataBox.energy(lowerStateClctIdx));
            // [eV]

            constexpr float_X scalingConstantPhotonFrequency
                = static_cast<float_X>(picongpu::UNITCONV_eV_to_Joule / (2 * pi * hbar_SI) * picongpu::UNIT_TIME);
            // J/(eV) / (Js) * s/UNIT_TIME = J/J * s/s * 1/(eV * UNIT_TIME);

            /// @attention actual SI frequency, NOT angular frequency
            float_X frequencyPhoton = deltaEnergyTransition * scalingConstantPhotonFrequency;
            // unit: 1/UNIT_TIME

            float_X ratio = static_cast<float_X>(
                multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(lowerStateClctIdx))
                / multiplicityConfigNumber<S_ConfigNumber>(atomicStateDataBox.configNumber(upperStateClctIdx)));
            // unit: unitless

            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CALCULATION_HOT_DEBUG)
                debugChecksMultiplicity(ratio, lowerStateClctIdx, upperStateClctIdx, atomicStateDataBox);

            // (2 * pi * e^2)/(eps_0 * m_e * c^3 * s/UNIT_TIME) = (2 * pi * e^2 * mue_0) / (m_e * c * s/UNIT_TIME)
            constexpr float_X scalingConstantRate
                = static_cast<float_X>((2. * pi * e_SI * e_SI * mue0_SI) / (m_e_SI * c_SI * picongpu::UNIT_TIME));
            /* (N/A^2 * (As)^2) / (kg * m/s * s/UNIT_TIME) = (A^2/A^2 *s^2 * N * UNIT_TIME) / (kg * m * s/s)
             * = (s^2 * kg*m/(s^2) * UNIT_TIME) / ( kg * m) = s^2/(s^2) (kg*m)/(kg*m) * UNIT_TIME = UNIT_TIME
             */
            // unit: UNIT_TIME

            /* [(2 * pi * e^2)/(eps_0 * m_e * c^3)] * nu^2 * g_new/g_old * faax
             * taken from https://en.wikipedia.org/wiki/Einstein_coefficients
             * s * (1/s)^2 = 1/s
             */
            return scalingConstantRate * frequencyPhoton * frequencyPhoton * ratio
                * boundBoundTransitionDataBox.absorptionOscillatorStrength(transitionCollectionIndex);
            // UNIT_TIME * 1/(UNIT_TIME^2) * unitless * unitless = 1/UNIT_TIME
            // unit: 1/UNIT_TIME
        }

        /// @todo radiativeBoundBoundCrossSection
        /// @todo rateRadiativeBoundBoundTransition
    };
} // namespace picongpu::particles::atomicPhysics2::rateCalculation
