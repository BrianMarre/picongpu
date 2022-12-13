/* Copyright 2022 Brian Marre, Axel Huebl
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

//! @todo necessary?, Brian Marre, 2022
//#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicData.hpp"

#include <pmacc/algorithms/math.hpp>

/** @file implements calculation of rates for bound-bound atomic physics transitions
 *
 * based on the rate calculation of FLYCHK, as extracted by Axel Huebl in the original
 *  flylite prototype.
 *
 * References:
 * - Axel Huebl
 *  first flylite prototype, not published
 *
 *  - R. Mewe.
 *  "Interpolation formulae for the electron impact excitation of ions in
 *  the H-, He-, Li-, and Ne-sequences."
 *  Astronomy and Astrophysics 20, 215 (1972)
 *
 *  - H.-K. Chung, R.W. Lee, M.H. Chen.
 *  "A fast method to generate collisional excitation cross-sections of
 *  highly charged ions in a hot dense matter"
 *  High Energy Dennsity Physics 3, 342-352 (2007)
 */

namespace picongpu::particles::atomicPhysics2::rateCalculation
{
    /** functor providing rates and cross section calculation
     *
     * @tparam T_IonSpecies resolved typename of the ion species
     * @attention atomic data box input data is assumed to be in eV
     */
    template<typename T_IonSpecies>
    class BoundBoundTransitionRates
    {
        /** gaunt factor like suppression of cross section
         *
         * @param energyDifference difference of energy between atomic states, [eV]
         * @param energyElectron energy of electron, [eV]
         * @param indexTransition internal index of transition in atomicDataBox
         *      use findIndexTransition method of atomicDataBox and screen for not found value
         *
         * @attention no range check for indexTransition outside debug compile, invalid memory access otherwise
         *
         * @return unit: unitless
         */
        HDINLINE static float_X gauntFactor(
            float_X energyDifference, // [eV]
            float_X energyElectron, // [eV]
            uint32_t indexTransition, // unitless
            //AtomicDataBox atomicDataBox)
        {
            // get gaunt coeficients, unit: unitless
            // float_X const A = atomicDataBox.getCinx1(indexTransition);
            // float_X const B = atomicDataBox.getCinx2(indexTransition);
            // float_X const C = atomicDataBox.getCinx3(indexTransition);
            // float_X const D = atomicDataBox.getCinx4(indexTransition);
            // float_X const a = atomicDataBox.getCinx5(indexTransition);

            // calculate gaunt Factor
            float_X const U = energyElectron / energyDifference; // unit: unitless
            float_X const g = A * math::log(U) + B + C / (U + a) + D / ((U + a) * (U + a)); // unitless

            return g * (U > 1.0); // unitless
        }

        /** energyDifference of atomicPhysics transition
         *
         * @param newStateCollectionIndex collection index of final state of transition
         * @param oldStateCollectionIndex collection index of initial state of transition
         * @param atomicStateDataBox dataBox of atomic state property data
         *
         * @return unit [eV]
         * @todo really necessary?, Brian Marre, 2022
         */
        HDINLINE static float_X energyDifference(
            uint32_t const newStateCollectionIndex,
            uint32_t const oldStateCollectionIndex,
            atomicData::AtomicStateDataBox atomicStateDataBox)
        {
            return (
                atomicStateDataBox.getEnergy(newStateCollectionIndex)
                - atomicStateDataBox.getEnergy(oldStateCollectionIndex)); // [eV]
        }

    public:
        /** returns the cross section for atomicPhysics collisional de-/excitation
         *
         * @attention no check whether electron energy high enough for transition,
         *      should be checked by caller
         *
         * @param energyElectron kinetic electron energy only [eV]
         * @param boundBoundTransitionDataBoxound box containing 
         *
         * @return unit: b(barn = 10^(-28) m^2)
         */
        HDINLINE static float_X collisionalExcitationCrosssection(
            float_X energyElectron, // [eV]
            uint32_t const transitionCollectionIndex,
            atomicData::ChargeStateOrgaDataBox chargeStateOrga
            atomicData::BoundBoundTransitionDataBox boundBoundTransitionDataBox)
        {
            float_X m_energyDifference = energyDifference(
                oldConfigNumber,
                newConfigNumber,
                atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

            // case no physical transition possible, since insufficient electron energy
            if(m_energyDifference > energyElectron)
                return 0._X;

            float_X Ratio; // unitless

            // excitation or deexcitation
            if(m_energyDifference < 0._X)
            {
                // deexcitation
                m_energyDifference = -m_energyDifference; // unit: ATOMIC_UNIT_ENERGY

                // ratio due to multiplicity
                // unitless/unitless * (AU + AU) / AU = unitless
                Ratio = static_cast<float_X>((Multiplicity(newConfigNumber)) / (Multiplicity(oldConfigNumber)))
                    * (energyElectron + m_energyDifference) / energyElectron; // unitless

                // security check for NaNs in Ratio and debug outputif present
                if(!(Ratio == Ratio)) // only true if nan
                {
                    printf(
                        "Warning: NaN in ratio calculation, ask developer for more information\n"
                        "   newIdx %u ,oldIdx %u ,energyElectron_SI %f ,energyDifference_m %f",
                        static_cast<uint32_t>(newConfigNumber),
                        static_cast<uint32_t>(oldConfigNumber),
                        energyElectron,
                        m_energyDifference);
                }

                energyElectron = energyElectron + m_energyDifference; // unit; ATOMIC_UNIT_ENERGY
            }
            else
            {
                // excitation
                Ratio = 1._X; // unitless
            }

            // unitless * unitless = unitless
            float_X const collisionalOscillatorStrength
                = Ratio * atomicDataBox.getCollisionalOscillatorStrength(transitionIndex); // unitless

            // physical constants
            // (unitless * m)^2 / unitless = m^2
            float_X c0_SI = float_X(
                8._X * math::pow(picongpu::PI * picongpu::SI::BOHR_RADIUS, 2.0_X) / math::sqrt(3._X)); // uint: m^2, SI

            // scaling constants * E_Ry^2/deltaE_Trans^2 * f * deltaE_Trans/E_kin
            // m^2 * (AUE/AUE)^2 * unitless * AUE/AUE * unitless<-[ J, J, unitless, unitless ] = m^2
            // AUE =^= ATOMIC_UNIT_ENERGY
            float_X crossSection_SI = c0_SI * (1._X / 4._X) / (m_energyDifference * m_energyDifference)
                * collisionalOscillatorStrength * (m_energyDifference / energyElectron)
                * gauntFactor(m_energyDifference,
                              energyElectron,
                              transitionIndex,
                              atomicDataBox); // unit: m^2, SI

            // safeguard against negative cross sections due to imperfect approximations
            if(crossSection_SI < 0._X)
            {
                return 0._X;
            }

            return crossSection_SI;
        }
    }

} // namespace picongpu::particles::atomicPhysics2::rateCalculation
