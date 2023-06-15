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

//! @file helper function for getting deltaEnergy of transition

#pragma once

// need atomicPhysics_Debug.param
#include "picongpu/simulation_defines.hpp"

namespace picongpu::particles::atomicPhysics2
{
    struct DeltaEnergyTransition
    {
        /** ionizationEnergy between two arbitrary ionization States
         *
         * @attention in debug compile returns -1. if upperStateChargeState < lowerStateChargeState
         * @return unit: eV
         */
        template<typename T_ChargeStateDataBox>
        HDINLINE static float_X ionizationEnergy(
            uint8_t const upperStateChargeState,
            uint8_t const lowerStateChargeState,
            T_ChargeStateDataBox const chargeStateDataBox)
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CALCULATION_HOT_DEBUG)
            {
                if(upperStateChargeState < lowerStateChargeState)
                {
                    printf("atomicPhysics ERROR: upper and lower state inverted in ionizationEnergy() call\n");
                    return -1._X; // eV
                }
            }

            float_X sumIonizationEnergies = 0._X; // eV
            for(uint8_t k = lowerStateChargeState; k < upperStateChargeState; k++)
            {
                sumIonizationEnergies += chargeStateDataBox.ionizationEnergy(k); // eV
            }

            return sumIonizationEnergies; // eV
        }

        /** get energy difference between upper and lower state of a transition
         *
         * @tparam T_isIonizing whether transition is ionizing
         *
         * @param atomicStateBox deviceDataBox giving access to atomic state property data
         * @param transitionBox deviceDataBox giving access to transition property data,
         * @param chargeStateBox optional deviceDataBox giving access to charge state property data
         *  required if T_isIonizing = true
         *
         * @return unit: eV
         */
        template<
            bool T_isIonizing,
            typename T_AtomicStateDataBox,
            typename T_TransitionDataBox,
            typename... T_ChargeStateDataBox>
        HDINLINE static float_X get(
            uint32_t const transitionIndex,
            T_AtomicStateDataBox const atomicStateDataBox,
            T_TransitionDataBox const transitionDataBox,
            T_ChargeStateDataBox... chargeStateDataBox)
        {
            using CollectionIdx = typename T_TransitionDataBox::S_TransitionDataBox::Idx;
            using ConfigNumberIdx = typename T_AtomicStateDataBox::Idx;
            using ConfigNumber = typename T_AtomicStateDataBox::ConfigNumber;

            CollectionIdx const lowerStateCollectionIndex
                = transitionDataBox.lowerStateCollectionIndex(transitionIndex);
            CollectionIdx const upperStateCollectionIndex
                = transitionDataBox.upperStateCollectionIndex(transitionIndex);

            // difference initial and final excitation energy
            float_X deltaEnergy = atomicStateDataBox.energy(upperStateCollectionIndex)
                - atomicStateDataBox.energy(lowerStateCollectionIndex); // eV

            if constexpr(T_isIonizing)
            {
                // ionizing electronic interactive processClassGroup
                ConfigNumberIdx const lowerStateConfigNumber
                    = atomicStateDataBox.configNumber(lowerStateCollectionIndex);
                ConfigNumberIdx const upperStateConfigNumber
                    = atomicStateDataBox.configNumber(upperStateCollectionIndex);

                uint8_t const lowerStateChargeState = ConfigNumber::getIonizationState(lowerStateConfigNumber);
                uint8_t const upperStateChargeState = ConfigNumber::getIonizationState(upperStateConfigNumber);

                // + ionization energy
                deltaEnergy += DeltaEnergyTransition::ionizationEnergy<T_ChargeStateDataBox...>(
                    upperStateChargeState,
                    lowerStateChargeState,
                    chargeStateDataBox...); // eV
            }

            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_DELTA_ENERGY_HOT_DEBUG)
                if(deltaEnergy < 0._X)
                    printf(
                        "atomicPhysics ERROR: negative energy in DeltaEnergyTransition::get(...) call\n"
                        "\t processClassGroup %u, transitionIndex: %u, lowerStateClctIdx: %u, upperStateClctIdx: %u,"
                        " deltaEnergy[eV]:  %.8f \n",
                        static_cast<uint32_t>(T_TransitionDataBox::processClassGroup),
                        transitionIndex,
                        lowerStateCollectionIndex,
                        upperStateCollectionIndex,
                        deltaEnergy);
            return deltaEnergy;
        }
    };
} // namespace picongpu::particles::atomicPhysics2
