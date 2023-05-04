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

#pragma once

#include "picongpu/particles/atomicPhysics2/processClass/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/ProcessIDConversion.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/TransitionDataClass.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::processClass
{
    /** table[processID] = processClass
     *
     * @tparam T_numEntry number of entries in conversion Table
     *
     * @tparam electronicExcitation is channel active?
     * @tparam electronicDeexcitation is channel active?
     * @tparam spontaneousDeexcitation is channel active?
     * @tparam autonomousIonization is channel active?
     * @tparam electonicIonization is channel active?
     * @tparam fieldIonization is channel active?
     */
    template<
        uint8_t T_numEntry,
        TransitionDataClass T_TransitionDataClass,
        bool T_electronicExcitation,
        bool T_electronicDeexcitation,
        bool T_spontaneousDeexcitation,
        bool T_electronicIonization,
        bool T_autonomousIonization,
        bool T_fieldIonization>
    struct ProcessIDConversionTable
    {
        //! table[processID] = processClass
        uint8_t table[T_numEntry];

        using S_ProcessIDConversion = ProcessIDConversion<
            T_electronicExcitation,
            T_electronicDeexcitation,
            T_spontaneousDeexcitation,
            T_electronicIonization,
            T_autonomousIonization,
            T_fieldIonization>;

        /** constructor
         *
         * @tparam T_TransitionDataClass type of transition, e.g. bound-bound_Up, ...
         */
        constexpr ProcessIDConversionTable() : table()
        {
            for(uint8_t i = 0u; i < T_numEntry; ++i)
            {
                if constexpr(T_TransitionDataClass == TransitionDataClass::boundBound_Down)
                    table[i] = S_ProcessIDConversion::getProcessClassBoundBound_Down(i);
                if constexpr(T_TransitionDataClass == TransitionDataClass::boundBound_Up)
                    table[i] = S_ProcessIDConversion::getProcessClassBoundBound_Up(i);
                if constexpr(T_TransitionDataClass == TransitionDataClass::boundFree_Down)
                    table[i] = S_ProcessIDConversion::getProcessClassBoundFreeDown(i);
                if constexpr(T_TransitionDataClass == TransitionDataClass::boundFree_Up)
                    table[i] = S_ProcessIDConversion::getProcessClassBoundFree_Up(i);
                if constexpr(T_TransitionDataClass == TransitionDataClass::autonomous_Down)
                    table[i] = S_ProcessIDConversion::getProcessClassAutonomous_Down(i);
                if constexpr(T_TransitionDataClass == TransitionDataClass::autonomous_Up)
                    table[i] = S_ProcessIDConversion::getProcessClassAutonomous_Up(i);
                else
                    table[i] = 255u;
            }
        }
    };

} // namespace picongpu::particles::atomicPhysics2::processClass
