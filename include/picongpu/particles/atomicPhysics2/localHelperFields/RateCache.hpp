/* Copyright 2022 Brian Marre
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

#include "picongpu/param/atomicPhysics2_Debug.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::localHelperFields
{
    /** @class cache of one row, column, or the diagonal of the rate matrix
     *
     * @tparam T_numberAtomicStates number of entries in cache
     *
     * @attention invalidated every time the local electron spectrum changes
     */
    template<uint16_t T_numberAtomicStates>
    class RateCache
    {
    public:
        constexpr uint16_t numberAtomicStates = T_numberAtomicStates;

    private:
        float_X rates[T_numberAtomicStates] = {0}; // unit: 1/(Dt_PIC)

    public:
        /** set/update cache entry
         *
         * @param collectionIndex collection Index of atomic state
         * @param rate rate of transition, [1/(Dt_PIC)]
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        HDINLINE void updateRate(uint16_t const collectionIndex, float_X rate)
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CACHE_DEBUG)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in updateRate() call");
                    return;
                }

            rates[collectionIndex] = rate;
            return;
        }

        /** get cached rate for an atomic state
         *
         * @param collectionIndex collection Index of atomic state
         * @return rate of transition, [1/Dt_PIC]
         *
         * @attention no range checks outside a debug compile, invalid memory access on failure
         */
        HDINLINE float_X getRate(uint16_t const collectionIndex)
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_RATE_CACHE_DEBUG)
                if(collectionIndex >= numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of range in getRate() call");
                    return 0._X;
                }

            return this->rates[collectionIndex];
        }
    };
} // namespace picongpu::particles::atomicPhysics2::localHelperFields
