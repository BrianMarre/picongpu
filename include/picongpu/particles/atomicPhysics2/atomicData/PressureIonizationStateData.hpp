/* Copyright 2022-2023 Brian Marre
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

#include <cstdint>

//! @file implements storage table of pressure ionization state for each atomic state

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing pressure ionization states
     *
     * stores atomic state collectionIndex of pressure ionization state for each atomic state
     *
     * @tparam T_CollectionIndexType dataType used for atomicState collectionIndex
     */
    template<typename T_CollectionIndexType>
    class PressureIonizationStateDataBox
    {
    public:
        using CollectionIdx = T_CollectionIndexType;
        using BoxCollectionIndex = pmacc::DataBox<pmacc::PitchedBox<T_CollectionIndexType, 1u>>;
    private:
        //! collectionIndex of pressure ionization state for each atomic state
        BoxCollectionIndex m_boxCollectionIndex;

    public:
        /** constructor
         *
         * @param boxCollectionIndex dataBox of pressure ionization state collection index
         * @param numberAtomicStates number of atomic states
         */
        AtomicStateDataBox(BoxCollectionIndex boxCollectionIndex)
            : m_boxCollectionIndex(boxCollectionIndex)
        {
        }

        /** store pressure ionization state collectionIndex for the given atomic state
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         *  numberAtomicStates
         *
         * @param state collectionIndex of an atomic state
         * @param pressureIonizationState collectionIndex of it's pressure pressureIonizationState
         */
        HINLINE void store(CollectionIdx const state, CollectionIdx const pressureIonizationState)
        {
            if constexpr(picongpu::atomicPhysics2::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of bounds atomic state store call");
                    return;
                }
        }
    }
} // namespace picongpu::particles::atomicPhysics2::atomicData
