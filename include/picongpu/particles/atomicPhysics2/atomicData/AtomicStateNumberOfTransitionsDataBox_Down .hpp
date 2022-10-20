/* Copyright 2022 Brian Marre, Sergei Bastrakov
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

#include "picongpu/particles/atomicPhysics2/atomicData/Data.hpp"

#include <cstdint>

/** @file implements storage of numberInBlock data for each atomic state with downward transitions
 *
 * e.g. for autonomous transitions
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** data box storing atomic state numberInBlock for downward-only transitions
                 *
                 * for use on device.
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateNumberOfTransitionsDataBox_Down : public Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    /** start collection index of the block of autonomous transitions
                     * from the atomic state in the collection of autonomous transitions
                     */
                    BoxNumber m_boxNumberOfTransitions;

                    /// @todo transitions from configNumber 0u?

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param boxNumberTransitions number of transitions from the atomic state
                     */
                    AtomicStateNumberOfTransitionsDataBox_Down(
                        BoxNumber boxNumberOfTransitions)
                        : m_boxNumberOfTransitions(boxNumberOfTransitions)
                    {
                    }

                    /** get start index of block of autonomous transitions from atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber numberOfTransitions(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberOfTransitions(collectionIndex);
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
