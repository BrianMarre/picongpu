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

/**@file implements storage of numberInBlock data for each atomic state with up- and downward transitions
 *
 * e.g. for bound-bound and bound-free transitions
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {

                /** data box storing atomic state numberInBlock for up- and downward-transitions
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
                class AtomicStateNumberOfTransitionsDataBox_UpDown : public Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    //! start collection index of the block of upward transitions from the atomic state in the corresponding upward collection
                    BoxNumber m_boxNumberOfTransitionsUp;

                    //! start collection index of the block of downward transitions from the atomic state in the corresponding upward collection
                    BoxNumber m_boxNumberOfTransitionsDown;

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param numberInBlockAtomicStatesUp start collection index of the block of
                     *  transitions in the corresponding upward transition collection
                     * @param numberInBlockAtomicStatesDown start collection index of the block of
                     *  transitions in the corresponding downward transition collection
                     */
                    AtomicStateNumberOfTransitionsDataBox_UpDown(
                        BoxNumber boxNumberOfTransitionsUp,
                        BoxNumber boxNumberOfTransitionsDown)
                        : m_boxNumberOfTransitionsUp(boxNumberOfBlockTransitionsUp)
                        , m_boxNumberOfTransitionsDown(boxNumberOfBlockTransitionsDown)
                    {
                    }

                    /** get number of transitions in block of transitions upward from the atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     *
                     * @attention no range check
                     */
                    TypeNumber numberOfTransitionsUp(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberOfTransitionsUp(collectionIndex);
                    }

                    /** get number of transitions in block of transitions downward from the atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     *
                     * @attention no range check
                     */
                    TypeNumber numberOfTransitionsDown(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberOfTransitionsDown(collectionIndex);
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
