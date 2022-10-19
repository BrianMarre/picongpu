/* Copyright 2022 Sergei Bastrakov, Brian Marre
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

/** @file implements the storage of atomic state orga data for the autonomous transitions
 *
 * The atomic state data consists of the following data sets:
 *
 * - collection of atomic states orga data for autonomous transitions (sorted blockwise by ionization state ascending)
 *    [ (number of transitions,
 *       startIndex of transition block)]
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** data box storing atomic state orga data for autonomous transitions
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
                class AtomicStateOrgaDataBox_Autonomous : Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    //! number of autonomous transitions from the atomic state
                    BoxNumber m_boxNumberTransitions;
                    /** start collection index of the block of autonomous transitions
                     * from the atomic state in the collection of autonomous transitions
                     */
                    BoxNumber m_boxStartIndexBlockTransitions;

                    /// @todo transitions from configNumber 0u?

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param boxNumberTransitions number of autonomous transitions from the atomic state
                     * @param startIndexBlockAtomicStates start collection index of block of
                     *  autonomous transitions in autonomous transition collection
                     */
                    AtomicStateOrgaDataBox_Autonomous(
                        BoxNumber boxNumberTransitions,
                        BoxNumber boxStartIndexBlockTransitions)
                        : m_boxNumberTransitions(boxNumberTransitions)
                        , m_boxStartIndexBlockTransitions(boxStartIndexBlockTransitions)
                    {
                    }

                    /** get number of autonomous transitions from an atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber numberTransitions(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberTransitions(collectionIndex);
                    }

                    /** get start index of block of autonomous transitions from atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber startIndexBlockTransitions(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitions(collectionIndex);
                    }

                };


            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu