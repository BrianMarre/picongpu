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

/** @file implements the storage of atomic state orga data for the bound-bound transitions
 *
 * The atomic state data consists of the following data sets:
 *
 * - collection of atomic states orga data for bound-bound transitions (sorted blockwise by ionization state ascending)
 *    [ (number of transitions          up,
 *       startIndex of transition block up,
 *       number of transitions          down,
 *       startIndex of transition block down)]
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** data box storing atomic state orga data for bound-bound transitions
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
                class AtomicStateOrgaDataBox_BoundBound : Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    //! number of bound-bound transitions from the atomic state upward
                    BoxNumber m_boxNumberTransitionsUp;
                    /** start collection index of the block of upward bound-bound transitions
                     * from the atomic state in the collection of upward bound-bound transitions
                     */
                    BoxNumber m_boxStartIndexBlockTransitionsUp;
                    //! number of bound-bound transitions from the atomic state downward
                    BoxNumber m_boxNumberTransitionsDown;
                    /** start collection index of the block of downward bound-bound transitions
                     * from the atomic state in the collection of downward bound-bound transitions
                     */
                    BoxNumber m_boxStartIndexBlockTransitionsDown;

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param boxNumberTransitionsUp number of upward bound-bound transitions from the atomic state
                     * @param startIndexBlockAtomicStatesUp start collection index of the block of
                     *  bound-bound transitions in the upward bound-bound transition collection
                     * @param boxNumberTransitionsDown number of downward bound-bound transitions from the atomic state
                     * @param startIndexBlockAtomicStatesDown start collection index of the block of
                     *  bound-bound transitions in the downward bound-bound transition collection
                     */
                    AtomicStateOrgaDataBox_BoundBound(
                        BoxNumber boxNumberTransitionsUp,
                        BoxNumber boxStartIndexBlockTransitionsUp,
                        BoxNumber boxNumberTransitionsDown,
                        BoxNumber boxStartIndexBlockTransitionsDown)
                        : m_boxNumberTransitionsUp(boxNumberTransitionsUp)
                        , m_boxStartIndexBlockTransitionsUp(boxStartIndexBlockTransitionsUp)
                        , m_boxNumberTransitionsDown(boxNumberTransitionsDown)
                        , m_boxStartIndexBlockTransitionsDown(boxStartIndexBlockTransitionsDown)
                    {
                    }

                    /** get number of upward bound-bound transitions from an atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber numberTransitionsUp(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberTransitionsUp(collectionIndex);
                    }

                    /** get number of downward bound-bound transitions from an atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber numberTransitionsDown(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberTransitionsDown(collectionIndex);
                    }

                    /** get start index of block of autonomous transitions from atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber startIndexBlockTransitionsUp(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitionsUp(collectionIndex);
                    }

                    /** get start index of block of autonomous transitions from atomic state
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber startIndexBlockTransitionsDown(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitionsDown(collectionIndex);
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu