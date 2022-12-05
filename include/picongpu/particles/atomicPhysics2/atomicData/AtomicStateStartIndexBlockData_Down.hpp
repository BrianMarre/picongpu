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

#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

/** @file implements base class of atomic state start index block data with up- and downward transitions
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
                /** data box storing for each atomic state the startIndexBlock for downward-only transitions
                 *
                 * for use on device.
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateStartIndexBlockDataBox_Down : public DataBox< T_Number, T_Value, T_atomicNumber >
                {
                public:
                    using S_DataBox = DataBox<T_Number, T_Value, T_atomicNumber>;
                private:
                    /** start collection index of the block of autonomous transitions
                     * from the atomic state in the collection of autonomous transitions
                     */
                    typename S_DataBox::BoxNumber m_boxStartIndexBlockTransitions;

                    /// @todo transitions from configNumber 0u?

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     *
                     * @param boxNumberTransitions number of autonomous transitions from the atomic state
                     * @param startIndexBlockAtomicStates start collection index of block of
                     *  autonomous transitions in autonomous transition collection
                     */
                    AtomicStateStartIndexBlockDataBox_Down(
                        typename S_DataBox::BoxNumber boxStartIndexBlockTransitions)
                        : m_boxStartIndexBlockTransitions(boxStartIndexBlockTransitions)
                    {
                    }

                    //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
                    void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber startIndexDown)
                    {
                        m_boxStartIndexBlockTransitions[collectionIndex] = startIndexDown;
                    }

                    /** get start index of block of autonomous transitions from atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
                     */
                    typename S_DataBox::TypeNumber startIndexBlockTransitions(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitions(collectionIndex);
                    }

                };

                /** complementing buffer class
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template< typename T_Number, typename T_Value, uint8_t T_atomicNumber>
                class AtomicStateStartIndexBlockDataBuffer_Down : public DataBuffer<T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using dataBoxType
                        = AtomicStateStartIndexBlockDataBox_Down< T_Number, T_Value, T_atomicNumber>;
                    using S_DataBuffer = DataBuffer<T_Number, T_Value, T_atomicNumber>;

                private:
                    std::unique_ptr< typename S_DataBuffer::BufferNumber > bufferStartIndexBlockTransitionsDown;

                public:
                    HINLINE AtomicStateStartIndexBlockDataBuffer_Down(uint32_t numberAtomicStates)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutAtomicStates = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

                        bufferStartIndexBlockTransitionsDown.reset( new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
                    }

                    HINLINE AtomicStateStartIndexBlockDataBox_Down< T_Number, T_Value, T_atomicNumber>
                    getHostDataBox()
                    {
                        return AtomicStateStartIndexBlockDataBox_Down<
                            T_Number,
                            T_Value,
                            T_atomicNumber>(bufferStartIndexBlockTransitionsDown->getHostBuffer().getDataBox());
                    }

                    HINLINE AtomicStateStartIndexBlockDataBox_Down< T_Number, T_Value, T_atomicNumber> getDeviceDataBox()
                    {
                        return AtomicStateStartIndexBlockDataBox_Down<  T_Number, T_Value, T_atomicNumber>(
                            bufferStartIndexBlockTransitionsDown->getDeviceBuffer().getDataBox());
                    }

                    HDINLINE void hostToDevice()
                    {
                        bufferStartIndexBlockTransitionsDown->hostToDevice();
                    }

                    HDINLINE void deviceToHost()
                    {
                        bufferStartIndexBlockTransitionsDown->deviceToHost();
                    }
                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
