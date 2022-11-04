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

/** @file implements base class of atomic state start index block data with up- and downward transitions
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

                /** data box storing atomic state startIndexBlock for up- and downward-transitions
                 *
                 * for use on device.
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateStartIndexBlockDataBox_UpDown : public DataBox< T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using dataBoxType
                        = AtomicStateStartIndexBlockDataBox_UpDown< T_Number, T_Value, T_atomicNumber>;
                    using S_DataBox = DataBox<T_Number, T_Value, T_atomicNumber>;

                private:
                    //! start collection index of the block of downward transitions from the atomic state in the corresponding upward collection                    BoxNumber m_boxStartIndexBlockTransitionsDown;
                    typename S_DataBox::BoxNumber m_boxStartIndexBlockTransitionsDown;
                    //! start collection index of the block of upward transitions from the atomic state in the corresponding upward collection
                    typename S_DataBox::BoxNumber m_boxStartIndexBlockTransitionsUp;

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param startIndexBlockAtomicStatesUp start collection index of the block of
                     *  bound-bound transitions in the upward bound-bound transition collection
                     * @param startIndexBlockAtomicStatesDown start collection index of the block of
                     *  bound-bound transitions in the downward bound-bound transition collection
                     */
                    AtomicStateStartIndexBlockDataBox_UpDown(
                        typename S_DataBox::BoxNumber boxStartIndexBlockTransitionsDown,
                        typename S_DataBox::BoxNumber boxStartIndexBlockTransitionsUp)
                        : m_boxStartIndexBlockTransitionsDown(boxStartIndexBlockTransitionsDown)
                        , m_boxStartIndexBlockTransitionsUp(boxStartIndexBlockTransitionsUp)
                    {
                    }

                    //! @attention no range check
                    void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber startIndexDown)
                    {
                        m_boxStartIndexBlockTransitionsDown[collectionIndex] = startIndexDown;
                    }

                    //! @attention no range check
                    void storeUp(uint32_t const collectionIndex, typename S_DataBox::TypeNumber startIndexUp)
                    {
                        m_boxStartIndexBlockTransitionsUp[collectionIndex] = startIndexUp;
                    }

                    /** get start index of block of transitions downward from atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    typename S_DataBox::TypeNumber startIndexBlockTransitionsDown(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitionsDown(collectionIndex);
                    }

                    /** get start index of block of transitions upward from atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    typename S_DataBox::TypeNumber startIndexBlockTransitionsUp(uint32_t const collectionIndex) const
                    {
                        return m_boxStartIndexBlockTransitionsUp(collectionIndex);
                    }

                };

                /** complementing buffer class
                 *
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 */
                template<
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateStartIndexBlockDataBuffer_UpDown : public DataBuffer< T_Number, T_Value, T_atomicNumber >
                {
                public:
                    using S_DataBuffer = DataBuffer<T_Number, T_Value, T_atomicNumber>;

                private:
                    std::unique_ptr< typename S_DataBuffer::BufferNumber > bufferStartIndexBlockTransitionsDown;
                    std::unique_ptr< typename S_DataBuffer::BufferNumber > bufferStartIndexBlockTransitionsUp;

                public:
                    HINLINE AtomicStateStartIndexBlockDataBuffer_UpDown(uint32_t numberAtomicStates)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutAtomicStates = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

                        bufferStartIndexBlockTransitionsDown.reset( new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
                        bufferStartIndexBlockTransitionsUp.reset( new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
                    }

                    HINLINE AtomicStateStartIndexBlockDataBox_UpDown< T_Number, T_Value, T_atomicNumber> getHostDataBox()
                    {
                        return AtomicStateStartIndexBlockDataBox_UpDown< T_Number, T_Value, T_atomicNumber>(
                            bufferStartIndexBlockTransitionsDown->getHostBuffer().getDataBox(),
                            bufferStartIndexBlockTransitionsUp->getHostBuffer().getDataBox());

                    }

                    HINLINE AtomicStateStartIndexBlockDataBox_UpDown< T_Number, T_Value, T_atomicNumber> getDeviceDataBox()
                    {
                        return AtomicStateStartIndexBlockDataBox_UpDown< T_Number, T_Value, T_atomicNumber>(
                            bufferStartIndexBlockTransitionsDown->getDeviceBuffer().getDataBox(),
                            bufferStartIndexBlockTransitionsUp->getDeviceBuffer().getDataBox());
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferStartIndexBlockTransitionsDown->hostToDevice();
                        bufferStartIndexBlockTransitionsUp->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
