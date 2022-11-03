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
                template<typename T_Number, typename T_Value, uint8_t T_atomicNumber>
                class AtomicStateNumberOfTransitionsDataBox_UpDown : public DataBox<T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using S_DataBox = DataBox<T_Number, T_Value, T_atomicNumber>;

                private:
                    //! start collection index of the block of upward transitions from the atomic state in the corresponding upward collection
                    typename S_DataBox::BoxNumber m_boxNumberOfTransitionsUp;

                    //! start collection index of the block of downward transitions from the atomic state in the corresponding upward collection
                    typename S_DataBox::BoxNumber m_boxNumberOfTransitionsDown;
                    //! offset of transition type in chooseTransition selection
                    typename S_DataBox::BoxNumber m_boxOffset;

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
                     * @param boxOffset offset of transition type in chooseTransition selection
                     */
                    AtomicStateNumberOfTransitionsDataBox_UpDown(
                        typename S_DataBox::BoxNumber boxNumberOfTransitionsUp,
                        typename S_DataBox::BoxNumber boxNumberOfTransitionsDown,
                        typename S_DataBox::BoxNumber boxOffset)
                        : m_boxNumberOfTransitionsUp(boxNumberOfTransitionsUp)
                        , m_boxNumberOfTransitionsDown(boxNumberOfTransitionsDown)
                        , m_boxOffset(boxOffset)
                    {
                    }

                    //! @attention no range check
                    void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const numberDown)
                    {
                        m_boxNumberOfTransitionsDown[collectionIndex] = numberDown;
                    }

                    //! @attention no range check
                    void storeUp(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const numberUp)
                    {
                        m_boxNumberOfTransitionsUp[collectionIndex] = numberUp;
                    }

                    //! @attention no range check
                    void storeOffset(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const offset)
                    {
                        m_boxOffset[collectionIndex] = offset;
                    }

                    /** get number of transitions in block of transitions upward from the atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     *
                     * @attention no range check
                     */
                    typename S_DataBox::TypeNumber numberOfTransitionsUp(uint32_t const collectionIndex) const
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
                    typename S_DataBox::TypeNumber numberOfTransitionsDown(uint32_t const collectionIndex) const
                    {
                        return m_boxNumberOfTransitionsDown(collectionIndex);
                    }

                    /** get offset of transition type for the atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    typename S_DataBox::TypeNumber offset(uint32_t const collectionIndex)
                    {
                        return m_boxOffset(collectionIndex);
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
                class AtomicStateNumberOfTransitionsDataBuffer_UpDown : public DataBuffer< T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using dataBoxType = AtomicStateNumberOfTransitionsDataBox_UpDown<
                        T_Number,
                        T_Value,
                        T_atomicNumber>;
                    using S_DataBuffer = DataBuffer<T_Number, T_Value, T_atomicNumber>;

                private:
                    std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberOfTransitionsDown;
                    std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberOfTransitionsUp;
                    std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferOffset;

                public:
                    HINLINE AtomicStateNumberOfTransitionsDataBuffer_UpDown(uint32_t numberAtomicStates)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutAtomicStates = pmacc::GridLayout<1>(numberAtomicStates, guardSize);

                        bufferNumberOfTransitionsDown.reset(new
                                                            typename S_DataBuffer::BufferValue(layoutAtomicStates));
                        bufferNumberOfTransitionsUp.reset(new typename S_DataBuffer::BufferValue(layoutAtomicStates));
                        bufferOffset.reset(new typename S_DataBuffer::BufferValue(layoutAtomicStates));
                    }

                    HINLINE AtomicStateNumberOfTransitionsDataBox_UpDown<T_Number, T_Value, T_atomicNumber>
                    getHostDataBox()
                    {
                        return AtomicStateNumberOfTransitionsDataBox_UpDown<T_Number, T_Value, T_atomicNumber>(
                            bufferNumberOfTransitionsDown->getHostBuffer().getDataBox(),
                            bufferNumberOfTransitionsUp->getHostBuffer().getDataBox());
                    }

                    HINLINE AtomicStateNumberOfTransitionsDataBox_UpDown<T_Number, T_Value, T_atomicNumber>
                    getDeviceDataBox()
                    {
                        return AtomicStateNumberOfTransitionsDataBox_UpDown<T_Number, T_Value, T_atomicNumber>(
                            bufferNumberOfTransitionsDown->getDeviceBuffer().getDataBox(),
                            bufferNumberOfTransitionsUp->getDeviceBuffer().getDataBox());
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferNumberOfTransitionsDown->hostToDevice();
                        bufferNumberOfTransitionsUp->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
