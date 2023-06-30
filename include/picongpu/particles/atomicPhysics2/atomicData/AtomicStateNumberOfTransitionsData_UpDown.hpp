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

#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"
#include "picongpu/particles/atomicPhysics2/processClass/ProcessClassGroup.hpp"

#include <cstdint>
#include <memory>

/**@file implements storage of numberInBlock data for each atomic state with up- and downward transitions
 *
 * e.g. for bound-bound and bound-free transitions
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing atomic state numberInBlock for up- and downward-transitions
     *
     * for use on device.
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_Number,
        typename T_Value,
        particles::atomicPhysics2::processClass::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateNumberOfTransitionsDataBox_UpDown : public DataBox<T_Number, T_Value>
    {
    public:
        using S_DataBox = DataBox<T_Number, T_Value>;
        static constexpr auto processClassGroup = T_ProcessClassGroup;

    private:
        //! start collection index of the block of upward transitions from the atomic state in the corresponding upward
        //! collection
        typename S_DataBox::BoxNumber m_boxNumberOfTransitionsUp;

        //! start collection index of the block of downward transitions from the atomic state in the corresponding
        //! upward collection
        typename S_DataBox::BoxNumber m_boxNumberOfTransitionsDown;
        //! offset of transition type in chooseTransition selection
        typename S_DataBox::BoxNumber m_boxOffset;

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise by charge state
         *  and secondary ascending by configNumber.
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

        //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
        ALPAKA_FN_HOST void storeDown(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const numberDown)
        {
            m_boxNumberOfTransitionsDown[collectionIndex] = numberDown;
        }

        //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
        ALPAKA_FN_HOST void storeUp(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const numberUp)
        {
            m_boxNumberOfTransitionsUp[collectionIndex] = numberUp;
        }

        //! @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
        ALPAKA_FN_HOST void storeOffset(uint32_t const collectionIndex, typename S_DataBox::TypeNumber const offset)
        {
            m_boxOffset[collectionIndex] = offset;
        }

        /** get number of transitions in block of transitions upward from the atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
         */
        HDINLINE typename S_DataBox::TypeNumber numberOfTransitionsUp(uint32_t const collectionIndex) const
        {
            return m_boxNumberOfTransitionsUp(collectionIndex);
        }

        /** get number of transitions in block of transitions downward from the atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
         */
        HDINLINE typename S_DataBox::TypeNumber numberOfTransitionsDown(uint32_t const collectionIndex) const
        {
            return m_boxNumberOfTransitionsDown(collectionIndex);
        }

        /** get offset of transition type for the atomic state
         *
         * @param collectionIndex atomic state collection index
         *
         * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
         * @attention no range check, invalid memory access if collectionIndex >= numberAtomicStates
         */
        HDINLINE typename S_DataBox::TypeNumber offset(uint32_t const collectionIndex) const
        {
            return m_boxOffset(collectionIndex);
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ProcessClassGroup processClassGroup current data corresponds to
     */
    template<
        typename T_Number,
        typename T_Value,
        particles::atomicPhysics2::processClass::ProcessClassGroup T_ProcessClassGroup>
    class AtomicStateNumberOfTransitionsDataBuffer_UpDown : public DataBuffer<T_Number, T_Value>
    {
    public:
        using DataBoxType = AtomicStateNumberOfTransitionsDataBox_UpDown<T_Number, T_Value, T_ProcessClassGroup>;
        using S_DataBuffer = DataBuffer<T_Number, T_Value>;

    private:
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberOfTransitionsDown;
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberOfTransitionsUp;
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferOffset;

    public:
        HINLINE AtomicStateNumberOfTransitionsDataBuffer_UpDown(uint32_t numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferNumberOfTransitionsDown.reset(new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
            bufferNumberOfTransitionsUp.reset(new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
            bufferOffset.reset(new typename S_DataBuffer::BufferNumber(layoutAtomicStates, false));
        }

        HINLINE DataBoxType getHostDataBox()
        {
            return DataBoxType(
                bufferNumberOfTransitionsDown->getHostBuffer().getDataBox(),
                bufferNumberOfTransitionsUp->getHostBuffer().getDataBox(),
                bufferOffset->getHostBuffer().getDataBox());
        }

        HINLINE DataBoxType getDeviceDataBox()
        {
            return DataBoxType(
                bufferNumberOfTransitionsDown->getDeviceBuffer().getDataBox(),
                bufferNumberOfTransitionsUp->getDeviceBuffer().getDataBox(),
                bufferOffset->getDeviceBuffer().getDataBox());
        }

        HINLINE void hostToDevice()
        {
            bufferNumberOfTransitionsDown->hostToDevice();
            bufferNumberOfTransitionsUp->hostToDevice();
            bufferOffset->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferNumberOfTransitionsDown->deviceToHost();
            bufferNumberOfTransitionsUp->deviceToHost();
            bufferOffset->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
