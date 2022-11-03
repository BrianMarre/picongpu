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
                 * @param boxOffset offset of transition type in chooseTransition selection
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateNumberOfTransitionsDataBox_Down : public Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using dataBoxType
                        = AtomicStateNumberOfTransitionsDataBox_Down<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                private:
                    /** start collection index of the block of autonomous transitions
                     * from the atomic state in the collection of autonomous transitions
                     */
                    BoxNumber m_boxNumberOfTransitions;
                    //! offset of transition type in chooseTransition selection
                    BoxNumber m_boxOffset;

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
                        BoxNumber boxNumberOfTransitions,
                        BoxNumber boxOffset)
                        : m_boxNumberOfTransitions(boxNumberOfTransitions)
                        , m_boxOffset(boxOffset)
                    {
                    }

                    //! @attention no range check
                    void storeDown(uint32_t const collectionIndex, TypeNumber const numberDown)
                    {
                        m_boxNumberOfTransitionsDown[collectionIndex] = numberDown;
                    }

                    //! @attention no range check
                    void storeOffset(uint32_t const collectionIndex, TypeNumber offset)
                    {
                        m_boxOffset[collectionOffset] = offset;
                    }

                    /** get number of transitions from atomic state
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

                    /** get offset of transition type for the atomic state
                     *
                     * @param collectionIndex atomic state collection index
                     *
                     * get collectionIndex from atomicStateDataBox.findStateCollectionIndex(configNumber)
                     * @attention no range check
                     */
                    TypeNumber offset(uint32_t const collectionOffset)
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
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    uint8_t T_atomicNumber>
                class AtomicStateNumberOfTransitionsDataBuffer_Down : public DataBuffer< T_Number, T_Value, T_atomicNumber>
                {
                    std::unique_ptr< BufferNumber > bufferNumberOfTransitionsBlockTransitionsDown;

                public:
                    HINLINE AtomicStateNumberOfTransitionsDataBuffer_Down(uint32_t numberAtomicStates)
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutAtomicStates = pmacc::GridLayout<1>(numberAtomicStates, guardSize);

                        bufferNumberOfTransitionsBlockTransitionsDown.reset( new BufferValue(layoutAtomicStates));
                    }

                    HINLINE AtomicStateNumberOfTransitionsDataBox_Down< T_DataBoxType, T_Number, T_Value, T_atomicNumber> getHostDataBox()
                    {
                        return AtomicStateNumberOfTransitionsDataBox_Down< T_DataBoxType, T_Number, T_Value, T_atomicNumber>(
                            bufferNumberOfTransitionsBlockTransitionsDown->getHostBuffer().getDataBox());

                    }

                    HINLINE AtomicStateNumberOfTransitionsDataBox_Down<T_DataBoxType, T_Number, T_Value, T_atomicNumber> getDeviceDataBox()
                    {
                        return AtomicStateNumberOfTransitionsDataBox_Down< T_DataBoxType, T_Number, T_Value, T_atomicNumber>(
                            bufferNumberOfTransitionsBlockTransitionsDown->getDeviceBuffer().getDataBox());
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferNumberOfTransitionsBlockTransitionsDown->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
