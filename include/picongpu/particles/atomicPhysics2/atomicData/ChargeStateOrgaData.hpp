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

/** @file implements the storage of charge state orga data
 *
 * The charge state data consists of the following data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (number of atomicStates of this charge state,
 *         startIndex of block of atomicStates in atomicState collection) ]
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** data box storing charge state orga data
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
                    uint8_t T_atomicNumber // element
                    >
                class ChargeStateOrgaDataBox : public DataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                    //! number of atomic states associated with the charge state, @todo could be left out to reduce memory but increase seek time in missing state case
                    BoxNumber m_boxNumberAtomicStates;
                    //! start collection index of block of atomic states for charge state in collection of AtomicStates
                    BoxNumber m_boxStartIndexBlockAtomicStates;

                public:
                    /** constructor
                     *
                     * @attention charge state data must be sorted by ascending charge and
                     *  the completely ionized state is left out.
                     *
                     * @param numberAtomicStates number of atomicStates for the charge state
                     * @param startIndexBlockAtomicStates start collection index of block of atomicStates in atomicState
                     */
                    ChargeStateOrgaDataBox(
                        BoxNumber numberAtomicStates,
                        BoxNumber startIndexBlockAtomicStates)
                        : m_boxNumberAtomicStates(numberAtomicStates)
                        , m_boxStartIndexBlockAtomicStates(startIndexBlockAtomicStates)
                    {
                    }

                    /** store data
                     *
                     * @attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
                     *
                     * @param chargeState
                     * @param numberAtomicStates number of atomic states associated with this charge state
                     * @param startIndex start collectionIndex of block of atomic states in atomicState collection with
                     * this charge state
                     */
                    HINLINE void store(uint8_t const chargeState, T_Number numberAtomicStates, T_Number startIndex)
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= T_atomicNumber)
                        {
                            throw std::runtime_error("atomicPhysics ERROR: outside range call");
                            return;
                        }
                        m_boxNumberAtomicStates[chargeState] = numberAtomicStates;
                        m_boxStartIndexBlockAtomicStates[chargeState] = startIndex;
                    }

                    //! @attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
                    HDINLINE T_Number numberAtomicStates(uint8_t chargeState)
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= T_atomicNumber)
                        {
                            printf("atomicPhysics ERROR: outside range call\n");
                            returnstatic_cast<T_Number>(0u);
                        }
                        return m_boxNumberAtomicStates[chargeState];
                    }

                    //! @attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
                    T_Number startIndexBlockAtomicStates(uint8_t chargeState)
                    {
                        // debug only
                        /// @todo find correct compile guard, Brian Marre, 2022
                        if(collectionIndex >= T_atomicNumber)
                        {
                            printf("atomicPhysics ERROR: outside range call");
                            return static_cast<T_Number>(0u);
                        }
                        return m_boxStartIndexBlockAtomicStates[chargeState];
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
                class ChargeStateOrgaDataBuffer : public DataBuffer< T_Number, T_Value, T_atomicNumber>
                {
                    std::unique_ptr< BufferNumber > bufferNumberAtomicStates;
                    std::unique_ptr< BufferNumber > bufferStartIndexBlockAtomicStates;

                public:
                    HINLINE ChargeStateOrgaDataBuffer()
                    {
                        auto const guardSize = pmacc::DataSpace<1>::create(0);
                        auto const layoutChargeStates = pmacc::GridLayout<1>(T_atomicNumber, guardSize);

                        bufferNumberAtomicStates.reset( new BufferNumber(layoutChargeStates));
                        bufferStartIndexBlockAtomicStates.reset( new BufferNumber(layoutChargeStates));
                    }

                    HINLINE ChargeStateOrgaDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber> getHostDataBox()
                    {
                        return ChargeStateOrgaDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>(
                            bufferNumberAtomicStates->getHostBuffer().getDataBox(),
                            bufferStartIndexBlockAtomicStates->getHostBuffer().getDataBox());
                    }

                    HINLINE TransitionDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber> getDeviceDataBox()
                    {
                        return ChargeStateOrgaDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>(
                            bufferNumberAtomicStates->getDeviceBuffer().getDataBox(),
                            bufferStartIndexBlockAtomicStates->getDeviceBuffer().getDataBox());
                    }

                    HINLINE void syncToDevice()
                    {
                        bufferNumberAtomicStates->hostToDevice();
                        bufferStartIndexBlockAtomicStates->hostToDevice();
                    }

                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
