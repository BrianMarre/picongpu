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

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

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

namespace picongpu::particles::atomicPhysics2::atomicData
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
        typename T_Number,
        typename T_Value,
        uint8_t T_atomicNumber // element
        >
    class ChargeStateOrgaDataBox : public DataBox<T_Number, T_Value, T_atomicNumber>
    {
    public:
        using S_DataBox = DataBox<T_Number, T_Value, T_atomicNumber>;

    private:
        //! number of atomic states associated with the charge state, @todo could be left out to reduce memory but
        //! increase seek time in missing state case
        typename S_DataBox::BoxNumber m_boxNumberAtomicStates;
        //! start collection index of block of atomic states for charge state in collection of AtomicStates
        typename S_DataBox::BoxNumber m_boxStartIndexBlockAtomicStates;

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
            typename S_DataBox::BoxNumber numberAtomicStates,
            typename S_DataBox::BoxNumber startIndexBlockAtomicStates)
            : m_boxNumberAtomicStates(numberAtomicStates)
            , m_boxStartIndexBlockAtomicStates(startIndexBlockAtomicStates)
        {
        }

        /** store data
         *
         * @attention NEVER call with chargeState > T_atomicNumber, otherwise invalid memory access
         *
         * @param collectionIndex charge state of state, used as index for charge states
         * @param numberAtomicStates number of atomic states associated with this charge state
         * @param startIndex start collectionIndex of block of atomic states in atomicState collection with
         * this charge state
         */
        HINLINE void store(uint32_t const collectionIndex, T_Number numberAtomicStates, T_Number startIndex)
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_COLD_DEBUG)
                if(collectionIndex > static_cast<uint32_t>(T_atomicNumber))
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of range call store() chargeState orga data");
                    return;
                }

            m_boxNumberAtomicStates[collectionIndex] = numberAtomicStates;
            m_boxStartIndexBlockAtomicStates[collectionIndex] = startIndex;
        }

        //! @attention NEVER call with chargeState > T_atomicNumber, otherwise invalid memory access
        HDINLINE T_Number numberAtomicStates(uint8_t chargeState)
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_HOT_DEBUG)
                if(chargeState > static_cast<uint32_t>(T_atomicNumber))
                {
                    printf("atomicPhysics ERROR: out of range numberAtomicStates() call\n");
                    return static_cast<T_Number>(0._X);
                }

            return m_boxNumberAtomicStates[chargeState];
        }

        //! @attention NEVER call with chargeState == T_atomicNumber, otherwise invalid memory access
        HDINLINE T_Number startIndexBlockAtomicStates(uint8_t chargeState)
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_HOT_DEBUG)
                if(chargeState > static_cast<uint32_t>(T_atomicNumber))
                {
                    printf("atomicPhysics ERROR: out of range startIndexBlockAtomicStates() call\n");
                    return static_cast<T_Number>(0._X);
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
    template<typename T_Number, typename T_Value, uint8_t T_atomicNumber>
    class ChargeStateOrgaDataBuffer : public DataBuffer<T_Number, T_Value, T_atomicNumber>
    {
    public:
        using S_DataBuffer = DataBuffer<T_Number, T_Value, T_atomicNumber>;
        using S_ChargeStateorgaDataBox = ChargeStateOrgaDataBox<T_Number, T_Value, T_atomicNumber>;


    private:
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferNumberAtomicStates;
        std::unique_ptr<typename S_DataBuffer::BufferNumber> bufferStartIndexBlockAtomicStates;

    public:
        HINLINE ChargeStateOrgaDataBuffer()
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutChargeStates
                = pmacc::GridLayout<1>(T_atomicNumber + 1u, guardSize)
                      .getDataSpaceWithoutGuarding(); // +1 for completely ionized charge state

            bufferNumberAtomicStates.reset(new typename S_DataBuffer::BufferNumber(layoutChargeStates, false));
            bufferStartIndexBlockAtomicStates.reset(new
                                                    typename S_DataBuffer::BufferNumber(layoutChargeStates, false));
        }

        HINLINE S_ChargeStateorgaDataBox getHostDataBox()
        {
            return ChargeStateOrgaDataBox<T_Number, T_Value, T_atomicNumber>(
                bufferNumberAtomicStates->getHostBuffer().getDataBox(),
                bufferStartIndexBlockAtomicStates->getHostBuffer().getDataBox());
        }

        HINLINE S_ChargeStateorgaDataBox getDeviceDataBox()
        {
            return ChargeStateOrgaDataBox<T_Number, T_Value, T_atomicNumber>(
                bufferNumberAtomicStates->getDeviceBuffer().getDataBox(),
                bufferStartIndexBlockAtomicStates->getDeviceBuffer().getDataBox());
        }

        HDINLINE void hostToDevice()
        {
            bufferNumberAtomicStates->hostToDevice();
            bufferStartIndexBlockAtomicStates->hostToDevice();
        }

        HDINLINE void deviceToHost()
        {
            bufferNumberAtomicStates->deviceToHost();
            bufferStartIndexBlockAtomicStates->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
