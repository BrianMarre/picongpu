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


#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>

/** @file implements the storage of atomic state property data
 *
 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
 *      class, while index always refers to a collection index.
 *      The configNumber of a given state is always the same, its collection index depends
 *      on input file,it should therefore only be used internally!
 */

namespace picongpu::particles::atomicPhysics2::atomicData
{
    /** data box storing state property data
     *
     * for use on device.
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
     *      typically uint64_t
     * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
     *
     * @attention ConfigNumber specifies the number of a state as defined by the configNumber
     *      class, while index always refers to a collection index.
     *      The configNumber of a given state is always the same, its collection index depends
     *      on input file,it should therefore only be used internally!
     */
    template<typename T_Number, typename T_Value, typename T_ConfigNumberDataType, uint8_t T_atomicNumber>
    class AtomicStateDataBox : public DataBox<T_Number, T_Value, T_atomicNumber>
    {
    public:
        using Idx = T_ConfigNumberDataType;
        using BoxConfigNumber = pmacc::DataBox<pmacc::PitchedBox<T_ConfigNumberDataType, 1u>>;

        using S_AtomicStateTuple = AtomicStateTuple<T_Value, Idx>;
        using S_DataBox = DataBox<T_Number, T_Value, T_atomicNumber>;

    private:
        //! configNumber of atomic state, sorted block wise by ionization state
        BoxConfigNumber m_boxConfigNumber;
        //! energy respective to ground state of ionization state[eV], sorted block wise by ionizatioState
        typename S_DataBox::BoxValue m_boxStateEnergy; // unit: eV
        uint32_t m_numberAtomicStates;

    public:
        /** constructor
         *
         * @attention atomic state data must be sorted block-wise ascending by
         *  charge state and secondary ascending by configNumber.
         *
         * @param boxConfigNumber dataBox of atomic state configNumber(fancy index)
         * @param boxStateEnergy dataBox of energy respective to ground state of ionization state [eV]
         * @param numberAtomicStates number of atomic states
         */
        AtomicStateDataBox(
            BoxConfigNumber boxConfigNumber,
            typename S_DataBox::BoxValue boxStateEnergy,
            uint32_t numberAtomicStates)
            : m_boxConfigNumber(boxConfigNumber)
            , m_boxStateEnergy(boxStateEnergy)
            , m_numberAtomicStates(numberAtomicStates)
        {
        }

        /** store atomic state in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfill all ordering and content assumptions of constructor!
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param tuple tuple containing data of atomic state
         */
        HINLINE void store(uint32_t const collectionIndex, S_AtomicStateTuple& tuple)
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_COLD_DEBUG)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    throw std::runtime_error("atomicPhysics ERROR: out of bounds atomic state store call");
                    return;
                }

            m_boxConfigNumber[collectionIndex] = std::get<0>(tuple);
            m_boxStateEnergy[collectionIndex] = std::get<1>(tuple);
        }

        /** returns collection index of atomic state in dataBox with given ConfigNumber
         *
         * @attention do not use to get energy of atomic state, use getEnergy() directly instead!
         *
         * @param configNumber ... configNumber of atomic state
         * @param startIndexBlock ... start index for search, not required but faster,
         *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         * @param numberAtomicStatesForChargeState ... number of atomic states in model with charge state
         *  of configNumber, not required but faster,
         *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         *
         * @return returns numStates if not found
         */
        HDINLINE uint32_t findStateCollectionIndex(
            Idx const configNumber, // 0
            uint32_t const numberAtomicStatesForChargeState, // 1
            uint32_t const startIndexBlock = 0u) const // 10
        {
            /// @todo replace linear search, BrianMarre, 2022
            // search for state in dataBox
            for(uint32_t i = 0; i < numberAtomicStatesForChargeState; i++)
            {
                if(m_boxConfigNumber(i + startIndexBlock) == configNumber)
                {
                    return i + startIndexBlock;
                }
            }

            // atomic state not found return known bad value
            return m_numberAtomicStates;
        }

        /**returns the energy of the given state respective to the ground state of its ionization
         *
         * @param ConfigNumber ... configNumber of atomic state
         * @param startIndexBlock ... start index for search, not required but faster,
         *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         * @param numberAtomicStatesForChargeState ... number of atomic states in model with charge state
         *  of configNumber, not required but faster,
         *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
         *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
         *
         * @return unit: eV
         */
        HDINLINE typename S_DataBox::TypeValue getEnergy(
            Idx const configNumber,
            uint32_t const numberAtomicStatesForChargeState,
            uint32_t const startIndexBlock = 0u) const
        {
            // special case completely ionized ion
            if(configNumber == 0u)
                return static_cast<typename S_DataBox::TypeValue>(0.0_X);

            uint32_t collectionIndex
                = findStateCollectionIndex(configNumber, startIndexBlock, numberAtomicStatesForChargeState);

            // atomic state not found, return zero, by definition isolated state
            if(collectionIndex == m_numberAtomicStates)
            {
                return static_cast<typename S_DataBox::TypeValue>(0._X);
            }

            // standard case
            return m_boxStateEnergy(collectionIndex);
        }

        /** returns configNumber of state corresponding to given index
         *
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         * @param collectionIndex index of data box entry to query
         */
        HDINLINE Idx configNumber(uint32_t const collectionIndex) const
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_HOT_DEBUG)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds atomic state configNumber call\n");
                    return static_cast<typename S_DataBox::TypeValue>(0._X);
                }

            return this->m_boxConfigNumber(collectionIndex);
        }

        /** directly query energy dataBox entry, use getEnergy() unless you know what you are doing!
         *
         * @attention does not respond correctly for configNumber == 0, since data for this
         *  case is not stored explicitly
         * @attention no range check outside debug compile, invalid memory access if collectionIndex >=
         * numberAtomicStates
         * @param collectionIndex index of data box entry to query
         */
        HDINLINE typename S_DataBox::TypeValue stateEnergy(uint32_t const collectionIndex) const
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_HOT_DEBUG)
                if(collectionIndex >= m_numberAtomicStates)
                {
                    printf("atomicPhysics ERROR: out of bounds atomic state energy call\n");
                    return static_cast<typename S_DataBox::TypeValue>(0._X);
                }

            return this->m_boxStateEnergy(collectionIndex);
        }

        //! directly query get number of known atomic states
        HDINLINE uint32_t getNumberAtomicStatesTotal()
        {
            return m_numberAtomicStates;
        }
    };

    /** complementing buffer class
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
     */
    template<typename T_Number, typename T_Value, typename T_ConfigNumberDataType, uint8_t T_atomicNumber>
    class AtomicStateDataBuffer : public DataBuffer<T_Number, T_Value, T_atomicNumber>
    {
    public:
        using Idx = T_ConfigNumberDataType;
        using BufferConfigNumber = pmacc::HostDeviceBuffer<T_ConfigNumberDataType, 1u>;
        using S_DataBuffer = DataBuffer<T_Number, T_Value, T_atomicNumber>;

    private:
        std::unique_ptr<BufferConfigNumber> bufferConfigNumber;
        std::unique_ptr<typename S_DataBuffer::BufferValue> bufferStateEnergy;

        uint32_t m_numberAtomicStates;

    public:
        HINLINE AtomicStateDataBuffer(uint32_t numberAtomicStates) : m_numberAtomicStates(numberAtomicStates)
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutAtomicStates
                = pmacc::GridLayout<1>(numberAtomicStates, guardSize).getDataSpaceWithoutGuarding();

            bufferConfigNumber.reset(new BufferConfigNumber(layoutAtomicStates, false));
            bufferStateEnergy.reset(new typename S_DataBuffer::BufferValue(layoutAtomicStates, false));
        }

        HINLINE AtomicStateDataBox<T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getHostDataBox()
        {
            return AtomicStateDataBox<T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                bufferConfigNumber->getHostBuffer().getDataBox(),
                bufferStateEnergy->getHostBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        HINLINE AtomicStateDataBox<T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber> getDeviceDataBox()
        {
            return AtomicStateDataBox<T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>(
                bufferConfigNumber->getDeviceBuffer().getDataBox(),
                bufferStateEnergy->getDeviceBuffer().getDataBox(),
                m_numberAtomicStates);
        }

        //! get number of known atomic states
        HINLINE uint32_t getNumberAtomicStatesTotal() const
        {
            return m_numberAtomicStates;
        }

        HDINLINE void hostToDevice()
        {
            bufferConfigNumber->hostToDevice();
            bufferStateEnergy->hostToDevice();
        }

        HDINLINE void deviceToHost()
        {
            bufferConfigNumber->deviceToHost();
            bufferStateEnergy->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::atomicData
