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

/** @file implements the storage of atomic state property data
 *
 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
 *      class, while index always refers to a collection index.
 *      The configNumber of a given state is always the same, its collection index depends
 *      on input file,it should therefore only be used internally!
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
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
                 * @tparam T_numberAtomicStates number of atomic states to be stored
                 *
                 * @attention ConfigNumber specifies the number of a state as defined by the configNumber
                 *      class, while index always refers to a collection index.
                 *      The configNumber of a given state is always the same, its collection index depends
                 *      on input file,it should therefore only be used internally!
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber,
                    uint32_t T_numberAtomicStates>
                class AtomicStateDataBox : public Data<T_DataBoxType, T_Number, T_Value, T_atomicNumber>
                {
                public:
                    using Idx = T_ConfigNumberDataType;
                    using BoxConfigNumber = T_DataBoxType<T_ConfigNumberDataType>;

                    constexpr static uint32_t numberAtomicStates = T_numberAtomicStates;

                private:
                    //! configNumber of atomic state, sorted block wise by ionization state
                    BoxStateConfigNumber m_boxConfigNumber;
                    //! energy respective to ground state of ionization state[eV], sorted block wise by ionizatioState
                    BoxValue m_boxStateEnergy; // unit: eV

                public:
                    /** constructor
                     *
                     * @attention atomic state data must be sorted block-wise by charge state
                     *  and secondary ascending by configNumber.
                     * @attention the completely ionized state must be left out.
                     *
                     * @param boxConfigNumber dataBox of atomic state configNumber(fancy index)
                     * @param boxStateEnergy dataBox of energy respective to ground state of ionization state [eV]
                     */
                    AtomicStateDataBox(
                        BoxStateConfigNumber boxConfigNumber,
                        BoxValue boxStateEnergy)
                        : m_boxConfigNumber(boxConfigNumber)
                        , m_boxStateEnergy(boxStateEnergy)
                    {
                    }

                    /** returns collection index of atomic state in dataBox with given ConfigNumber
                     *
                     * @param configNumber ... configNumber of atomic state
                     * @param startIndexBlock ... start index for search, not required but faster,
                     *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
                     *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
                     * @param numberAtomicStates ... number of atomic states in model with charge state
                     *  of configNumber, not required but faster,
                     *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
                     *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
                     *
                     * @return returns numStates if not found or configNumber == 0u(never stored explicitly)
                     */
                    HDINLINE uint32_t findStateCollectionIndex(
                        Idx const configNumber,
                        uint32_t const startIndexBlock = 0u,
                        uint32_t const numberAtomicStatesForChargeState = T_numberAtomicStates) const
                    {
                        // special case completely ionized ion, is never stored
                        if(configNumber == 0u)
                            return T_numberAtomicStates;

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
                        return T_numberAtomicStates;
                    }

                    /**returns the energy of the given state respective to the ground state of its ionization
                     *
                     * @param ConfigNumber ... configNumber of atomic state
                     * @param startIndexBlock ... start index for search, not required but faster,
                     *  is available from chargeStateOrgaDataBox.startIndexBlockAtomicStates(chargeState)
                     *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
                     * @param numberAtomicStates ... number of atomic states in model with charge state
                     *  of configNumber, not required but faster,
                     *  is available from chargeStateOrgaDataBox.numberAtomicStates(chargeState).
                     *  with chargeState available from ConfigNumber::getIonizatioState(configNumber)
                     *
                     * @return unit: eV
                     */
                    HDINLINE TypeValue getEnergy(Idx const configNumber) const
                    {
                        // special case completely ionized ion
                        if(configNumber == 0u)
                            return static_cast<TypeValue>(0.0_X);

                        uint32_t collectionIndex = findStateCollectionIndex(configNumber);

                        // atomic state not found, return zero, by definition isolated state
                        if(collectionIndex == numberAtomicStates)
                        {
                            return static_cast<TypeValue>(0._X);
                        }

                        // standard case
                        return m_boxStateEnergy(collectionIndex);
                    }

                    //! returns state corresponding to given index
                    HDINLINE Idx configNumber(uint32_t const collectionIndex) const
                    {
                        return this->m_boxConfigNumber(collectionIndex);
                    }

                    HDINLINE TypeValue stateEnergy(uint32_t const collectionIndex) const
                    {
                        return this->m_boxStateEnergy(collectionIndex);
                    }
                };

            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
