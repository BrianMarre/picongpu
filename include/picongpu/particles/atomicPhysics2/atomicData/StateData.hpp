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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"

#include <cstdint>
#include <memory>
#include <utility>

/** @file implements the storage of atomic state data, including ionization states
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each
 * atomicPhysics ion species. These model's parameters are provided by the user as a
 * human readable file of specified format at runtime.
 *
 *  PIConGPU itself does include only basic model data for ionization, the data required
 *  for ADK, Thomas-Fermi and BSI ionization, the data required for the atomicPhysics
 *  modeling is kept separate from PIConGPU itself due to license requirements.
 *
 * This file is read at the start of the simulation and stored in several DataBoxes for
 *  later us on gpu.
 *
 * Consequently always two different classes handle atomic data:
 * - a data class ... implements
 *                      * reading of the atomicData input file
 *                      * export to the DataBox for device side use
 *                      * host side storage of atomicData
 * - a DataBox class ... deviceSide storage and access to atomicData
 *
 * The atomic state data consists of different data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (ionization energy, [eV]
 *         screenedCharge, [eV]
 *         number of atomicStates,
 *         startIndex of block of atomicStates in atomicState list) ]
 *
 * - list of levels (sorted blockwise by ionization state list)
 *    [(configNumber, [ionization state can be retrieved from here]
 *      energy respective to ground state of ionization state)] [eV]
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
                 * Data describing available transitions for each state stored separately
                 * in @todo .
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
                 *      typically uint64_t
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 * @tparam T_numberAtomicStates number of atomic states to be stored
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint8_t T_atomicNumber, // number ionization states-1
                    uint32_t T_numberAtomicStates>
                class StateDataBox
                {
                public:
                    using Idx = T_ConfigNumberDataType;
                    using BoxNumber = T_DataBoxType<T_Number>;
                    using BoxValue = T_DataBoxType<T_Value>;
                    using BoxConfigNumber = T_DataBoxType<T_ConfigNumberDataType>;
                    using BoxTransitionIdx = T_DataBoxType<T_TransitionIndexDataType>;

                    using TypeNumber = T_Number;
                    using TypeValue = T_Value;
                    using DataBoxType = T_DataBoxType;

                    /// @todo necessary?
                    constexpr static uint8_t atomicNumber = T_atomicNumber;
                    constexpr static uint32_t numberAtomicStates = T_numberAtomicStates;

                private:
                    // ionization state data
                    BoxValue m_boxIonizationEnergy; // unit: eV
                    BoxValue m_boxScreenedCharge; // unit: e
                    BoxNumber m_boxNumberAtomicStates;
                    BoxNumber m_boxStartIndexBlockAtomicStates;

                    // atomic state data storage
                    BoxStateConfigNumber m_boxConfigNumber;
                    BoxValue m_boxStateEnergy; // unit: eV

                public:
                    /** constructor
                     *
                     * charge state data is sorted by ascending charge
                     * @param ionizationEnergy dataBox of ionization energy[eV] for charge states
                     * @param screenedCharge dataBox of screenedCharge[e] for charge states
                     * @param numberAtomicStates dataBox number of atomicStates for the charge state
                     * @param startIndexBlockAtomicStates dataBox startIndex of block of atomicStates in atomicState
                     * list for charge state
                     *
                     * atomic state data is sorted block-wise by charge state and secondary ascending by configNumber
                     * @param configNumber [ionization state can be retrieved from here]
                     * @param energy dataBox of energy respective to ground state of ionization state [eV]
                     */
                    StateDataBox(
                        // ionization data
                        BoxValue ionizationEnergy,
                        BoxValue screenedCharge,
                        BoxNumber numberAtomicStates,
                        BoxNumber startIndexBlockAtomicStates,
                        // atomic state data
                        BoxStateConfigNumber configNumber,
                        BoxValue energy)
                        : m_boxIonizationEnergy(ionizationEnergy)
                        , m_boxScreenedCharge(screenedCharge)
                        , m_boxNumberAtomicStates(numberAtomicStates)
                        , m_boxStartIndexBlockAtomicStates(startIndexBlockAtomicStates)
                        , m_boxConfigNumber(configNumber)
                        , m_boxStateEnergy(energy)
                    {
                    }

                    /** returns collection index of atomic state in dataBox with given ConfigNumber
                     *
                     * @param configNumber configNumber of atomic state
                     *
                     * @return returns numStates if not found
                     */
                    HDINLINE uint32_t findStateCollectionIndex(Idx const configNumber) const
                    {
                        // special case completely ionized ion
                        if(configNumber == 0u)
                            return 0u;

                        uint8_t ionizationState = stateRepresentation::ConfigNumber::getIonizationState(configNumber);

                        T_Number startIndexBlock = m_startIndexBlockAtomicStates(ionizationState);
                        T_Number numberAtomicStates = m_numberAtomicStates(ionizationState);

                        /// @todo replace linear search BrianMarre 2022
                        // search for state in dataBox
                        for(uint32_t i = startIndexBlock; i < numberAtomicStates; i++)
                        {
                            if(m_boxConfigNumber(i) == configNumber)
                            {
                                return i
                            }
                        }

                        // atomic state not found return known bad value
                        return numberAtomicStates;
                    }

                    /**returns the energy of the given state respective to the ground state of its ionization
                     *
                     * @param ConfigNumber configNumber of atomic state
                     * @return unit: eV
                     */
                    HDINLINE T_Value getEnergy(Idx const configNumber) const
                    {
                        // special case completely ionized ion
                        if(configNumber == 0u)
                            return 0.0_X;

                        uint32_t collectionIndex = findStateCollectionIndex(configNumber);

                        // atomic state not found, return zero
                        if(collectionIndex == numberAtomicStates)
                        {
                            return static_cast<T_ValueType>(0);
                        }

                        // standard case
                        return m_boxStateEnergy(collectionIndex);
                    }

                    //! returns state corresponding to given index
                    HDINLINE Idx getAtomicStateConfigNumberIndex(uint32_t const StateCollectionIndex) const
                    {
                        return this->m_boxConfigNumber(indexState);
                    }
                };

                template<uint8_t T_atomicNumber, typename T_ConfigNumberDataType = uint64_t>
                class AtomicData
                {
                public:
                }
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu