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

// charge state data
#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/ChargeStateOrgaData.hpp"

// atomic state data
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateNumberOfTransitionsData_Down.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateNumberOfTransitionsData_UpDown.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateStartIndexBlockData_Down.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicStateStartIndexBlockData_UpDown.hpp"

// transition data
#include "picongpu/particles/atomicPhysics2/atomicData/AutonomousTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundBoundTransitionData.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/BoundFreeTransitionData.hpp"

// precomputed cache for transition selection kernel
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionSelectionDataBox.hpp"

// conversion of configNumber to charge state for checking
#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"

// Host DeviceBuffer for storage of data
#include <pmacc/memory/buffers/HostDeviceBuffer.tpp>

#include <cstdint>
#include <string>
#include <tuple>

#include <list>
#include <stdexcept>
#include <fstream>

#include <memory>

/** @file gathers atomic data storage implementations and implements filling on runtime
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each
 * atomicPhysics ion species.
 * These model's parameters are provided by the user as a .txt file of specified format
 * at runtime.
 *
 *  PIConGPU itself only includes charge state data, for ADK-, Thomas-Fermi- and BSI-ionization.
 *  All other atomic state data is kept separate from PIConGPU itself, due to license requirements.
 *
 * This file is read at the start of the simulation and stored distributed in several
 *  objects by set of kernels.
 *
 * Storage includes:
 *  - charge state property data [ChargeStateDataBox.hpp]
 *      * ionization energy
 *      * screened charge
 *  - charge state orga data [ChargeStateOrgaDataBox.hpp]
 *      * number of atomic states for charge state
 *      * start index block for charge state
 * - atomic state property data [AtomicStateDataBox.hpp]
 *      * configNumber
 *      * state energy, above ground state
 * - atomic state orga data
 *      [AtomicStateNumberOfTransitionsDataBox_Down, AtomicStateNumberOfTransitionsDataBox_UpDown]
 *       * number of transitions (up-/)down for each atomic state,
 *          by type of transition(bound-bound/bound-free/autonomous)
 *       * offset in transition selection ordering for each atomic state
 *      [AtomicStateStartIndexBlockDataBox_Down, AtomicStateStartIndexBlockDataBox_UpDown]
 *       * start index of block in transition collection index for atomic state,
 *          by type of transition(bound-bound/bound-free/autonomous)
 * - transition property data[BoundBoundTransitionDataBox, BoundFreeTransitionDataBox, AutonomousTransitionDataBox]
 *      * parameters for cross section calculation for each modeled transition
 *
 * (orga data describes the structure of the property data for faster lookups, lookups are
 *  are always possible without it, but are possible non performant)
 */

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace atomicData
            {
                /** gathering of all atomicPhyiscs input data
                 *
                 * @tparam T_DataBoxType dataBox type used for storage
                 * @tparam T_Number dataType used for number storage, typically uint32_t
                 * @tparam T_Value dataType used for value storage, typically float_X
                 * @tparam T_ConfigNumberDataType dataType used for configNumber storage,
                 *      typically uint64_t
                 * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
                 * @tparam T_n_max maximum principal quantum number contained in data
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint32_t T_atomicNumber,
                    uint8_t T_n_max>
                class AtomicData
                {
                public:
                    using DataBoxType = T_DataBoxType;
                    using TypeNumber = T_Number;
                    using TypeValue = T_Value;
                    using Idx = T_ConfigNumberDataType;

                    //using BufferValue = pmacc::GridBuffer<TypeValue, 1>;
                    //using BufferNumber = pmacc::GridBuffer<TypeNumber, 1>;
                    //using BufferConfigNumber = pmacc::GridBuffer<T_ConfigNumberDataType, 1>;

                    using ChargeStateTuple = std::tuple<
                        uint8_t,    // charge state
                        TypeValue,  // ionization energy[eV]
                        TypeValue>; // screened charge[e]

                    using AtomicStateTuple = std::tuple<
                        Idx,        // configNumber
                        TypeValue>; // energy over ground [eV]

                    using BoundBoundTransitionTuple = std::tuple<
                        TypeValue, // collisional oscillator strength
                        TypeValue, // absorption oscillator strength
                        TypeValue, // cinx1 gaunt tunnel coefficient
                        TypeValue, // cinx2
                        TypeValue, // cinx3
                        TypeValue, // cinx4
                        TypeValue, // cinx5
                        Idx,       // lowerState
                        Idx>;      // upperState

                    using BoundFreeTransitionTuple = std::tuple<
                        TypeValue, // cinx1 cross section parameter
                        TypeValue, // cinx2
                        TypeValue, // cinx3
                        TypeValue, // cinx4
                        TypeValue, // cinx5
                        TypeValue, // cinx6
                        TypeValue, // cinx7
                        TypeValue, // cinx8
                        Idx,       // lowerState
                        Idx>;      // upperState

                    using AutonomousTransitionTuple = std::tuple<
                        T_Value, // rate [1/s]
                        Idx,     // lowerState
                        Idx>;    // upperState

                    constexpr uint8_t atomicNumber = T_atomicNumber;
                    constexpr uint8_t n_max = T_n_max;

                    // S_* for shortened name
                    using S_ChargeStateDataBox =
                        ChargeStateDataBox< T_DataBoxType, T_Number, T_Value, T_atomicNumber>;
                    using S_ChargeStateOrgaDataBox =
                        ChargeStateOrgaDataBox< T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    using S_AtomicStateDataBox =
                        AtomicStateDataBox< T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber >;
                    using S_AtomicStateStartIndexBlockDataBox_UpDown =
                        AtomicStateStartIndexBlockDataBox_UpDown< T_DataBoxType, T_Number, T_Value, T_atomicNumber >;
                    using S_AtomicStateStartIndexBlockDataBox_Down =
                        AtomicStateStartIndexBlockDataBox_Down< T_DataBoxType, T_Number, T_Value, T_atomicNumber >;
                    using S_AtomicStateNumberOfTransitionsDataBox_UpDown =
                        AtomicStateNumberOfTransitionsDataBox_UpDown<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;
                    using S_AtomicStateNumberOfTransitionsDataBox_Down =
                        AtomicStateNumberTransitionsBuffer_UpDown<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    using S_BoundBoundTransitionDataBox =
                        BoundBoundTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;
                    using S_BoundFreeTransitionDataBox =
                        BoundFreeTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;
                    using S_AutonomousTransitionDataBox =
                        AutonomousTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;

                    using S_TranstionTransitionSelectionDataBox =
                        TransitionSelectionDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    // buffer types
                private:
                    // pointers to storage
                    // charge state data
                    std::unique_ptr<ChargeStateDataBuffer> chargeStateDataBuffer;
                    std::unique_ptr<ChargeStateOrgaDataBuffer> chargeStateOrgaDataBuffer;

                    // atomic property data
                    std::unique_ptr<AtomicStateDataBuffer> atomicStateDataBuffer;
                    // atomic orga data
                    std::unique_ptr<AtomicStateStartIndexBlockDataBuffer_UpDown> atomicStateStartIndexBlockDataBuffer_BoundBound;
                    std::unique_ptr<AtomicStateStartIndexBlockDataBuffer_UpDown> atomicStateStartIndexBlockDataBuffer_BoundFree;
                    std::unique_ptr<AtomicStateStartIndexBlockDataBuffer_Down> atomicStateStartIndexBlockDataBuffer_Autonomous;
                    std::unique_ptr<AtomicStateNumberOfTransitionsDataBuffer_UpDown> atomicStateNumberOfTransitionsDataBuffer_BoundBound;
                    std::unique_ptr<AtomicStateNumberOfTransitionsDataBuffer_UpDown> atomicStateNumberOfTransitionsDataBuffer_BoundFree;
                    std::unique_ptr<AtomicStateNumberOfTransitionsDataBuffer_Down> atomicStateNumberOfTransitionsDataBuffer_Autonomous;

                    // transition data
                    std::unique_ptr<BoundBoundTransitionDataBuffer> boundBoundTransitionDataBuffer;
                    std::unique_ptr<BoundFreeTransitionDataBuffer> boundFreeTransitionDataBuffer;
                    std::unique_ptr<AutonomousTransitionDataBuffer> autonomousTransitionDataBuffer;

                    // transition selection data
                    std::unique_ptr<TransitionSelectionDataBuffer> transitionSelectionDataBuffer;

                    uint32_t m_numberAtomicStates = 0u;

                    uint32_t m_numberBoundBoundTransitions = 0u;
                    uint32_t m_numberBoundFreeTransitions = 0u;
                    uint32_t m_numberAutonomousTransitions = 0u;

                    //! open file
                    HINLINE static std::ifstream openFile(std::string fileName, std::string fileContent)
                    {
                        std::ifstream file(fileName);

                        // check for success
                        if(!file)
                        {
                            throw std::runtime_error(
                                "atomicPhysics ERROR: could not open "+ fileContent + ": " + fileName);
                        }

                        return file;
                    }

                    /// @todo generalize to single template filling template tuple from line
                    /** read charge state data file
                     *
                     * @attention assumes input to already fulfills all ordering and unit assumptions
                     *   - charge state data is sorted by ascending charge
                     *   - the completely ionized state is left out
                     *
                     * @return returns empty list if file not found/accessible
                     */
                    std::list<ChargeStateTuple> readChargeStates(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "charge state data");
                        if( !file )
                            return std::list<BoundBoundTransitionTuple>{};

                        std::list<ChargeStateTuple> chargeStateList;

                        TypeValue ionizationEnergy;
                        TypeValue screenedCharge;
                        uint8_t chargeState;
                        uint8_t numberChargeStates = 0u;

                        while( file >> chargeState >> ionizationEnergy >> screenedCharge )
                        {
                            if (chargeState == T_atomicNumber)
                                throw std::runtime_error("charge state " + std::to_string(chargeState)
                                    + " should not be included in input file for Z = " + std::to_string(T_atomicNumber));

                            ChargeStateTuple item = std::make_tuple(
                                chargeState,
                                ionizationEnergy, // [eV]
                                screenedCharge)   // [e]

                            chargeStateList.push_back(item);

                            numberChargeStates++;
                        }

                        if(numberChargeStates > T_atomicNumber)
                            throw std::runtime_error("atomicPhysics ERROR: too many charge states, num > Z: " + std::to_string(T_atomicNumber));

                        return chargeStateList;
                    }

                    /** read atomic state data file
                     *
                     * @attention assumes input to already fulfills all ordering and unit assumptions
                     *   - atomic state data is sorted block wise by charge state and secondary by ascending configNuber
                     *   - the completely ionized state is left out
                     *
                     * @return returns empty list if file not found/accessible
                     */
                    std::list<AtomicStateTuple> readAtomicStates(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "atomic state data");
                        if( !file )
                            return std::list<BoundBoundTransitionTuple>{};

                        std::list<AtomicStateTuple> atomicStateList;

                        double stateConfigNumber;
                        TypeValue energyOverGround;

                        while( file >> stateConfigNumber >> energyOverGround )
                        {
                            AtomicStateTuple item = std::make_tuple(
                                static_cast<Idx>(stateConfigNumber), //unitless
                                energyOverGround) // [eV]

                            atomicStateList.push_back(item);

                            numberAtomicStates++;
                        }

                        return atomicStateList;
                    }

                    /** read bound-bound transitions data file
                     *
                     * @attention assumes input to already fulfills all ordering and unit assumptions
                     *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state configNumber
                     *
                     * @return returns empty list if file not found/accessible
                     */
                    std::list<BoundBoundTransitionTuple> readBoundBoundTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "bound-bound transition data");
                        if( !file )
                            return std::list<BoundBoundTransitionTuple>{};

                        std::list<BoundBoundTransitionTuple> boundBoundTransitions;

                        double idxLower;
                        double idxUpper;
                        TypeNumber collisionalOscillatorStrength;
                        TypeNumber absorptionOscillatorStrength;

                        // gauntCoeficients
                        TypeNumber cinx1;
                        TypeNumber cinx2;
                        TypeNumber cinx3;
                        TypeNumber cinx4;
                        TypeNumber cinx5;

                        while(file >> idxLower >> idxUpper
                            >> collisionalOscillatorStrength >> absorptionOscillatorStrength
                            >> cinx1 >> cinx2 >> cinx3 >> cinx4 >> cinx5 )
                        {
                            Idx stateLower = static_cast<Idx>(idxLower);
                            Idx stateUpper = static_cast<Idx>(idxUpper);

                            // protection against circle transitions
                            if(stateLower == stateUpper)
                            {
                                std::cout
                                << "atomicPhysics ERROR: circular transitions are not supported,"
                                "treat steps separately" << std::endl;
                                continue;
                            }

                            BoundBoundTransitionTuple item = std::make_tuple(
                                collisionalOscillatorStrength,
                                absorptionOscillatorStrength,
                                cinx1,
                                cinx2,
                                cinx3,
                                cinx4,
                                cinx5,
                                stateLower,
                                stateUpper);

                            boundBoundTransitions.push_back(item);
                            numberBoundBoundTransitions++;
                        }
                        return boundBoundTransitions;
                    }

                    /** read bound-free transitions data file
                     *
                     * @attention assumes input to already fulfills all ordering and unit assumptions
                     *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state configNumber
                     *
                     * @return returns empty list if file not found/accessible
                     */
                    std::list<BoundFreeTransitionTuple> readBoundFreeTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "bound-free transition data");
                        if( !file )
                            return std::list<BoundFreeTransitionTuple>{};

                        std::list<BoundFreeTransitionTuple> boundFreeTransitions;

                        double idxLower;
                        double idxUpper;

                        // gauntCoeficients
                        TypeNumber cinx1;
                        TypeNumber cinx2;
                        TypeNumber cinx3;
                        TypeNumber cinx4;
                        TypeNumber cinx5;
                        TypeNumber cinx6;
                        TypeNumber cinx7;
                        TypeNumber cinx8;

                        while(file >> idxLower >> idxUpper
                            >> cinx1 >> cinx2 >> cinx3 >> cinx4 >> cinx5 >> cinx6 >> cinx7 >> cinx8)
                        {
                            Idx stateLower = static_cast<Idx>(idxLower);
                            Idx stateUpper = static_cast<Idx>(idxUpper);

                            // protection against circle transitions
                            if(stateLower == stateUpper)
                            {
                                std::cout
                                << "atomicPhysics ERROR: circular transitions are not supported,"
                                "treat steps separately" << std::endl;
                                continue;
                            }

                            BoundFreeTransitionTuple item = std::make_tuple(
                                cinx1,
                                cinx2,
                                cinx3,
                                cinx4,
                                cinx5,
                                cinx6,
                                cinx7,
                                cinx8,
                                stateLower,
                                stateUpper);

                            boundFreeTransitions.push_back(item);
                            numberBoundFreeTransitions++;
                        }
                        return boundFreeTransitions;
                    }

                    /** read autonomous transitions data file
                     *
                     * @attention assumes input to already fulfills all ordering and unit assumptions
                     *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state configNumber
                     *
                     * @return returns empty list if file not found/accessible
                     */
                    std::list<AutonomousTransitionTuple> readAutonomousTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "autonomous transition data");
                        if( !file )
                            return std::list<AutonomousTransitionTuple>{};

                        std::list<AutonomousTransitionTuple> autonomousTransitions;

                        double idxLower;
                        double idxUpper;

                        // unit: 1/s
                        TypeNumber rate;

                        while(file >> idxLower >> idxUpper >> rate)
                        {
                            Idx stateLower = static_cast<Idx>(idxLower);
                            Idx stateUpper = static_cast<Idx>(idxUpper);

                            // protection against circle transitions
                            if(stateLower == stateUpper)
                            {
                                std::cout
                                << "atomicPhysics ERROR: circular transitions are not supported,"
                                "treat steps separately" << std::endl;
                                continue;
                            }

                            AutonomousTransitionTuple item = std::make_tuple(
                                rate,
                                stateLower,
                                stateUpper);

                            autonomousTransitions.push_back(item);
                            numberAutonomousTransitions++;
                        }
                        return autonomousTransitions;
                    }

                    /** check charge state list
                     *
                     * @throws runtime error if duplicate charge state, missing charge state,
                     *  order broken, completely ionized state included or unphysical charge state
                     */
                    void checkChargeStateList(std::list<ChargeStateTuple>& const chargeStateList)
                    {
                        std::list<ChargeStateTuple>::iterator iter = chargeStateList.begin();

                        uint8_t chargeState = 1u;
                        uint8_t lastChargeState;
                        uint8_t currentChargeState;

                        lastChargeState = std::get<0>(*iter);
                        iter++;

                        if (lastChargeState != 0)
                            throw std::runtime_error("atomicPhysics ERROR: charge state 0 not first charge state");

                        for (iter; iter != chargeStateList.end(); iter++)
                        {
                            currentChargeState = std::get<0>(*iter);

                            // duplicate atomic state
                            if ( currentChargeState == lastChargeState )
                                throw std::runtime_error("atomicPhysics ERROR: duplicate charge state");

                            // ordering
                            if ( not(currentChargeState > lastChargeState) )
                                throw std::runtime_error("atomicPhysics ERROR: charge state ordering wrong");

                            // missing charge state
                            if ( not (currentChargeState == chargeState) )
                                throw std::runtime_error("atomicPhysics ERROR: charge state missing");

                            // completely ionized state
                            if (chargeState == T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: completely ionized charge state found");

                            // unphysical state
                            if (chargeState == T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                            chargeState++;
                            lastChargeState = currentChargeState;
                        }
                    }

                    /** check atomic state list
                     *
                     * @throws runtime error if duplicate atomic state, primary order broken,
                     *  secondary order broken, completely ionized state found, or unphysical charge state found
                     */
                    void checkAtomicStateList(std::list<AtomicStateTuple>& const atomicStateList)
                    {
                        std::list<ChargeStateTuple>::iterator iter = atomicStateList.begin();

                        Idx lastAtomicStateConfigNumber;
                        uint8_t lastChargeState;

                        Idx currentAtomicStateConfigNumber;
                        uint8_t currentChargeState;

                        lastAtomicStateConfigNumber = std::get<0>(*iter);
                        lastChargeState = stateRepresentation::ConfigNumber<
                            Idx,
                            T_n_max,
                            T_atomicNumber>::getIonizationState(lastAtomicStateConfigNumber);

                        iter++;

                        for (; iter != chargeStateList.end(); iter++)
                        {
                            currentAtomicStateConfigNumber = std::get<0>(*iter);
                            currentChargeState = stateRepresentation::ConfigNumber<
                                Idx,
                                T_n_max,
                                T_atomicNumber>::getIonizationState(currentAtomicStateConfigNumber);

                            // duplicate atomic state
                            if (currentAtomicStateConfigNumber == lastAtomicStateConfigNumber)
                                throw std::runtime_error("atomicPhysics ERROR: duplicate atomic state");
                            // later duplicate will break ordering

                            // primary/secondary order
                            if ( currentChargeState == lastChargeState )
                                // same block
                                if ( currentAtomicStateConfigNumber < lastAtomicStateConfigNumber )
                                    throw std::runtime_error("atomicPhysics ERROR: wrong secondary ordering of atomic states");
                            else
                                // next block
                                if ( currentChargeState > lastChargeState )
                                    throw std::runtime_error("atomicPhysics ERROR: wrong primary ordering of atomic state");

                            // completely ionized atomic state
                            if ( currentChargeState == T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: completely ionized charge state found");

                            // unphysical atomic state
                            if ( currentChargeState > T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                            lastChargeState = currentChargeState;
                            lastAtomicStateConfigNumber = currentAtomicStateConfigNumber;
                        }
                    }

                    //! helper function giving back transition type name
                    template<typename T_TransitionTuple>
                    HINLINE constexpr std::string getStringTransitionType() = 0;

                    template<>
                    HINLINE constexpr std::string getStringTransitionType<BoundBoundTransitionTuple>()
                    {
                        return "bound-bound"
                    }

                    template<>
                    HINLINE constexpr std::string getStringTransitionType<BoundFreeTransitionTuple>()
                    {
                        return "bound-free"
                    }

                    template<>
                    HINLINE constexpr std::string getStringTransitionType<AutonomousTransitionTuple>()
                    {
                        return "autonomous"
                    }

                    //! helper function checking for charge states compatibility with transition type
                    template<typename T_TransitionTuple>
                    HINLINE bool checkForTransitionType(uint8_t lowerChargeState, uint8_t upperChargeState) = 0;

                    template<>
                    HINLINE bool wrongForTransitionType<BoundBoundTransitionTuple>(uint8_t lowerChargeState, uint8_t upperChargeState)
                    {
                        if (lowerChargeState == upperChargeState)
                            return false; // correct case
                        return true; // error case
                    }

                    template<>
                    HINLINE bool wrongForTransitionType<BoundFreeTransitionTuple>(uint8_t lowerChargeState, uint8_t upperChargeState)
                    {
                        if (lowerChargeState > upperChargeState)
                            return false; // correct case
                        return true; // error case
                    }

                    template<>
                    HINLINE bool wrongForTransitionType<AutonomousTransitionTuple>(uint8_t lowerChargeState, uint8_t upperChargeState)
                    {
                        return false; // charge state might change or not
                    }


                    /** check transition list
                     *
                     * @throws runtime error if primary order broken,
                     *  secondary order broken, transition from/to unphysical charge state found,
                     *  wrong transition type for lower/upper charge state pair
                     */
                    template<typename T_TransitionTuple>
                    void checkTransitionList(std::list<T_TransitionTuple>& const transitionList)
                    {
                        constexpr uint8_t tupleSize = std::tuple_size<T_TransitionTuple>;
                        std::string transitionType = getStringTransitionType<T_TransitionTuple>();

                        std::list<T_TransitionTuple>::iterator iter = transitionList.begin();

                        Idx lastLowerAtomicStateConfigNumber;
                        Idx lastUpperAtomicStateConfigNumber;
                        uint8_t lastLowerChargeState;
                        uint8_t lastUpperChargeState;

                        Idx currentLowerAtomicStateConfigNumber;
                        Idx currentUpperAtomicStateConfigNumber;
                        uint8_t currentLowerChargeState;
                        uint8_t currentUpperChargeState;

                        lastLowerAtomicStateConfigNumber = std::get<tupelSize - 2u>(*iter);
                        lastUpperAtomicStateConfigNumber = std::get<tupelSize - 1u>(*iter);
                        lastLowerChargeState = stateRepresentation::ConfigNumber<
                            Idx,
                            T_n_max,
                            T_atomicNumber>::getIonizationState(lastLowerAtomicStateConfigNumber);
                        lastUpperChargeState = stateRepresentation::ConfigNumber<
                            Idx,
                            T_n_max,
                            T_atomicNumber>::getIonizationState(lastUpperAtomicStateConfigNumber);
                        iter++;

                        // transitionType <-> charge state pair
                        if ( wrongForTransitionType<T_TransitionTuple>(lastLowerChargeState, lastUpperChargeState) )
                            throw std::runtime_error("atomicPhysics ERROR: wrong upper-/lower charge state pair for transition type" + transitionType);

                        // unphysical lower charge state
                        if (lastLowerChargeState > T_atomicNumber)
                            throw std::runtime_error("atomicPhysics ERROR: unphysical lower charge State");

                        // unphysical upper charge state
                        if (lastUpperChargeState > T_atomicNumber)
                            throw std::runtime_error("atomicPhysics ERROR: unphysical upper charge State");

                        for (; iter != chargeStateList.end(); iter++)
                        {
                            currentLowerAtomicStateConfigNumber = std::get<tupelSize - 2u>(*iter);
                            currentUpperAtomicStateConfigNumber = std::get<tupelSize - 1u>(*iter);
                            currentLowerChargeState = stateRepresentation::ConfigNumber<
                                Idx,
                                T_n_max,
                                T_atomicNumber>::getIonizationState(currentLowerAtomicStateConfigNumber);
                            currentUpperChargeState = stateRepresentation::ConfigNumber<
                                    Idx,
                                    T_n_max,
                                    T_atomicNumber>::getIonizationState(currentUpperAtomicStateConfigNumber);

                            // primary/secondary order
                            if (lastLowerAtomicStateConfigNumber == currentLowerAtomicStateConfigNumber)
                                // same block
                                if ( currentUpperAtomicStateConfigNumber < lastUpperAtomicStateConfigNumber )
                                    throw std::runtime_error("atomicPhysics ERROR: wrong secondary ordering of "
                                        + transitionType + " transitions" );
                            else
                                // next block
                                if ( lastLowerAtomicStateConfigNumber > currentLowerAtomicStateConfigNumber )
                                    throw std::runtime_error("atomicPhysics ERROR: wrong primary ordering of " + transitionType + " transition");

                            // transitionType <-> charge state pair
                            if ( wrongForTransitionType<T_TransitionTuple>(currentLowerChargeState, currentUpperChargeState) )
                                throw std::runtime_error("atomicPhysics ERROR: wrong upper-/lower charge state pair for transition type" + transitionType);

                            // unphysical lower charge state
                            if (currentLowerChargeState > T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: unphysical lower charge State in " + transitionType + " transitions");

                            // unphysical upper charge state
                            if (currentUpperChargeState > T_atomicNumber)
                                throw std::runtime_error("atomicPhysics ERROR: unphysical upper charge State in " + transitionType + " transitions");

                            lastLowerChargeState = currentLowerChargeState;
                            lastUpperChargeState = currentUpperChargeState;
                            lastLowerAtomicStateConfigNumber = currentLowerAtomicStateConfigNumber;
                            lastUpperAtomicStateConfigNumber = currentUpperAtomicStateConfigNumber;
                        }
                    }

                    //! init buffers, @attention all readMethods must have been executed before exactly once!
                    HINLINE void initBuffers()
                    {
                    // charge state data
                    chargeStateDataBuffer.reset( new ChargeStateDataBuffer());
                    chargeStateOrgaDataBuffer.reset( new ChargeStateOrgaDataBuffer());

                    // atomic property data
                    atomicStateDataBuffer.reset( new AtomicStateDataBuffer(numberAtomicStates));
                    // atomic orga data
                    atomicStateStartIndexBlockDataBuffer_BoundBound.reset( new AtomicStateStartIndexBlockDataBuffer_UpDown(numberAtomicStates));
                    atomicStateStartIndexBlockDataBuffer_BoundFree.reset( new AtomicStateStartIndexBlockDataBuffer_UpDown(numberAtomicStates));
                    atomicStateStartIndexBlockDataBuffer_Autonomous.reset( new AtomicStateStartIndexBlockDataBuffer_Down(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_BoundBound.reset( new AtomicStateNumberOfTransitionsDataBuffer_UpDown(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_BoundFree.reset( new AtomicStateNumberOfTransitionsDataBuffer_UpDown(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_Autonomous.reset( new AtomicStateNumberOfTransitionsDataBuffer_Down(numberAtomicStates));

                    // transition data
                    boundBoundTransitionDataBuffer.reset( new BoundBoundTransitionDataBuffer(numberBoundBoundTransitions));
                    boundFreeTransitionDataBuffer.reset( new BoundFreeTransitionDataBuffer(numberBoundFreeTransitions));
                    autonomousTransitionDataBuffer.reset( new AutonomousTransitionDataBuffer(numberAutonomousTransitions));

                    // transition selection data
                    transitionSelectionDataBuffer.reset( new TransitionSelectionDataBuffer(numberAtomicStates));
                    }

                public:
                    /** read input files and create/fill data boxes
                     *
                     * @param fileChargeData path to file containing charge state data
                     * @param fileAtomicStateData path to file containing atomic state data
                     * @param fileTransitionData path to file containing atomic state data
                     */
                    AtomicData(
                        std::string fileChargeData,
                        std::string fileAtomicStateData,
                        std::string fileBoundBoundTransitionData,
                        std::string fileBoundFreeTransitionData,
                        std::string fileAutonomousTransitionData)
                    {
                        // read in files
                        std::list<ChargeStateTuple> chargeStates = readChargeStates(fileChargeData);
                        std::list<AtomicStateTuple> atomicStates = readAtomicStates(fileAtomicStateData);

                        std::list<BoundBoundTransitionTuple> boundBoundTransitions = readBoundBoundTransitions(fileBoundBoundTransitionData);
                        std::list<BoundFreeTransitionTuple> boundFreeTransitions = readBoundFreeTransitions(fileBoundFreeTransitionData);
                        std::list<AutonomousTransitionTuple> autonomousTransitions = readAutonomousTransitions(fileAutonomousTransitionData);

                        // check assumptions
                        checkChargeStateList(chargeStateList);
                        checkAtomicStateList(atomicStateList);
                        checkTransitionList<BoundBoundTransitionTuple>(boundBoundTransitions);
                        checkTransitionList<BoundFreeTransitionTuple>(boundFreeTransitions);
                        checkTransitionList<AutonomousTransitionTuple>(autonomousTransitions);

                        // initialize buffers
                        initBuffers();

                        /// @todo fill into property data box
                        /// @todo compute orga data boxes
                        // (synchronize to device)
                    }

                    /// @todo
                    void syncToDevice()
                    {
                        //// charge state data
                        //chargeStateDataBox.syncToDevice();
                        //chargeStateOrgaDataBox.syncToDevice();
                        //
                        //// atomic property data
                        //atomicStateDataBox.syncToDevice();
                        //// atomic orga data
                        //atomicStateStartIndexBlockDataBox_BoundBound.syncToDevice();
                        //atomicStateStartIndexBlockDataBox_BoundFree.syncToDevice();
                        //atomicStateStartIndexBlockDataBox_Autonomous.syncToDevice();
                        //atomicStateNumberOfTransitionsDataBox_BoundBound.syncToDevice();
                        //atomicStateNumberOfTransitionsDataBox_BoundFree.syncToDevice();
                        //atomicStateNumberOfTransitionsDataBox_Autonomous.syncToDevice();
                        //
                        //// transition data
                        //boundBoundTransitionDataBox.syncToDevice();
                        //boundFreeTransitionDataBox.syncToDevice();
                        //autonomousTransitionDataBox.syncToDevice();
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu