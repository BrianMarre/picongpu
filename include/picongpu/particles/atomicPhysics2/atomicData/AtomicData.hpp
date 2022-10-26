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

// tuple definitions
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
// helper stuff for transition tuples
#include "picongpu/particles/atomicPhysics2/atomicData/CompareTransitionTupel.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTupel.hpp"

#include <cstdint>
#include <fstream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

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
 * - transition property data[BoundBoundTransitionDataBuffer, BoundFreeTransitionDataBuffer,
 * AutonomousTransitionDataBuffer]
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
                 * @tparam electronicExcitation is channel active?
                 * @tparam electronicDeexcitation is channel active?
                 * @tparam spontaneousDeexcitation is channel active?
                 * @tparam autonomousIonization is channel active?
                 * @tparam electonicIonization is channel active?
                 * @tparam fieldIonization is channel active?
                 */
                template<
                    typename T_DataBoxType,
                    typename T_Number,
                    typename T_Value,
                    typename T_ConfigNumberDataType,
                    uint32_t T_atomicNumber,
                    uint8_t T_n_max,
                    bool electronicExcitation,
                    bool electronicDeexcitation,
                    bool spontaneousDeexcitation,
                    bool autonomousIonization,
                    bool electonicIonization,
                    bool fieldIonization> /// @todo add photonic channels
                class AtomicData
                {
                public:
                    using DataBoxType = T_DataBoxType;
                    using TypeNumber = T_Number;
                    using TypeValue = T_Value;
                    using Idx = T_ConfigNumberDataType;

                    constexpr uint8_t atomicNumber = T_atomicNumber;
                    constexpr uint8_t n_max = T_n_max;

                    // tuples: S_* for shortened name
                    using S_ChargeStateTuple = ChargeStateTuple<T_Value>;
                    using S_AtomicStateTuple = AtomicStateTuple<TypeValue, Idx>;
                    using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;
                    using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;
                    using S_AutonomousTransitionTuple = AutonomousTransitionTuple<TypeValue, Idx>;

                    // buffers: S_* for shortened name
                    using S_ChargeStateDataBuffer
                        = ChargeStateDataBuffer<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;
                    using S_ChargeStateOrgaDataBuffer
                        = ChargeStateOrgaDataBuffer<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    using S_AtomicStateDataBuffer = AtomicStateDataBuffer<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>;
                    using S_AtomicStateStartIndexBlockDataBuffer_UpDown = AtomicStateStartIndexBlockDataBuffer_UpDown<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_atomicNumber>;
                    using S_AtomicStateStartIndexBlockDataBuffer_Down
                        = AtomicStateStartIndexBlockDataBuffer_Down<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;
                    using S_AtomicStateNumberOfTransitionsDataBuffer_UpDown
                        = AtomicStateNumberOfTransitionsDataBuffer_UpDown<
                            T_DataBoxType,
                            T_Number,
                            T_Value,
                            T_atomicNumber>;
                    using S_AtomicStateNumberOfTransitionsDataBuffer_Down
                        = AtomicStateNumberTransitionsBuffer_UpDown<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    using S_BoundBoundTransitionDataBuffer = BoundBoundTransitionDataBuffer<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>;
                    using S_BoundFreeTransitionDataBuffer = BoundFreeTransitionDataBuffer<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>;
                    using S_AutonomousTransitionDataBuffer = AutonomousTransitionDataBuffer<
                        T_DataBoxType,
                        T_Number,
                        T_Value,
                        T_ConfigNumberDataType,
                        T_atomicNumber>;

                    using S_TranstionTransitionSelectionDataBuffer
                        = TransitionSelectionDataBuffer<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    // dataBoxes: S_* for shortened name
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
                    using S_AtomicStateNumberOfTransitionsDataBox_Down
                        = AtomicStateNumberTransitionsBox_UpDown<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                    using S_BoundBoundTransitionDataBox =
                        BoundBoundTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;
                    using S_BoundFreeTransitionDataBox =
                        BoundFreeTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;
                    using S_AutonomousTransitionDataBox =
                        AutonomousTransitionDataBox<T_DataBoxType, T_Number, T_Value, T_ConfigNumberDataType, T_atomicNumber>;

                    using S_TranstionTransitionSelectionDataBox =
                        TransitionSelectionDataBox<T_DataBoxType, T_Number, T_Value, T_atomicNumber>;

                private:
                    // pointers to storage
                    // charge state data
                    std::unique_ptr<S_ChargeStateDataBuffer> chargeStateDataBuffer;
                    std::unique_ptr<S_ChargeStateOrgaDataBuffer> chargeStateOrgaDataBuffer;

                    // atomic property data
                    std::unique_ptr<S_AtomicStateDataBuffer> atomicStateDataBuffer;
                    // atomic orga data
                    std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown>
                        atomicStateStartIndexBlockDataBuffer_BoundBound;
                    std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown>
                        atomicStateStartIndexBlockDataBuffer_BoundFree;
                    std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_Down>
                        atomicStateStartIndexBlockDataBuffer_Autonomous;
                    std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_UpDown>
                        atomicStateNumberOfTransitionsDataBuffer_BoundBound;
                    std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_UpDown>
                        atomicStateNumberOfTransitionsDataBuffer_BoundFree;
                    std::unique_ptr<S_AtomicStateNumberOfTransitionsDataBuffer_Down>
                        atomicStateNumberOfTransitionsDataBuffer_Autonomous;

                    // transition data, normal
                    std::unique_ptr<S_BoundBoundTransitionDataBuffer> boundBoundTransitionDataBuffer;
                    std::unique_ptr<S_BoundFreeTransitionDataBuffer> boundFreeTransitionDataBuffer;
                    std::unique_ptr<S_AutonomousTransitionDataBuffer> autonomousTransitionDataBuffer;

                    // transition data, inverted
                    std::unique_ptr<S_BoundBoundTransitionDataBuffer> inverseBoundBoundTransitionDataBuffer;
                    std::unique_ptr<S_BoundFreeTransitionDataBuffer> inverseBoundFreeTransitionDataBuffer;
                    std::unique_ptr<S_AutonomousTransitionDataBuffer> inverseAutonomousTransitionDataBuffer;

                    // transition selection data
                    std::unique_ptr<S_TransitionSelectionDataBuffer> transitionSelectionDataBuffer;

                    /// @todo inverse lists transitions

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
                    std::list<S_ChargeStateTuple> readChargeStates(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "charge state data");
                        if( !file )
                            return std::list<S_BoundBoundTransitionTuple>{};

                        std::list<S_ChargeStateTuple> chargeStateList;

                        TypeValue ionizationEnergy;
                        TypeValue screenedCharge;
                        uint8_t chargeState;
                        uint8_t numberChargeStates = 0u;

                        while( file >> chargeState >> ionizationEnergy >> screenedCharge )
                        {
                            if (chargeState == T_atomicNumber)
                                throw std::runtime_error("charge state " + std::to_string(chargeState)
                                    + " should not be included in input file for Z = " + std::to_string(T_atomicNumber));

                            S_ChargeStateTuple item = std::make_tuple(
                                                          chargeState,
                                                          ionizationEnergy, // [eV]
                                                          screenedCharge) // [e]

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
                    std::list<S_AtomicStateTuple> readAtomicStates(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "atomic state data");
                        if( !file )
                            return std::list<S_BoundBoundTransitionTuple>{};

                        std::list<S_AtomicStateTuple> atomicStateList;

                        double stateConfigNumber;
                        TypeValue energyOverGround;

                        while( file >> stateConfigNumber >> energyOverGround )
                        {
                            S_AtomicStateTuple item = std::make_tuple(
                                                          static_cast<Idx>(stateConfigNumber), // unitless
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
                    std::list<S_BoundBoundTransitionTuple> readBoundBoundTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "bound-bound transition data");
                        if( !file )
                            return std::list<S_BoundBoundTransitionTuple>{};

                        std::list<S_BoundBoundTransitionTuple> boundBoundTransitions;

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

                            S_BoundBoundTransitionTuple item = std::make_tuple(
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
                    std::list<S_BoundFreeTransitionTuple> readBoundFreeTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "bound-free transition data");
                        if( !file )
                            return std::list<S_BoundFreeTransitionTuple>{};

                        std::list<S_BoundFreeTransitionTuple> boundFreeTransitions;

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

                            S_BoundFreeTransitionTuple item = std::make_tuple(
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
                    std::list<S_AutonomousTransitionTuple> readAutonomousTransitions(std::string fileName)
                    {
                        std::ifstream file = openFile(fileName, "autonomous transition data");
                        if( !file )
                            return std::list<S_AutonomousTransitionTuple>{};

                        std::list<S_AutonomousTransitionTuple> autonomousTransitions;

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

                            S_AutonomousTransitionTuple item = std::make_tuple(rate, stateLower, stateUpper);

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
                    void checkChargeStateList(std::list<S_ChargeStateTuple>& const chargeStateList)
                    {
                        std::list<S_ChargeStateTuple>::iterator iter = chargeStateList.begin();

                        uint8_t chargeState = 1u;
                        uint8_t lastChargeState;
                        uint8_t currentChargeState;

                        lastChargeState = std::get<0>(*iter);
                        iter++;

                        if (lastChargeState != 0)
                            throw std::runtime_error("atomicPhysics ERROR: charge state 0 not first charge state");

                        for(; iter != chargeStateList.end(); iter++)
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
                    void checkAtomicStateList(std::list<S_AtomicStateTuple>& const atomicStateList)
                    {
                        std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

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
                    HINLINE constexpr std::string getStringTransitionType<S_BoundBoundTransitionTuple>()
                    {
                        return "bound-bound"
                    }

                    template<>
                    HINLINE constexpr std::string getStringTransitionType<S_BoundFreeTransitionTuple>()
                    {
                        return "bound-free"
                    }

                    template<>
                    HINLINE constexpr std::string getStringTransitionType<S_AutonomousTransitionTuple>()
                    {
                        return "autonomous"
                    }

                    //! helper function checking for charge states compatibility with transition type
                    template<typename T_TransitionTuple>
                    HINLINE bool checkForTransitionType(uint8_t lowerChargeState, uint8_t upperChargeState) = 0;

                    template<>
                    HINLINE bool wrongForTransitionType<S_BoundBoundTransitionTuple>(
                        uint8_t lowerChargeState,
                        uint8_t upperChargeState)
                    {
                        if (lowerChargeState == upperChargeState)
                            return false; // correct case
                        return true; // error case
                    }

                    template<>
                    HINLINE bool wrongForTransitionType<S_BoundFreeTransitionTuple>(
                        uint8_t lowerChargeState,
                        uint8_t upperChargeState)
                    {
                        if (lowerChargeState > upperChargeState)
                            return false; // correct case
                        return true; // error case
                    }

                    template<>
                    HINLINE bool wrongForTransitionType<S_AutonomousTransitionTuple>(
                        uint8_t lowerChargeState,
                        uint8_t upperChargeState)
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

                        lastLowerAtomicStateConfigNumber = getLowerStateConfigNumber<Idx, TypeValue>(*iter);
                        lastUpperAtomicStateConfigNumber = getUpperStateConfigNumber<Idx, TypeValue>(*iter);
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
                    chargeStateDataBuffer.reset(new S_ChargeStateDataBuffer<>());
                    chargeStateOrgaDataBuffer.reset(new S_ChargeStateOrgaDataBuffer());

                    // atomic property data
                    atomicStateDataBuffer.reset(new S_AtomicStateDataBuffer(numberAtomicStates));
                    // atomic orga data
                    atomicStateStartIndexBlockDataBuffer_BoundBound.reset(
                        new S_AtomicStateStartIndexBlockDataBuffer_UpDown(numberAtomicStates));
                    atomicStateStartIndexBlockDataBuffer_BoundFree.reset(
                        new S_AtomicStateStartIndexBlockDataBuffer_UpDown(numberAtomicStates));
                    atomicStateStartIndexBlockDataBuffer_Autonomous.reset(
                        new S_AtomicStateStartIndexBlockDataBuffer_Down(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_BoundBound.reset(
                        new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_BoundFree.reset(
                        new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown(numberAtomicStates));
                    atomicStateNumberOfTransitionsDataBuffer_Autonomous.reset(
                        new S_AtomicStateNumberOfTransitionsDataBuffer_Down(numberAtomicStates));

                    // transition data
                    boundBoundTransitionDataBuffer.reset(
                        new S_BoundBoundTransitionDataBuffer(numberBoundBoundTransitions));
                    boundFreeTransitionDataBuffer.reset(
                        new S_BoundFreeTransitionDataBuffer(numberBoundFreeTransitions));
                    autonomousTransitionDataBuffer.reset(
                        new S_AutonomousTransitionDataBuffer(numberAutonomousTransitions));

                    inverseBoundBoundTransitionDataBuffer.reset(
                        new S_BoundBoundTransitionDataBuffer(numberBoundBoundTransitions));
                    inverseBoundFreeTransitionDataBuffer.reset(
                        new S_BoundFreeTransitionDataBuffer(numberBoundFreeTransitions));
                    inverseAutonomousTransitionDataBuffer.reset(
                        new S_AutonomousTransitionDataBuffer(numberAutonomousTransitions));

                    // transition selection data
                    transitionSelectionDataBuffer.reset(new S_TransitionSelectionDataBuffer(numberAtomicStates));
                    }

                    /** fill pure data storage buffer from list
                     *
                     * @tparam T_Tuple type of tuple
                     * @tparam T_DataBox type of dataBox
                     * @tparam T_Buffer type of buffer, automatically deduce able
                     *
                     * @param list correctly ordered list of data tuples to store
                     */
                    template<typename T_Tuple, typename T_DataBox, typename T_Buffer>
                    void storeData(std::list<T_Tuple>& const list, T_Buffer buffer)
                    {
                        std::list<T_Tuple>::iterator iter = list.begin();

                        T_DataBox hostBox = buffer->getHostDataBox();
                        uint32_t i = 0u;

                        for(; iter != list.end(); iter++)
                        {
                            hostBox.store(i, *iter);
                            i++;
                        }
                        buffer->syncToDevice();
                    }

                    void fillChargeStateOrgaData(std::list<S_AtomicStateTuple> atomicStateList)
                    {
                        std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

                        S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox
                            = chargeStateOrgaDataBuffer->getHostDataBox();

                        uint8_t currentChargeState;

                        // read first entry
                        uint8_t lastChargeState
                            = stateRepresentation::ConfigNumber::getIonizationState<Idx, T_n_max, T_atomicNumber>(
                                std::get<0>(*iter));

                        TypeNumber numberStates = 1u;
                        TypeNumber startIndexLastBlock = 0u;
                        iter++;

                        // iterate over rest of the list
                        T_Number i = 1u;
                        for(; iter != list.end(); iter++)
                        {
                            currentChargeState
                                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                                    std::get<0>(*iter));

                            if(currentChargeState != lastChargeState)
                            {
                                // new block
                                chargeStateOrgaDataHostBox.store(lastChargeState, numberStates, startIndexLastBlock);
                                numberStates = 1u;
                                startIndexLastBlock = i;
                                lastChargeState = currentChargeState;
                            }
                            else
                            {
                                // same block
                                numberStates += 1u;
                            }

                            i++
                        }
                        // finish last block
                        chargeStateOrgaDataHostBox.store(lastChargeState, numberStates, startIndexLastBlock);

                        chargeStateOrgaDataBuffer->syncToDevice()
                    }

                    //! @attention assumes that transitionList is sorted by lower state block wise
                    template<
                        typename T_NumberBuffer,
                        typename T_NumberBox,
                        typename T_StartIndexBuffer,
                        typename T_StartIndexBox>
                    void fill_UpTransition_OrgaData(
                        std::list<T_Tuple> transitionList,
                        T_NumberBuffer numberBuffer,
                        T_StartIndexBuffer startIndexBuffer)
                    {
                        std::list<T_Tuple>::iterator iter = transitionList.begin();

                        S_AtomicStateDataBox atomicStateDataBox = atomicStateDataBuffer->getHostDataBox();
                        S_ChargeStateOrgaDataBox chargeStateOrgaDataBox = chargeStateOrgaDataBuffer->getHostDataBox();

                        T_StartIndexBox startIndexHostBox = startIndexBuffer->getHostDataBox();
                        T_NumberBox numberHostBox = numberBuffer->getHostDataBox();

                        // read first entry
                        Idx lastLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);
                        TypeNumber numberInBlock = 1u;
                        TypeNumber lastStartIndex = 0u;
                        iter++;

                        uint8_t lastChargeState;
                        uint32_t lastAtomicStateCollectionIndex;
                        Idx currentLower;

                        // iterate over rest of the list
                        TypeNumber i = 1u;
                        for(; iter != list.end(); iter++)
                        {
                            currentLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);

                            if(currentLower != lastLower)
                            {
                                // new block
                                lastChargeState = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::
                                    getIonizationState(lastLower);

                                lastAtomicStateCollectionIndex = atomicStateDataBox.findStateCollectionIndex(
                                    lastLower,
                                    // completely ionized state can never be lower state of an transition
                                    chargeStateOrgaDataBox.numberAtomicStates(lastChargeState),
                                    chargeStateOrgaDataBox.startIndexBlockAtomicStates(lastChargeState));

                                startIndexHostBox.storeUp(lastAtomicStateCollectionIndex, lastStartIndex);
                                numberHostBox.storeUp(lastAtomicStateCollectionIndex, numberInBlock);

                                numberInBlock = 1u;
                                lastStartIndex = i;
                                lastLower = currentLower;
                            }
                            else
                            {
                                // same block
                                numberInBlock += 1u;
                            }

                            i++;
                        }

                        // finish last block
                        lastChargeState
                            = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                                lastLower);

                        lastAtomicStateCollectionIndex = atomicStateDataBox.findStateCollectionIndex(
                            lastLower,
                            // completely ionized state can never be lower state of an transition
                            chargeStateOrgaDataBox.numberAtomicStates(lastChargeState),
                            chargeStateOrgaDataBox.startIndexBlockAtomicStates(lastChargeState));

                        startIndexHostBox.storeUp(lastAtomicStateCollectionIndex, lastStartIndex);
                        numberHostBox.storeUp(lastAtomicStateCollectionIndex, numberInBlock);

                        startIndexBuffer->syncToDevice();
                        numberBuffer->syncToDevice();
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
                        std::list<S_ChargeStateTuple> chargeStates = readChargeStates(fileChargeData);
                        std::list<S_AtomicStateTuple> atomicStates = readAtomicStates(fileAtomicStateData);

                        std::list<S_BoundBoundTransitionTuple> boundBoundTransitions
                            = readBoundBoundTransitions(fileBoundBoundTransitionData);
                        std::list<S_BoundFreeTransitionTuple> boundFreeTransitions
                            = readBoundFreeTransitions(fileBoundFreeTransitionData);
                        std::list<S_AutonomousTransitionTuple> autonomousTransitions
                            = readAutonomousTransitions(fileAutonomousTransitionData);

                        // check assumptions
                        checkChargeStateList(chargeStateList);
                        checkAtomicStateList(atomicStateList);
                        checkTransitionList<S_BoundBoundTransitionTuple>(boundBoundTransitions);
                        checkTransitionList<S_BoundFreeTransitionTuple>(boundFreeTransitions);
                        checkTransitionList<S_AutonomousTransitionTuple>(autonomousTransitions);

                        // initialize buffers
                        initBuffers();

                        // fill data buffers
                        storeData<S_ChargeStateTuple, S_ChargeStateDataBox>(chargeStates, chargeStateDataBuffer);
                        storeData<S_AtomicStateTuple, S_AtomicStateDataBox>(atomicStatesStates, atomicStateDataBuffer);

                        storeData<S_BoundBoundTransitionTuple, S_BoundBoundTransitionDataBox>(
                            boundBoundTransitions,
                            boundBoundTransitionDataBuffer);
                        storeData<S_BoundFreeTransitionTuple, S_BoundFreeTransitionDataBox>(
                            boundFreeTransitions,
                            boundFreeTransitionDataBuffer);
                        storeData<S_AutonomousTransitionTuple, S_AutonomousTransitionDataBox>(
                            autonomousTransitions,
                            autonomousTransitionDataBuffer);

                        // fill orga buffers, for up direction
                        fillChargeStateOrgaData(atomicStates);
                        fill_UpTransition_OrgaData<
                            S_AtomicStateNumberOfTransitionsDataBox_UpDown,
                            S_AtomicStateStartIndexBlockDataBox_UpDown>(
                            boundBoundTransitions,
                            atomicStateNumberOfTransitionsDataBuffer_BoundBound,
                            atomicStateStartIndexBlockDataBuffer_BoundBound)
                            fill_UpTransition_OrgaData<
                                S_AtomicStateNumberOfTransitionsDataBox_UpDown,
                                S_AtomicStateStartIndexBlockDataBox_UpDown>(
                                boundBoundTransitions,
                                atomicStateNumberOfTransitionsDataBuffer_BoundFree,
                                atomicStateStartIndexBlockDataBuffer_BoundFree)

                            // resort transition lists                                          // sort by upper state
                            boundBoundTransitions.sort(CompareTransitionTupel<TypeNumber, Idx, false>());
                        boundFreeTransitions.sort(CompareTransitionTupel<TypeNumber, Idx, false>());
                        autonomousTransitions.sort(CompareTransitionTupel<TypeNumber, Idx, false>());

                        //      store also in inverse order
                        storeData<S_BoundBoundTransitionTuple, S_BoundBoundTransitionDataBox>(
                            boundBoundTransitions,
                            boundBoundTransitionDataBuffer);
                        storeData<S_BoundFreeTransitionTuple, S_BoundFreeTransitionDataBox>(
                            boundFreeTransitions,
                            boundFreeTransitionDataBuffer);
                        storeData<S_AutonomousTransitionTuple, S_AutonomousTransitionDataBox>(
                            autonomousTransitions,
                            autonomousTransitionDataBuffer);

                        /// @todo fill orga buffers for inverse


                        // fill transitioSelectionBuffer

                        /// @todo call for each species

                        this->syncToDevice()
                    }

                    void syncToDevice()
                    {
                        // charge state data
                        chargeStateDataBuffer->syncToDevice();
                        chargeStateOrgaDataBuffer->syncToDevice();

                        // atomic property data
                        atomicStateDataBuffer->syncToDevice();
                        // atomic orga data
                        atomicStateStartIndexBlockDataBuffer_BoundBound->syncToDevice();
                        atomicStateStartIndexBlockDataBuffer_BoundFree->syncToDevice();
                        atomicStateStartIndexBlockDataBuffer_Autonomous->syncToDevice();
                        atomicStateNumberOfTransitionsDataBuffer_BoundBound->syncToDevice();
                        atomicStateNumberOfTransitionsDataBuffer_BoundFree->syncToDevice();
                        atomicStateNumberOfTransitionsDataBuffer_Autonomous->syncToDevice();

                        // transition data
                        boundBoundTransitionDataBuffer->syncToDevice();
                        boundFreeTransitionDataBuffer->syncToDevice();
                        autonomousTransitionDataBuffer->syncToDevice();
                    }

                };
            } // namespace atomicData
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
