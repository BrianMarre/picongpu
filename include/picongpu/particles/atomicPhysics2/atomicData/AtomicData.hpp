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

#include "picongpu/param/atomicPhysics2_Debug.param"

#include <pmacc/dataManagement/ISimulationData.hpp>

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
#include "picongpu/particles/atomicPhysics2/atomicData/TransitionSelectionData.hpp"

// conversion of configNumber to charge state for checking
#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"

// tuple definitions
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicTuples.def"
// helper stuff for transition tuples
#include "picongpu/particles/atomicPhysics2/atomicData/CompareTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTuple.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

// debug only
//#include <typeinfo>
//#include <cxxabi.h>

/** @file gathers atomic data storage implementations and implements filling them on runtime
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each
 * atomicPhysics ion species.
 * These model's parameters are provided by the user as a .txt file of specified format
 * (see documentation) at runtime.
 *
 *  PIConGPU itself only includes charge state data, for ADK-, Thomas-Fermi- and BSI-ionization.
 *  All other atomic state data is kept separate from PIConGPU itself, due to license requirements.
 *
 * This file is read at the start of the simulation and stored distributed over the following:
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
 *  are always possible without it, but are possibly non performant)
 *
 * For each of data subsets exists a dataBox class, container class holding the actual data in pmacc::dataBox'es,
 *  and a dataBuffer class, a container class holding pmacc::buffers in turn holding the pmacc::dataBox'es.
 *
 * Each data Buffer will create on demand a host or device side dataBox class object which
 *  in turn gives access to the data.
 */

namespace picongpu::particles::atomicPhysics2::atomicData
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
        typename T_Number,
        typename T_Value,
        typename T_ConfigNumberDataType,
        uint32_t T_atomicNumber,
        uint8_t T_n_max,
        bool T_electronicExcitation,
        bool T_electronicDeexcitation,
        bool T_spontaneousDeexcitation,
        bool T_electronicIonization,
        bool T_autonomousIonization,
        bool T_fieldIonization> /// @todo add photonic channels, Brian Marre, 2022
    class AtomicData : public pmacc::ISimulationData
    {
    public:
        using TypeNumber = T_Number;
        using TypeValue = T_Value;
        using Idx = T_ConfigNumberDataType;

        static constexpr uint8_t atomicNumber = T_atomicNumber;
        static constexpr uint8_t n_max = T_n_max;

        // tuples: S_* for shortened name
        using S_ChargeStateTuple = ChargeStateTuple<TypeValue>;
        using S_AtomicStateTuple = AtomicStateTuple<TypeValue, Idx>;
        using S_BoundBoundTransitionTuple = BoundBoundTransitionTuple<TypeValue, Idx>;
        using S_BoundFreeTransitionTuple = BoundFreeTransitionTuple<TypeValue, Idx>;
        using S_AutonomousTransitionTuple = AutonomousTransitionTuple<TypeValue, Idx>;

        // dataBuffers: S_* for shortened name
        using S_ChargeStateDataBuffer = ChargeStateDataBuffer<TypeNumber, TypeValue, T_atomicNumber>;
        using S_ChargeStateOrgaDataBuffer = ChargeStateOrgaDataBuffer<TypeNumber, TypeValue, T_atomicNumber>;

        using S_AtomicStateDataBuffer
            = AtomicStateDataBuffer<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_AtomicStateStartIndexBlockDataBuffer_UpDown
            = AtomicStateStartIndexBlockDataBuffer_UpDown<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateStartIndexBlockDataBuffer_Down
            = AtomicStateStartIndexBlockDataBuffer_Down<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateNumberOfTransitionsDataBuffer_UpDown
            = AtomicStateNumberOfTransitionsDataBuffer_UpDown<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateNumberOfTransitionsDataBuffer_Down
            = AtomicStateNumberOfTransitionsDataBuffer_Down<TypeNumber, TypeValue, T_atomicNumber>;

        using S_BoundBoundTransitionDataBuffer
            = BoundBoundTransitionDataBuffer<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_BoundFreeTransitionDataBuffer
            = BoundFreeTransitionDataBuffer<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_AutonomousTransitionDataBuffer
            = AutonomousTransitionDataBuffer<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;

        using S_TransitionSelectionDataBuffer = TransitionSelectionDataBuffer<TypeNumber, TypeValue, T_atomicNumber>;

        // dataBoxes: S_* for shortened name
        using S_ChargeStateDataBox = ChargeStateDataBox<TypeNumber, TypeValue, T_atomicNumber>;
        using S_ChargeStateOrgaDataBox = ChargeStateOrgaDataBox<TypeNumber, TypeValue, T_atomicNumber>;

        using S_AtomicStateDataBox = AtomicStateDataBox<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_AtomicStateStartIndexBlockDataBox_UpDown
            = AtomicStateStartIndexBlockDataBox_UpDown<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateStartIndexBlockDataBox_Down
            = AtomicStateStartIndexBlockDataBox_Down<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateNumberOfTransitionsDataBox_UpDown
            = AtomicStateNumberOfTransitionsDataBox_UpDown<TypeNumber, TypeValue, T_atomicNumber>;
        using S_AtomicStateNumberOfTransitionsDataBox_Down
            = AtomicStateNumberOfTransitionsDataBox_Down<TypeNumber, TypeValue, T_atomicNumber>;

        using S_BoundBoundTransitionDataBox
            = BoundBoundTransitionDataBox<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_BoundFreeTransitionDataBox
            = BoundFreeTransitionDataBox<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;
        using S_AutonomousTransitionDataBox
            = AutonomousTransitionDataBox<TypeNumber, TypeValue, T_ConfigNumberDataType, T_atomicNumber>;

        using S_TransitionSelectionDataBox = TransitionSelectionDataBox<TypeNumber, TypeValue, T_atomicNumber>;

    private:
        // pointers to storage
        // charge state data
        std::unique_ptr<S_ChargeStateDataBuffer> chargeStateDataBuffer;
        std::unique_ptr<S_ChargeStateOrgaDataBuffer> chargeStateOrgaDataBuffer;

        // atomic property data
        std::unique_ptr<S_AtomicStateDataBuffer> atomicStateDataBuffer;
        // atomic orga data
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown> atomicStateStartIndexBlockDataBuffer_BoundBound;
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_UpDown> atomicStateStartIndexBlockDataBuffer_BoundFree;
        std::unique_ptr<S_AtomicStateStartIndexBlockDataBuffer_Down> atomicStateStartIndexBlockDataBuffer_Autonomous;
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

        const std::string m_speciesName;

        //! open file
        HINLINE static std::ifstream openFile(std::string fileName, std::string fileContent)
        {
            std::ifstream file(fileName);

            // check for success
            if(!file)
            {
                throw std::runtime_error("atomicPhysics ERROR: could not open " + fileContent + ": " + fileName);
            }

            return file;
        }

        /** read charge state data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - charge state data is sorted by ascending charge
         *   - the completely ionized state is left out
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_ChargeStateTuple> readChargeStates(std::string fileName)
        {
            std::ifstream file = openFile(fileName, "charge state data");
            if(!file)
                return std::list<S_ChargeStateTuple>{};

            std::list<S_ChargeStateTuple> chargeStateList;

            TypeValue ionizationEnergy;
            TypeValue screenedCharge;
            uint32_t chargeState;
            uint8_t numberChargeStates = 0u;

            while(file >> chargeState >> ionizationEnergy >> screenedCharge)
            {
                if(chargeState == static_cast<uint32_t>(T_atomicNumber))
                    throw std::runtime_error(
                        "charge state " + std::to_string(chargeState)
                        + " should not be included in input file for Z = " + std::to_string(T_atomicNumber));

                S_ChargeStateTuple item = std::make_tuple(
                    chargeState,
                    ionizationEnergy, // [eV]
                    screenedCharge); // [e]

                chargeStateList.push_back(item);

                numberChargeStates++;
            }

            if(numberChargeStates > T_atomicNumber)
                throw std::runtime_error(
                    "atomicPhysics ERROR: too many charge states, num > Z: " + std::to_string(T_atomicNumber));

            return chargeStateList;
        }

        /** read atomic state data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - atomic state data is sorted block wise by charge state and secondary by ascending
         * configNumber
         *   - the completely ionized state is left out
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_AtomicStateTuple> readAtomicStates(std::string fileName)
        {
            std::ifstream file = openFile(fileName, "atomic state data");
            if(!file)
                return std::list<S_AtomicStateTuple>{};

            std::list<S_AtomicStateTuple> atomicStateList;

            double stateConfigNumber;
            TypeValue energyOverGround;

            while(file >> stateConfigNumber >> energyOverGround)
            {
                S_AtomicStateTuple item = std::make_tuple(
                    static_cast<Idx>(stateConfigNumber), // unitless
                    energyOverGround); // [eV]

                atomicStateList.push_back(item);

                m_numberAtomicStates++;
            }

            return atomicStateList;
        }

        /** read bound-bound transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_BoundBoundTransitionTuple> readBoundBoundTransitions(std::string fileName)
        {
            std::ifstream file = openFile(fileName, "bound-bound transition data");
            if(!file)
                return std::list<S_BoundBoundTransitionTuple>{};

            std::list<S_BoundBoundTransitionTuple> boundBoundTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;
            TypeValue collisionalOscillatorStrength;
            TypeValue absorptionOscillatorStrength;

            // gauntCoeficients
            TypeValue cxin1;
            TypeValue cxin2;
            TypeValue cxin3;
            TypeValue cxin4;
            TypeValue cxin5;

            while(file >> idxLower >> idxUpper >> collisionalOscillatorStrength >> absorptionOscillatorStrength
                  >> cxin1 >> cxin2 >> cxin3 >> cxin4 >> cxin5)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                S_BoundBoundTransitionTuple item = std::make_tuple(
                    collisionalOscillatorStrength,
                    absorptionOscillatorStrength,
                    cxin1,
                    cxin2,
                    cxin3,
                    cxin4,
                    cxin5,
                    stateLower,
                    stateUpper);

                boundBoundTransitions.push_back(item);
                m_numberBoundBoundTransitions++;
            }

            return boundBoundTransitions;
        }

        /** read bound-free transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_BoundFreeTransitionTuple> readBoundFreeTransitions(std::string fileName)
        {
            std::ifstream file = openFile(fileName, "bound-free transition data");
            if(!file)
                return std::list<S_BoundFreeTransitionTuple>{};

            std::list<S_BoundFreeTransitionTuple> boundFreeTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;

            // gauntCoeficients
            TypeValue cxin1;
            TypeValue cxin2;
            TypeValue cxin3;
            TypeValue cxin4;
            TypeValue cxin5;
            TypeValue cxin6;
            TypeValue cxin7;
            TypeValue cxin8;

            while(file >> idxLower >> idxUpper >> cxin1 >> cxin2 >> cxin3 >> cxin4 >> cxin5 >> cxin6 >> cxin7 >> cxin8)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                S_BoundFreeTransitionTuple item
                    = std::make_tuple(cxin1, cxin2, cxin3, cxin4, cxin5, cxin6, cxin7, cxin8, stateLower, stateUpper);

                boundFreeTransitions.push_back(item);
                m_numberBoundFreeTransitions++;
            }
            return boundFreeTransitions;
        }

        /** read autonomous transitions data file
         *
         * @attention assumes input to already fulfills all ordering and unit assumptions
         *   - transition data is sorted block wise by lower atomic state and secondary by ascending by upper state
         * configNumber
         *
         * @return returns empty list if file not found/accessible
         */
        ALPAKA_FN_HOST std::list<S_AutonomousTransitionTuple> readAutonomousTransitions(std::string fileName)
        {
            std::ifstream file = openFile(fileName, "autonomous transition data");
            if(!file)
                return std::list<S_AutonomousTransitionTuple>{};

            std::list<S_AutonomousTransitionTuple> autonomousTransitions;

            uint64_t idxLower;
            uint64_t idxUpper;

            // unit: 1/s
            TypeValue rate;

            while(file >> idxLower >> idxUpper >> rate)
            {
                Idx stateLower = static_cast<Idx>(idxLower);
                Idx stateUpper = static_cast<Idx>(idxUpper);

                // protection against circle transitions
                if(stateLower == stateUpper)
                {
                    std::cout << "atomicPhysics ERROR: circular transitions are not supported,"
                                 "treat steps separately"
                              << std::endl;
                    continue;
                }

                const S_AutonomousTransitionTuple item = std::make_tuple(rate, stateLower, stateUpper);

                autonomousTransitions.push_back(item);
                m_numberAutonomousTransitions++;
            }
            return autonomousTransitions;
        }

        /** check charge state list
         *
         * @throws runtime error if duplicate charge state, missing charge state,
         *  order broken, completely ionized state included or unphysical charge state
         */
        ALPAKA_FN_HOST void checkChargeStateList(std::list<S_ChargeStateTuple>& chargeStateList)
        {
            typename std::list<S_ChargeStateTuple>::iterator iter = chargeStateList.begin();

            uint8_t chargeState = 1u;
            uint32_t lastChargeState;
            uint8_t currentChargeState;

            if(iter == chargeStateList.end())
                throw std::runtime_error("atomicPhysics ERROR: empty charge state list");
            lastChargeState = std::get<0>(*iter);

            iter++;

            if(lastChargeState != 0u)
                throw std::runtime_error("atomicPhysics ERROR: charge state 0 not first charge state");

            for(; iter != chargeStateList.end(); iter++)
            {
                currentChargeState = std::get<0>(*iter);

                // duplicate atomic state
                if(currentChargeState == lastChargeState)
                    throw std::runtime_error("atomicPhysics ERROR: duplicate charge state");

                // ordering
                if(not(currentChargeState > lastChargeState))
                    throw std::runtime_error("atomicPhysics ERROR: charge state ordering wrong");

                // missing charge state
                if(not(currentChargeState == chargeState))
                    throw std::runtime_error("atomicPhysics ERROR: charge state missing");

                // completely ionized state
                if(chargeState == T_atomicNumber)
                    throw std::runtime_error("atomicPhysics ERROR: completely ionized charge state found");

                // unphysical state
                if(chargeState == T_atomicNumber)
                    throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                chargeState++;
                lastChargeState = currentChargeState;
            }
        }

        /** check atomic state list
         *
         * @throws runtime error if duplicate atomic state, primary order broken,
         *  secondary order broken, or unphysical charge state found
         */
        ALPAKA_FN_HOST void checkAtomicStateList(std::list<S_AtomicStateTuple>& atomicStateList)
        {
            typename std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

            Idx currentAtomicStateConfigNumber;
            uint8_t currentChargeState;

            // empty transition list
            if(iter == atomicStateList.end())
                return;

            Idx lastAtomicStateConfigNumber = std::get<0>(*iter);
            uint8_t lastChargeState
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                    lastAtomicStateConfigNumber);

            iter++;

            for(; iter != atomicStateList.end(); iter++)
            {
                currentAtomicStateConfigNumber = std::get<0>(*iter);
                currentChargeState
                    = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                        currentAtomicStateConfigNumber);

                // duplicate atomic state
                if(currentAtomicStateConfigNumber == lastAtomicStateConfigNumber)
                    throw std::runtime_error("atomicPhysics ERROR: duplicate atomic state");
                // later duplicate will break ordering

                // primary/secondary order
                if(currentChargeState == lastChargeState)
                    // same block
                    if(currentAtomicStateConfigNumber < lastAtomicStateConfigNumber)
                        throw std::runtime_error("atomicPhysics ERROR: wrong secondary ordering of atomic states");
                    else
                        // next block
                        if(currentChargeState > lastChargeState)
                        throw std::runtime_error("atomicPhysics ERROR: wrong primary ordering of atomic state");

                // completely ionized atomic state is allowed as upper state

                // unphysical atomic state
                if(currentChargeState > T_atomicNumber)
                    throw std::runtime_error("atomicPhysics ERROR: unphysical charge state found");

                lastChargeState = currentChargeState;
                lastAtomicStateConfigNumber = currentAtomicStateConfigNumber;
            }
        }

        /** check transition list
         *
         * @throws runtime error if primary order broken,
         *  secondary order broken, transition from/to unphysical charge state found,
         *  wrong transition type for lower/upper charge state pair
         */
        template<typename T_TransitionTuple>
        ALPAKA_FN_HOST void checkTransitionList(std::list<T_TransitionTuple>& transitionList)
        {
            std::string transitionType = getStringTransitionType<T_TransitionTuple>();

            typename std::list<T_TransitionTuple>::iterator iter = transitionList.begin();

            Idx currentLowerAtomicStateConfigNumber;
            Idx currentUpperAtomicStateConfigNumber;
            uint8_t currentLowerChargeState;
            uint8_t currentUpperChargeState;

            // empty transition list
            if(iter == transitionList.end())
                return;

            // read first entry
            Idx lastLowerAtomicStateConfigNumber = getLowerStateConfigNumber<Idx, TypeValue>(*iter);
            Idx lastUpperAtomicStateConfigNumber = getUpperStateConfigNumber<Idx, TypeValue>(*iter);
            uint8_t lastLowerChargeState
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                    lastLowerAtomicStateConfigNumber);
            uint8_t lastUpperChargeState
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                    lastUpperAtomicStateConfigNumber);
            iter++;

            // transitionType <-> charge state pair
            if(wrongForTransitionType<T_TransitionTuple>(lastLowerChargeState, lastUpperChargeState))
                throw std::runtime_error(
                    "atomicPhysics ERROR: wrong last upper-/lower charge state pair for transition type "
                    + transitionType);

            // unphysical lower charge state
            if(lastLowerChargeState > T_atomicNumber)
                throw std::runtime_error("atomicPhysics ERROR: unphysical lower charge State");

            // unphysical upper charge state
            if(lastUpperChargeState > T_atomicNumber)
                throw std::runtime_error("atomicPhysics ERROR: unphysical upper charge State");

            for(; iter != transitionList.end(); iter++)
            {
                currentLowerAtomicStateConfigNumber = getLowerStateConfigNumber(*iter);
                currentUpperAtomicStateConfigNumber = getUpperStateConfigNumber(*iter);
                currentLowerChargeState
                    = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                        currentLowerAtomicStateConfigNumber);
                currentUpperChargeState
                    = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                        currentUpperAtomicStateConfigNumber);

                // primary/secondary order
                if(lastLowerAtomicStateConfigNumber == currentLowerAtomicStateConfigNumber)
                    // same block
                    if(currentUpperAtomicStateConfigNumber < lastUpperAtomicStateConfigNumber)
                        throw std::runtime_error(
                            "atomicPhysics ERROR: wrong secondary ordering of " + transitionType + " transitions");
                    else
                        // next block
                        if(lastLowerAtomicStateConfigNumber > currentLowerAtomicStateConfigNumber)
                        throw std::runtime_error(
                            "atomicPhysics ERROR: wrong primary ordering of " + transitionType + " transition");

                // transitionType <-> charge state pair
                if(wrongForTransitionType<T_TransitionTuple>(currentLowerChargeState, currentUpperChargeState))
                    throw std::runtime_error(
                        "atomicPhysics ERROR: wrong current upper-/lower charge state pair for transition "
                        "type "
                        + transitionType);

                // unphysical lower charge state
                if(currentLowerChargeState > T_atomicNumber)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: unphysical lower charge State in " + transitionType + " transitions");

                // unphysical upper charge state
                if(currentUpperChargeState > T_atomicNumber)
                    throw std::runtime_error(
                        "atomicPhysics ERROR: unphysical upper charge State in " + transitionType + " transitions");

                lastLowerChargeState = currentLowerChargeState;
                lastUpperChargeState = currentUpperChargeState;
                lastLowerAtomicStateConfigNumber = currentLowerAtomicStateConfigNumber;
                lastUpperAtomicStateConfigNumber = currentUpperAtomicStateConfigNumber;
            }
        }

        //! init buffers, @attention all readMethods must have been executed exactly once before!
        ALPAKA_FN_HOST void initBuffers()
        {
            // charge state data
            chargeStateDataBuffer.reset(new S_ChargeStateDataBuffer());
            chargeStateOrgaDataBuffer.reset(new S_ChargeStateOrgaDataBuffer());

            // atomic property data
            atomicStateDataBuffer.reset(new S_AtomicStateDataBuffer(m_numberAtomicStates));
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_UpDown(m_numberAtomicStates));
            atomicStateStartIndexBlockDataBuffer_BoundFree.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_UpDown(m_numberAtomicStates));
            atomicStateStartIndexBlockDataBuffer_Autonomous.reset(
                new S_AtomicStateStartIndexBlockDataBuffer_Down(m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_BoundBound.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown(m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_BoundFree.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_UpDown(m_numberAtomicStates));
            atomicStateNumberOfTransitionsDataBuffer_Autonomous.reset(
                new S_AtomicStateNumberOfTransitionsDataBuffer_Down(m_numberAtomicStates));

            // transition data
            boundBoundTransitionDataBuffer.reset(new S_BoundBoundTransitionDataBuffer(m_numberBoundBoundTransitions));
            boundFreeTransitionDataBuffer.reset(new S_BoundFreeTransitionDataBuffer(m_numberBoundFreeTransitions));
            autonomousTransitionDataBuffer.reset(new S_AutonomousTransitionDataBuffer(m_numberAutonomousTransitions));

            inverseBoundBoundTransitionDataBuffer.reset(
                new S_BoundBoundTransitionDataBuffer(m_numberBoundBoundTransitions));
            inverseBoundFreeTransitionDataBuffer.reset(
                new S_BoundFreeTransitionDataBuffer(m_numberBoundFreeTransitions));
            inverseAutonomousTransitionDataBuffer.reset(
                new S_AutonomousTransitionDataBuffer(m_numberAutonomousTransitions));

            // transition selection data
            transitionSelectionDataBuffer.reset(new S_TransitionSelectionDataBuffer(m_numberAtomicStates));
        }

        /** fill pure data storage buffers from list
         *
         * @tparam T_Tuple type of tuple
         * @tparam T_DataBox type of dataBox
         * @tparam T_Buffer type of buffer, automatically deduce able
         *
         * @param list correctly ordered list of data tuples to store
         * @attention does not sync to device, must be synced externally explicitly
         */
        template<typename T_Tuple, typename T_DataBox>
        ALPAKA_FN_HOST void storeData(std::list<T_Tuple>& list, T_DataBox hostBox)
        {
            typename std::list<T_Tuple>::iterator iter = list.begin();

            uint32_t i = 0u;

            for(; iter != list.end(); iter++)
            {
                hostBox.store(i, *iter);
                i++;
            }
        }

        /** fill the charge orga data buffer
         *
         * @attention assumes that the atomic states are sorted block wise by charge state
         *
         * @param atomicStateList list of all atomicStates, sorted block wise by charge state
         */
        ALPAKA_FN_HOST void fillChargeStateOrgaData(std::list<S_AtomicStateTuple> atomicStateList)
        {
            typename std::list<S_AtomicStateTuple>::iterator iter = atomicStateList.begin();

            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            uint8_t currentChargeState;

            // empty atomic state list
            if(iter == atomicStateList.end())
                return;

            // read first entry as first last entry
            uint8_t lastChargeState
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                    std::get<0>(*iter));

            TypeNumber numberStates = 1u;
            TypeNumber startIndexLastBlock = 0u;
            iter++;

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != atomicStateList.end(); iter++)
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

                i++;
            }
            // finish last block
            chargeStateOrgaDataHostBox.store(lastChargeState, numberStates, startIndexLastBlock);

            chargeStateOrgaDataBuffer->hostToDevice();
        }

        /** fill the upward atomic state orga buffers for a transition groups
         *
         * i.e. number of transitions and start index, up( and down) for each atomic state
         *  for a transition group(bound-bound, bound-free)
         *
         * @attention assumes that transitionList is sorted by lower state block wise
         * @attention changes have to synced to device separately
         */
        template<typename T_Tuple>
        ALPAKA_FN_HOST void fill_UpTransition_OrgaData(
            std::list<T_Tuple> transitionList,
            S_AtomicStateNumberOfTransitionsDataBox_UpDown numberHostBox,
            S_AtomicStateStartIndexBlockDataBox_UpDown startIndexHostBox)
        {
            typename std::list<T_Tuple>::iterator iter = transitionList.begin();

            // quick lockup data
            S_AtomicStateDataBox atomicStateDataHostBox = atomicStateDataBuffer->getHostDataBox();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            uint8_t lastChargeState;
            uint32_t lastAtomicStateCollectionIndex;
            Idx currentLower; // transitions up from a state have the state as lower state

            // empty transition list
            if(iter == transitionList.end())
                return;

            // read first entry
            Idx lastLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);
            TypeNumber numberInBlock = 1u;
            TypeNumber lastStartIndex = 0u;
            iter++;

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != transitionList.end(); iter++)
            {
                currentLower = getLowerStateConfigNumber<Idx, TypeValue>(*iter);

                if(currentLower != lastLower)
                {
                    // new block
                    lastChargeState
                        = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                            lastLower); // will always be < T_atomicNumber, since
                                        // q=T_atomicNumber may never be a lower state

                    lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                        lastLower,
                        // completely ionized state can never be lower state of an transition
                        chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                        chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

                    if(lastAtomicStateCollectionIndex >= atomicStateDataHostBox.getNumberAtomicStatesTotal())
                        throw std::runtime_error("atomicPhysics ERROR: atomic state not found");

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
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(lastLower);

            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastLower,
                // completely ionized state can never be lower state of an transition
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            startIndexHostBox.storeUp(lastAtomicStateCollectionIndex, lastStartIndex);
            numberHostBox.storeUp(lastAtomicStateCollectionIndex, numberInBlock);
        }

        /** fill the downward atomic state orga buffers for a transition groups
         *
         * i.e. number of transitions and start index, up( and down) of each atomic state
         *  for a transition group(bound-bound, bound-free, autonomous)
         *
         * @attention assumes that transitionList is sorted by upper state block wise
         */
        template<typename T_Tuple, typename T_NumberHostBox, typename T_StartIndexHostBox>
        ALPAKA_FN_HOST void fill_DownTransition_OrgaData(
            std::list<T_Tuple> transitionList,
            T_NumberHostBox numberHostBox,
            T_StartIndexHostBox startIndexHostBox)
        {
            typename std::list<T_Tuple>::iterator iter = transitionList.begin();

            S_AtomicStateDataBox atomicStateDataHostBox = atomicStateDataBuffer->getHostDataBox();
            S_ChargeStateOrgaDataBox chargeStateOrgaDataHostBox = chargeStateOrgaDataBuffer->getHostDataBox();

            // empty transition list
            if(iter == transitionList.end())
                return;

            // read first entry
            Idx lastUpper = getUpperStateConfigNumber<Idx, TypeValue>(
                *iter); // transitions down from a state have the state as upper
            TypeNumber numberInBlock = 1u;
            TypeNumber lastStartIndex = 0u;
            iter++;

            uint8_t lastChargeState;
            uint32_t lastAtomicStateCollectionIndex;
            Idx currentUpper;

            // iterate over rest of the list
            TypeNumber i = 1u;
            for(; iter != transitionList.end(); iter++)
            {
                currentUpper = getUpperStateConfigNumber<Idx, TypeValue>(*iter);

                if(currentUpper != lastUpper)
                {
                    // new block
                    lastChargeState
                        = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(
                            lastUpper);

                    lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                        lastUpper,
                        // completely ionized state can never be lower state of an transition
                        chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                        chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

                    if(lastAtomicStateCollectionIndex >= atomicStateDataHostBox.getNumberAtomicStatesTotal())
                        throw std::runtime_error("atomicPhysics ERROR: atomic state not found");

                    startIndexHostBox.storeDown(lastAtomicStateCollectionIndex, lastStartIndex);
                    numberHostBox.storeDown(lastAtomicStateCollectionIndex, numberInBlock);

                    numberInBlock = 1u;
                    lastStartIndex = i;
                    lastUpper = currentUpper;
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
                = stateRepresentation::ConfigNumber<Idx, T_n_max, T_atomicNumber>::getIonizationState(lastUpper);

            // debug only
            std::cout << "charge State: " << static_cast<uint16_t>(lastChargeState) << std::endl;
            std::cout << "CN: " << lastUpper << std::endl;

            lastAtomicStateCollectionIndex = atomicStateDataHostBox.findStateCollectionIndex(
                lastUpper,
                // completely ionized state can never be lower state of an transition
                chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState),
                chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState));

            // debug only
            std::cout << "startIndex: " << chargeStateOrgaDataHostBox.startIndexBlockAtomicStates(lastChargeState)
                      << ", numberStates: " << chargeStateOrgaDataHostBox.numberAtomicStates(lastChargeState)
                      << std::endl;
            std::cout << "collectionIndex: " << lastAtomicStateCollectionIndex << std::endl;
            std::cout << "numberInBlock: " << numberInBlock << std::endl;

            startIndexHostBox.storeDown(lastAtomicStateCollectionIndex, lastStartIndex);
            numberHostBox.storeDown(lastAtomicStateCollectionIndex, numberInBlock);
        }

        /** fill the transition selectionBuffer
         *
         * @tparam electronicExcitation is channel active?
         * @tparam electronicDeexcitation is channel active?
         * @tparam spontaneousDeexcitation is channel active?
         * @tparam autonomousIonization is channel active?
         * @tparam electonicIonization is channel active?
         * @tparam fieldIonization is channel active?
         */
        template<
            bool electronicExcitation,
            bool electronicDeexcitation,
            bool spontaneousDeexcitation,
            bool electronicIonization,
            bool autonomousIonization,
            bool fieldIonization>
        ALPAKA_FN_HOST void fillTransitionSelectionDataBufferAndSetOffsets()
        {
            S_TransitionSelectionDataBox transitionSelectionDataHostBox
                = transitionSelectionDataBuffer->getHostDataBox();

            S_AtomicStateNumberOfTransitionsDataBox_UpDown hostBoxNumberBoundBound
                = atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox();
            S_AtomicStateNumberOfTransitionsDataBox_UpDown hostBoxNumberBoundFree
                = atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox();
            S_AtomicStateNumberOfTransitionsDataBox_Down hostBoxNumberAutonomous
                = atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox();

            TypeNumber numberPhysicalTransitionsTotal;

            for(uint32_t i = 0u; i < m_numberAtomicStates; i++)
            {
                numberPhysicalTransitionsTotal = 0u;

                // bound-bound transitions
                hostBoxNumberBoundBound.storeOffset(i, 0u);
                if constexpr(electronicDeexcitation)
                    numberPhysicalTransitionsTotal += hostBoxNumberBoundBound.numberOfTransitionsDown(i);
                if constexpr(spontaneousDeexcitation)
                    numberPhysicalTransitionsTotal += hostBoxNumberBoundBound.numberOfTransitionsDown(i);

                if constexpr(electronicExcitation)
                    numberPhysicalTransitionsTotal += hostBoxNumberBoundBound.numberOfTransitionsUp(i);

                // bound-free transitions
                hostBoxNumberBoundFree.storeOffset(i, numberPhysicalTransitionsTotal);
                if constexpr(electronicIonization)
                    numberPhysicalTransitionsTotal += hostBoxNumberBoundFree.numberOfTransitionsUp(i);
                if constexpr(fieldIonization)
                    numberPhysicalTransitionsTotal += hostBoxNumberBoundFree.numberOfTransitionsUp(i);
                /// recombination missing, @todo implement recombination, Brian Marre

                // autonomousTransitions
                hostBoxNumberAutonomous.storeOffset(i, numberPhysicalTransitionsTotal);
                if constexpr(autonomousIonization)
                    numberPhysicalTransitionsTotal += hostBoxNumberAutonomous.numberOfTransitions(i);

                transitionSelectionDataHostBox.store(i, numberPhysicalTransitionsTotal);
            }

            // sync offsets
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();

            // sync transition selection data
            transitionSelectionDataBuffer->hostToDevice();
        }

    public:
        /** read input files and create/fill data boxes
         *
         * @param fileChargeData path to file containing charge state data
         * @param fileAtomicStateData path to file containing atomic state data
         * @param fileTransitionData path to file containing atomic state data
         */
        AtomicData(
            std::string fileChargeStateData,
            std::string fileAtomicStateData,
            std::string fileBoundBoundTransitionData,
            std::string fileBoundFreeTransitionData,
            std::string fileAutonomousTransitionData,
            std::string speciesName)
            : m_speciesName(speciesName)
        {
            // read in files
            //      state data
            std::list<S_ChargeStateTuple> chargeStates = readChargeStates(fileChargeStateData);
            std::list<S_AtomicStateTuple> atomicStates = readAtomicStates(fileAtomicStateData);

            //      transition data
            std::list<S_BoundBoundTransitionTuple> boundBoundTransitions
                = readBoundBoundTransitions(fileBoundBoundTransitionData);
            std::list<S_BoundFreeTransitionTuple> boundFreeTransitions
                = readBoundFreeTransitions(fileBoundFreeTransitionData);
            std::list<S_AutonomousTransitionTuple> autonomousTransitions
                = readAutonomousTransitions(fileAutonomousTransitionData);

            //      sort by lower transition, optional since input files already sorted
            // boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, Idx,true>());
            // boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, Idx, true>());
            // autonomousTransitions.sort(CompareTransitionTupel<TypeValue, Idx,true>());

            // check assumptions
            checkChargeStateList(chargeStates);
            checkAtomicStateList(atomicStates);
            checkTransitionList<S_BoundBoundTransitionTuple>(boundBoundTransitions);
            checkTransitionList<S_BoundFreeTransitionTuple>(boundFreeTransitions);
            checkTransitionList<S_AutonomousTransitionTuple>(autonomousTransitions);

            // initialize buffers
            initBuffers();

            // fill data buffers
            //      states
            storeData<S_ChargeStateTuple, S_ChargeStateDataBox>(chargeStates, chargeStateDataBuffer->getHostDataBox());
            chargeStateDataBuffer->hostToDevice();

            storeData<S_AtomicStateTuple, S_AtomicStateDataBox>(atomicStates, atomicStateDataBuffer->getHostDataBox());
            atomicStateDataBuffer->hostToDevice();

            //      transitions
            storeData<S_BoundBoundTransitionTuple, S_BoundBoundTransitionDataBox>(
                boundBoundTransitions,
                boundBoundTransitionDataBuffer->getHostDataBox());
            boundBoundTransitionDataBuffer->hostToDevice();

            storeData<S_BoundFreeTransitionTuple, S_BoundFreeTransitionDataBox>(
                boundFreeTransitions,
                boundFreeTransitionDataBuffer->getHostDataBox());
            boundFreeTransitionDataBuffer->hostToDevice();

            storeData<S_AutonomousTransitionTuple, S_AutonomousTransitionDataBox>(
                autonomousTransitions,
                autonomousTransitionDataBuffer->getHostDataBox());
            autonomousTransitionDataBuffer->hostToDevice();

            // fill orga data buffers 1,)
            //          charge state
            fillChargeStateOrgaData(atomicStates);

            //          atomic states, up direction
            fill_UpTransition_OrgaData<S_BoundBoundTransitionTuple>(
                boundBoundTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox());
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice(),
                atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();

            fill_UpTransition_OrgaData<S_BoundFreeTransitionTuple>(
                boundFreeTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox());
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice(),
                atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();

            // autonomous transitions are always only downward

            // re-sort by upper state
            boundBoundTransitions.sort(CompareTransitionTupel<TypeValue, Idx, false>());
            boundFreeTransitions.sort(CompareTransitionTupel<TypeValue, Idx, false>());
            autonomousTransitions.sort(CompareTransitionTupel<TypeValue, Idx, false>());

            // store transition data in inverse order
            storeData<S_BoundBoundTransitionTuple, S_BoundBoundTransitionDataBox>(
                boundBoundTransitions,
                inverseBoundBoundTransitionDataBuffer->getHostDataBox());
            inverseBoundBoundTransitionDataBuffer->hostToDevice();

            storeData<S_BoundFreeTransitionTuple, S_BoundFreeTransitionDataBox>(
                boundFreeTransitions,
                inverseBoundFreeTransitionDataBuffer->getHostDataBox());
            inverseBoundFreeTransitionDataBuffer->hostToDevice();

            storeData<S_AutonomousTransitionTuple, S_AutonomousTransitionDataBox>(
                autonomousTransitions,
                inverseAutonomousTransitionDataBuffer->getHostDataBox());
            inverseAutonomousTransitionDataBuffer->hostToDevice();

            // fill orga data buffers 2.)
            //      atomic state, down direction
            fill_DownTransition_OrgaData<
                S_BoundBoundTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_UpDown,
                S_AtomicStateStartIndexBlockDataBox_UpDown>(
                boundBoundTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox());

            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();

            fill_DownTransition_OrgaData<
                S_BoundFreeTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_UpDown,
                S_AtomicStateStartIndexBlockDataBox_UpDown>(
                boundFreeTransitions,
                atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox());

            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();

            fill_DownTransition_OrgaData<
                S_AutonomousTransitionTuple,
                S_AtomicStateNumberOfTransitionsDataBox_Down,
                S_AtomicStateStartIndexBlockDataBox_Down>(
                autonomousTransitions,
                atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox(),
                atomicStateStartIndexBlockDataBuffer_Autonomous->getHostDataBox());

            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_Autonomous->hostToDevice();

            // fill transitionSelectionBuffer
            fillTransitionSelectionDataBufferAndSetOffsets<
                T_electronicExcitation,
                T_electronicDeexcitation,
                T_spontaneousDeexcitation,
                T_electronicIonization,
                T_autonomousIonization,
                T_fieldIonization>();

            // just to be sure
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_ATOMIC_DATA_DEBUG_SYNC_TO_HOST)
                this->hostToDevice();
        }

        void hostToDevice()
        {
            // charge state data
            chargeStateDataBuffer->hostToDevice();
            chargeStateOrgaDataBuffer->hostToDevice();

            // atomic property data
            atomicStateDataBuffer->hostToDevice();
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_BoundFree->hostToDevice();
            atomicStateStartIndexBlockDataBuffer_Autonomous->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->hostToDevice();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->hostToDevice();

            // transition data
            boundBoundTransitionDataBuffer->hostToDevice();
            boundFreeTransitionDataBuffer->hostToDevice();
            autonomousTransitionDataBuffer->hostToDevice();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->hostToDevice();
            inverseBoundFreeTransitionDataBuffer->hostToDevice();
            inverseAutonomousTransitionDataBuffer->hostToDevice();

            transitionSelectionBuffer->hostToDevice();
        }

        void deviceToHost()
        {
            // charge state data
            chargeStateDataBuffer->deviceToHost();
            chargeStateOrgaDataBuffer->deviceToHost();

            // atomic property data
            atomicStateDataBuffer->deviceToHost();
            // atomic orga data
            atomicStateStartIndexBlockDataBuffer_BoundBound->deviceToHost();
            atomicStateStartIndexBlockDataBuffer_BoundFree->deviceToHost();
            atomicStateStartIndexBlockDataBuffer_Autonomous->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_BoundBound->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_BoundFree->deviceToHost();
            atomicStateNumberOfTransitionsDataBuffer_Autonomous->deviceToHost();

            // transition data
            boundBoundTransitionDataBuffer->deviceToHost();
            boundFreeTransitionDataBuffer->deviceToHost();
            autonomousTransitionDataBuffer->deviceToHost();

            // inverse transition data
            inverseBoundBoundTransitionDataBuffer->deviceToHost();
            inverseBoundFreeTransitionDataBuffer->deviceToHost();
            inverseAutonomousTransitionDataBuffer->deviceToHost();

            transitionSelectionBuffer->deviceToHost();
        }

        // charge states
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_ChargeStateDataBox getChargeStateDataDataBox()
        {
            if constexpr(hostData)
                return chargeStateDataBuffer->getHostDataBox();
            else
                return chargeStateDataBuffer->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_ChargeStateOrgaDataBox getChargeStateOrgaDataBox()
        {
            if constexpr(hostData)
                return chargeStateOrgaDataBuffer->getHostDataBox();
            else
                return chargeStateOrgaDataBuffer->getDeviceDataBox();
        }

        // atomic states
        //      property data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateDataBox getAtomicStateDataDataBox()
        {
            if constexpr(hostData)
                return atomicStateDataBuffer->getHostDataBox();
            else
                return atomicStateDataBuffer->getDeviceDataBox();
        }

        //      start index orga data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_UpDown getBoundBoundStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_BoundBound->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_BoundBound->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_UpDown getBoundFreeStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_BoundFree->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_BoundFree->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateStartIndexBlockDataBox_Down getAutonomousStartIndexBlockDataBox()
        {
            if constexpr(hostData)
                return atomicStateStartIndexBlockDataBuffer_Autonomous->getHostDataBox();
            else
                return atomicStateStartIndexBlockDataBuffer_Autonomous->getDeviceDataBox();
        }

        //      number transitions orga data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_UpDown getBoundBoundNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_BoundBound->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_BoundBound->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_UpDown getBoundFreeNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_BoundFree->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_BoundFree->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AtomicStateNumberOfTransitionsDataBox_Down getAutonomousNumberTransitionsDataBox()
        {
            if constexpr(hostData)
                return atomicStateNumberOfTransitionsDataBuffer_Autonomous->getHostDataBox();
            else
                return atomicStateNumberOfTransitionsDataBuffer_Autonomous->getDeviceDataBox();
        }

        // transition data, normal
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_BoundBoundTransitionDataBox getBoundBoundTransitionDataBox()
        {
            if constexpr(hostData)
                return boundBoundTransitionDataBuffer->getHostDataBox();
            else
                return boundBoundTransitionDataBuffer->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_BoundFreeTransitionDataBox getBoundFreeTransitionDataBox()
        {
            if constexpr(hostData)
                return boundFreeTransitionDataBuffer->getHostDataBox();
            else
                return boundFreeTransitionDataBuffer->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AutonomousTransitionDataBox getAutonomousTransitionDataBox()
        {
            if constexpr(hostData)
                return autonomousTransitionDataBuffer->getHostDataBox();
            else
                return autonomousTransitionDataBuffer->getDeviceDataBox();
        }

        // transition data, inverted
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_BoundBoundTransitionDataBox getInverseBoundBoundTransitionDataBox()
        {
            if constexpr(hostData)
                return inverseBoundBoundTransitionDataBuffer->getHostDataBox();
            else
                return inverseBoundBoundTransitionDataBuffer->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_BoundFreeTransitionDataBox getInverseBoundFreeTransitionDataBox()
        {
            if constexpr(hostData)
                return inverseBoundFreeTransitionDataBuffer->getHostDataBox();
            else
                return inverseBoundFreeTransitionDataBuffer->getDeviceDataBox();
        }

        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_AutonomousTransitionDataBox getInverseAutonomousTransitionDataBox()
        {
            if constexpr(hostData)
                return inverseAutonomousTransitionDataBuffer->getHostDataBox();
            else
                return inverseAutonomousTransitionDataBuffer->getDeviceDataBox();
        }

        // transition selection data
        //! @tparam hostData true: get hostDataBox, false: get DeviceDataBox
        template<bool hostData>
        S_TransitionSelectionDataBox getTransitionSelectionDataBox()
        {
            if constexpr(hostData)
                return transitionSelectionDataBuffer->getHostDataBox();
            else
                return transitionSelectionDataBuffer->getDeviceDataBox();
        }

        // debug queries
        uint32_t getNumberAtomicStates() const
        {
            return m_numberAtomicStates;
        }
        uint32_t getNumberBoundBoundTransitions() const
        {
            return m_numberBoundBoundTransitions;
        }
        uint32_t getNumberBoundFreeTransitions() const
        {
            return m_numberBoundFreeTransitions;
        }
        uint32_t getNumberAutonomousTransitions() const
        {
            return m_numberAutonomousTransitions;
        }

        //! == deviceToHost, required by ISimulationData
        void synchronize() override
        {
            this->deviceToHost();
        }

        //! required by ISimulationData
        std::string getUniqueId() override
        {
            return m_speciesName + "_atomicData";
        }
    };

} // namespace picongpu::particles::atomicPhysics2::atomicData
