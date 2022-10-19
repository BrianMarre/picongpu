/* Copyright 2020-2022 Sergei Bastrakov, Brian Marre
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

#include "picongpu/param/physicalConstants.param"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

#include <cstdint>
#include <memory>
#include <utility>


/** @file implements the storage of atomic state and transition data
 *
 * The atomicPhysics step relies on a model of atomic states and transitions for each atomic-
 * Physics ion species. The model's parameters are provided by the user as .txt file of
 * specified format at runtime, external to PIConGPU itself due to license requirements.
 *
 * This file is read at the start of the simulation and stored in an instance of the
 * atomicData Database implemented in this file.
 *
 * too different classes give access to atomic data:
 * - AtomicDataDB ... implements
 *                         * reading of the atomicData input file
 *                         * export to the DataBox for device side use
 *                         * host side storage of atomicData
 * - AtomicDataBox ... deviceSide storage and access to atomicData
 *
 * The atomic data actually consists of 7 different data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (ionization energy, [eV]
 *         number of atomicStates,
 *         startIndex of block of atomicStates in atomicState list) ]
 *
 * - list of levels (sorted blockwise by ionization state list)
 *    [(configNumber, [see electronDistribution]
 *      energy respective to ground state of ionization state, [eV]
 *
 *      number of transitions          up b,
 *      startIndex of transition block up b,
 *      number of transitions          down b,
 *      startIndex of transition block down b,
 *
 *      number of transitions          up f,
 *      startIndex of transition block up f,
 *      number of transitions          down f,
 *      startIndex of transition block down f,
 *
 *      number of transitions          up a,
 *      startIndex of transition block up a,
 *      number of transitions          down a,
 *      startIndex of transition block down a)]
 *
 * - bound-bound(b) transitions, list (sorted blockwise by lower State according to state list)
 *    [(collisionalOscillatorStrength,
 *      absorptionOscillatorStrength,
 *      gaunt coefficent 1,
 *      gaunt coefficent 2,
 *      gaunt coefficent 3,
 *      gaunt coefficent 4,
 *      gaunt coefficent 5,
 *      upper state configNumber)]
 *
 * - b reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * - bound-free(f) transitions, list (sorted blockwise by lower State according to state list)
 *    [(phicx coefficent 1,
 *      phicx coefficent 2,
 *      phicx coefficent 3,
 *      phicx coefficent 4,
 *      phicx coefficent 5,
 *      phicx coefficent 6,
 *      phicx coefficent 7,
 *      phicx coefficent 8,
 *      upper state configNumber)]
 *
 * - f reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * - autonomous transitions(a), list (sorted blockwise by lower atomic, according to state list)
 *    [(rate, [1/s]
 *      upper state configNumber)]
 *
 * - f reverse loockup list: sorted blockwise by upper State
 *    [ index Transition,
 *      lower state configNumber]
 *
 * NOTE: - configNumber specifies the number of a state as defined by the configNumber class
 *       - index always refers to a collection index
 *      the configNumber of a given state is always the same, its collection index depends on
 *      input file, => should only be used internally
 */


namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            template<
                // Data box types for atomic data on host and device
                typename T_DataBoxType,
                typename T_Number, // uint32_t
                typename T_Value, // float_X
                typename T_ConfigNumberDataType, // uint64_t
                typename T_TransitionIndexDataType, // uint32_t

                uint32_t T_atomicNumber, // number ionization states-1
                uint32_t T_numberAtomicStates,
                uint32_t T_numberAtomicTransitions>
            class AtomicDataBox
            {
            public:
                using Idx = T_ConfigNumberDataType;
                using BoxNumber = T_DataBoxType<T_Number>;
                using BoxValue = T_DataBoxType<T_Value>;
                using BoxConfigNumber = T_DataBoxType<T_ConfigNumberDataType>;
                using BoxTransitionIdx = T_DataBoxType<T_TransitionIndexDataType>;

            private:
                // ionization state storage
                BoxValue m_boxIonization_ionizationEnergy; // unit: eV
                BoxValue m_boxIonization_screenedCharge; // unit: e
                BoxNumber m_boxIonization_numberAtomicStates;
                BoxNumber m_boxIonization_startIndexBlockAtomicStates;

                // atomic state storage
                BoxStateConfigNumber m_boxState_configNumber;
                BoxValue m_boxState_energy; // unit: eV

                BoxNumber m_boxState_numberTransitions_up_b;
                BoxNumber m_boxState_startIndexBlockTransitions_up_b;
                BoxNumber m_boxState_numberTransitions_down_b;
                BoxNumber m_boxState_startIndexBlockTransitions_down_b;

                BoxNumber m_boxState_numberTransitions_up_f;
                BoxNumber m_boxState_startIndexBlockTransitions_up_f;
                BoxNumber m_boxState_numberTransitions_down_f;
                BoxNumber m_boxState_startIndexBlockTransitions_down_f;

                BoxNumber m_boxState_numberTransitions_up_a;
                BoxNumber m_boxState_startIndexBlockTransitions_up_a;
                BoxNumber m_boxState_numberTransitions_down_a;
                BoxNumber m_boxState_startIndexBlockTransitions_down_a;

                // transitionStorage bound-bound
                BoxValue m_boxTransitionB_collisionalOscillatorStrength; // unitless
                BoxValue m_boxTransitionB_absorptionOscillatorStrength; // unitless
                BoxValue m_boxTransitionB_cinx1; // unitless
                BoxValue m_boxTransitionB_cinx2; // unitless
                BoxValue m_boxTransitionB_cinx3; // unitless
                BoxValue m_boxTransitionB_cinx4; // unitless
                BoxValue m_boxTransitionB_cinx5; // unitless
                BoxStateConfigNumber m_boxTransitionB_UpperConfigNumber; // lower config number is available via index

                // reverse lockupTable bound-bound
                BoxTransitionIdx m_boxTransitionReverseB_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseB_lowerConfigNumber;

                // transitionStorage bound-free
                BoxValue m_boxTransitionF_cinx1; // unitless
                BoxValue m_boxTransitionF_cinx2; // unitless
                BoxValue m_boxTransitionF_cinx3; // unitless
                BoxValue m_boxTransitionF_cinx4; // unitless
                BoxValue m_boxTransitionF_cinx5; // unitless
                BoxValue m_boxTransitionF_cinx6; // unitless
                BoxValue m_boxTransitionF_cinx7; // unitless
                BoxValue m_boxTransitionF_cinx8; // unitless
                BoxStateConfigNumber m_boxTransitionF_upperConfigNumber; // lower config number is available via index

                // reverse lockupTable bound-free
                BoxTransitionIdx m_boxTransitionReverseF_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseF_lowerConfigNumber;

                // transitionStorage autonomous
                ///@todo better unit?
                BoxValue m_boxTransitionA_Rate; // unit: 1/s
                BoxConfigNumber m_boxTransitionA_upperStateConfigNumber;

                // reverse lockup-autonomous
                BoxTransitionIdx m_boxTransitionReverseA_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseA_lowerConfigNumber;

            public:
                AtomicDataBox(
                    BoxValue boxIonization_ionizationEnergy, // unit: eV
                    BoxValue boxIonization_screenedCharge, // unit: e
                    BoxNumber boxIonization_numberAtomicStates, // unitless
                    BoxNumber boxIonization_startIndexBlockAtomicStates, // unitless
                    BoxStateConfigNumber boxState_configNumber,
                    BoxValue boxState_energy, // unit: eV
                    BoxNumber boxState_numberTransitions_up_b,
                    BoxNumber boxState_startIndexBlockTransitions_up_b,
                    BoxNumber boxState_numberTransitions_down_b,
                    BoxNumber boxState_startIndexBlockTransitions_down_b,
                    BoxNumber boxState_numberTransitions_up_f,
                    BoxNumber boxState_startIndexBlockTransitions_up_f,
                    BoxNumber boxState_numberTransitions_down_f,
                    BoxNumber boxState_startIndexBlockTransitions_down_f,
                    BoxNumber boxState_numberTransitions_up_a,
                    BoxNumber boxState_startIndexBlockTransitions_up_a,
                    BoxNumber boxState_numberTransitions_down_a,
                    BoxNumber boxState_startIndexBlockTransitions_down_a,
                    BoxValue boxTransitionB_collisionalOscillatorStrength, // unitless
                    BoxValue boxTransitionB_absorptionOscillatorStrength, // unitless
                    BoxValue boxTransitionB_cinx1, // unitless
                    BoxValue boxTransitionB_cinx2, // unitless
                    BoxValue boxTransitionB_cinx3, // unitless
                    BoxValue boxTransitionB_cinx4, // unitless
                    BoxValue boxTransitionB_cinx5, // unitless
                    BoxStateConfigNumber boxTransitionB_UpperConfigNumber,
                    BoxTransitionIdx boxTransitionReverseB_transitionIndex,
                    BoxConfigNumber boxTransitionReverseB_lowerConfigNumber,
                    BoxValue boxTransitionF_cinx1, // unitless
                    BoxValue boxTransitionF_cinx2, // unitless
                    BoxValue boxTransitionF_cinx3, // unitless
                    BoxValue boxTransitionF_cinx4, // unitless
                    BoxValue boxTransitionF_cinx5, // unitless
                    BoxValue boxTransitionF_cinx6, // unitless
                    BoxValue boxTransitionF_cinx7, // unitless
                    BoxValue boxTransitionF_cinx8, // unitless
                    BoxStateConfigNumber boxTransitionF_upperConfigNumber,
                    BoxTransitionIdx boxTransitionReverseF_transitionIndex,
                    BoxConfigNumber boxTransitionReverseF_lowerConfigNumber,
                    BoxValue boxTransitionA_Rate, // unit: 1/s
                    BoxConfigNumber boxTransitionA_upperStateConfigNumber,
                    BoxTransitionIdx boxTransitionReverseA_transitionIndex,
                    BoxConfigNumber boxTransitionReverseA_lowerConfigNumber)
                    : m_boxIonization_ionizationEnergy(boxIonization_ionizationEnergy)
                    , m_boxIonization_screenedCharge(boxIonization_screenedCharge)
                    , m_boxIonization_numberAtomicStates(boxIonization_numberAtomicStates)
                    , m_boxIonization_startIndexBlockAtomicStates(boxIonization_startIndexBlockAtomicStates)
                    , m_boxState_configNumber(boxState_configNumber)
                    , m_boxState_energy(boxState_energy)
                    , m_boxState_numberTransitions_up_b(boxState_numberTransitions_up_b)
                    , m_boxState_startIndexBlockTransitions_up_b(boxState_startIndexBlockTransitions_up_b)
                    , m_boxState_numberTransitions_down_b(boxState_numberTransitions_down_b)
                    , m_boxState_startIndexBlockTransitions_down_b(boxState_startIndexBlockTransitions_down_b)
                    , m_boxState_numberTransitions_up_f(boxState_numberTransitions_up_f)
                    , m_boxState_startIndexBlockTransitions_up_f(boxState_startIndexBlockTransitions_up_f)
                    , m_boxState_numberTransitions_down_f(boxState_numberTransitions_down_f)
                    , m_boxState_startIndexBlockTransitions_down_f(boxState_startIndexBlockTransitions_down_f)
                    , m_boxState_numberTransitions_up_a(boxState_numberTransitions_up_a)
                    , m_boxState_startIndexBlockTransitions_up_a(boxState_startIndexBlockTransitions_up_a)
                    , m_boxState_numberTransitions_down_a(boxState_numberTransitions_down_a)
                    , m_boxState_startIndexBlockTransitions_down_a(boxState_startIndexBlockTransitions_down_a)
                    , m_boxTransitionB_collisionalOscillatorStrength(boxTransitionB_collisionalOscillatorStrength)
                    , m_boxTransitionB_absorptionOscillatorStrength(boxTransitionB_absorptionOscillatorStrength)
                    , m_boxTransitionB_cinx1(boxTransitionB_cinx1)
                    , m_boxTransitionB_cinx2(boxTransitionB_cinx2)
                    , m_boxTransitionB_cinx3(boxTransitionB_cinx3)
                    , m_boxTransitionB_cinx4(boxTransitionB_cinx4)
                    , m_boxTransitionB_cinx5(boxTransitionB_cinx5)
                    , m_boxTransitionB_UpperConfigNumber(boxTransitionB_UpperConfigNumber)
                    , m_boxTransitionReverseB_transitionIndex(boxTransitionReverseB_transitionIndex)
                    , m_boxTransitionReverseB_lowerCondfigNumber(boxTransitionReverseB_lowerCondfigNumber)
                    , m_boxTransitionF_cinx1(boxTransitionF_cinx1)
                    , m_boxTransitionF_cinx2(boxTransitionF_cinx2)
                    , m_boxTransitionF_cinx3(boxTransitionF_cinx3)
                    , m_boxTransitionF_cinx4(boxTransitionF_cinx4)
                    , m_boxTransitionF_cinx5(boxTransitionF_cinx5)
                    , m_boxTransitionF_cinx6(boxTransitionF_cinx6)
                    , m_boxTransitionF_cinx7(boxTransitionF_cinx7)
                    , m_boxTransitionF_cinx8(boxTransitionF_cinx8)
                    , m_boxTransitionF_upperConfigNumber(boxTransitionF_upperConfigNumber)
                    , m_boxTransitionReverseF_transitionIndex(boxTransitionReverseF_transitionIndex)
                    , m_boxTransitionReverseF_lowerCondfigNumber(boxTransitionReverseF_lowerCondfigNumber)
                    , m_boxTransitionA_Rate(boxTransitionA_Rate)
                    , m_boxTransitionA_upperStateConfigNumber(boxTransitionA_upperStateConfigNumber)
                    , m_boxTransitionReverseA_transitionIndex(boxTransitionReverseA_transitionIndex)
                    , m_boxTransitionReverseA_lowerCondfigNumber(boxTransitionReverseA_lowerCondfigNumber)
                {
                }

                /** returns index of atomic state in databox, if returns numStates state not found
                 *
                 * @TODO: replace linear search @BrianMarre, 2021
                 */
                HDINLINE uint32_t findState(Idx const stateConfigNumber) const
                {
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        if(this->m_boxStateConfigNumber(i) == stateConfigNumber)
                            return i;
                    }
                    return m_numStates;
                }

                /** returns index of transition in databox, if retunrs numberTransitions not found
                 *
                 * @TODO: replace linear search
                 */
                HDINLINE uint32_t findTransition(Idx const lowerConfigNumber, Idx const upperConfigNumber) const
                {
                    // search for lowerConfigNumber in state list
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                        if(m_boxStateConfigNumber(i) == lowerConfigNumber)
                        {
                            // search in corresponding block in transitions box
                            for(uint32_t j = 0u; j < this->m_boxNumTransitions(i); j++)
                            {
                                // Does Lower state have at least one transition?,
                                // otherwise StartIndexBlockTransition == m_maxNumberTransitions
                                if((this->m_boxStartIndexBlockTransitions(i) < (this->m_maxNumberTransitions)) &&
                                   // is correct upperConfigNumber?
                                   this->m_boxUpperConfigNumber(this->m_boxStartIndexBlockTransitions(i) + j)
                                       == upperConfigNumber)
                                    return this->m_boxStartIndexBlockTransitions(i) + j;
                            }
                        }
                    return this->m_numberTransitions;
                }

                /** searches for transition to upper state in block of transitions of lower State,
                 *  returns index in databox of this transition if found, or m_numberTransitions if not
                 */
                HDINLINE uint32_t
                findTransitionInBlock(uint32_t const indexLowerState, Idx const upperConfigNumber) const
                {
                    uint32_t startIndexBlock = this->m_boxStartIndexBlockTransitions(indexLowerState);

                    for(uint32_t i = 0u; i < this->m_boxNumTransitions(indexLowerState); i++)
                    {
                        if(this->m_boxUpperConfigNumber(startIndexBlock + i) == upperConfigNumber)
                            return this->m_boxStartIndexBlockTransitions(indexLowerState) + i;
                    }
                    return this->m_numberTransitions;
                }

                /** returns upper states ConfigNumber of the transition
                 *
                 * @param indexTransition ... collection index of transition,
                 *  available using findTransition() and findTransitionInBlock()
                 */
                HDINLINE Idx getUpperConfigNumberTransition(uint32_t const indexTransition) const
                {
                    return this->m_boxUpperConfigNumber(indexTransition);
                }

                /** returns number of Transitions in dataBox with state as lower state
                 *
                 *  @param stateIndex ... collection index of state, available using findState()
                 */
                HDINLINE uint32_t getNumberTransitions(uint32_t const indexState) const
                {
                    if(indexState < m_numStates)
                        return this->m_boxNumTransitions(indexState);
                    return 0u;
                }

                // returns start index of the block of transitions with state as lower state
                HDINLINE uint32_t getStartIndexBlock(uint32_t const indexState) const
                {
                    return this->m_boxStartIndexBlockTransitions(indexState);
                }

                // number of Transitions stored in this box
                HDINLINE uint32_t getNumTransitions() const
                {
                    return this->m_numberTransitions;
                }

                // number of atomic states stored in this box
                HDINLINE uint32_t getNumStates() const
                {
                    return this->m_numStates;
                }

                HDINLINE T_ValueType getCollisionalOscillatorStrength(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCollisionalOscillatorStrength(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getCinx1(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx1(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getCinx2(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx2(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getCinx3(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx3(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getCinx4(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx4(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getCinx5(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxCinx5(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE T_ValueType getAbsorptionOscillatorStrength(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numberTransitions)
                        return this->m_boxAbsorptionOscillatorStrength(indexTransition);
                    return static_cast<T_ValueType>(0);
                }

                HDINLINE void addLevel(
                    Idx const ConfigNumber, // must be index as defined in ConfigNumber
                    T_ValueType const energy // unit: eV
                )
                {
                    /** add a level to databox
                     *
                     *  NOTE: - must be called sequentially!
                     *        - assumes no more levels are added than memory is available,
                     */

                    if(this->m_numStates < m_maxNumberStates)
                    {
                        this->m_boxStateConfigNumber[this->m_numStates] = ConfigNumber;
                        this->m_boxStateEnergy[this->m_numStates] = energy;
                        this->m_boxNumTransitions[this->m_numStates] = 0u;
                        this->m_boxStartIndexBlockTransitions[this->m_numStates] = this->m_maxNumberTransitions;
                        this->m_numStates += 1u;
                    }
                }

                /** add transition to atomic data box
                 *
                 *  NOTE: must be called block wise and sequentially!
                 *  - block wise: add all transitions of one lower ConfigNumber before moving on
                 *      to the next lowerConfigNumber value
                 */
                HDINLINE void addTransition(
                    Idx const lowerConfigNumber, // must be index as defined in ConfigNumber
                    Idx const upperConfigNumber, // must be index as defined in ConfigNumber
                    T_ValueType const collisionalOscillatorStrength, // unit: unitless
                    T_ValueType const gauntCoefficent1, // unit: unitless
                    T_ValueType const gauntCoefficent2, // unit: unitless
                    T_ValueType const gauntCoefficent3, // unit: unitless
                    T_ValueType const gauntCoefficent4, // unit: unitless
                    T_ValueType const gauntCoefficent5, // unit: unitless
                    T_ValueType const absorptionOscillatorStrength) // unitless
                {
                    // get dataBox index of lowerConfigNumber
                    uint32_t collectionIndex = this->findState(lowerConfigNumber);

                    // check transition actually found
                    if(collectionIndex == this->m_numStates)
                    {
                        printf("ERROR: Tried adding transition without adding lower level first");
                        return;
                    }

                    // set start index block in transition collection if first transition of this lowerConfigNumber
                    if(this->m_boxStartIndexBlockTransitions(collectionIndex) == m_maxNumberTransitions)
                    {
                        this->m_boxStartIndexBlockTransitions(collectionIndex) = m_numberTransitions;
                    }

                    // check not too many transitions
                    if((this->m_numberTransitions < m_maxNumberTransitions))
                    {
                        // input transition data
                        this->m_boxUpperConfigNumber[m_numberTransitions] = upperConfigNumber;
                        this->m_boxCollisionalOscillatorStrength[m_numberTransitions] = collisionalOscillatorStrength;
                        this->m_boxCinx1[m_numberTransitions] = gauntCoefficent1;
                        this->m_boxCinx2[m_numberTransitions] = gauntCoefficent2;
                        this->m_boxCinx3[m_numberTransitions] = gauntCoefficent3;
                        this->m_boxCinx4[m_numberTransitions] = gauntCoefficent4;
                        this->m_boxCinx5[m_numberTransitions] = gauntCoefficent5;
                        this->m_boxAbsorptionOscillatorStrength[m_numberTransitions] = absorptionOscillatorStrength;

                        // update context
                        this->m_numberTransitions += 1u;
                        this->m_boxNumTransitions(collectionIndex) += 1u;
                    }
                }
            };


            // atomic data box host-device storage,
            // to be used from the host side only
            template<uint8_t T_atomicNumber, typename T_ConfigNumberDataType = uint64_t>
            class AtomicData
            {
            public:
                // type declarations
                using Idx = T_ConfigNumberDataType;
                using BufferValue = pmacc::GridBuffer<float_X, 1>;
                using BufferNumber = pmacc::GridBuffer<uint32_t, 1>;
                using BufferConfigNumber = pmacc::GridBuffer<T_ConfigNumberDataType, 1>;

                // data storage
                using InternalDataBoxTypeValue = pmacc::DataBox<pmacc::PitchedBox<float_X, 1>>;
                using InternalDataBoxTypeNumber = pmacc::DataBox<pmacc::PitchedBox<uint32_t, 1>>;
                using InternalDataBoxTypeConfigNumber = pmacc::DataBox<pmacc::PitchedBox<T_ConfigNumberDataType, 1>>;

                // acess datatype used on device
                using DataBoxType = AtomicDataBox<
                    T_atomicNumber,
                    InternalDataBoxTypeValue,
                    InternalDataBoxTypeNumber,
                    InternalDataBoxTypeConfigNumber,
                    T_ConfigNumberDataType>;

            private:
                // pointers to storage
                std::unique_ptr<BufferValue>
                    dataStateEnergy; // unit: eV, @TODO change to ATOMIC_UNIT_ENERGY?, BrianMarre, 2021
                std::unique_ptr<BufferNumber> dataNumTransitions; // unit: unitless
                std::unique_ptr<BufferNumber> dataStartIndexBlockTransitions; // unit: unitless
                std::unique_ptr<BufferConfigNumber> dataConfigNumber; // unit: unitless

                std::unique_ptr<BufferConfigNumber> dataUpperConfigNumber; // unit: unitless
                std::unique_ptr<BufferValue> dataCollisionalOscillatorStrength; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx1; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx2; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx3; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx4; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx5; // unit: unitless
                std::unique_ptr<BufferValue> dataAbsorptionOscillatorStrength; // unit: unitless

                // number of states included in atomic data
                uint32_t m_maxNumberStates;
                uint32_t m_maxNumberTransitions;

            public:
                HINLINE AtomicData(uint32_t maxNumberStates, uint32_t maxNumberTransitions)
                {
                    m_maxNumberStates = maxNumberStates;
                    m_maxNumberTransitions = maxNumberTransitions;

                    // get values for init of databox
                    auto sizeStates = pmacc::DataSpace<1>::create(m_maxNumberStates);
                    auto sizeTransitions = pmacc::DataSpace<1>::create(m_maxNumberTransitions);

                    auto const guardSize = pmacc::DataSpace<1>::create(0);

                    auto const layoutStates = pmacc::GridLayout<1>(sizeStates, guardSize);
                    auto const layoutTransitions = pmacc::GridLayout<1>(sizeTransitions, guardSize);

                    // create Buffers on stack and store pointer to it as member
                    // states data
                    dataConfigNumber.reset(new BufferConfigNumber(layoutStates));
                    dataStateEnergy.reset(new BufferValue(layoutStates));
                    dataNumTransitions.reset(new BufferNumber(layoutStates));
                    dataStartIndexBlockTransitions.reset(new BufferNumber(layoutStates));

                    // transition data
                    dataUpperConfigNumber.reset(new BufferConfigNumber(layoutTransitions));
                    dataCollisionalOscillatorStrength.reset(new BufferValue(layoutTransitions));
                    dataCinx1.reset(new BufferValue(layoutTransitions));
                    dataCinx2.reset(new BufferValue(layoutTransitions));
                    dataCinx3.reset(new BufferValue(layoutTransitions));
                    dataCinx4.reset(new BufferValue(layoutTransitions));
                    dataCinx5.reset(new BufferValue(layoutTransitions));
                    dataAbsorptionOscillatorStrength.reset(new BufferValue(layoutTransitions));
                }

                //! Get the host data box for the rate matrix values
                HINLINE DataBoxType getHostDataBox(uint32_t numStates, uint32_t numberTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getHostBuffer().getDataBox(),
                        dataNumTransitions->getHostBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getHostBuffer().getDataBox(),
                        dataConfigNumber->getHostBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        // dataLowerConfigNumber->getHostBuffer().getDataBox(),
                        dataUpperConfigNumber->getHostBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                        dataCinx1->getHostBuffer().getDataBox(),
                        dataCinx2->getHostBuffer().getDataBox(),
                        dataCinx3->getHostBuffer().getDataBox(),
                        dataCinx4->getHostBuffer().getDataBox(),
                        dataCinx5->getHostBuffer().getDataBox(),
                        dataAbsorptionOscillatorStrength->getHostBuffer().getDataBox(),
                        numberTransitions,
                        this->m_maxNumberTransitions);
                }

                //! Get the device data box for the rate matrix values
                HINLINE DataBoxType getDeviceDataBox(uint32_t numStates, uint32_t numberTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getDeviceBuffer().getDataBox(),
                        dataNumTransitions->getDeviceBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getDeviceBuffer().getDataBox(),
                        dataConfigNumber->getDeviceBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        dataUpperConfigNumber->getDeviceBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getDeviceBuffer().getDataBox(),
                        dataCinx1->getDeviceBuffer().getDataBox(),
                        dataCinx2->getDeviceBuffer().getDataBox(),
                        dataCinx3->getDeviceBuffer().getDataBox(),
                        dataCinx4->getDeviceBuffer().getDataBox(),
                        dataCinx5->getDeviceBuffer().getDataBox(),
                        dataAbsorptionOscillatorStrength->getDeviceBuffer().getDataBox(),
                        numberTransitions,
                        this->m_maxNumberTransitions);
                }

                void syncToDevice()
                {
                    dataStateEnergy->hostToDevice();
                    dataNumTransitions->hostToDevice();
                    dataStartIndexBlockTransitions->hostToDevice();
                    dataConfigNumber->hostToDevice();

                    dataUpperConfigNumber->hostToDevice();
                    dataCollisionalOscillatorStrength->hostToDevice();
                    dataCinx1->hostToDevice();
                    dataCinx2->hostToDevice();
                    dataCinx3->hostToDevice();
                    dataCinx4->hostToDevice();
                    dataCinx5->hostToDevice();
                    dataAbsorptionOscillatorStrength->hostToDevice();
                }
            };

        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
