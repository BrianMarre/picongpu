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
                // reverse lockupTable bound-bound
                BoxTransitionIdx m_boxTransitionReverseB_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseB_lowerConfigNumber;

                // reverse lockupTable bound-free
                BoxTransitionIdx m_boxTransitionReverseF_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseF_lowerConfigNumber;

                // reverse lockup-autonomous
                BoxTransitionIdx m_boxTransitionReverseA_transitionIndex;
                BoxConfigNumber m_boxTransitionReverseA_lowerConfigNumber;

            public:

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

            };

        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
