/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov,
 *                     Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>
#include "picongpu/param/physicalConstants.param"

#include <cstdint>
#include <memory>
#include <utility>

#pragma once

// debug only
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            /** too different classes giving acess to atomic data:
             * - base class ... implements actual functionality
             * dataBox class ... provides acess implementation for actual storage in box
             *      encapsulates index shift currently
             */

            // Data box type for rate matrix on host and device
            template<
                uint8_t T_atomicNumber,
                typename T_DataBoxValue,
                typename T_DataBoxNumber,
                typename T_DataBoxStateIdx,
                typename T_ConfigNumberDataType>
            class AtomicDataBox
            {
            public:
                using DataBoxValue = T_DataBoxValue;
                using DataBoxNumber = T_DataBoxNumber;
                using DataBoxStateIdx = T_DataBoxStateIdx;
                using Idx = T_ConfigNumberDataType;
                using ValueType = typename DataBoxValue::ValueType;

            private:
                uint32_t m_numStates;
                uint32_t m_maxNumberStates;
                DataBoxValue m_boxStateEnergy;
                DataBoxNumber m_boxNumTransitions;
                DataBoxNumber m_boxStartIndexBlockTransitions;
                DataBoxStateIdx m_boxStateIdx;

                // debug only, should be private later
            public:
                DataBoxValue m_boxCollisionalOscillatorStrength;

            private:
                uint32_t m_numTransitions;
                uint32_t m_maxNumberTransitions;
                DataBoxValue m_boxCinx1;
                DataBoxValue m_boxCinx2;
                DataBoxValue m_boxCinx3;
                DataBoxValue m_boxCinx4;
                DataBoxValue m_boxCinx5;
                // DataBoxStateIdx m_boxLowerIdx;
                DataBoxStateIdx m_boxUpperIdx;


            public:
                // Constructor
                AtomicDataBox(
                    DataBoxValue boxStateEnergy,
                    DataBoxNumber boxNumTransitions,
                    DataBoxNumber boxStartIndexBlockTransitions,
                    DataBoxStateIdx boxStateIdx,
                    uint32_t numStates,
                    uint32_t maxNumberStates,

                    // DataBoxStateIdx boxLowerIdx,
                    DataBoxStateIdx boxUpperIdx,
                    DataBoxValue boxCollisionalOscillatorStrength,
                    DataBoxValue boxCinx1,
                    DataBoxValue boxCinx2,
                    DataBoxValue boxCinx3,
                    DataBoxValue boxCinx4,
                    DataBoxValue boxCinx5,
                    uint32_t numTransitions,
                    uint32_t maxNumberTransitions)
                    : m_boxStateEnergy(boxStateEnergy)
                    , m_boxNumTransitions(boxNumTransitions)
                    , m_boxStartIndexBlockTransitions(boxStartIndexBlockTransitions)
                    , m_boxStateIdx(boxStateIdx)
                    //, m_boxLowerIdx(boxLowerIdx)
                    , m_boxUpperIdx(boxUpperIdx)
                    , m_boxCollisionalOscillatorStrength(boxCollisionalOscillatorStrength)
                    , m_boxCinx1(boxCinx1)
                    , m_boxCinx2(boxCinx2)
                    , m_boxCinx3(boxCinx3)
                    , m_boxCinx4(boxCinx4)
                    , m_boxCinx5(boxCinx5)
                    , m_numStates(numStates)
                    , m_maxNumberStates(maxNumberStates)
                    , m_numTransitions(numTransitions)
                    , m_maxNumberTransitions(maxNumberTransitions)
                {
                }

                // get energy, respective to ground state, of atomic state
                // @param idx ... configNumber of atomic state
                // return unit: ATOMIC_UNIT_ENERGY
                // TODO: replace dumb linear search @BrianMarre 2021
                HDINLINE ValueType operator()(Idx const idx) const
                {
                    // one is a special case
                    if(idx == 0)
                        return 0.0_X;

                    // search for state in list
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        if(m_boxStateIdx(i) == idx)
                        {
                            // NOTE: unit conversion should be done in 64 bit
                            return float_X(float_64(m_boxStateEnergy(i)) * UNITCONV_eV_to_AU);
                        }
                    }
                    // atomic state not found return zero
                    return static_cast<ValueType>(0);
                }


                // returns state corresponding to given index
                HDINLINE Idx getAtomicStateConfigNumberIndex(uint32_t const indexState) const
                {
                    return this->m_boxStateIdx(indexState);
                }

                // returns index of state in databox, numStates equal to not found
                // @TODO: replace stupid linear search @BrianMarre, 2021
                HDINLINE uint32_t findState(Idx const stateIdx) const
                {
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                    {
                        if(this->m_boxStateIdx(i) == stateIdx)
                            return i;
                    }
                    return m_numStates;
                }

                // returns index of transition in databox, if equal to numTransitions not found
                // @TODO: replace linear search
                HDINLINE uint32_t findTransition(Idx const lowerIdx, Idx const upperIdx) const
                {
                    // search for lowerIdx in state list
                    for(uint32_t i = 0u; i < this->m_numStates; i++)
                        if(m_boxStateIdx(i) == lowerIdx)
                        {
                            // search in corresponding block in transitions box
                            for(uint32_t j = 0u; j < this->m_boxNumTransitions(i); j++)
                            {
                                // Does Lower state have at least one transition?,
                                // otherwise StartIndexBlockTransition == m_maxNumberTransitions
                                if((this->m_boxStartIndexBlockTransitions(i) < (this->m_maxNumberTransitions)) &&
                                   // is correct upperIdx?
                                   this->m_boxUpperIdx(this->m_boxStartIndexBlockTransitions(i) + j) == upperIdx)
                                    return this->m_boxStartIndexBlockTransitions(i) + j;
                            }
                        }
                    return this->m_numTransitions;
                }

                // searches for transition to upper state in block of transitions of lower State,
                // returns index in databox of this transition if found, or m_numTransitions if not
                HDINLINE uint32_t findTransitionInBlock(uint32_t const indexLowerState, Idx const upperIdx) const
                {
                    uint32_t startIndexBlock = this->m_boxStartIndexBlockTransitions(indexLowerState);

                    // debug only
                    /*std::cout << "        indexLowerState " << indexLowerState <<
                        " upperState " << upperIdx << " startIndexBlock " << startIndexBlock
                        << " numTransitions " << this->m_boxNumTransitions(indexLowerState) << std::endl;*/

                    for(uint32_t i = 0u; i < this->m_boxNumTransitions(indexLowerState); i++)
                    {
                        // debug only
                        /*std::cout << "            transitionIndex: " << startIndexBlock + i
                            << " upperIdxTransition " << this->m_boxUpperIdx(startIndexBlock + i) << std::endl;*/

                        if(this->m_boxUpperIdx(startIndexBlock + i) == upperIdx)
                            return this->m_boxStartIndexBlockTransitions(indexLowerState) + i;
                    }
                    return this->m_numTransitions;
                }

                // returns upper states Idx of the transition
                HDINLINE Idx getUpperIdxTransition(uint32_t const indexTransition) const
                {
                    return this->m_boxUpperIdx(indexTransition);
                }

                // returns number of Transitions in dataBox with state as lower state
                // stateIndex ... collection Index of state, available using findState()
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
                    return this->m_numTransitions;
                }

                // number of atomic states stored in this box
                HDINLINE uint32_t getNumStates() const
                {
                    return this->m_numStates;
                }

                HDINLINE ValueType getCollisionalOscillatorStrength(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCollisionalOscillatorStrength(indexTransition);

                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx1(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCinx1(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx2(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCinx2(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx3(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCinx3(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx4(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCinx4(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE ValueType getCinx5(uint32_t const indexTransition) const
                {
                    if(indexTransition < this->m_numTransitions)
                        return this->m_boxCinx5(indexTransition);
                    return static_cast<ValueType>(0);
                }

                HDINLINE constexpr static uint8_t getAtomicNumber()
                {
                    return T_atomicNumber;
                }

                // must be called sequentially!
                // assumes no more levels are added than memory is available
                //
                HDINLINE void addLevel(
                    Idx const idx, // must be index as defined in ConfigNumber
                    ValueType const energy // unit: eV
                )
                {
                    if(this->m_numStates < m_maxNumberStates)
                    {
                        this->m_boxStateIdx[this->m_numStates] = idx;
                        this->m_boxStateEnergy[this->m_numStates] = energy;
                        this->m_boxNumTransitions[this->m_numStates] = 0u;
                        this->m_boxStartIndexBlockTransitions[this->m_numStates] = this->m_maxNumberTransitions;
                        this->m_numStates += 1u;
                    }
                }

                // must be called block wise and sequentially!
                //  block wise: add all transitions of one lower Idx before moving on
                // to the next lowerIdx value
                HDINLINE void addTransition(
                    Idx const lowerIdx, // must be index as defined in ConfigNumber
                    Idx const upperIdx, // must be index as defined in ConfigNumber
                    ValueType const collisionalOscillatorStrength, // unit: unitless
                    ValueType const gauntCoefficent1, // unit: unitless
                    ValueType const gauntCoefficent2, // unit: unitless
                    ValueType const gauntCoefficent3, // unit: unitless
                    ValueType const gauntCoefficent4, // unit: unitless
                    ValueType const gauntCoefficent5) // unit: unitless
                {
                    // get dataBox index of lowerIdx
                    uint32_t collectionIndex = this->findState(lowerIdx);

                    // check transition actually found
                    if(collectionIndex == this->m_numStates)
                    {
                        printf("ERROR: Tried adding transition without adding lower level first");
                        return;
                    }

                    // set start index block in transition collection if first transition of this lowerIdx
                    if(this->m_boxStartIndexBlockTransitions(collectionIndex) == m_maxNumberTransitions)
                    {
                        this->m_boxStartIndexBlockTransitions(collectionIndex) = m_numTransitions;
                    }

                    // check not too many transitions
                    if((this->m_numTransitions < m_maxNumberTransitions))
                    {
                        // input transition data
                        this->m_boxUpperIdx[m_numTransitions] = upperIdx;
                        this->m_boxCollisionalOscillatorStrength[m_numTransitions] = collisionalOscillatorStrength;
                        this->m_boxCinx1[m_numTransitions] = gauntCoefficent1;
                        this->m_boxCinx2[m_numTransitions] = gauntCoefficent2;
                        this->m_boxCinx3[m_numTransitions] = gauntCoefficent3;
                        this->m_boxCinx4[m_numTransitions] = gauntCoefficent4;
                        this->m_boxCinx5[m_numTransitions] = gauntCoefficent5;

                        // update context
                        this->m_numTransitions += 1u;
                        this->m_boxNumTransitions(collectionIndex) += 1u;
                    }
                }
            };


            // Rate matrix host-device storage,
            // to be used from the host side only
            template<uint8_t T_atomicNumber, typename T_ConfigNumberDataType = uint64_t>
            class AtomicData
            {
                // type declarations
            public:
                using Idx = T_ConfigNumberDataType;
                using BufferValue = pmacc::GridBuffer<float_X, 1>;
                using BufferNumber = pmacc::GridBuffer<uint32_t, 1>;
                using BufferIdx = pmacc::GridBuffer<T_ConfigNumberDataType, 1>;
                // data storage
                using InternalDataBoxTypeValue = pmacc::DataBox<pmacc::PitchedBox<float_X, 1>>;
                using InternalDataBoxTypeNumber = pmacc::DataBox<pmacc::PitchedBox<uint32_t, 1>>;
                using InternalDataBoxTypeIdx = pmacc::DataBox<pmacc::PitchedBox<T_ConfigNumberDataType, 1>>;

                // acess datatype used on device
                using DataBoxType = AtomicDataBox<
                    T_atomicNumber,
                    InternalDataBoxTypeValue,
                    InternalDataBoxTypeNumber,
                    InternalDataBoxTypeIdx,
                    T_ConfigNumberDataType>;

            private:
                // pointers to storage
                std::unique_ptr<BufferValue> dataStateEnergy; // unit: eV
                std::unique_ptr<BufferNumber> dataNumTransitions; // unit: unitless
                std::unique_ptr<BufferNumber> dataStartIndexBlockTransitions; // unit: unitless
                std::unique_ptr<BufferIdx> dataIdx; // unit: unitless


                std::unique_ptr<BufferValue> dataCollisionalOscillatorStrength; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx1; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx2; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx3; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx4; // unit: unitless
                std::unique_ptr<BufferValue> dataCinx5; // unit: unitless
                // std::unique_ptr<BufferIdx> dataLowerIdx; // unit: unitless
                std::unique_ptr<BufferIdx> dataUpperIdx; // unit: unitless

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
                    dataStateEnergy.reset(new BufferValue(layoutStates));
                    dataNumTransitions.reset(new BufferNumber(layoutStates));
                    dataStartIndexBlockTransitions.reset(new BufferNumber(layoutStates));
                    dataIdx.reset(new BufferIdx(layoutStates));

                    // transition data
                    dataCollisionalOscillatorStrength.reset(new BufferValue(layoutTransitions));
                    dataCinx1.reset(new BufferValue(layoutTransitions));
                    dataCinx2.reset(new BufferValue(layoutTransitions));
                    dataCinx3.reset(new BufferValue(layoutTransitions));
                    dataCinx4.reset(new BufferValue(layoutTransitions));
                    dataCinx5.reset(new BufferValue(layoutTransitions));
                    // dataLowerIdx.reset(new BufferIdx(layoutTransitions));
                    dataUpperIdx.reset(new BufferIdx(layoutTransitions));
                }

                //! Get the host data box for the rate matrix values
                HINLINE DataBoxType getHostDataBox(uint32_t numStates, uint32_t numTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getHostBuffer().getDataBox(),
                        dataNumTransitions->getHostBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getHostBuffer().getDataBox(),
                        dataIdx->getHostBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        // dataLowerIdx->getHostBuffer().getDataBox(),
                        dataUpperIdx->getHostBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                        dataCinx1->getHostBuffer().getDataBox(),
                        dataCinx2->getHostBuffer().getDataBox(),
                        dataCinx3->getHostBuffer().getDataBox(),
                        dataCinx4->getHostBuffer().getDataBox(),
                        dataCinx5->getHostBuffer().getDataBox(),
                        numTransitions,
                        this->m_maxNumberTransitions);
                }

                //! Get the device data box for the rate matrix values
                HINLINE DataBoxType getDeviceDataBox(uint32_t numStates, uint32_t numTransitions)
                {
                    return DataBoxType(
                        dataStateEnergy->getDeviceBuffer().getDataBox(),
                        dataNumTransitions->getDeviceBuffer().getDataBox(),
                        dataStartIndexBlockTransitions->getDeviceBuffer().getDataBox(),
                        dataIdx->getDeviceBuffer().getDataBox(),
                        numStates,
                        this->m_maxNumberStates,

                        // dataLowerIdx->getDeviceBuffer().getDataBox(),
                        dataUpperIdx->getDeviceBuffer().getDataBox(),
                        dataCollisionalOscillatorStrength->getHostBuffer().getDataBox(),
                        dataCinx1->getDeviceBuffer().getDataBox(),
                        dataCinx2->getDeviceBuffer().getDataBox(),
                        dataCinx3->getDeviceBuffer().getDataBox(),
                        dataCinx4->getDeviceBuffer().getDataBox(),
                        dataCinx5->getDeviceBuffer().getDataBox(),
                        numTransitions,
                        this->m_maxNumberTransitions);
                }

                void syncToDevice()
                {
                    dataStateEnergy->hostToDevice();
                    dataNumTransitions->hostToDevice();
                    dataStartIndexBlockTransitions->hostToDevice();
                    dataIdx->hostToDevice();

                    dataCollisionalOscillatorStrength->hostToDevice();
                    dataCinx1->hostToDevice();
                    dataCinx2->hostToDevice();
                    dataCinx3->hostToDevice();
                    dataCinx4->hostToDevice();
                    dataCinx5->hostToDevice();
                    // dataLowerIdx->hostToDevice();
                    dataUpperIdx->hostToDevice();
                }
            };

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
