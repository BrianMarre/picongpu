/* Copyright 2022 Brian Marre, Rene Widera
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

#include "picongpu/particles/atomicPhysics2/electronDistribution/LogSpaceHistogram.hpp"
#include "picongpu/particles/atomicPhysics2/atomicData/AtomicData.hpp"
#include "picongpu/particles/atomicPhysics2/stateRepresentation/ConfigNumber.hpp"


#include <cstdint>
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace debug
            {
                //! debug only, print content and bins of histogram to console, serial and cpu build only
                template<typename T_Histogram>
                void printHistogramToConsole(T_Histogram const& histogram)
                {
                    constexpr uint32_t numBins = T_Histogram::numberBins;

                    std::cout << "histogram: base=" << histogram.getBase();
                    std::cout << " numBins=" << T_Histogram::numberBins;
                    std::cout << " maxE=" << T_Histogram::maxEnergy << std::endl;

                    float_X centralEnergy;
                    float_X binWidth;

                    for(uint32_t i = 0u; i < numBins; i++)
                    {
                        // binIndex
                        std::cout << "\t "<< i;

                        // central bin energy [eV] and binWidth [eV]
                        centralEnergy = histogram.getBinEnergy(i);
                        binWidth = histogram.getBinWidth(i);

                        std::cout << "(" << centralEnergy - binWidth/2._X << ", "<<  centralEnergy + binWidth/2._X << "] :";

                        // bin data, [w0, DeltaW, DeltaEnergy, binOverSubscribed]
                        std::cout << " [w0, Dw, DE, o?]: [";
                        std::cout << histogram.getBinWeight0(i) << ", ";
                        std::cout << histogram.getBinDeltaWeight(i) << ", ";
                        std::cout << histogram.getBinDeltaEnergy(i) << ", ";
                        std::cout << histogram.isBinOverSubscribed(i) << "]";
                        std::cout << std::endl;
                    }
                    std::cout << "\t overFlow: w0=" << histogram.getOverflowWeight() << std::endl;
                }

                //! debug only, write atomic data to console, must be called serially and cpu build only
                template<typename T_AtomicData, bool T_printTransitionData, bool T_printInverseTransitions>
                std::unique_ptr<T_AtomicData> printAtomicDataToConsole(std::unique_ptr<T_AtomicData> atomicData)
                {
                    uint32_t numberAtomicStates = atomicData->getNumberAtomicStates();
                    uint32_t numberBoundBoundTransitions = atomicData->getNumberBoundBoundTransitions();
                    uint32_t numberBoundFreeTransitions = atomicData->getNumberBoundFreeTransitions();
                    uint32_t numberAutonomousTransitions = atomicData->getNumberAutonomousTransitions();

                    // basic numbers
                    std::cout << "AtomicNumber: " << static_cast<uint16_t>(T_AtomicData::atomicNumber) << "(#s "
                              << numberAtomicStates << ", #b " << numberBoundBoundTransitions << ", #f "
                              << numberBoundFreeTransitions << ", #a " << numberAutonomousTransitions << ")"
                              << std::endl;

                    // ChargeState data
                    auto chargeStateDataBox
                        = atomicData->template getChargeStateDataDataBox<true>(); // true: get hostDataBox
                    auto chargeStateOrgaBox = atomicData->template getChargeStateOrgaDataBox<true>();

                    std::cout << "ChargeState Data index : (E_ionization[eV], Z_screened[e]) [#AtomicStates, "
                                 "startIndexBlock]"
                              << std::endl;
                    for (uint8_t i = 0u; i < T_AtomicData::atomicNumber; i++)
                    {
                        std::cout << "\t" << static_cast<uint16_t>(i) << ":( "
                                  << chargeStateDataBox.ionizationEnergy(i) << ", "
                                  << chargeStateDataBox.screenedCharge(i) << " ) [ "
                                  << chargeStateOrgaBox.numberAtomicStates(i) << ", "
                                  << chargeStateOrgaBox.startIndexBlockAtomicStates(i) << " ]" << std::endl;
                    }
                    //      completely ionized state
                    std::cout << "\t" << static_cast<uint16_t>(T_AtomicData::atomicNumber) << ":( "
                                  << "na" << ", "
                                  << "na" << " ) [ "
                                  << chargeStateOrgaBox.numberAtomicStates(T_AtomicData::atomicNumber) << ", "
                                  << chargeStateOrgaBox.startIndexBlockAtomicStates(T_AtomicData::atomicNumber) << " ]" << std::endl;


                    // AtomicState data
                    auto atomicStateDataBox
                        = atomicData->template getAtomicStateDataDataBox<true>(); // true: get hostDataBox

                    auto boundBoundStartIndexBox = atomicData->template getBoundBoundStartIndexBlockDataBox<true>();
                    auto boundFreeStartIndexBox = atomicData->template getBoundFreeStartIndexBlockDataBox<true>();
                    auto autonomousStartIndexBox = atomicData->template getAutonomousStartIndexBlockDataBox<true>();

                    auto boundBoundNumberTransitionsBox
                        = atomicData->template getBoundBoundNumberTransitionsDataBox<true>();
                    auto boundFreeNumberTransitionsBox
                        = atomicData->template getBoundFreeNumberTransitionsDataBox<true>();
                    auto autonomousNumberTransitionsBox
                        = atomicData->template getAutonomousNumberTransitionsDataBox<true>();

                    auto transitionSelectionBox = atomicData->template getTransitionSelectionDataBox<true>();

                    using S_ConfigNumber = stateRepresentation::ConfigNumber<
                        uint64_t,
                        T_AtomicData::n_max,
                        T_AtomicData::atomicNumber>;

                    std::cout << "AtomicState Data -- index : [ConfigNumber, chargeState, levelVector]: E_overGround"
                              << std::endl;
                    std::cout << "\t b/f/a: [#TransitionsUp/]#TransitionsDown, [startIndexUp/]startIndexDown (offset)" << std::endl;
                    std::cout << "\t transitionSelectionData: #pyhsical transitions" << std::endl;
                    for(uint32_t i = 0u; i < numberAtomicStates; i++)
                    {
                        uint64_t stateConfigNumber = static_cast<uint64_t>(atomicStateDataBox.configNumber(i));

                        std::cout << "\t" << i << " : [" << stateConfigNumber << ", "
                                  << static_cast<uint16_t>(S_ConfigNumber::getIonizationState(stateConfigNumber))
                                  << ", ";

                        auto levelVector = S_ConfigNumber::getLevelVector(stateConfigNumber);
                        std::cout << "( ";
                        for (uint8_t j=0u; j < T_AtomicData::n_max; j++)
                        {
                            std::cout << static_cast<uint16_t>(levelVector[j]) << ", ";
                        }

                        std::cout << ")";
                        std::cout << "]: " << atomicStateDataBox.stateEnergy(i) << std::endl;
                        std::cout << "\t\t b: "
                            << boundBoundNumberTransitionsBox.numberOfTransitionsUp(i) << "/"
                            << boundBoundNumberTransitionsBox.numberOfTransitionsDown(i) << ", "
                            << boundBoundStartIndexBox.startIndexBlockTransitionsUp(i) << "/"
                            << boundBoundStartIndexBox.startIndexBlockTransitionsDown(i)
                            << " (" << boundBoundNumberTransitionsBox.offset(i) << ")" << std::endl;
                        std::cout << "\t\t f: "
                            << boundFreeNumberTransitionsBox.numberOfTransitionsUp(i) << "/"
                            << boundFreeNumberTransitionsBox.numberOfTransitionsDown(i) << ", "
                            << boundFreeStartIndexBox.startIndexBlockTransitionsUp(i) << "/"
                            << boundFreeStartIndexBox.startIndexBlockTransitionsDown(i)
                            << " (" << boundFreeNumberTransitionsBox.offset(i) << ")"  << std::endl;
                        std::cout << "\t\t a: "
                            << autonomousNumberTransitionsBox.numberOfTransitions(i) << ", "
                            << autonomousStartIndexBox.startIndexBlockTransitions(i)
                            << " (" << boundBoundNumberTransitionsBox.offset(i) << ")" << std::endl;
                        std::cout << "\t\t physical transitions: " << transitionSelectionBox.numberTransitions(i) << std::endl;
                    }

                    // transitionData
                    if constexpr (T_printTransitionData)
                    {
                        // bound-bound transitions
                        auto boundBoundTransitionDataBox = atomicData->template getBoundBoundTransitionDataBox<true>();
                        std::cout << "bound-bound transition" << std::endl;
                        std::cout << "index :(low, up), C: , A: \"Gaunt\"( <1>, <2>, ...)" << std::endl;
                        for(uint32_t i = 0; i < numberBoundBoundTransitions; i++)
                        {
                            std::cout << i << "("
                                << boundBoundTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << boundBoundTransitionDataBox.upperConfigNumberTransition(i) << ")"
                                << ", C: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                                << ", A: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i)
                                << " \"Gaunt\"( " << boundBoundTransitionDataBox.cxin1(i) << ", "
                                << boundBoundTransitionDataBox.cxin2(i) << ", "
                                << boundBoundTransitionDataBox.cxin3(i) << ", "
                                << boundBoundTransitionDataBox.cxin4(i) << ", "
                                << boundBoundTransitionDataBox.cxin5(i) << " )" << std::endl;
                        }

                        // bound-free transitions
                        auto boundFreeTransitionDataBox = atomicData->template getBoundFreeTransitionDataBox<true>();
                        std::cout << "bound-free transition" << std::endl;
                        std::cout << "index :(low, up), Coeff( <1>, <2>, ...)" << std::endl;
                        for(uint32_t i = 0; i < numberBoundFreeTransitions; i++)
                        {
                            std::cout << i << "("
                                << boundFreeTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << boundFreeTransitionDataBox.upperConfigNumberTransition(i) << ")"
                                << "Coeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                                << boundFreeTransitionDataBox.cxin2(i) << ", "
                                << boundFreeTransitionDataBox.cxin3(i) << ", "
                                << boundFreeTransitionDataBox.cxin4(i) << ", "
                                << boundFreeTransitionDataBox.cxin5(i) << ", "
                                << boundFreeTransitionDataBox.cxin6(i) << ", "
                                << boundFreeTransitionDataBox.cxin7(i) << ", "
                                << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
                        }

                        // autonomous transitions
                        auto autonomousTransitionDataBox = atomicData->template getAutonomousTransitionDataBox<true>();

                        std::cout << "autonomous transitions" << std::endl;
                        std::cout << "index :(low, up), rate" << std::endl;

                        for(uint32_t i = 0; i < numberAutonomousTransitions; i++)
                        {
                            std::cout << i << "("
                                << autonomousTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << autonomousTransitionDataBox.upperConfigNumberTransition(i) << ") "
                                << autonomousTransitionDataBox.rate(i)<< std::endl;
                        }
                    }

                    // inverse transitionData
                    if constexpr (T_printInverseTransitions)
                    {
                        // bound-bound transitions
                        auto boundBoundTransitionDataBox
                            = atomicData->template getInverseBoundBoundTransitionDataBox<true>();
                        std::cout << "inverse bound-bound transition" << std::endl;
                        std::cout << "index :(low, up), C: , A: \"Gaunt\"( <1>, <2>, ...)" << std::endl;
                        for(uint32_t i = 0; i < numberBoundBoundTransitions; i++)
                        {
                            std::cout << i << "("
                                << boundBoundTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << boundBoundTransitionDataBox.upperConfigNumberTransition(i) << ")"
                                << ", C: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                                << ", A: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i)
                                << "\"Gaunt\"(" << boundBoundTransitionDataBox.cxin1(i) << ", "
                                << boundBoundTransitionDataBox.cxin2(i) << ", "
                                << boundBoundTransitionDataBox.cxin3(i) << ", "
                                << boundBoundTransitionDataBox.cxin4(i) << ", "
                                << boundBoundTransitionDataBox.cxin5(i) << ")" << std::endl;
                        }

                        // bound-free transitions
                        auto boundFreeTransitionDataBox
                            = atomicData->template getInverseBoundFreeTransitionDataBox<true>();
                        std::cout << "inverse bound-free transition" << std::endl;
                        std::cout << "index :(low, up), Coeff( <1>, <2>, ...)" << std::endl;
                        for(uint32_t i = 0; i < numberBoundFreeTransitions; i++)
                        {
                            std::cout << i << "("
                                << boundFreeTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << boundFreeTransitionDataBox.upperConfigNumberTransition(i) << ")"
                                << "Coeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                                << boundFreeTransitionDataBox.cxin2(i) << ", "
                                << boundFreeTransitionDataBox.cxin3(i) << ", "
                                << boundFreeTransitionDataBox.cxin4(i) << ", "
                                << boundFreeTransitionDataBox.cxin5(i) << ", "
                                << boundFreeTransitionDataBox.cxin6(i) << ", "
                                << boundFreeTransitionDataBox.cxin7(i) << ", "
                                << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
                        }

                        // autonomous transitions
                        auto autonomousTransitionDataBox
                            = atomicData->template getInverseAutonomousTransitionDataBox<true>();

                        std::cout << "inverse autonomous transitions" << std::endl;
                        std::cout << "index :(low, up), rate" << std::endl;

                        for(uint32_t i = 0; i < numberAutonomousTransitions; i++)
                        {
                            std::cout << i << "("
                                << autonomousTransitionDataBox.lowerConfigNumberTransition(i) << ", "
                                << autonomousTransitionDataBox.upperConfigNumberTransition(i) << ") "
                                << autonomousTransitionDataBox.rate(i)<< std::endl;
                        }

                        return atomicData;
                    }
                }

            } // namespace debug
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
