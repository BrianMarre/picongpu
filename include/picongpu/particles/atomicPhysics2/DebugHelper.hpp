/* Copyright 2022-2023 Brian Marre, Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/particles/atomicPhysics2/atomicData/GetStateFromTransitionTuple.hpp"
#include "picongpu/particles/atomicPhysics2/enums/TransitionOrdering.hpp"

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>

#include <cstdint>
#include <iostream>
#include <string>

namespace picongpu::particles::atomicPhysics2::debug
{
    namespace enums = picongpu::particles::atomicPhysics2::enums;

    //! debug only, write atomic data to console, @attention serially and cpu build only
    template<typename T_AtomicData, bool T_printTransitionData, bool T_printInverseTransitions>
    std::unique_ptr<T_AtomicData> printAtomicDataToConsole(std::unique_ptr<T_AtomicData> atomicData)
    {
        std::cout << std::endl << "**AtomicData DEBUG Output**" << std::endl;

        // process configuration
        std::cout << "process configuration:" << std::endl;
        std::cout << "\t Electronic Excitation:    " << ((T_AtomicData::switchElectronicExcitation) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Electronic DeExcitation:  "
                  << ((T_AtomicData::switchElectronicDeexcitation) ? "true" : "false") << std::endl;
        std::cout << "\t Spontaneous DeExcitation: "
                  << ((T_AtomicData::switchSpontaneousDeexcitation) ? "true" : "false") << std::endl;
        std::cout << "\t Electronic Ionization:    " << ((T_AtomicData::switchElectronicIonization) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Autonomous ionization:    " << ((T_AtomicData::switchAutonomousIonization) ? "true" : "false")
                  << std::endl;
        std::cout << "\t Field Ionization:         " << ((T_AtomicData::switchFieldIonization) ? "true" : "false")
                  << std::endl;

        uint32_t numberAtomicStates = atomicData->getNumberAtomicStates();
        uint32_t numberBoundBoundTransitions = atomicData->getNumberBoundBoundTransitions();
        uint32_t numberBoundFreeTransitions = atomicData->getNumberBoundFreeTransitions();
        uint32_t numberAutonomousTransitions = atomicData->getNumberAutonomousTransitions();

        // basic numbers
        std::cout << "Basic Statistics:" << std::endl;
        std::cout << "AtomicNumber: " << static_cast<uint16_t>(T_AtomicData::ConfigNumber::atomicNumber) << "(#s "
                  << numberAtomicStates << ", #b " << numberBoundBoundTransitions << ", #f "
                  << numberBoundFreeTransitions << ", #a " << numberAutonomousTransitions << ")" << std::endl;

        // chargeState data
        auto chargeStateDataBox = atomicData->template getChargeStateDataDataBox<true>(); // true: get hostDataBox
        auto chargeStateOrgaBox = atomicData->template getChargeStateOrgaDataBox<true>();

        std::cout << "ChargeState Data" << std::endl;
        std::cout << "index : (E_ionization[eV], Z_screened[e]) [#AtomicStates, startIndexBlock]" << std::endl;
        for(uint8_t i = 0u; i < T_AtomicData::ConfigNumber::atomicNumber; i++)
        {
            std::cout << "\t" << static_cast<uint16_t>(i) << ":( " << chargeStateDataBox.ionizationEnergy(i) << ", "
                      << chargeStateDataBox.screenedCharge(i) << " ) [ " << chargeStateOrgaBox.numberAtomicStates(i)
                      << ", " << chargeStateOrgaBox.startIndexBlockAtomicStates(i) << " ]" << std::endl;
        }

        //      completely ionized state
        std::cout << "\t" << static_cast<uint16_t>(T_AtomicData::ConfigNumber::atomicNumber) << ":( "
                  << "na"
                  << ", "
                  << "na"
                  << " ) [ " << chargeStateOrgaBox.numberAtomicStates(T_AtomicData::ConfigNumber::atomicNumber) << ", "
                  << chargeStateOrgaBox.startIndexBlockAtomicStates(T_AtomicData::ConfigNumber::atomicNumber) << " ]"
                  << std::endl;


        // AtomicState data
        auto atomicStateDataBox = atomicData->template getAtomicStateDataDataBox<true>(); // true: get hostDataBox
        auto pressureIonizationStateDataBox = atomicData->template getPressureIonizationStateDataBox<true>();

        auto boundBoundStartIndexBox = atomicData->template getBoundBoundStartIndexBlockDataBox<true>();
        auto boundFreeStartIndexBox = atomicData->template getBoundFreeStartIndexBlockDataBox<true>();
        auto autonomousStartIndexBox = atomicData->template getAutonomousStartIndexBlockDataBox<true>();

        auto boundBoundNumberTransitionsBox = atomicData->template getBoundBoundNumberTransitionsDataBox<true>();
        auto boundFreeNumberTransitionsBox = atomicData->template getBoundFreeNumberTransitionsDataBox<true>();
        auto autonomousNumberTransitionsBox = atomicData->template getAutonomousNumberTransitionsDataBox<true>();

        using S_ConfigNumber = stateRepresentation::
            ConfigNumber<uint64_t, T_AtomicData::ConfigNumber::numberLevels, T_AtomicData::ConfigNumber::atomicNumber>;

        // state data
        std::cout << "AtomicState Data" << std::endl;
        std::cout << "index : [ConfigNumber, chargeState, levelVector]: E_overGround, PressureIonizationState"
                  << std::endl;
        std::cout << "\t b/f/a: [#TransitionsUp/]#TransitionsDown, [startIndexUp/]startIndexDown" << std::endl;
        for(uint32_t stateCollectionIndex = 0u; stateCollectionIndex < numberAtomicStates; stateCollectionIndex++)
        {
            uint64_t const stateConfigNumber
                = static_cast<uint64_t>(atomicStateDataBox.configNumber(stateCollectionIndex));
            auto const stateLevelVector = S_ConfigNumber::getLevelVector(stateConfigNumber);
            auto const pressureIonizationStateCollectionIndex
                = pressureIonizationStateDataBox.pressureIonizationState(stateCollectionIndex);
            auto const levelVectorPressureIonizationState = S_ConfigNumber::getLevelVector(
                atomicStateDataBox.configNumber(pressureIonizationStateCollectionIndex));

            std::cout << "\t" << stateCollectionIndex << " : [" << stateConfigNumber << ", "
                      << static_cast<uint16_t>(S_ConfigNumber::getChargeState(stateConfigNumber)) << ", "
                      << precisionCast<uint16_t>(stateLevelVector).toString(",", "()")
                      << "]: " << atomicStateDataBox.energy(stateCollectionIndex) << ",\t"
                      << precisionCast<uint16_t>(levelVectorPressureIonizationState).toString(",", "()") << std::endl;
            std::cout << "\t\t b: " << boundBoundNumberTransitionsBox.numberOfTransitionsUp(stateCollectionIndex)
                      << "/" << boundBoundNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex) << ", "
                      << boundBoundStartIndexBox.startIndexBlockTransitionsUp(stateCollectionIndex) << "/"
                      << boundBoundStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex) << std::endl;
            std::cout << "\t\t f: " << boundFreeNumberTransitionsBox.numberOfTransitionsUp(stateCollectionIndex) << "/"
                      << boundFreeNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex) << ", "
                      << boundFreeStartIndexBox.startIndexBlockTransitionsUp(stateCollectionIndex) << "/"
                      << boundFreeStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex) << std::endl;
            std::cout << "\t\t a: " << autonomousNumberTransitionsBox.numberOfTransitionsDown(stateCollectionIndex)
                      << ", " << autonomousStartIndexBox.startIndexBlockTransitionsDown(stateCollectionIndex)
                      << std::endl;
        }

        // transitionData
        if constexpr(T_printTransitionData)
        {
            // bound-bound transitions
            auto boundBoundTransitionDataBox
                = atomicData->template getBoundBoundTransitionDataBox<true, enums::TransitionOrdering::byLowerState>();
            std::cout << "bound-bound transition" << std::endl;
            std::cout << "index :(low, up), dE, C, A, \"Gaunt\"( <1>, <2>, ...)" << std::endl;
            for(uint32_t i = 0; i < numberBoundBoundTransitions; i++)
            {
                std::cout << i << "\t(" << boundBoundTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundBoundTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(i, atomicStateDataBox, boundBoundTransitionDataBox)
                          << ",\tC: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                          << ",\tA: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i) << "\t\"Gaunt\"( "
                          << boundBoundTransitionDataBox.cxin1(i) << ", " << boundBoundTransitionDataBox.cxin2(i)
                          << ", " << boundBoundTransitionDataBox.cxin3(i) << ", "
                          << boundBoundTransitionDataBox.cxin4(i) << ", " << boundBoundTransitionDataBox.cxin5(i)
                          << " )" << std::endl;
            }

            // bound-free transitions
            auto boundFreeTransitionDataBox
                = atomicData->template getBoundFreeTransitionDataBox<true, enums::TransitionOrdering::byLowerState>();
            std::cout << "bound-free transition" << std::endl;
            std::cout << "index (low, up), dE, Coeff( <1>, <2>, ...)" << std::endl;
            for(uint32_t i = 0; i < numberBoundFreeTransitions; i++)
            {
                std::cout << i << "\t(" << boundFreeTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundFreeTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 boundFreeTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << "\tCoeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                          << boundFreeTransitionDataBox.cxin2(i) << ", " << boundFreeTransitionDataBox.cxin3(i) << ", "
                          << boundFreeTransitionDataBox.cxin4(i) << ", " << boundFreeTransitionDataBox.cxin5(i) << ", "
                          << boundFreeTransitionDataBox.cxin6(i) << ", " << boundFreeTransitionDataBox.cxin7(i) << ", "
                          << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
            }

            // autonomous transitions
            auto autonomousTransitionDataBox
                = atomicData->template getAutonomousTransitionDataBox<true, enums::TransitionOrdering::byLowerState>();

            std::cout << "autonomous transitions" << std::endl;
            std::cout << "index :(low, up), dE, rate [1/Dt_PIC]" << std::endl;

            for(uint32_t i = 0; i < numberAutonomousTransitions; i++)
            {
                std::cout << i << "\t(" << autonomousTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << autonomousTransitionDataBox.upperStateCollectionIndex(i) << ") "
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 autonomousTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << ",\trate: " << autonomousTransitionDataBox.rate(i) << std::endl;
            }
            std::cout << std::endl;
        }

        // inverse transitionData
        if constexpr(T_printInverseTransitions)
        {
            // bound-bound transitions
            auto boundBoundTransitionDataBox
                = atomicData->template getBoundBoundTransitionDataBox<true, enums::TransitionOrdering::byUpperState>();
            std::cout << "inverse bound-bound transition" << std::endl;
            std::cout << "index :(low, up), dE, C, A, \"Gaunt\"( <1>, <2>, ...)" << std::endl;
            for(uint32_t i = 0; i < numberBoundBoundTransitions; i++)
            {
                std::cout << i << "\t(" << boundBoundTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundBoundTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(i, atomicStateDataBox, boundBoundTransitionDataBox)
                          << ",\tC: " << boundBoundTransitionDataBox.collisionalOscillatorStrength(i)
                          << ",\tA: " << boundBoundTransitionDataBox.absorptionOscillatorStrength(i) << "\t\"Gaunt\"("
                          << boundBoundTransitionDataBox.cxin1(i) << ", " << boundBoundTransitionDataBox.cxin2(i)
                          << ", " << boundBoundTransitionDataBox.cxin3(i) << ", "
                          << boundBoundTransitionDataBox.cxin4(i) << ", " << boundBoundTransitionDataBox.cxin5(i)
                          << ")" << std::endl;
            }

            // bound-free transitions
            auto boundFreeTransitionDataBox
                = atomicData->template getBoundFreeTransitionDataBox<true, enums::TransitionOrdering::byUpperState>();
            std::cout << "inverse bound-free transition" << std::endl;
            std::cout << "index :(low, up), dE, Coeff( <1>, <2>, ...)" << std::endl;
            for(uint32_t i = 0; i < numberBoundFreeTransitions; i++)
            {
                std::cout << i << "\t(" << boundFreeTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << boundFreeTransitionDataBox.upperStateCollectionIndex(i) << ")"
                          << ",\tdE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 boundFreeTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << "\tCoeff(" << boundFreeTransitionDataBox.cxin1(i) << ", "
                          << boundFreeTransitionDataBox.cxin2(i) << ", " << boundFreeTransitionDataBox.cxin3(i) << ", "
                          << boundFreeTransitionDataBox.cxin4(i) << ", " << boundFreeTransitionDataBox.cxin5(i) << ", "
                          << boundFreeTransitionDataBox.cxin6(i) << ", " << boundFreeTransitionDataBox.cxin7(i) << ", "
                          << boundFreeTransitionDataBox.cxin8(i) << ")" << std::endl;
            }

            // autonomous transitions
            auto autonomousTransitionDataBox
                = atomicData->template getAutonomousTransitionDataBox<true, enums::TransitionOrdering::byUpperState>();

            std::cout << "inverse autonomous transitions" << std::endl;
            std::cout << "index :(low, up), dE, rate" << std::endl;

            for(uint32_t i = 0; i < numberAutonomousTransitions; i++)
            {
                std::cout << i << "\t(" << autonomousTransitionDataBox.lowerStateCollectionIndex(i) << ", "
                          << autonomousTransitionDataBox.upperStateCollectionIndex(i) << ") "
                          << ", dE: "
                          << DeltaEnergyTransition::get(
                                 i,
                                 atomicStateDataBox,
                                 autonomousTransitionDataBox,
                                 // eV, ionization potential depression is dynamics dependent, therefore set to zero
                                 // for debug
                                 0._X,
                                 chargeStateDataBox)
                          << ", rate: " << autonomousTransitionDataBox.rate(i) << std::endl;
            }
            std::cout << std::endl;
        }
        return atomicData;
    }

    //! debug only, write atomicPhysics attributes to console, @attention serial and cpu build only
    template<typename T_Ion>
    void printAtomicPhysicsIonToConsole(T_Ion const& ion)
    {
        std::cout << "ID: " << ion[particleId_] << std::endl;
        std::cout << "\t - weighting: " << ion[weighting_] << std::endl;

        std::cout << "\t - momentum: (" << ion[momentum_].toString(",", "") << ")" << std::endl;
        std::cout << "\t - position: (" << ion[position_].toString(",", "") << ")" << std::endl;
        std::cout << "\t - atomicPhysicsData:" << std::endl;
        std::cout << "\t\t - atomicStateCollectionIndex: " << ion[atomicStateCollectionIndex_] << std::endl;
        std::cout << "\t\t - processClass: " << static_cast<uint16_t>(ion[processClass_]) << std::endl;
        std::cout << "\t\t - transitionIndex: " << ion[transitionIndex_] << std::endl;
        std::cout << "\t\t - binIndex: " << ion[binIndex_] << std::endl;
        std::cout << "\t\t - accepted: " << ((ion[accepted_]) ? "true" : "false") << std::endl;
        std::cout << "\t\t - boundElectrons: " << ion[boundElectrons_] << std::endl;
    }

    //! debug only. write transition tuple to console, @attention serial and cpu build only
    template<typename T_Tuple, typename T_Idx, typename T_Value, typename T_ConfigNumber>
    void printTransitionTupleToConsole(T_Tuple const& tuple)
    {
        T_Idx const upperAtomicState
            = picongpu::particles::atomicPhysics2::atomicData::getUpperStateConfigNumber<T_Idx, T_Value>(tuple);
        T_Idx const lowerAtomicState
            = picongpu::particles::atomicPhysics2::atomicData::getLowerStateConfigNumber<T_Idx, T_Value>(tuple);
        uint8_t const upperChargeState = T_ConfigNumber::getChargeState(upperAtomicState);
        uint8_t const lowerChargeState = T_ConfigNumber::getChargeState(lowerAtomicState);

        std::cout << "State : " << static_cast<uint16_t>(lowerChargeState) << ", " << lowerAtomicState << ", "
                  << static_cast<uint16_t>(upperChargeState) << ", " << upperAtomicState << std::endl;
    }
} // namespace picongpu::particles::atomicPhysics2::debug
