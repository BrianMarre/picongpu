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

//@TODO: change normalisation, such that [simga] = UNIT_LENGTH^2 and [densities] = UNIT_LENGTH^3, Brian Marre 2021
//@TODO: change normailzation time remaining to UNIT_TIME, Brian Marre 2021

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>


#include "picongpu/particles/particleToGrid/derivedAttributes/Density.hpp"
#inlcude "picongpu/param/atomicPhysics.param"

#include <cstdint>

// debug only
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            // for now 32-bit hardcoded, should cover even the most extensive state and transition lists
            using DistributionInt = pmacc::random::distributions::Uniform<uint32_t>;
            using RngFactoryInt = particles::functor::misc::Rng<DistributionInt>;
            using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
            using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;
            using RandomGenInt = RngFactoryInt::RandomGen;
            using RandomGenFloat = RngFactoryFloat::RandomGen;

            /** actual rate equation solver
             *
             * \return updates value of timeRemaining_SI
             *
             * this method does one step of the rate solving algorithm, it is called
             * by atomicPhysics.kernel until time remaining of the current particle
             * becomes zero
             *
             * basic:
             *  1.) randomly choose viable transition
             *  2.) choose a random bin of energy histogram of electrons to interact with
             *  3.) calculate rate of change into this new state, with choosen electron energy
             *  3.) calculate the quasiProbability = rate * dt
             *  4.) if (quasiProbability > 1):
             *      - try to update histogram
             *      - change ion atomic state to new state
             *      - reduce time by 1/rate, mean time between such changes
             *     else:
             *      if ( quasiProbability < 0 && oldState == newState ):
             *          - must choose different state, try again
             *      else:
             *          - decide randomly with quasiProbability if change to new state
             *          if we change state:
             *              - change ion state
             *
             * @TODO: Refactor needed to reduce complexity, Brian Marre 2021
             */
            template<
                typename T_AtomicRate,
                typename T_Acc,
                typename T_Ion,
                typename T_AtomicDataBox,
                typename T_Histogram>
            DINLINE void processIon(
                T_Acc const& acc,
                RandomGenInt& randomGenInt,
                RandomGenFloat& randomGenFloat,
                T_Ion ion,
                float_X& timeRemaining_SI,
                T_AtomicDataBox const atomicDataBox,
                T_Histogram* histogram,
                uint16_t globalLoopCounter,
                uint16_t localLoopCounter)
            {
                // case of no electrons in current super cell
                if(histogram->getNumBins() == 0)
                {
                    // debug only
                    // std::cout << "          no electrons present" << std::endl;
                    printf("         no electrons present in one super cell\n");

                    timeRemaining_SI = 0._X;
                    return;
                }

                // workaround: the types may be obtained in a better fashion
                // TODO: replace with better version
                auto configNumber = ion[atomicConfigNumber_];
                using ConfigNumber = decltype(configNumber);
                using ConfigNumberDataType = decltype(ion[atomicConfigNumber_].getStateIndex()); // @TODO: ? shorten

                using AtomicRate = T_AtomicRate;

                ConfigNumberDataType oldState;
                ConfigNumberDataType newState;
                uint32_t newStateIndex;
                uint32_t oldStateIndex;
                uint32_t transitionIndex;

                uint8_t processIndex;

                float_X deltaEnergyTransition;
                uint16_t histogramIndex;
                float_X energyElectron;

                // conversion factors
                constexpr float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
                constexpr auto numCellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                // read out old state
                oldState = ion[atomicConfigNumber_].getStateIndex(); // config number
                oldStateIndex = atomicDataBox.findState(oldState); // collection index of atomic state

                // debug only
                // std::cout << "            start transition search, local loop " << localLoopCounter << std::endl;

                uint16_t loopCounterTransitionSearch = 0u;
                bool transitionFound = false;

                // randomly select viable Transition
                while(true)
                {
                    // get a random new state index
                    newStateIndex = randomGenInt() % atomicDataBox.getNumStates();
                    newState = atomicDataBox.getAtomicStateConfigNumberIndex(newStateIndex);

                    // get random process,
                    // @TODO get available processes from species on compile time
                    // 1: free electron interaction, 2: +photonic spontanous deexcitation
                    process = randomGentInt() % picongpu::atomicPhysics::numProcesses;

                    // choose random histogram collection index
                    histogramIndex = static_cast<uint16_t>(randomGenInt()) % histogram->getNumBins();
                    // get energy of histogram bin with this collection index
                    energyElectron = histogram->getEnergyBin(
                        acc,
                        histogramIndex,
                        atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                    // no change always viable transition, @TODO: circle transitions?, BrianMarre, 2021
                    if(newState == oldState)
                        break;

                    // search for transition from oldState to newState, BEWARE: first arg. is list index, second is
                    // state index
                    transitionIndex = atomicDataBox.findTransitionInBlock(oldStateIndex, newState);

                    // found transition?
                    if(transitionIndex != atomicDataBox.getNumTransitions())
                    {
                        transitionFound = true;
                    }
                    else
                    {
                        // search for Transition to oldState from newState
                        transitionIndex = atomicDataBox.findTransitionInBlock(newStateIndex, oldState);

                        // found transition?
                        if(transitionIndex != atomicDataBox.getNumTransitions())
                        {
                            transitionFound = true;
                        }
                    }

                    // debug only
                    /*std::cout << "    loopCount " << loopCounterTransitionSearch << " oldState " << oldState
                        << " newState " << newState << " transitionFound? " << transitionFound
                        << " transitionIndex " << transitionIndex<< std::endl;*/

                    if(transitionFound)
                    {
                        // check wether transition is actually possible with choosen energy bin
                        deltaEnergyTransition = AtomicRate::energyDifference(acc, oldState, newState, atomicDataBox);
                        // unit: ATOMIC_UNIT_ENERGY

                        // free electron interaction
                        if(process == 0u)
                        {
                            if(deltaEnergyTransition <= energyElectron)
                            {
                                freeElectronInteraction<T_AtomicRate>(
                                    acc,
                                    randomGenFloat,
                                    ion,
                                    timeRemaining_SI, // unit: s, SI
                                    atomicDataBox,
                                    histogram,
                                    oldStateIndex,
                                    oldState, // unit: untiless
                                    newStateIndex,
                                    newState, // unit: unitless
                                    transitionIndex,
                                    histogramIndex,
                                    energyElectron, // unit: ATOMIC_UNIT_ENERGY
                                    deltaEnergyTransition, // unit: ATOMIC_UNIT_ENERGY
                                    globalLoopCounter,
                                    localLoopCounter);
                                break;
                            }
                            else
                            {
                                transitionFound = false;
                            }
                        }
                        // spontaneous photon emission
                        else if(process == 1u)
                        {
                            if(deltaEnergyTransition <= 0)
                            {
                                // call spontaneous photon emission function
                                break;
                            }
                            else
                            {
                                transitionFound = false;
                            }
                        }
                    }

                    // debug only
                    /*std::cout << "    no valid transition" << std::endl;*/

                    // retry if no transition between states found
                    loopCounterTransitionSearch++;
                }
            }

            template<
                typename T_AtomicRate,
                typename T_Acc,
                typename T_Ion,
                typename T_ConfigNumberDataType,
                typename T_AtomicDataBox,
                typename T_Histogram>
            DINLINE void freeElectronInteraction(
                T_Acc const& acc,
                RandomGenFloat& randomGenFloat,
                T_Ion ion,
                float_X& timeRemaining_SI, // unit: s, SI
                T_AtomicDataBox const atomicDataBox,
                T_Histogram* histogram,
                uint32_t oldStateIndex,
                T_ConfigNumberDataType oldState, // unit: unitless
                uint32_t newStateIndex,
                T_ConfigNumberDataType newState, // unit: unitless
                uint32_t transitionIndex,
                uint16_t histogramIndex,
                float_X energyElectron, // unit: ATOMIC_UNIT_ENERGY
                float_X deltaEnergyTransition, // unit: ATOMIC_UNIT_ENERGY
                uint16_t globalLoopCounter,
                uint16_t localLoopCounter)
            {
                //{ preparation
                float_X energyElectronBinWidth;

                float_X rate_SI;
                float_X quasiProbability;
                float_X deltaEnergy;

                float_X densityElectrons;

                // get width of histogram bin with this collection index
                energyElectronBinWidth = histogram->getBinWidth(
                    acc,
                    true, // answer to question: directionPositive?
                    histogram->getLeftBoundaryBin(histogramIndex), // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                // calculate density of electrons based on weight of electrons in this bin
                // REMEMBER: histogram is filled by direct add of particle[weighting_]
                // and weighting_ is "number of real particles"
                densityElectrons
                    = (histogram->getWeightBin(histogramIndex) + histogram->getDeltaWeightBin(histogramIndex))
                    / (numCellsPerSuperCell * picongpu::CELL_VOLUME * UNIT_VOLUME * energyElectronBinWidth);
                // # / ( # * Volume * m^3/Volume * AU )
                // = # / (m^3 * AU) => unit: 1/(m^3 * AU)

                // debug only
                // std::cout << "        densityElectrons " << densityElectrons << std::endl;

                //{ check for numerical problems in density Electrons
                // check for nan
                if(!(densityElectrons == densityElectrons))
                {
                    printf("ERROR: densityElectrons in rate solver is nan\n");
                    // debug only output
                    /*std::cout << "histogramIndex " << histogramIndex << " WeightBin "
                              << histogram->getWeightBin(histogramIndex) << " DeltaWeight "
                              << histogram->getDeltaWeightBin(histogramIndex) << " binWidth " << energyElectronBinWidth
                              << std::endl;*/
                }
                // check for inf
                if(densityElectrons == 1.0 / 0.0)
                {

                    printf("ERROR: densityElectrons in rate solver is +-inf\n");
                    // debug only output
                    /*std::cout << "histogramIndex " << histogramIndex << " WeightBin "
                              << histogram->getWeightBin(histogramIndex) << " DeltaWeight "
                              << histogram->getDeltaWeightBin(histogramIndex) << " binWidth " << energyElectronBinWidth
                              << std::endl;*/
                }
                //}

                // calculating quasiProbability for special case of keeping in current state
                if(oldState == newState)
                {
                    // R_(i->i) = - sum_f( R_(i->f), rate_SI = - R_(i->i),
                    // R ... rate, i ... initial state, f ... final state
                    rate_SI = AtomicRate::totalRate(
                        acc,
                        oldState, // unitless
                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                        energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                        densityElectrons, // unit: 1/(m^3*AU), SI
                        atomicDataBox); // unit: 1/s, SI @TODO: update total rate calculation

                    quasiProbability = 1._X - rate_SI * timeRemaining_SI;
                    deltaEnergy = 0._X;
                    // std::cout << "old state == new state " << (oldState==newState) << std::endl;
                }
                else
                {
                    // calculating quasiProbability for standard case of different newState

                    rate_SI = AtomicRate::RateFreeElectronInteraction(
                        acc,
                        oldState, // unitless
                        newState, // unitless
                        transitionIndex, // unitless
                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                        energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                        densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                        atomicDataBox); // unit: 1/s, SI

                    // get the change of electron energy in bin due to transition
                    deltaEnergy = (-deltaEnergyTransition) * ion[weighting_];
                    // unit: ATOMIC_UNIT_ENERGY, scaled with number of ions represented

                    quasiProbability = rate_SI * timeRemaining_SI;
                }

                float_X affectedWeighting = ion[weighting_];

                // debug only
                /*std::cout << "particleID " << ion[particleId_] << " globalLoopCounter " << globalLoopCounter
                    << " localLoopCounter " << localLoopCounter
                    << " timeRemaining " << timeRemaining_SI << " oldState "
                    << oldState << " newState " << newState << " energyElectron " << energyElectron
                    << " energyElectronBinWidth " << energyElectronBinWidth << " densityElectrons "
                    << densityElectrons << " histogramIndex " << histogramIndex << " quasiProbability "
                    << quasiProbability << " rateSI " << rate_SI << std::endl;*/
                //} end preparation

                rateEquationSolverForExternalSpectrum(
                    acc,
                    ion,
                    timeRemaining_SI,
                    histogram,
                    histogramIndex,
                    atomicDataBox,
                    energyElectron,
                    quasiProbability,
                    rate_SI,
                    deltaEnergyTransition,
                    deltaEnergy,
                    affectedWeighting,
                    oldState,
                    newState)
            }

            template<typename T_Acc, typename T_Ion, typename T_ConfigNumberDataType, typename T_AtomicDataBox>
            DINLINE void rateEquationSolverForExternalSpectrum(
                T_Acc const& acc,
                RandomGenFloat& randomGenFloat,
                T_Ion ion,
                float_X& timeRemaining_SI,
                T_Histogram* histogram,
                uint16_t histogramIndex,
                T_AtomicDataBox const atomicDataBox,
                float_X energyElectron,
                float_X quasiProbability,
                float_X rate_SI,
                float_X deltaEnergyTransition,
                float_X deltaEnergy,
                float_X affectedWeighting,
                T_ConfigNumberDataType oldState,
                T_ConfigNumberDataType newState)
            {
                if(quasiProbability >= 1.0_X)
                {
                    // case: more than one change per time remaining
                    // -> change once and reduce time remaining by mean time between such transitions
                    //  can only happen in the case of newState != oldstate, since otherwise 1 - ( >0 ) < 1
                    //  unless current state is isolated state

                    // debug only
                    // std::cout << "    intermediate state" << std::endl;

                    // case: no transition possible, due to isolated atomic state
                    if(oldState == newState)
                    {
                        // debug only
                        /*std::cout << "      no transition possible " << std::endl;*/
                        timeRemaining_SI = 0._X;
                        return;
                    }

                    // try to remove electrons from bin, returns false if not enough
                    // electrons in bin to interact with entire macro ion
                    bool sufficentWeightInBin
                        = histogram->tryRemoveWeightFromBin(acc, histogramIndex, affectedWeighting);

                    // do state change randomly even if not enough weight, Note: might be problematic for very large
                    // number of workers?
                    if(!sufficentWeightInBin)
                    {
                        affectedWeighting
                            = histogram->getWeightBin(histogramIndex) + histogram->getDeltaWeightBin(histogramIndex);
                        if(randomGenFloat() <= affectedWeighting / ion[weighting_])
                        {
                            histogram->removeWeightFromBin(acc, histogramIndex, affectedWeighting);
                            sufficentWeightInBin = true;
                            deltaEnergy = deltaEnergy * affectedWeighting / ion[weighting_];
                        }
                    }

                    if(sufficentWeightInBin)
                    {
                        // debug only
                        // std::cout << "  change accepted" << std::endl;

                        ion[atomicConfigNumber_] = newState;

                        // record change of energy in bin in original bin
                        histogram->addDeltaEnergy(acc, histogramIndex, deltaEnergy);
                        // shift weight of interaction electron to new bin
                        // for further interactions
                        histogram->shiftWeight(
                            acc,
                            energyElectron - deltaEnergyTransition, // new electron energy, unit: ATOMIC_UNIT_ENERGY
                            affectedWeighting,
                            atomicDataBox);

                        // reduce time remaining by mean time between interactions
                        timeRemaining_SI -= 1.0_X / rate_SI;

                        // safeguard against numerical error
                        if(rate_SI < 0)
                        {
                            // case: timeRemaining < 0: should not happen
                            // last resort to avoid infinte loop
                            timeRemaining_SI = 0._X;
                            printf("ERROR: time remaining < 0, in rate solver");
                        }
                    }
                }
                else
                {
                    if(quasiProbability < 0._X)
                    {
                        // case: newState != oldState
                        // quasiProbability can only be > 0, since AtomicRate::Rate( )>=0
                        // and timeRemaining >= 0
                        if(oldState != newState)
                        {
                            timeRemaining_SI = 0._X;
                            printf("ERROR: negative time remaining encountered in rate solver");
                        }

                        // debug only
                        // std::cout << "    unphysical rate" << std::endl;

                        // case: newState == oldState
                        // on average change from original state into new more than once
                        // in timeRemaining
                        // => can not remain in current state -> must choose new state
                    }
                    else if(randomGenFloat() <= quasiProbability)
                    {
                        // case change only possible once
                        // => randomly change to newState in time remaining

                        // try to remove weight from eectron bin, to cover entire macro ion
                        bool sufficentWeightInBin
                            = histogram->tryRemoveWeightFromBin(acc, histogramIndex, affectedWeighting);

                        if(!sufficentWeightInBin)
                        {
                            affectedWeighting = histogram->getWeightBin(histogramIndex)
                                + histogram->getDeltaWeightBin(histogramIndex);
                            if(randomGenFloat() <= affectedWeighting / ion[weighting_])
                            {
                                histogram->removeWeightFromBin(acc, histogramIndex, affectedWeighting);
                                sufficentWeightInBin = true;
                                deltaEnergy = deltaEnergy * affectedWeighting / ion[weighting_];
                            }
                        }

                        if(sufficentWeightInBin)
                        {
                            // debug only
                            // std::cout << "  change accepted" << std::endl;

                            // change ion state
                            ion[atomicConfigNumber_] = newState;

                            // record change of energy in bin in original bin
                            histogram->addDeltaEnergy(acc, histogramIndex, deltaEnergy);
                            // shift weight of interaction electron to new bin
                            // for further interactions
                            histogram->shiftWeight(
                                acc,
                                energyElectron
                                    - deltaEnergyTransition, // new electron energy, unit: ATOMIC_UNIT_ENERGY
                                affectedWeighting,
                                atomicDataBox);

                            // complete timeRemaining is used up
                            timeRemaining_SI = 0.0_X;
                        }

                        // debug only
                        // std::cout << "    final state" << std::endl;
                    }
                }
            }

            template<typename T_Acc, typename T_Ion, typename T_ConfigNumberDataType>
            DINLINE void rateEquationSolverSpontaneousTransition(
                T_Acc const& acc,
                RandomGenFloat& randomGenFloat,
                T_Ion ion,
                float_X& timeRemaining_SI,
                float_X quasiProbability,
                float_X rate_SI,
                float_X deltaEnergyTransition,
                float_X affectedWeighting,
                T_ConfigNumberDataType oldState,
                T_ConfigNumberDataType newState)
            {
                if(quasiProbability >= 1.0_X)
                {
                    // case: more than one change per time remaining
                    // -> change once and reduce time remaining by mean time between such transitions
                    //  can only happen in the case of newState != oldstate, since otherwise 1 - ( >0 ) < 1
                    //  unless current state is isolated state

                    // debug only
                    // std::cout << "    intermediate state" << std::endl;

                    // case: no transition possible, due to isolated atomic state
                    if(oldState == newState)
                    {
                        // debug only
                        /*std::cout << "      no transition possible " << std::endl;*/
                        timeRemaining_SI = 0._X;
                        return;
                    }

                    // debug only
                    // std::cout << "  change accepted" << std::endl;

                    ion[atomicConfigNumber_] = newState;

                    // @TODO: spawn photon with frequency deltaEnergyTransition and affectedWeight

                    // reduce time remaining by mean time between interactions
                    timeRemaining_SI -= 1.0_X / rate_SI;

                    // safeguard against numerical error
                    if(rate_SI < 0)
                    {
                        // case: timeRemaining < 0: should not happen
                        // last resort to avoid infinte loop
                        timeRemaining_SI = 0._X;
                        printf("ERROR: time remaining < 0, in rate solver");
                    }
                }
                else
                {
                    if(quasiProbability < 0._X)
                    {
                        // case: newState != oldState
                        // quasiProbability can only be > 0, since AtomicRate::Rate( )>=0
                        // and timeRemaining >= 0
                        if(oldState != newState)
                        {
                            timeRemaining_SI = 0._X;
                            printf("ERROR: negative time remaining encountered in rate solver");
                        }

                        // debug only
                        // std::cout << "    unphysical rate" << std::endl;

                        // case: newState == oldState
                        // on average change from original state into new more than once
                        // in timeRemaining
                        // => can not remain in current state -> must choose new state
                    }
                    else if(randomGenFloat() <= quasiProbability)
                    {
                        // case change only possible once
                        // => randomly change to newState in time remaining

                        // debug only
                        // std::cout << "  change accepted" << std::endl;

                        // change ion state
                        ion[atomicConfigNumber_] = newState;

                        // @TODO: spawn photon with frequency deltaEnergyTransition and affectedWeight

                        // complete timeRemaining is used up
                        timeRemaining_SI = 0.0_X;

                        // debug only
                        // std::cout << "    final state" << std::endl;
                    }
                }
            }

            template<
                typename T_AtomicRate,
                typename T_Acc,
                typename T_Ion,
                typename T_ConfigNumberDataType,
                typename T_AtomicDataBox,
                typename T_Histogram>
            DINLINE void spontaneousPhotonEmission(
                T_Acc const& acc,
                RandomGenFloat& randomGenFloat,
                T_Ion ion,
                float_X& timeRemaining_SI,
                T_AtomicDataBox const atomicDataBox,
                T_Histogram* histogram,
                uint32_t oldStateIndex,
                T_ConfigNumberDataType oldState,
                uint32_t newStateIndex,
                T_ConfigNumberDataType newState,
                uint32_t transitionIndex,
                float_X deltaEnergyTransition,
                uint16_t globalLoopCounter,
                uint16_t localLoopCounter)
            {
                float_X rate_SI;
                float_X quasiProbability;
                float_X deltaEnergy;

                // calculating quasiProbability for special case of keeping in current state, @TODO: think about that
                // more
                if(oldState == newState)
                {
                    // R_(i->i) = - sum_f( R_(i->f), rate_SI = - R_(i->i),
                    // R ... rate, i ... initial state, f ... final state
                    /*rate_SI = AtomicRate::totalRateSpontaneousPhotonEmission(
                        acc,
                        oldState, // unitless
                        atomicDataBox); // unit: 1/s, SI @TODO: update total rate calculation

                    quasiProbability = 1._X - rate_SI * timeRemaining_SI;
                    deltaEnergy = 0._X;*/
                    // std::cout << "old state == new state " << (oldState==newState) << std::endl;
                }
                else
                {
                    // calculating quasiProbability for standard case of different newState

                    rate_SI = AtomicRate::RateSpontaneousPhotonEmission(
                        acc,
                        oldState, // unitless
                        newState, // unitless
                        transitionIndex, // unitless
                        atomicDataBox); // unit: 1/s, SI

                    // get the change of electron energy in bin due to transition
                    deltaEnergy = (-deltaEnergyTransition) * ion[weighting_];
                    // unit: ATOMIC_UNIT_ENERGY, scaled with number of ions represented

                    quasiProbability = rate_SI * timeRemaining_SI;
                }

                float_X affectedWeighting = ion[weighting_];

                // debug only
                /*std::cout << "particleID " << ion[particleId_] << " globalLoopCounter " << globalLoopCounter
                    << " localLoopCounter " << localLoopCounter
                    << " timeRemaining " << timeRemaining_SI << " oldState "
                    << oldState << " newState " << newState << " energyElectron " << energyElectron
                    << " energyElectronBinWidth " << energyElectronBinWidth << " densityElectrons "
                    << densityElectrons << " histogramIndex " << histogramIndex << " quasiProbability "
                    << quasiProbability << " rateSI " << rate_SI << std::endl;*/
                //} end preparation

                //= as function in atomic rate (2 * pi * e^2)/(eps_0 * m_e * c^3) * g_new/g_old * faax

                rateEquationSolverSpontaneousTransition(
                    acc,
                    randomGenFloat,
                    ion,
                    quasiProbability,
                    rate_SI,
                    deltaEnergyTransition,
                    affectedWeighting,
                    oldState,
                    newState)
            }

            // Fill the histogram return via the last parameter
            // should be called inside the AtomicPhysicsKernel
            template<
                uint32_t T_numWorkers,
                typename T_AtomicRate,
                typename T_Acc,
                typename T_Mapping,
                typename T_IonBox,
                typename T_AtomicDataBox,
                typename T_Histogram>
            DINLINE void solveRateEquation(
                T_Acc const& acc,
                T_Mapping mapper,
                RngFactoryInt rngFactoryInt,
                RngFactoryFloat rngFactoryFloat,
                T_IonBox ionBox,
                T_AtomicDataBox const atomicDataBox,
                T_Histogram* histogram,
                bool debug)
            {
                using namespace mappings::threads;

                //// todo: express framesize better, not via supercell size
                constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorkers = T_numWorkers;
                using ParticleDomCfg = IdxConfig<frameSize, numWorkers>;

                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                pmacc::DataSpace<simDim> const supercellIdx(
                    mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));

                auto frame = ionBox.getLastFrame(supercellIdx);
                auto particlesInSuperCell = ionBox.getSuperCell(supercellIdx).getSizeLastFrame();

                // Offset without guards for random numbers
                auto const supercellLocalOffset = supercellIdx - mapper.getGuardingSuperCells();
                pmacc::mappings::threads::WorkerCfg<numWorkers> workerCfg(workerIdx);

                auto generatorInt = rngFactoryInt(acc, supercellLocalOffset, workerCfg);
                auto generatorFloat = rngFactoryFloat(acc, supercellLocalOffset, workerCfg);

                ForEachIdx<IdxConfig<1, numWorkers>> onlyMaster{workerIdx};

                // debug only
                // std::cout << "start rate application" << std::endl;

                // go over frames
                while(frame.isValid())
                {
                    // debug only
                    // std::cout << "  frame processing" << std::endl;

                    // all Ions of current frame processed
                    PMACC_SMEM(acc, allIonsProcessed, bool);
                    // init
                    onlyMaster([&](uint32_t const, uint32_t const) { allIonsProcessed = false; });

                    // create one instance of timeRemaining for each virtual worker and init with
                    // remaining time to pic time step at the beginning
                    memory::CtxArray<float_X, ParticleDomCfg> timeRemaining_SI(picongpu::SI::DELTA_T_SI);

                    // local loop counter variable
                    memory::CtxArray<uint16_t, ParticleDomCfg> localLoopCounter(0);

                    // debug only
                    uint16_t globalLoopCounter = 0u;

                    while(!allIonsProcessed)
                    {
                        onlyMaster([&](uint32_t const, uint32_t const) { allIonsProcessed = true; });

                        ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const idx) {
                            // debug only
                            /*std::cout << "reset finish switch:loopCounter " << globalLoopCounter << " timeRemaining "
                                     << timeRemaining_SI[idx] << std::endl;*/
                            if((linearIdx < particlesInSuperCell) && (timeRemaining_SI[idx] > 0._X))
                            {
                                auto particle = frame[linearIdx];
                                processIon<T_AtomicRate>(
                                    acc,
                                    generatorInt,
                                    generatorFloat,
                                    particle,
                                    timeRemaining_SI[idx],
                                    atomicDataBox,
                                    histogram,
                                    globalLoopCounter,
                                    localLoopCounter[idx]);

                                // debug only
                                globalLoopCounter++;
                                localLoopCounter[idx]++;

                                if(timeRemaining_SI[idx] > 0._X)
                                {
                                    allIonsProcessed = false;
                                }
                            }
                        });

                        cupla::__syncthreads(acc);

                        onlyMaster([&](uint32_t const, uint32_t const) { histogram->updateWithNewShiftBins(); });

                        cupla::__syncthreads(acc);
                    }


                    // get the next frmae once done with the current one.
                    frame = ionBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
