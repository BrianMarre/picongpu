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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/random/distributions/Uniform.hpp>


#include "picongpu/particles/particleToGrid/derivedAttributes/Density.hpp"

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
             * by atomicPhysics.kernel until time remaining becomes zero
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
             *  TODO: Refactor this to reduce complexity
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
                uint16_t loopCounter)
            {
                // case of no electrons in current super cell
                if(histogram->getNumBins() == 0)
                {
                    timeRemaining_SI = 0._X;
                    return;
                }

                // debug only
                // std::cout << "        process Ion" << std::endl;

                // workaround: the types may be obtained in a better fashion
                // TODO: relace with better version
                auto configNumber = ion[atomicConfigNumber_];
                using ConfigNumber = decltype(configNumber);
                using ConfigNumberDataType = decltype(ion[atomicConfigNumber_].getStateIndex()); // ? shorten

                // debug only
                // std::cout << "        got ion configNumber object" << std::endl;

                using AtomicRate = T_AtomicRate;

                ConfigNumberDataType oldState;
                ConfigNumberDataType newState;
                uint32_t newStateIndex;
                uint32_t oldStateIndex;
                uint32_t transitionIndex;

                float_X deltaEnergyTransition;
                uint16_t histogramIndex;
                float_X energyElectron;

                // conversion factors
                constexpr float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
                constexpr auto numCellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                // read out old state
                oldState = ion[atomicConfigNumber_].getStateIndex(); // config number
                oldStateIndex = atomicDataBox.findState(oldState); // collection index

                // choose random histogram collection index
                histogramIndex = static_cast<uint16_t>(randomGenInt()) % histogram->getNumBins();
                // get energy of histogram bin with this collection index
                energyElectron = histogram->getEnergyBin(
                    acc,
                    histogramIndex,
                    atomicDataBox); // unit: ATOMIC_UNIT_ENERGY

                // debug only
                // std::cout << "            start transition search, loop " << loopCounter << std::endl;

                uint16_t loopCounterTransitionSearch = 0u;
                bool transitionFound = false;

                // randomly select viable Transition
                while(true)
                {
                    // get a random new state index
                    newStateIndex = randomGenInt() % atomicDataBox.getNumStates();
                    newState = atomicDataBox.getAtomicStateConfigNumberIndex(newStateIndex);

                    // no change always viable transition
                    if(newState == oldState)
                        break;

                    // search for transition from oldState to newState
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

                        if(deltaEnergyTransition <= energyElectron)
                        {
                            break;
                        }
                        else
                        {
                            transitionFound = false;
                        }
                    }

                    // debug only
                    //std::cout << "    no valid transition" << std::endl;

                    // retry if no transition between states found
                    loopCounterTransitionSearch++;
                }

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
                    histogram->getInitialGridWidth(), // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox);

                // calculate density of electrons based on weight of electrons in this bin
                densityElectrons
                    = (histogram->getWeightBin(histogramIndex) + histogram->getDeltaWeightBin(histogramIndex))
                    * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                    / (numCellsPerSuperCell * picongpu::CELL_VOLUME * UNIT_VOLUME * energyElectronBinWidth);
                // (weighting * #/weighting) /
                //      ( # * Volume * m^3/Volume * AU )
                // = # / (m^3 * AU) => unit: 1/(m^3 * AU)

                // debug only
                // std::cout << "        densityElectrons " << densityElectrons << std::endl;

                // check for nan
                if(!(densityElectrons < 0) && !(densityElectrons >= 0))
                {
                    printf("ERROR: densityElectrons in rate solver is nan or inf\n");
                }

                // debug only
                // std::cout << "check if old == new" << std::endl;

                if(oldState == newState)
                {
                    // calculating quasiProbability for special case of keeping in current state

                    // R_(i->i) = - sum_f( R_(i->f), rate_SI = - R_(i->i),
                    // R ... rate, i ... initial state, f ... final state
                    rate_SI = AtomicRate::totalRate(
                        acc,
                        oldState, // unitless
                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                        energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                        densityElectrons, // unit: 1/(m^3*AU), SI
                        atomicDataBox); // unit: 1/s, SI

                    quasiProbability = 1._X - rate_SI * timeRemaining_SI;
                    deltaEnergy = 0._X;

                    // debug only
                    // std::cout << "  yes" << std::endl;
                }
                else
                {
                    // calculating quasiProbability for standard case of different newState

                    rate_SI = AtomicRate::Rate(
                        acc,
                        oldState, // unitless
                        newState, // unitless
                        transitionIndex, // unitless
                        energyElectron, // unit: ATOMIC_UNIT_ENERGY
                        energyElectronBinWidth, // unit: ATOMIC_UNIT_ENERGY
                        densityElectrons, // unit: 1/(m^3*ATOMIC_UNIT_ENERGY)
                        atomicDataBox); // unit: 1/s, SI

                    // get the change of electron energy in bin due to transition
                    deltaEnergy = (-deltaEnergyTransition) * ion[weighting_]
                        * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                    // unit: ATOMIC_UNIT_ENERGY, scaled with number of ions represented

                    quasiProbability = rate_SI * timeRemaining_SI;
                    // debug only
                    // std::cout << "  no" << std::endl;
                }

                float_X affectedWeighting = ion[weighting_];

                // debug only
                // std::cout << "call to processIon" << std::endl;
                /*std::cout << "loopCounter " << loopCounter << " timeRemaining " << timeRemaining_SI << " oldState "
                          << oldState << " newState " << newState << " energyElectron " << energyElectron
                          << " energyElectronBinWidth " << energyElectronBinWidth << " densityElectrons "
                          << densityElectrons << " histogramIndex " << histogramIndex << " quasiProbability "
                          << quasiProbability << " rateSI " << rate_SI << std::endl;*/

                if(quasiProbability >= 1.0_X)
                {
                    // case: more than one change per time remaining
                    // -> change once and reduce time remaining by mean time between such transitions
                    //  can only happen in the case of newState != olstate, since otherwise 1 - ( >0 ) < 1

                    // debug only
                    // std::cout << "    intermediate state" << std::endl;

                    // case: no transition possible, due to isolated atomic state
                    if(oldState == newState)
                    {
                        timeRemaining_SI = 0._X;
                        return;
                    }

                    // try to remove electrons from bin, returns false if not enough
                    // electrons in bin to interact with entire macro ion
                    bool sufficentElectronsInBin
                        = histogram->tryRemoveWeightFromBin(acc, histogramIndex, affectedWeighting);

                    if(!sufficentElectronsInBin)
                    {
                        affectedWeighting
                            = histogram->getWeightBin(histogramIndex) + histogram->getDeltaWeightBin(histogramIndex);
                        if(randomGenFloat() <= affectedWeighting / ion[weighting_])
                        {
                            histogram->removeWeightFromBin(acc, histogramIndex, affectedWeighting);
                            sufficentElectronsInBin = true;
                            deltaEnergy = deltaEnergy * affectedWeighting / ion[weighting_];
                        }
                    }

                    if(sufficentElectronsInBin)
                    {
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
                        bool sufficentElectronsInBin
                            = histogram->tryRemoveWeightFromBin(acc, histogramIndex, affectedWeighting);

                        if(!sufficentElectronsInBin)
                        {
                            affectedWeighting = histogram->getWeightBin(histogramIndex)
                                + histogram->getDeltaWeightBin(histogramIndex);
                            if(randomGenFloat() <= affectedWeighting / ion[weighting_])
                            {
                                histogram->removeWeightFromBin(acc, histogramIndex, affectedWeighting);
                                sufficentElectronsInBin = true;
                                deltaEnergy = deltaEnergy * affectedWeighting / ion[weighting_];
                            }
                        }

                        if(sufficentElectronsInBin)
                        {
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

                    // debug only
                    uint16_t loopCounter = 0u;

                    while(!allIonsProcessed)
                    {
                        onlyMaster([&](uint32_t const, uint32_t const) { allIonsProcessed = true; });

                        ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const idx) {
                            // debug only
                            // std::cout << "reset finish switch:loopCounter " << loopCounter << " timeRemaining "
                            //          << timeRemaining_SI[idx] << std::endl;
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
                                    loopCounter);

                                // debug only
                                loopCounter++;

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
