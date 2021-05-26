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
            // for now 32-bit hardcoded, that should be
            // underlying type of config number
            using DistributionInt = pmacc::random::distributions::Uniform<uint32_t>;
            using RngFactoryInt = particles::functor::misc::Rng<DistributionInt>;
            using DistributionFloat = pmacc::random::distributions::Uniform<float_X>;
            using RngFactoryFloat = particles::functor::misc::Rng<DistributionFloat>;
            using RandomGenInt = RngFactoryInt::RandomGen;
            using RandomGenFloat = RngFactoryFloat::RandomGen;

            /** actual rate equation solver
             *
             * basic steps:
             * so long as time is remaining
             *  1.) choose a new state by choosing a random integer in the atomic state index
             *      range.
             *  2.) choose a random bin of energy histogram of electrons to interact with
             *  3.) calculate rate of change into this new state, with choosen electron energy
             *  3.) calculate the quasiProbability = rate * dt
             *  4.) if (quasiProbability > 1):
             *      - change ion atomic state to new state
             *      - reduce time by 1/rate, mean time between such changes
             *      - start again at 1.)
             *     else:
             *      if ( quasiProbability < 0 ):
             *          - start again at 1.)
             *      else:
             *          - decide randomly with quasiProbability if change to new state
             *          if we change state:
             *              - change ion state
             *  5.) finish
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
                T_AtomicDataBox const atomicDataBox,
                T_Histogram* histogram,
                bool debug)
            {
                // case of no electrons in current super cell
                if(histogram->getNumBins() == 0)
                    return;

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

                uint16_t histogramIndex;
                float_X energyElectron;
                float_X energyElectronBinWidth;

                // conversion factors
                constexpr float_64 UNIT_VOLUME = UNIT_LENGTH * UNIT_LENGTH * UNIT_LENGTH;
                constexpr auto numCellsPerSuperCell = pmacc::math::CT::volume<SuperCellSize>::type::value;

                float_X densityElectrons;

                float_X rate_SI;
                float_X deltaEnergyTransition;
                float_X deltaEnergy;
                float_X quasiProbability;

                // debug only
                uint16_t loopCounter = 0u;

                // set remaining time to pic time step at the beginning
                float_X timeRemaining_SI = picongpu::SI::DELTA_T_SI;

                // debug only
                // std::cout << "            start state change loop" << std::endl;

                while(timeRemaining_SI > 0.0_X)
                {
                    // read out old state index
                    oldState = ion[atomicConfigNumber_].getStateIndex();
                    // get collection index of old State
                    oldStateIndex = atomicDataBox.findState(oldState);

                    // randomly select viable Transition
                    while(true)
                    {
                        // get a random new state index
                        newStateIndex = randomGenInt() % atomicDataBox.getNumStates();
                        newState = atomicDataBox.getAtomicStateConfigNumberIndex(newStateIndex);

                        // debug only
                        /*std::cout << "    RateSolver: oldState " << oldState
                                  << " newState " << newState << " numTransitions "
                                  << atomicDataBox.getNumTransitions() << std::endl;*/

                        if(newState == oldState)
                            break;

                        // search for transition
                        transitionIndex = atomicDataBox.findTransitionInBlock(oldStateIndex, newState);

                        // debug only
                        // std::cout << "    RateSolver_1: transitionIndex " << transitionIndex << std::endl;

                        // found transition?
                        if(transitionIndex != atomicDataBox.getNumTransitions())
                        {
                            break;
                        }

                        // search for reverse Transition
                        transitionIndex = atomicDataBox.findTransitionInBlock(newStateIndex, oldState);

                        // debug only
                        // std::cout << "    RateSolver_2: transitionIndex " << transitionIndex << std::endl;

                        // found transition?
                        if(transitionIndex != atomicDataBox.getNumTransitions())
                        {
                            break;
                        }
                        // retry if no transition between states found
                    }

                    deltaEnergyTransition = AtomicRate::energyDifference(acc, oldIdx, newIdx, atomicDataBox);

                    // choose random histogram collection index
                    histogramIndex = static_cast<uint16_t>(randomGenInt()) % histogram->getNumBins();
                    // get energy of histogram bin with this collection index
                    energyElectron = histogram->getEnergyBin(
                        acc,
                        histogramIndex,
                        atomicDataBox); // unit: ATOMIC_UNIT_ENERGY
                    // get width of histogram bin with this collection index

                    if(deltaEnergyTransition > energyElectron)
                    {
                        // electrons of bin lack sufficient energy for choosen transition
                        // => need to choose new transition
                        continue;
                    }

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
                    //      ( numCellsPerSuperCell * Volume * m^3/Volume * AU )
                    // = # / (m^3 * AU) => unit: 1/(m^3 * AU)

                    // check for nan or inf
                    if(densityElectrons + 1 == densityElectrons)
                    {
                        printf("ERROR: densityElectrons in rate solver is nan or inf");
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
                            densityElectrons, // unit: 1/(m^3*J), SI
                            atomicDataBox); // unit: 1/s, SI

                        // get the change of electron energy
                        deltaEnergy = (-deltaEnergyTransition) * ion[weighting_]
                            * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE;
                        // unit: ATOMIC_UNIT_ENERGY

                        quasiProbability = rate_SI * timeRemaining_SI;
                        // debug only
                        // std::cout << "  no" << std::endl;
                    }

                    // debug only
                    if(debug)
                    {
                        // std::cout << "ping" << std::endl;
                        /*std::cout << "loopCounter " << loopCounter << " timeRemaining " << timeRemaining_SI
                                  << " oldState " << oldState << " newState " << newState << " energyElectron "
                                  << energyElectron << " energyElectronBinWidth " << energyElectronBinWidth
                                  << " densityElectrons " << densityElectrons << " histogramIndex " << histogramIndex
                                  << " quasiProbability " << quasiProbability << " rateSI " << rate_SI << std::endl;*/
                    }

                    if(quasiProbability >= 1.0_X)
                    {
                        // case: more than one change per time remaining
                        // -> change once and reduce time remaining by mean time between such transitions
                        //  can only happen in the case of newState != olstate, since otherwise 1 - ( >0 ) < 1

                        // debug only
                        // std::cout << "    intermediate state" << std::endl;

                        // record energy removed or added to electrons
                        bool sufficentElectronsInBin = histogram->tryRemoveWeightFromBin(
                            acc,
                            histogramIndex, // unitless
                            ion[weighting_] // unit: ATOMIC_UNIT_ENERGY
                        );

                        if(sufficentElectronsInBin)
                        {
                            // change atomic state of ion
                            ion[atomicConfigNumber_] = newState;

                            histogram->addDeltaEnergy(acc, histogramIndex, deltaEnergy);

                            newElectronBin = histogram->get

                            histogram->addDeltaWeight(
                                acc,
                                
                                

                            // reduce time remaining
                            timeRemaining_SI -= 1.0_X / rate_SI;


                            if(rate_SI < 0)
                                // case timeRemaining < 0: shoule dnot happen
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
                            // quasiProbability can only be > 0, since AtomicRate::Rate( )>0
                            // and timeRemaining > 0

                            // case: newState == oldState
                            // on average change from original state into new more than once
                            // in timeRemaining
                            // => can not remain in current state -> must choose new state

                            // debug only
                            // std::cout << "    unphysical rate" << std::endl;
                        }
                        else if(randomGenFloat() /*rand() / float_X(RAND_MAX)*/ <= quasiProbability)
                        {
                            // case change only possible once
                            // => randomly change to newState in time remaining

                            // record energy removed or added to electrons
                            bool sufficentEnergyInBin = histogram->tryAddEnergyToBin(
                                acc,
                                histogramIndex, // unitless
                                deltaEnergy // unit: ATOMIC_UNIT_ENERGY
                            );

                            if(sufficentEnergyInBin)
                            {
                                // change ion state
                                ion[atomicConfigNumber_] = newState;

                                // complete timeRemaining used
                                timeRemaining_SI = 0.0_X;
                            }

                            // debug only
                            // std::cout << "    final state" << std::endl;
                        }
                    }
                    loopCounter++;
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

                // go over frames
                while(frame.isValid())
                {
                    // parallel loop over all particles in the frame
                    ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                        // todo: check whether this if is necessary
                        if(linearIdx < particlesInSuperCell)
                        {
                            auto particle = frame[linearIdx];
                            processIon<T_AtomicRate>( // doOneStep
                                acc,
                                generatorInt,
                                generatorFloat,
                                particle,
                                atomicDataBox,
                                histogram,
                                debug);
                        }
                    });

                    // cupla::__syncthreads(acc);

                    frame = ionBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
