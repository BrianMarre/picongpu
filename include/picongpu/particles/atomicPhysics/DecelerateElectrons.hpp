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

#include "picongpu/particles/atomicPhysics/GetRealKineticEnergy.hpp"

#include <cstdint>

// debug only
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics
        {
            template<typename T_Acc, typename T_Electron, typename T_Histogram, typename T_AtomicDataBox>
            DINLINE void processElectron(
                T_Acc const& acc,
                T_Electron electron,
                T_Histogram const& histogram,
                T_AtomicDataBox atomicDataBox)
            {
                // TODO: cchoose algorithm by particle? @BrianMarre, 2021
                float_64 const energyPhysicalElectron
                    = picongpu::particles::atomicPhysics::GetRealKineticEnergy::KineticEnergy(electron)
                    / picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: ATOMIC_UNIT_ENERGY
                float_X const weightMacroParticle = electron[weighting_]; // unit: internal

                // look up in the histogram, which bin corresponds to this energy
                uint16_t binIndex = histogram.getBinIndex(
                    acc,
                    energyPhysicalElectron, // unit: ATOMIC_UNIT_ENERGY
                    atomicDataBox);

                // case: electron missing from histogram due to not enough histogram
                // bins/too few intermediate bins
                if(binIndex == histogram.getMaxNumberBins())
                    return;

                float_X const weightBin = histogram.getWeightBin(binIndex); // unitless
                float_X const deltaEnergyBin = histogram.getDeltaEnergyBin(binIndex);
                // unit: ATOMIC_UNIT_ENERGY

                /// @TODO: create attribute functor for pyhsical particle properties?, @BrianMarre, 2021
                constexpr auto c_SI = picongpu::SI::SPEED_OF_LIGHT_SI; // unit: m/s, SI
                auto m_e_rel = attribute::getMass(1.0_X, electron) * picongpu::UNIT_MASS * c_SI * c_SI
                    / picongpu::SI::ATOMIC_UNIT_ENERGY; // unit: ATOMIC_UNIT_ENERGY

                // distribute energy change as mean by weight on all electrons in bin
                float_64 newEnergyPhysicalElectron = energyPhysicalElectron
                    + static_cast<float_64>(deltaEnergyBin / (picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE * weightBin));
                // unit:: ATOMIC_UNIT_ENERGY

                // case: too much energy removed
                if(newEnergyPhysicalElectron < 0)
                    newEnergyPhysicalElectron = 0._X; // extract as much as possible, rest should be neglible

                float_64 newPhysicalElectronMomentum
                    = math::sqrt(newEnergyPhysicalElectron * (newEnergyPhysicalElectron + 2 * m_e_rel))
                    / (picongpu::SI::ATOMIC_UNIT_ENERGY * c_SI);
                // AU = ATOMIC_UNIT_ENERGY
                // sqrt(AU * (AU + AU)) / (AU/J) / c = sqrt(AU^2)/(AU/J) / c = J/c = kg*m^2/s^2/(m/s)
                // unit: kg*m/s, SI

                float_X previousMomentumVectorLength = pmacc::math::abs2(electron[momentum_]);
                // unit: internal, scaled

                // case: not moving electron
                if(previousMomentumVectorLength == 0._X)
                    previousMomentumVectorLength = 1._X; // no need to resize 0-vector

                // debug only
                float_64 previousPhysicalElectronMomentumEnergy = math::sqrt(
                    energyPhysicalElectron * (energyPhysicalElectron + 2 *m_e_rel))

                // if previous momentum == 0, discard electron
                electron[momentum_] *= 1 / previousMomentumVectorLength // get unity vector of momentum
                    * (newPhysicalElectronMomentum * electron[weighting_] // new momentum scaled and in internal units
                       * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE
                       / (picongpu::UNIT_MASS * picongpu::UNIT_LENGTH / picongpu::UNIT_TIME));
                // unit: internal units

                // debug only
                std::cout << "momentumMacroParticle[internal] " << previousMomentumVectorLength
                    << " byEnergy " << math::sqrt(energyPhysicalElectron * (energyPhysicalElectron + 2* m_e_rel)) / c_SI
                    << std::endl;

                /*std::cout << "weightParticle/Bin " << weightMacroParticle/weightBin
                    << " energyPhysicalElectron[AU] " << energyPhysicalElectron
                    << " deltaEnergyBinPerPhysicalParticel[AU] " << deltaEnergyBin
                        /(weightBin * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                    << " deltaEnergyPhysicalParticle " << newEnergyPhysicalElectron - energyPhysicalElectron
                    //<< " previousPhysicalMomentum [SI] "
                    //<< previousMomentumVectorLength * (picongpu::UNIT_MASS * picongpu::UNIT_LENGTH / picongpu::UNIT_TIME)
                    //    /(weightMacroParticle * picongpu::particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE)
                    //<< " newPhysicalElectronMomentum [SI] " 
                    //<< newPhysicalElectronMomentum
                    << std::endl;*/
            }

            // Fill the histogram return via the last parameter
            // should be called inside the AtomicPhysicsKernel
            template<
                uint32_t T_numWorkers,
                typename T_Acc,
                typename T_Mapping,
                typename T_ElectronBox,
                typename T_Histogram,
                typename T_AtomicDataBox>
            DINLINE void decelerateElectrons(
                T_Acc const& acc,
                T_Mapping mapper,
                T_ElectronBox electronBox,
                T_Histogram const& histogram,
                T_AtomicDataBox atomicDataBox)
            {
                using namespace mappings::threads;

                //// todo: express framesize better, not via supercell size
                constexpr uint32_t frameSize = pmacc::math::CT::volume<SuperCellSize>::type::value;
                constexpr uint32_t numWorkers = T_numWorkers;
                using ParticleDomCfg = IdxConfig<frameSize, numWorkers>;

                uint32_t const workerIdx = cupla::threadIdx(acc).x;

                pmacc::DataSpace<simDim> const supercellIdx(
                    mapper.getSuperCellIndex(DataSpace<simDim>(cupla::blockIdx(acc))));

                ForEachIdx<IdxConfig<1, numWorkers>> onlyMaster{workerIdx};

                auto frame = electronBox.getLastFrame(supercellIdx);
                auto particlesInSuperCell = electronBox.getSuperCell(supercellIdx).getSizeLastFrame();

                // go over frames
                while(frame.isValid())
                {
                    // parallel loop over all particles in the frame
                    ForEachIdx<ParticleDomCfg>{workerIdx}([&](uint32_t const linearIdx, uint32_t const) {
                        // todo: check whether this if is necessary
                        if(linearIdx < particlesInSuperCell)
                        {
                            auto particle = frame[linearIdx];
                            processElectron(acc, particle, histogram, atomicDataBox);
                        }
                    });

                    frame = electronBox.getPreviousFrame(frame);
                    particlesInSuperCell = frameSize;
                }
            }

        } // namespace atomicPhysics
    } // namespace particles
} // namespace picongpu
