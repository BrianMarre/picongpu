/* Copyright 2022-2023 Brian Marre
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

/** @file implements as an object an evenly distributed log-space Histogram, starting from argument 0
 */

#pragma once

#include "picongpu/simulation_defines.hpp" // need: picongpu/param/atomicPhysics2_Debug.param

#include "picongpu/particles/atomicPhysics2/electronDistribution/HistogramInterface.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics2::electronDistribution
{
    /** @class histogram of logarithmically evenly distributed bins
     *
     * The histogram uses (T_numberBins)-bins, logarithmically evenly distributed,
     *  to cover the range [0,maxEnergy) and one additional special high energy
     *  overflow bin.
     *
     * For every regular bins, the original binned accumulated weight w0,
     *  the already by transition used weight DeltaW, the accumulated energy change
     *  through transitions DeltaE and whether the bin was over subscribed, w0 < DeltaW
     *  last time we checked is stored.
     *
     * For the overflow bin only the total weight outside the range is stored,
     *  since not atomic transitions may use this bin due to it's unknown energy.
     *
     * @tparam T_maxEnergy maximum energy of the range covered, > 0, [eV]:float_X but stored as T_Storage
     * @tparam T_numberBins number of bins, does not include the overflow bin, unitless
     * @tparam T_Storage storage type of T_maxEnergy
     */
    template<typename T_Storage, uint32_t T_numberBins, T_Storage T_maxEnergy>
    class LogSpaceHistogram : HistogramInterface
    {
    public:
        static constexpr float_X maxEnergy = static_cast<float_X>(T_maxEnergy);
        static constexpr uint32_t numberBins = T_numberBins;

    private:
        float_X binWeights0[T_numberBins] = {0};
        float_X binDeltaWeights[T_numberBins] = {0};
        float_X binDeltaEnergy[T_numberBins] = {0};

        float_X overFlowBinWeight = 0._X;

        //! debug only bin Index range checks
        HDINLINE static bool debugCheckBinIndexInRange(uint32_t const binIndex)
        {
            if(binIndex >= T_numberBins)
            {
                printf("atomicPhysics ERROR: binIndex >= T_numberBins\n");
                return false;
            }
            return true;
        }

        HDINLINE static float_X computeBase()
        {
            auto const tmp = maxEnergy;
            return math::pow(tmp, 1._X / static_cast<float_X>(T_numberBins - 1u));
        }

    public:
        /** get binIndex for a given energy
         *
         * @attention for energy > maxEnergy, returns binIndex >= T_numberBins,
         *  unless using debug compile mode, check for energy > maxEnergy externally
         *
         * @param energy energy, >= 0, < maxEnergy, [eV]
         * @return corresponding binIndex, unitless
         */
        HDINLINE uint32_t getBinIndex(float_X const energy) const
        {
            // negative energies are always wrong
            /// @todo remove, doubling up?, Brian Marre, 2022
            PMACC_ASSERT_MSG(energy >= 0._X, "energies must be >= 0");

            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
            {
                if(energy < 0._X)
                {
                    printf("atomicPhysics ERROR: energy < 0 in histogram getBinIndex() call\n");
                    return 0u;
                }
                if(energy >= maxEnergy)
                {
                    printf("atomicPhysics ERROR: energy > maxEnergy in histogram getBinIndex() call\n");
                    return 0u;
                }
            }

            if(energy >= 1._X)
            {
                // standard bin
                return static_cast<uint32_t>(math::log(energy) / math::log(computeBase())) + 1u;
            }
            else
                return 0u; // first bin
        }

        /** check whether the physical energy is <= maxEnergy
         *
         * @param energy [eV]
         */
        HDINLINE bool inRange(float_X const energy) const
        {
            if(energy >= maxEnergy)
            {
                return false;
            }

            return true;
        }

        // query state-methods
        /** get the central Energy for a given binIndex
         *
         * @attention no range check outside a debug compile, check range externally!
         *
         * @param binIndex ... bin index , unitless
         * @return central energy of bin, [eV]
         */
        HDINLINE float_X getBinEnergy(uint32_t const binIndex) const
        {
            // check binIndex Boundaries
            /// @todo remove, since already covered?, Brian Marre, 2022
            PMACC_DEVICE_ASSERT_MSG((binIndex < T_numberBins), "binIndex must be < T_numberBins");

            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return 0u;

            if(binIndex == 0u)
                return 1._X / 2._X; //[eV]

            return (math::pow(computeBase(), static_cast<float_X>(binIndex - 1u))
                    + math::pow(computeBase(), static_cast<float_X>(binIndex)))
                / 2._X; // [eV]
        }

        /** get bin width
         *
         * @attention no range checks outside a debug compile, check range externally!
         *
         * @param binIndex ... index of bin, >= 0, < T_numberBins, unitless
         * @return binWidth, [eV]
         */
        HDINLINE float_X getBinWidth(uint32_t const binIndex) const
        {
            if(binIndex == 0u)
                return 1._X; //[eV]
            return (
                math::pow(LogSpaceHistogram::computeBase(), static_cast<float_X>(binIndex))
                - math::pow(LogSpaceHistogram::computeBase(), static_cast<float_X>(binIndex - 1u))); // [eV]
        }

        /** get weight0 entry for given binIndex
         *
         * @attention no range checks outside a debug compile, check range externally!
         *
         * @param binIndex ... index of bin, >= 0, < T_numberBins, unitless
         * @return weight of binned macro-electrons, unmodified, unitless
         */
        HDINLINE float_X getBinWeight0(uint32_t const binIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return 0._X;

            return this->binWeights0[binIndex];
        }

        /** get DeltaWeight entry for given binIndex
         *
         * @attention no range checks outside a debug compile, check range externally!
         *
         * @param binIndex ... index of bin, >= 0, < T_numberBins, unitless
         * @return weight of bin used by accepted transitions, unitless
         */
        HDINLINE float_X getBinDeltaWeight(uint32_t const binIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return 0._X;

            return this->binDeltaWeights[binIndex];
        }

        /** get DeltaE entry for given binIndex
         *
         * @attention no range checks outside a debug compile, check range externally!
         *
         * @param binIndex ... index of bin, >= 0, < T_numberBins, unitless
         * @return change of energy in this bin due to accepted transitions, [eV]
         */
        HDINLINE float_X getBinDeltaEnergy(uint32_t const binIndex) const
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return 0._X;

            return this->binDeltaEnergy[binIndex];
        }

        /** get accumulated weight of all previously binned particles with an
         *  energy >= maxEnergy
         */
        HDINLINE float_X getOverflowWeight() const
        {
            return this->overFlowBinWeight;
        }

        // change state-methods
        /** bin the particle, add weight to w0 of the corresponding bin
         *
         * particles with an energy > T_maxEnergy are binned in the overflowBin
         *
         * @tparam T_Acc ... accelerator
         *
         * @param acc ... description of the device to execute this on
         * @param energy ... physical particle energy, [eV]
         * @param weight ... weight of the macroParticle, unitless
         */
        template<typename T_Worker>
        HDINLINE void binParticle(T_Worker const& worker, float_X const energy, float_X const weight)
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(energy < 0)
                {
                    printf("atomicPhysics ERROR: energy < 0 in histogram binParticle() call\n");
                    return;
                }

            // overflow bin
            if(!inRange(energy))
            {
                cupla::atomicAdd(worker.getAcc(), &(this->overFlowBinWeight), weight);
                return;
            }

            // regular bin
            uint32_t binIndex = getBinIndex(energy);

            cupla::atomicAdd(worker.getAcc(), &(this->binWeights0[binIndex]), weight);
            return;
        }

        /** add to the deltaWeight of a given bin
         *
         * @attention no range check outside of debug compile
         *
         * @tparam T_Acc ... accelerator type
         *
         * @param acc ... description of the device to execute this on
         * @param binIndex ... physical particle energy, unitless
         * @param weight ... weight of the macroParticle, unitless
         */
        template<typename T_Worker>
        HDINLINE void addDeltaWeight(T_Worker const& worker, uint32_t const binIndex, float_X const weight)
        {
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return;

            cupla::atomicAdd(worker.getAcc(), &(this->binDeltaWeights[binIndex]), weight);
            return;
        }

        /** add to the deltaEnergy of a given bin
         *
         * @attention no range check outside of debug compile
         *
         * @tparam T_Acc ... accelerator type
         *
         * @param acc ... description of the device to execute this on
         * @param binIndex ... physical particle energy, unitless
         * @param weight ... weight of the macroParticle, unitless
         */
        template<typename T_Worker>
        HDINLINE void addDeltaEnergy(T_Worker const& worker, uint32_t const binIndex, float_X const deltaEnergy)
        {
            // debug only
            if constexpr(picongpu::atomicPhysics2::ATOMIC_PHYSICS_HISTOGRAM_DEBUG)
                if(!debugCheckBinIndexInRange(binIndex))
                    return;

            cupla::atomicAdd(worker.getAcc(), &(this->binDeltaEnergy[binIndex]), deltaEnergy);
            return;
        }

        //! returns number of calls we need to make to reset the histogram
        HDINLINE static constexpr uint32_t getNumberResetOps()
        {
            return numberBins;
        }

        HDINLINE static float_X getBase()
        {
            return computeBase();
        }
    };
} // namespace picongpu::particles::atomicPhysics2::electronDistribution
