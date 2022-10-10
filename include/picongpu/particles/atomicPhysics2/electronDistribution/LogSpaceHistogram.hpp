/* Copyright 2022 Brian Marre
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics2/electronDistribution/HistogramInterface.hpp"

#include <cstdint>

// debug only
#include <iostream>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace electronDistribution
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
                    bool binOverSubscribed[T_numberBins];

                    float_X overFlowBinWeight = 0._X;

                    //! debug only bin Index range checks
                    static DINLINE bool debugCheckBinIndexInRange(uint32_t const binIndex)
                    {
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(binIndex < 0)
                        {
                            std::cout << "Error: binIndex < 0" << std::endl;
                            return false;
                        }
                        if(binIndex >= T_numberBins)
                        {
                            std::cout << "Error: binIndex >= T_numberBins" << std::endl;
                            return false;
                        }
                        return true;
                    }

                    static DINLINE float_X computeBase()
                    {
                        return math::pow(maxEnergy, 1._X / static_cast<float_X>(T_numberBins - 1u));
                    }

                    /** get binIndex for a given energy
                     *
                     * BEWARE: for energy > maxEnergy, returns binIndex >= T_numberBins,
                     *  unless using debug compile mode, check for energy > maxEnergy externally
                     *
                     * @param energy energy, >= 0, < maxEnergy, [eV]
                     * @return corresponding binIndex, unitless
                     */
                    static DINLINE uint32_t getBinIndex(float_X const energy)
                    {
                        // negative energies are always wrong
                        PMACC_ASSERT_MSG(energy >= 0, "energies must be >= 0");

                        // debug only
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(energy < 0._X)
                        {
                            std::cout << "Error: energy < 0" << std::endl;
                            return 0u;
                        }
                        if(energy >= maxEnergy)
                        {
                            std::cout << "Error: energy > maxEnergy" << std::endl;
                            return 0u;
                        }


                        if(energy >= 1._X)
                        {
                            // standard bin
                            return static_cast<uint32_t>(math::log(energy) / math::log( computeBase() )) + 1u;
                        }
                        else
                            return 0u; // first bin
                    }

                public:
                    /** check whether the physical energy is <= maxEnergy */
                    static DINLINE bool inRange(float_X const energy)
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
                     * BEWARE: does no range check outside a debug compile, check range externally!
                     *
                     * @param binIndex ... bin index , unitless
                     * @return central energy of bin[eV]
                     */
                    HDINLINE float_X getBinEnergy(uint32_t const binIndex) const final
                    {
                        // check binIndex Boundaries
                        PMACC_DEVICE_ASSERT_MSG(
                            (binIndex >= 0u) and (binIndex < T_numberBins),
                            "binIndex must be >= 0 < T_numberBins");

                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(not debugCheckBinIndexInRange(binIndex))
                            return 0u;

                        if(binIndex == 0u)
                            return 1._X / 2._X; //[eV]

                        return (math::pow(computeBase(), static_cast<float_X>(binIndex - 1u))
                                + math::pow(computeBase(), static_cast<float_X>(binIndex)))
                            / 2._X; // [eV]
                    }

                    /** get bin width
                     *
                     * @param binIndex ... index of bin, >= 0, < T_numberBins, unitless
                     *
                     * @return binWidth, [eV]
                     */
                    DINLINE float_X getBinWidth(uint32_t const binIndex) const final
                    {
                        if(binIndex == 0u)
                            return 1._X; //[eV]
                        return (
                            math::pow(computeBase(), static_cast<float_X>(binIndex))
                            - math::pow(computeBase(), static_cast<float_X>(binIndex - 1u))); // [eV]
                    }

                    /** get w0 entry for given binIndex
                     *
                     * BEWARE: does no range checks outside a debug compile
                     */
                    DINLINE float_X getBinWeight0(uint32_t const binIndex) const final
                    {
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(not debugCheckBinIndexInRange(binIndex))
                            return 0._X;

                        return this->binWeights0[binIndex];
                    }

                    /** get DeltaW entry for given binIndex
                     *
                     * BEWARE: does no range checks outside a debug compile
                     */
                    DINLINE float_X getBinDeltaWeight(uint32_t const binIndex) const final
                    {
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(not debugCheckBinIndexInRange(binIndex))
                            return 0._X;

                        return this->binDeltaWeights[binIndex];
                    }

                    /** get DeltaE entry for given binIndex
                     *
                     * BEWARE: does no range checks outside a debug compile
                     */
                    DINLINE float_X getBinDeltaEnergy(uint32_t const binIndex) const final
                    {
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(not debugCheckBinIndexInRange(binIndex))
                            return 0._X;

                        return this->binDeltaEnergy[binIndex];
                    }

                    /** is the bin marked as oversubscribed ?
                     *
                     * BEWARE: does no range checks outside a debug compile
                     */
                    DINLINE bool isBinOverSubscribed(uint32_t const binIndex) const final
                    {
                        /// @todo find correct debug compile guard, Brian Marre, 2022
                        if(not debugCheckBinIndexInRange(binIndex))
                            return false;

                        return this->binOverSubscribed[binIndex];
                    }

                    /** get accumulated weight of all previously binned particles with an
                     *  energy >= maxEnergy
                     */
                    DINLINE float_X getOverflowWeight() const
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
                    template<typename T_Acc>
                    DINLINE void binParticle(T_Acc const& acc, float_X const energy, float_X const weight)
                    {
                        // debug only
                        ///@todo find correct debug compile guard, Brian Marre, 2022
                        if(energy < 0)
                        {
                            std::cout << "Error: energy < 0" << std::endl;
                            return;
                        }

                        // overflow bin
                        if(not inRange(energy))
                        {
                            cupla::atomicAdd(acc, &(this->overFlowBinWeight), weight);
                            return;
                        }

                        // regular bin
                        uint32_t binIndex = getBinIndex(energy);

                        cupla::atomicAdd(acc, &(this->binWeights0[binIndex]), weight);
                        return;
                    }

                    /** add to the deltaWeight of a given bin
                     *
                     * BEWARE: does no range check outside of debug compile
                     *
                     * @tparam T_Acc ... accelerator type
                     *
                     * @param acc ... description of the device to execute this on
                     * @param binIndex ... physical particle energy, unitless
                     * @param weight ... weight of the macroParticle, unitless
                     */
                    template<typename T_Acc>
                    DINLINE void addDeltaWeight(T_Acc const& acc, uint32_t const binIndex, float_X const weight)
                    {
                        // debug only
                        if(not debugCheckBinIndexInRange(binIndex))
                            return;

                        cupla::atomicAdd(acc, &(this->binDeltaWeights[binIndex]), weight);
                        return;
                    }

                    /** add to the deltaEnergy of a given bin
                     *
                     * BEWARE: does no range check outside of debug compile
                     *
                     * @tparam T_Acc ... accelerator type
                     *
                     * @param acc ... description of the device to execute this on
                     * @param binIndex ... physical particle energy, unitless
                     * @param weight ... weight of the macroParticle, unitless
                     */
                    template<typename T_Acc>
                    DINLINE void addDeltaEnergy(T_Acc const& acc, uint32_t const binIndex, float_X const deltaEnergy)
                    {
                        // debug only
                        if(not debugCheckBinIndexInRange(binIndex))
                            return;

                        cupla::atomicAdd(acc, &(this->binDeltaEnergy[binIndex]), deltaEnergy);
                        return;
                    }

                    //! BEWARE: does not check binIndex range outside a debug compile
                    DINLINE void setOversubscribed(uint32_t const binIndex) final
                    {
                        // debug only
                        if(not debugCheckBinIndexInRange(binIndex))
                            return;

                        this->binOverSubscribed[binIndex] = true;
                    }

                    //! returns number of calls we need to make to reset the histogram
                    DINLINE static constexpr uint32_t getNumberResetOps()
                    {
                        return numberBins;
                    }

                    float_X getBase() const
                    {
                        return computeBase();
                    }
                };

            } // namespace electronDistribution
        } // namespace atomicPhysics2
    } // namespace particles
}// namespace picongpu
