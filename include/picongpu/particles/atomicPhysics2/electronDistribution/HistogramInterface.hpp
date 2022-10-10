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

/** @file implements the interface all histograms must follow
 *
 * @attention some functions in this file are commented out, these functions are nervertheless required!
 */

#pragma once

#include <cstdint>

namespace picongpu
{
    namespace particles
    {
        namespace atomicPhysics2
        {
            namespace electronDistribution
            {
                class HistogramInterface
                {
                    //! get the central Energy for a given binIndex
                    virtual float_X getBinEnergy(uint32_t const binIndex) const = 0;

                    virtual float_X getBinWidth(uint32_t const binIndex) const = 0;

                    //! get weight of initially binned particles
                    virtual float_X getBinWeight0(uint32_t const binIndex) const = 0;

                    //! get reserved/already used weight of bin
                    virtual float_X getBinDeltaWeight(uint32_t const binIndex) const = 0;

                    //! get deltaEnergy of Bin
                    virtual float_X getBinDeltaEnergy(uint32_t const binIndex) const = 0;

                    //! was deltaWeight > weight0 ? on previous check?
                    virtual bool isBinOverSubscribed(uint32_t const binIndex) const = 0;

                    /* bin the particle, add weight to w0 of the corresponding bin
                     *
                     * @tparam T_Acc ... accelerator
                     *
                     * @param acc ... description of the device to execute this on
                     * @param energy ... physical particle energy, [eV]
                     * @param weight ... weight of the macroParticle, unitless
                     */
                    // template<typename T_Acc>
                    // virtual void binParticle(T_Acc const& acc, float_X const energy, float_X const weight) = 0;

                    /* add to the deltaWeight of a given bin
                     *
                     * @tparam T_Acc ... accelerator type
                     *
                     * @param acc ... description of the device to execute this on
                     * @param binIndex ... physical particle energy, unitless
                     * @param weight ... weight of the macroParticle, unitless
                     */
                    // template<typename T_Acc>
                    // virtual void addDeltaWeight(T_Acc const& acc, uint32_t const binIndex, float_X const weight) =
                    // 0;

                    /* add to the deltaEnergy of a given bin
                     *
                     * @tparam T_Acc ... accelerator type
                     *
                     * @param acc ... description of the device to execute this on
                     * @param binIndex ... physical particle energy, unitless
                     * @param weight ... weight of the macroParticle, unitless
                     */
                    // template<typename T_Acc>
                    // void addDeltaEnergy(T_Acc const& acc, uint32_t const binIndex, float_X const deltaEnergy) = 0;

                    virtual void setOversubscribed(uint32_t const binIndex) = 0;

                    //! returns number of calls we need to make to reset the histogram
                    // virtual static constexpr uint32_t getNumberResetOps() = 0;
                };
            } // namespace electronDistribution
        } // namespace atomicPhysics2
    } // namespace particles
} // namespace picongpu
