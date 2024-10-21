/* Copyright 2024 Brian Marre
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

// need simDim from dimension.param and SuperCellSize from memory.param
#include "picongpu/defines.hpp"

#include <pmacc/attribute/unroll.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    /**
     *
     * @param T_Extent pmacc compile time vector describing extent of cache in number of cell
     * @param T_StorageType type to use for storage
     */
    template<typename T_Extent, typename T_StorageType>
    struct FieldEnergyUseCache
    {
        using Extent = T_Extent;
        using StorageType = T_StorageType;

        constexpr uint32_t dim = Extent::dim;
        constexpr uint32_t numberCells = pmacc::math::CT::volume<typename T_Extent>::type::value;

        using CellIdx = pmacc::DataSpace<dim>;

    private:
        // eV
        StorageType fieldEnergyUsed[numberCells] = {0._X};

        /** get linear storage index
         *
         * @param cellIdx, vector index of cell
         *
         * @returns linear storage index corresponding to this cell
         */
        static constexpr uint32_t linearIndex(CellIdx const cellIdx)
        {
            uint32_t linearIndex = 0u;
            uint32_t stepWidth = 1u;
            constexpr CellIdx extent = Extent::toRT();

            constexpr uint8_t iExtent = dim;
            PMACC_UNROLL(iExtent)
            for(uint32_t i = 0u; i < iExtent; ++i)
            {
                if constexpr(picongpu::atomicPhysics::debug::fieldEnergyUsedCache::CELL_INDEX_RANGE_CHECKS)
                    if(cellIdx[i] >= extent[i])
                    {
                        printf("atomicPhysics ERROR: out of range in linearIndex() call to FieldEnergyUsedCachein\n");
                        return 0u;
                    }

                linearIndex += stepWidth * cellIdx;
                stepWidth *= superCellSize[i];
            }

            return linearIndex;
        }

    public:
        /** add to cache entry using atomics
         *
         * @param worker object containing the device and block information
         * @param localCellIndex vector index of cell to add energyUsed to
         * @param energy energy used, in eV
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         */
        template<typename T_Worker>
        HDINLINE void add(T_Worker const& worker, CellIdx const localCellIndex, float_X const energyUsed)
        {
            alpaka::atomicAdd(
                worker.getAcc(),
                &(this->fieldEnergyUsed[linearIndex(localCellIndex)]),
                rate,
                ::alpaka::hierarchy::Threads{});
        }

        /** add to cache entry, no atomics
         *
         * @tparam T_ChooseTransitionGroup ChooseTransitionGroup to add rate to
         *
         * @param localCellIndex vector index of cell to add energyUsed to
         * @param energyUsed energy used, in eV
         *
         * @attention no range checks outside a debug compile, invalid memory write on failure
         * @attention only use if only ever one thread accesses each rate cache entry!
         */
        template<particles::atomicPhysics::enums::ChooseTransitionGroup T_ChooseTransitionGroup>
        HDINLINE void add(CellIdx const localCellIndex, float_X const energyUsed)
        {
            rateEntries[linearIndex(localCellIndex)] += energyUsed;
            return;
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
