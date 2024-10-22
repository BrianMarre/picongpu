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

/** @file implements a super cell local cache of no-change transition rates for each
 *      atomic state of a species.
 *
 * no-change atomic physics transition rates(diagonal elements of rate matrix) are expensive
 *  to calculate and all have to be calculated anyway for the adaptive time step calculation.
 *
 * Therefore the are only calculated for all atomic states once per atomicPhysics substep
 *  and cached for use in the rate solver.
 *
 * Since no-change transition rates depend on the local electron spectrum, as well as all
 *  transition's parameters, they are super cell local, same as the electron spectrum.
 */

#pragma once

#include "picongpu/particles/atomicPhysics/SuperCellField.hpp"
#include "picongpu/particles/atomicPhysics/localHelperFields/FieldEnergyUsedCache.hpp"

#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <cstdint>
#include <string>

namespace picongpu::particles::atomicPhysics::localHelperFields
{
    //! debug only, write fieldEnergyUseCache to console
    struct PrintFieldEnergyUseCacheToConsole
    {
        // cpu version
        template<
            typename T_Acc,
            typename T_FieldEnergyUseCache HDINLINE auto operator()(
                T_Acc const&,
                T_FieldEnergyUseCache const& fieldEnergyUseCache,
                pmacc::DataSpace<picongpu::simDim> superCellIdx)
                const->std::enable_if_t<std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
            std::cout << "FieldEnergyUseCache: " << superCellFieldIdx.toString(",", "[]") << " (cell): energyUsed [eV]"
                      << std::endl;
            for(uint32_t linearIndex = 0u; linearIndex < T_FieldEnergyUseCache::numberCells; ++linearIndex)
            {
                std::cout << "\t"
                          << pmacc::math::mapToND(T_FieldEnergyUseCache::Extent::toRT(), static_cast<int>(linearIndex))
                                 .toString(",", "()");
                std::cout << ": " << fieldEnergyUseCache.energyUsed(linearIndex) << std::endl;
            }
        }

        // gpu version, does nothing
        template<typename T_Acc>
        HDINLINE auto operator()(T_Acc const&, float_X const timeStep, pmacc::DataSpace<picongpu::simDim> superCellIdx)
            const -> std::enable_if_t<!std::is_same_v<alpaka::Dev<T_Acc>, alpaka::DevCpu>>
        {
        }
    };

    /** superCell field of the per cell field energy use.
     *
     * @tparam T_MappingDescription description of local mapping from device to grid
     */
    template<typename T_MappingDescription>
    struct FieldEnergyUsedCacheField
        : public SuperCellField<
              FieldEnergyUsedCache<picongpu::SuperCellSize, float_X>,
              T_MappingDescription,
              false /*no guards*/>
    {
        FieldEnergyUsedCacheField(T_MappingDescription const& mappingDesc)
            : SuperCellField<
                FieldEnergyUsedCache<picongpu::SuperCellSize, float_X>,
                T_MappingDescription,
                false /*no guards*/>(mappingDesc)
        {
        }

        // required by ISimulationData
        std::string getUniqueId() override
        {
            return "FieldEnergyUseCacheField";
        }
    };
} // namespace picongpu::particles::atomicPhysics::localHelperFields
