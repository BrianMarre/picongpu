/* Copyright 2024 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software you can redistribute it and or modify
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

namespace picongpu::particles::creation::moduleInterfaces
{
    //! interface of SharedDataBoxIndexFunctor
    template<typename... T_KernelConfigOptions>
    struct SharedDataBoxIndexFunctor
    {
        //! may be overwritten by implementation
        using indexDim = picongpu::simDim;

        template<typename T_AreaMapping>
        HDINLINE static pmacc::DataSpace<indexDim> getIndex(
            T_AreaMapping const areaMapping,
            pmacc::DataSpace<picongpu::simDim> const superCellIdx);
    };
} // namespace picongpu::particles::creation::moduleInterfaces
