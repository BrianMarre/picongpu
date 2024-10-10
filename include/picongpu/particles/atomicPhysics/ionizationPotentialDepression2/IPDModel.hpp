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

//! @file implements generic kernel for filling the superCell ionization potential depression(IPD) superCell field

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2
{
    namespace detail
    {
        struct InputStruct
        {
            // in an implementation add all input parameters of the IPD Model as public members here
        };

        struct AccumulationStruct
        {
            // add all values we accumulate over all particles of the super Cell
        };
    } // namespace detail

    //! interface for IPDModels
    struct IPDModel
    {
        using SuperCellInputStruct = InputStruct;
        using SuperCellAccumulationStruct = AccumulationStruct;

        HDINLINE static float_X calculateSuperCellIPD(IPDSuperCellInputStruct const& input);
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression2
