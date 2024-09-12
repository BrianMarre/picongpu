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

//! @file interface for all electromagnetic field ionization potential depression(IPD) models

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/ipdModel/IPDModel.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::ipdModel
{
    struct FieldIPDModel
    {
        /** interface for calculating ionization potential depression(IPD)
         *
         * needs to implemented for each model
         *
         * @tparam T_IPDInput list of parameters of IPD calculation
         *
         * @param eFieldNorm norm of the electric field, internal units
         *
         * @return unit: eV, not weighted
         */
        template<typename... T_IPDInput>
        HDINLINE static float_X calculateIPD(float_X const eFieldNorm, T_IPDInput const... ipdInput);
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::ipdModel
