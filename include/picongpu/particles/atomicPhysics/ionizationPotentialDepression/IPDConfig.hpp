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

#include "picongpu/defines.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
{
    /** user facing and .param file save configuration of the IPD model
     *
     * @tparam T_MatterIPDModel model to use for matter contribution of IPD in the atomicPhysics stage
     * @tparam T_FieldIPDModel  model to use for field contribution of IPD in the atomicPhysics stage
     */
    template<typename T_MatterIPDModel, typename T_FieldIPDModel>
    struct IPDConfig
    {
        using MatterIPDModel = T_MatterIPDModel;
        using FieldIPDModel = T_FieldIPDModel;
    };
} // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression
