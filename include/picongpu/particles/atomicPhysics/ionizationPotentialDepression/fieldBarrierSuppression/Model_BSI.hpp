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

//! @file implementation of ionization potential depression(IPD) according to barrier suppression ionization model

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/FieldIPDModel.hpp"

namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::FieldBarrierSuppressionIPD
{
    struct Model_BSI : FieldIPDModel
    {
        /** calculate IPD due to the electric field
         *
         * @param eFieldNorm, in internal units
         * @param screenedCharge, in e
         *
         * @return unit: eV, not weighted
         */
        HDINLINE static float_X calculateIPD(float_X const eFieldNorm, float_X const screenedCharge)
        {
            return -2._X * math::sqrt(screenedCharge * eFieldNorm / picongpu::ATOMIC_UNIT_EFIELD);
        }
    } // namespace picongpu::particles::atomicPhysics::ionizationPotentialDepression::ipdModel
